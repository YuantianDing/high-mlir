//! `vector` dialect.
//!
//! Each operation is built by the function of its name (`vector::broadcast`,
//! `vector::transfer_read`, ...), with a `try_` variant returning a
//! [`TypeError`]. Accesses to memrefs (`load`, `gather`, `transfer_write`,
//! ...) are emitted in build order like every operation; on tensors they
//! return the new tensor.

use std::fmt::{self, Write};

use crate::{
    AffineMap, Attr, BuiltRegion, Closure, ClosureBuilder, Edges, HighMlir,
    IndexType, MemRefType, OpPrinter, TensorType, Ty, TypeError, Val, VectorType,
    arith::{FastMath, OpFlags, Operand, Operands, predicates},
    dialect::util::{check_rank, expect, indices, memref_ty, types, value_ty, vector_ty},
    memref::{Access, alignment_attr, check_alignment},
    mixed::{MixedList, MixedValues},
    types::{I1, I64, INDEX, TypeList},
    val::{OrPanic, Tuple},
};

predicates! {
    /// How `reduction`, `contract`, `scan`, ... combine elements, in MLIR's numbering.
    CombiningKind {
        Add = "add", Mul = "mul", MinUI = "minui", MinSI = "minsi", MinNumF = "minnumf", MaxUI = "maxui",
        MaxSI = "maxsi", MaxNumF = "maxnumf", And = "and", Or = "or", Xor = "xor", MinimumF = "minimumf",
        MaximumF = "maximumf",
    }
}

predicates! {
    /// The iterator type of a `vector.contract` dimension.
    IteratorType {
        Parallel = "parallel", Reduction = "reduction",
    }
}

predicates! {
    /// What `vector.print` prints after its value, in MLIR's numbering.
    Punctuation {
        NoPunctuation = "no_punctuation", NewLine = "newline", Comma = "comma", Open = "open", Close = "close",
    }
}

#[allow(clippy::derivable_impls)]
impl Default for Punctuation {
    fn default() -> Punctuation {
        Punctuation::NewLine
    }
}

impl From<()> for Punctuation {
    fn from(_: ()) -> Punctuation {
        Punctuation::NewLine
    }
}

fn kind_attr(kind: CombiningKind) -> (String, Attr) {
    ("kind".into(), Attr::Raw(format!("#vector.kind<{}>", kind.keyword())))
}

/// Checks that `kind` combines elements of type `element`.
fn check_kind(op: &str, kind: CombiningKind, element: &Ty) -> Result<(), TypeError> {
    use CombiningKind::*;
    let int = element.int_width().is_some() || element.is_index();
    let float = element.float_kind().is_some();
    let ok = match kind {
        Add | Mul => int || float,
        MinUI | MinSI | MaxUI | MaxSI | And | Or | Xor => int,
        MinNumF | MaxNumF | MinimumF | MaximumF => float,
    };
    expect(op, ok, || format!("`{}` does not combine `{element}`", kind.keyword()))
}

fn i64_list(values: &[i64]) -> String {
    format!("[{}]", values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "))
}

/// A type that must be a vector.
fn as_vector(op: &str, ty: &Ty) -> Result<VectorType, TypeError> {
    ty.as_vector().cloned().ok_or_else(|| TypeError::new(op, format!("expected a vector type, not `{ty}`")))
}

/// The number of elements of a fixed-length vector.
fn num_elements(v: &VectorType) -> u64 {
    v.dims().iter().product()
}

/// `vector.vscale`: the runtime multiple of scalable dimensions.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VectorScale;

/// `vector.vscale`
#[track_caller]
pub fn vscale() -> Val<IndexType> {
    Val::new_typed(VectorScale).or_panic()
}

impl HighMlir for VectorScale {
    fn name(&self) -> &str {
        "vector.vscale"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![INDEX.into()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        p.write_str("vector.vscale")
    }
}

/// `vector.step`: `[0, 1, 2, ..]` as a 1-D vector of `index`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Step {
    pub ty: Ty,
}

/// `vector.step` of type `ty`, a 1-D vector of `index`.
#[track_caller]
pub fn try_step(ty: impl Into<Ty>) -> Result<Val<VectorType>, TypeError> {
    let ty = ty.into();
    let v = as_vector("vector.step", &ty)?;
    expect("vector.step", v.rank() == 1 && v.element.is_index(), || format!("`{ty}` is not a 1-D vector of `index`"))?;
    Val::new_typed(Step { ty })
}

/// `vector.step` of type `ty`, a 1-D vector of `index`.
#[track_caller]
pub fn step(ty: impl Into<Ty>) -> Val<VectorType> {
    try_step(ty).or_panic()
}

impl HighMlir for Step {
    fn name(&self) -> &str {
        "vector.step"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.step : {}", self.ty)
    }
}

/// `$name %source : type to type` operations converting a value to `ty`.
macro_rules! conversion_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident, $check:ident;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub source: Val,
            pub ty: Ty,
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(source: impl Into<Val>, ty: impl Into<Ty>) -> Result<Val<VectorType>, TypeError> {
            let name = concat!("vector.", stringify!($f));
            let (source, ty) = (source.into(), ty.into());
            let src = value_ty(name, &source)?;
            let dst = as_vector(name, &ty)?;
            expect(name, $check(&src, &dst), || format!("cannot convert `{src}` to `{ty}`"))?;
            Val::new_typed($op { source, ty })
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f(source: impl Into<Val>, ty: impl Into<Ty>) -> Val<VectorType> {
            $try_f(source, ty).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("vector.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(std::iter::once(self.source.clone()))
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![self.ty.clone()]
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                write!(p, "{} {} : {} to {}", self.name(), p.operand(0), p.operand_type(0), self.ty)
            }
        }
    )*};
}

/// Same shape but the last dimension, which holds as many bits.
fn bitcast_ok(src: &Ty, dst: &VectorType) -> bool {
    let Some(s) = src.as_vector() else { return false };
    let (sd, dd) = (s.dims(), dst.dims());
    let bits = |v: &VectorType, d: &[u64]| v.element.bit_width().zip(d.last()).map(|(w, n)| w as u64 * n);
    s.rank() == dst.rank()
        && s.rank() > 0
        && sd[..sd.len() - 1] == dd[..dd.len() - 1]
        && s.scalable_dims() == dst.scalable_dims()
        && bits(s, &sd).is_some()
        && bits(s, &sd) == bits(dst, &dd)
}

/// A scalar of the element type, or a vector matching the trailing dimensions.
fn broadcast_ok(src: &Ty, dst: &VectorType) -> bool {
    match src.as_vector() {
        None => *src == dst.element,
        Some(s) => {
            let (sd, dd) = (s.dims(), dst.dims());
            s.element == dst.element
                && s.rank() <= dst.rank()
                && sd.iter().rev().zip(dd.iter().rev()).all(|(a, b)| *a == 1 || a == b)
        }
    }
}

/// The same elements in another shape.
fn shape_cast_ok(src: &Ty, dst: &VectorType) -> bool {
    src.as_vector().is_some_and(|s| {
        s.element == dst.element && num_elements(s) == num_elements(dst) && s.is_scalable() == dst.is_scalable()
    })
}

conversion_ops! {
    /// `vector.bitcast`: the same bits, with another element type.
    BitCast = bitcast / try_bitcast, bitcast_ok;
    /// `vector.broadcast` of a scalar or vector to a vector type.
    Broadcast = broadcast / try_broadcast, broadcast_ok;
    /// `vector.shape_cast`: the same elements in another shape.
    ShapeCast = shape_cast / try_shape_cast, shape_cast_ok;
}

/// `vector.type_cast`: a statically shaped memref as a 0-d memref of one vector.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TypeCast {
    pub memref: Val,
    pub ty: Ty,
}

/// `vector.type_cast` of a statically shaped, contiguous memref.
#[track_caller]
pub fn try_type_cast(memref: impl Into<Val>) -> Result<Val<MemRefType>, TypeError> {
    let name = "vector.type_cast";
    let memref = memref.into();
    let (src, m) = memref_ty(name, &memref)?;
    let shape: Option<Vec<u64>> = m.shape.iter().copied().collect();
    let ok = m.layout == crate::Layout::Identity && m.element.is_scalar();
    let Some(shape) = shape.filter(|_| ok) else {
        return Err(TypeError::new(name, format!("`{src}` is not a statically shaped, contiguous memref of scalars")));
    };
    let ty = MemRefType::new([], VectorType::new(shape, m.element.clone())).with_memory_space(m.memory_space).into();
    Val::new_typed(TypeCast { memref, ty })
}

/// `vector.type_cast` of a statically shaped, contiguous memref.
#[track_caller]
pub fn type_cast(memref: impl Into<Val>) -> Val<MemRefType> {
    try_type_cast(memref).or_panic()
}

impl HighMlir for TypeCast {
    fn name(&self) -> &str {
        "vector.type_cast"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.memref.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.type_cast {} : {} to {}", p.operand(0), p.operand_type(0), self.ty)
    }
}

fn check_mask_type(op: &str, ty: &Ty) -> Result<VectorType, TypeError> {
    let v = as_vector(op, ty)?;
    expect(op, v.element == I1, || format!("a mask must be a vector of `i1`, not `{ty}`"))?;
    Ok(v)
}

/// `vector.constant_mask`: the leading `sizes[i]` elements of each dimension are set.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ConstantMask {
    pub sizes: Vec<i64>,
    pub ty: Ty,
}

/// `vector.constant_mask` of type `ty`, a vector of `i1`.
#[track_caller]
pub fn try_constant_mask(sizes: impl IntoIterator<Item = i64>, ty: impl Into<Ty>) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.constant_mask";
    let (sizes, ty): (Vec<i64>, Ty) = (sizes.into_iter().collect(), ty.into());
    let v = check_mask_type(name, &ty)?;
    let ok = sizes.len() == v.rank() && sizes.iter().zip(v.dims()).all(|(&s, d)| s >= 0 && s as u64 <= d);
    expect(name, ok, || format!("sizes {} do not fit `{ty}`", i64_list(&sizes)))?;
    Val::new_typed(ConstantMask { sizes, ty })
}

/// `vector.constant_mask` of type `ty`, a vector of `i1`.
#[track_caller]
pub fn constant_mask(sizes: impl IntoIterator<Item = i64>, ty: impl Into<Ty>) -> Val<VectorType> {
    try_constant_mask(sizes, ty).or_panic()
}

impl HighMlir for ConstantMask {
    fn name(&self) -> &str {
        "vector.constant_mask"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("mask_dim_sizes".into(), Attr::i64_array(self.sizes.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.constant_mask {} : {}", i64_list(&self.sizes), self.ty)
    }
}

/// `vector.create_mask`: the leading `sizes[i]` elements of each dimension are set.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CreateMask {
    pub sizes: Vec<Val>,
    pub ty: Ty,
}

/// `vector.create_mask` of type `ty`; literal sizes become `index` constants.
#[track_caller]
pub fn try_create_mask(sizes: impl Operands, ty: impl Into<Ty>) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.create_mask";
    let (sizes, ty) = (indices(name, sizes)?, ty.into());
    let v = check_mask_type(name, &ty)?;
    check_rank(name, &ty, v.rank(), sizes.len())?;
    Val::new_typed(CreateMask { sizes, ty })
}

/// `vector.create_mask` of type `ty`; literal sizes become `index` constants.
#[track_caller]
pub fn create_mask(sizes: impl Operands, ty: impl Into<Ty>) -> Val<VectorType> {
    try_create_mask(sizes, ty).or_panic()
}

impl HighMlir for CreateMask {
    fn name(&self) -> &str {
        "vector.create_mask"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.sizes.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.create_mask {} : {}", p.operands(0..self.sizes.len()), self.ty)
    }
}

/// `vector.extract` of the element or sub-vector at `position`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Extract {
    pub source: Val,
    pub position: MixedValues,
}


impl Val<VectorType> {
    /// `vector.extract` from this vector; literal positions are static.
    #[track_caller]
    pub fn extract(&self, position: impl MixedList) -> Val {
        extract(self, position)
    }

    /// `vector.insert` of `value` into this vector; a literal takes the element type.
    #[track_caller]
    pub fn insert(&self, value: impl Operand, position: impl MixedList) -> Val<VectorType> {
        insert(value, self, position)
    }

    /// `vector.bitcast` to a vector of the same bits.
    #[track_caller]
    pub fn bitcast(&self, ty: impl Into<Ty>) -> Val<VectorType> {
        bitcast(self, ty)
    }

    /// `vector.shape_cast` to the same elements in another shape.
    #[track_caller]
    pub fn shape_cast(&self, ty: impl Into<Ty>) -> Val<VectorType> {
        shape_cast(self, ty)
    }

    /// `vector.extract_strided_slice` of the leading dimensions.
    #[track_caller]
    pub fn extract_strided_slice(
        &self,
        offsets: impl IntoIterator<Item = i64>,
        sizes: impl IntoIterator<Item = i64>,
        strides: impl IntoIterator<Item = i64>,
    ) -> Val<VectorType> {
        extract_strided_slice(self, offsets, sizes, strides)
    }

    /// `vector.insert_strided_slice` of this vector into `dest`.
    #[track_caller]
    pub fn insert_strided_slice(
        &self,
        dest: impl Into<Val>,
        offsets: impl IntoIterator<Item = i64>,
        strides: impl IntoIterator<Item = i64>,
    ) -> Val<VectorType> {
        insert_strided_slice(self, dest, offsets, strides)
    }

    /// `vector.fma`: `self * rhs + acc`.
    #[track_caller]
    pub fn fma(&self, rhs: impl Into<Val>, acc: impl Into<Val>) -> Val<VectorType> {
        fma(self, rhs, acc)
    }

    /// `vector.to_elements`: the elements in row-major order.
    #[track_caller]
    pub fn to_elements(&self) -> Tuple {
        to_elements(self)
    }

    /// `vector.shuffle` with `other` by `mask`.
    #[track_caller]
    pub fn shuffle(&self, other: impl Into<Val>, mask: impl IntoIterator<Item = i64>) -> Val<VectorType> {
        shuffle(self, other, mask)
    }

    /// `vector.interleave` with `other`.
    #[track_caller]
    pub fn interleave(&self, other: impl Into<Val>) -> Val<VectorType> {
        interleave(self, other)
    }

    /// `vector.deinterleave`: the even and the odd elements.
    #[track_caller]
    pub fn deinterleave(&self) -> (Val<VectorType>, Val<VectorType>) {
        deinterleave(self)
    }

    /// `vector.transpose`: dimension `i` of the result is dimension `permutation[i]`.
    #[track_caller]
    pub fn transpose(&self, permutation: impl IntoIterator<Item = i64>) -> Val<VectorType> {
        transpose(self, permutation)
    }

    /// `vector.reduction` of all elements (and `acc`) to a scalar.
    #[track_caller]
    pub fn reduction(&self, kind: CombiningKind, acc: Option<Val>, fastmath: impl Into<FastMath>) -> Val {
        reduction(kind, self, acc, fastmath)
    }

    /// `vector.multi_reduction` of the dimensions `dims` into `acc`.
    #[track_caller]
    pub fn multi_reduction(&self, kind: CombiningKind, acc: impl Into<Val>, dims: impl IntoIterator<Item = i64>) -> Val {
        multi_reduction(kind, self, acc, dims)
    }

    /// `vector.scan` along `reduction_dim`: the scan and the accumulated value.
    #[track_caller]
    pub fn scan(
        &self,
        kind: CombiningKind,
        initial_value: impl Into<Val>,
        reduction_dim: i64,
        inclusive: bool,
    ) -> (Val<VectorType>, Val) {
        scan(kind, self, initial_value, reduction_dim, inclusive)
    }

    /// `vector.outerproduct` with `rhs`, combined with `acc` by `kind`.
    #[track_caller]
    pub fn outerproduct(&self, rhs: impl Into<Val>, acc: Option<Val>, kind: CombiningKind) -> Val<VectorType> {
        outerproduct(self, rhs, acc, kind)
    }

    /// `vector.scalable.extract` of a fixed-length `ty` at `pos`.
    #[track_caller]
    pub fn scalable_extract(&self, pos: i64, ty: impl Into<Ty>) -> Val<VectorType> {
        scalable_extract(self, pos, ty)
    }

    /// `vector.scalable.insert` of this vector into `dest` at `pos`.
    #[track_caller]
    pub fn scalable_insert(&self, dest: impl Into<Val>, pos: i64) -> Val<VectorType> {
        scalable_insert(self, dest, pos)
    }

    /// `vector.store` of this vector into `base[indices]`.
    #[track_caller]
    pub fn store(&self, base: impl Into<Val>, indices: impl Operands) {
        store(self, base, indices, ())
    }
}

/// Vector accesses to memrefs.
impl Val<MemRefType> {
    /// `vector.type_cast`: this statically shaped memref as a 0-d memref of one vector.
    #[track_caller]
    pub fn type_cast(&self) -> Val<MemRefType> {
        type_cast(self)
    }

    /// `vector.load` of a vector `ty` from `self[indices]`.
    #[track_caller]
    pub fn load_vector(&self, indices: impl Operands, ty: impl Into<Ty>) -> Val<VectorType> {
        load(self, indices, ty, ())
    }

    /// `vector.transfer_read` of a vector `ty` from `self[indices]`.
    #[track_caller]
    pub fn transfer_read(
        &self,
        indices: impl Operands,
        padding: impl Operand,
        ty: impl Into<Ty>,
        opts: impl Into<Transfer>,
    ) -> Val<VectorType> {
        transfer_read(self, indices, padding, ty, opts)
    }

    /// `vector.transfer_write` of `value` into `self[indices]`.
    #[track_caller]
    pub fn transfer_write(&self, value: impl Into<Val>, indices: impl Operands, opts: impl Into<Transfer>) {
        transfer_write(value, self, indices, opts);
    }
}

/// Operations on values of any type.
impl Val {
    /// `vector.broadcast` of this scalar or vector to the vector type `ty`.
    #[track_caller]
    pub fn broadcast(&self, ty: impl Into<Ty>) -> Val<VectorType> {
        broadcast(self, ty)
    }

    /// `vector.print` of this value followed by `punctuation`.
    #[track_caller]
    pub fn print(&self, punctuation: impl Into<Punctuation>) {
        print(self, punctuation)
    }
}

/// `vector.extract`; literal positions are static.
#[track_caller]
pub fn try_extract(source: impl Into<Val>, position: impl MixedList) -> Result<Val, TypeError> {
    let name = "vector.extract";
    let source = source.into();
    let (ty, v) = vector_ty(name, &source)?;
    let position = MixedValues::new(name, position)?;
    expect(name, position.len() <= v.rank(), || format!("{} positions into `{ty}`", position.len()))?;
    Val::single(Extract { source, position })
}

/// `vector.extract`; literal positions are static.
#[track_caller]
pub fn extract(source: impl Into<Val>, position: impl MixedList) -> Val {
    try_extract(source, position).or_panic()
}

impl HighMlir for Extract {
    fn name(&self) -> &str {
        "vector.extract"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()).chain(self.position.dynamic.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.source.ty().as_vector().unwrap().trailing(self.position.len())]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("static_position".into(), Attr::i64_array(self.position.statics.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let pos = self.position.format(|k| p.operand(1 + k));
        let result = self.result_types().remove(0);
        write!(p, "vector.extract {}{pos} : {result} from {}", p.operand(0), p.operand_type(0))
    }
}

/// `vector.insert` of an element or sub-vector at `position`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Insert {
    pub value: Val,
    pub dest: Val,
    pub position: MixedValues,
}

/// `vector.insert`; a literal value takes the element type, literal positions are static.
#[track_caller]
pub fn try_insert(value: impl Operand, dest: impl Into<Val>, position: impl MixedList) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.insert";
    let dest = dest.into();
    let (ty, v) = vector_ty(name, &dest)?;
    let position = MixedValues::new(name, position)?;
    expect(name, position.len() <= v.rank(), || format!("{} positions into `{ty}`", position.len()))?;
    let expected = v.trailing(position.len());
    let value = value.into_val(&expected).map_err(|e| TypeError::new(name, e.message))?;
    let value_ty = value_ty(name, &value)?;
    expect(name, value_ty == expected, || format!("cannot insert `{value_ty}` at {} positions of `{ty}`", position.len()))?;
    Val::new_typed(Insert { value, dest, position })
}

/// `vector.insert`; a literal value takes the element type, literal positions are static.
#[track_caller]
pub fn insert(value: impl Operand, dest: impl Into<Val>, position: impl MixedList) -> Val<VectorType> {
    try_insert(value, dest, position).or_panic()
}

impl HighMlir for Insert {
    fn name(&self) -> &str {
        "vector.insert"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.value.clone(), self.dest.clone()].into_iter().chain(self.position.dynamic.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.dest.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("static_position".into(), Attr::i64_array(self.position.statics.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let pos = self.position.format(|k| p.operand(2 + k));
        let (v, d) = (p.operand(0), p.operand(1));
        write!(p, "vector.insert {v}, {d} {pos} : {} into {}", p.operand_type(0), p.operand_type(1))
    }
}

/// `vector.extract_strided_slice` of the leading dimensions.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ExtractStridedSlice {
    pub source: Val,
    pub offsets: Vec<i64>,
    pub sizes: Vec<i64>,
    pub strides: Vec<i64>,
}

/// `vector.extract_strided_slice` of the leading `offsets.len()` dimensions.
#[track_caller]
pub fn try_extract_strided_slice(
    source: impl Into<Val>,
    offsets: impl IntoIterator<Item = i64>,
    sizes: impl IntoIterator<Item = i64>,
    strides: impl IntoIterator<Item = i64>,
) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.extract_strided_slice";
    let source = source.into();
    let (ty, v) = vector_ty(name, &source)?;
    let (offsets, sizes, strides): (Vec<i64>, Vec<i64>, Vec<i64>) =
        (offsets.into_iter().collect(), sizes.into_iter().collect(), strides.into_iter().collect());
    let n = offsets.len();
    let fits = offsets.iter().zip(&sizes).zip(&strides).zip(v.dims()).all(|(((&o, &s), &t), d)| {
        o >= 0 && s > 0 && t == 1 && (o + s) as u64 <= d
    });
    expect(name, n <= v.rank() && sizes.len() == n && strides.len() == n && fits, || {
        format!("offsets {}, sizes {} and strides {} do not fit `{ty}`", i64_list(&offsets), i64_list(&sizes), i64_list(&strides))
    })?;
    Val::new_typed(ExtractStridedSlice { source, offsets, sizes, strides })
}

/// `vector.extract_strided_slice` of the leading `offsets.len()` dimensions.
#[track_caller]
pub fn extract_strided_slice(
    source: impl Into<Val>,
    offsets: impl IntoIterator<Item = i64>,
    sizes: impl IntoIterator<Item = i64>,
    strides: impl IntoIterator<Item = i64>,
) -> Val<VectorType> {
    try_extract_strided_slice(source, offsets, sizes, strides).or_panic()
}

impl HighMlir for ExtractStridedSlice {
    fn name(&self) -> &str {
        "vector.extract_strided_slice"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        let v = self.source.ty().as_vector().unwrap().clone();
        let dims: Vec<(u64, bool)> = v
            .dims()
            .into_iter()
            .enumerate()
            .map(|(i, d)| (self.sizes.get(i).map_or(d, |&s| s as u64), v.scalable_dims()[i]))
            .collect();
        vec![VectorType::scalable(dims, v.element.clone()).into()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![
            ("offsets".into(), Attr::i64_list(self.offsets.clone())),
            ("sizes".into(), Attr::i64_list(self.sizes.clone())),
            ("strides".into(), Attr::i64_list(self.strides.clone())),
        ]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (o, s, t) = (i64_list(&self.offsets), i64_list(&self.sizes), i64_list(&self.strides));
        let result = self.result_types().remove(0);
        let (src, src_ty) = (p.operand(0), p.operand_type(0));
        write!(p, "vector.extract_strided_slice {src} {{offsets = {o}, sizes = {s}, strides = {t}}} : {src_ty} to {result}")
    }
}

/// `vector.insert_strided_slice` of `value` into the trailing dimensions of `dest`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct InsertStridedSlice {
    pub value: Val,
    pub dest: Val,
    pub offsets: Vec<i64>,
    pub strides: Vec<i64>,
}

/// `vector.insert_strided_slice`; `offsets` index every dimension of `dest`.
#[track_caller]
pub fn try_insert_strided_slice(
    value: impl Into<Val>,
    dest: impl Into<Val>,
    offsets: impl IntoIterator<Item = i64>,
    strides: impl IntoIterator<Item = i64>,
) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.insert_strided_slice";
    let (value, dest) = (value.into(), dest.into());
    let ((vt, v), (dt, d)) = (vector_ty(name, &value)?, vector_ty(name, &dest)?);
    let (offsets, strides): (Vec<i64>, Vec<i64>) = (offsets.into_iter().collect(), strides.into_iter().collect());
    let ok = v.element == d.element && v.rank() <= d.rank() && offsets.len() == d.rank() && strides.len() == v.rank();
    expect(name, ok, || format!("cannot insert `{vt}` into `{dt}` at {}", i64_list(&offsets)))?;
    Val::new_typed(InsertStridedSlice { value, dest, offsets, strides })
}

/// `vector.insert_strided_slice`; `offsets` index every dimension of `dest`.
#[track_caller]
pub fn insert_strided_slice(
    value: impl Into<Val>,
    dest: impl Into<Val>,
    offsets: impl IntoIterator<Item = i64>,
    strides: impl IntoIterator<Item = i64>,
) -> Val<VectorType> {
    try_insert_strided_slice(value, dest, offsets, strides).or_panic()
}

impl HighMlir for InsertStridedSlice {
    fn name(&self) -> &str {
        "vector.insert_strided_slice"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.value.clone(), self.dest.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.dest.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![
            ("offsets".into(), Attr::i64_list(self.offsets.clone())),
            ("strides".into(), Attr::i64_list(self.strides.clone())),
        ]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (o, t) = (i64_list(&self.offsets), i64_list(&self.strides));
        let (v, d) = (p.operand(0), p.operand(1));
        let (vt, dt) = (p.operand_type(0), p.operand_type(1));
        write!(p, "vector.insert_strided_slice {v}, {d} {{offsets = {o}, strides = {t}}} : {vt} into {dt}")
    }
}

/// `vector.fma`: `lhs * rhs + acc` on float vectors.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Fma {
    pub lhs: Val,
    pub rhs: Val,
    pub acc: Val,
}

/// `vector.fma` of float vectors of one type.
#[track_caller]
pub fn try_fma(lhs: impl Into<Val>, rhs: impl Into<Val>, acc: impl Into<Val>) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.fma";
    let (lhs, rhs, acc) = (lhs.into(), rhs.into(), acc.into());
    let (ty, v) = vector_ty(name, &lhs)?;
    let same = value_ty(name, &rhs)? == ty && value_ty(name, &acc)? == ty;
    expect(name, same && v.element.float_kind().is_some(), || format!("operands must be float vectors of type `{ty}`"))?;
    Val::new_typed(Fma { lhs, rhs, acc })
}

/// `vector.fma` of float vectors of one type.
#[track_caller]
pub fn fma(lhs: impl Into<Val>, rhs: impl Into<Val>, acc: impl Into<Val>) -> Val<VectorType> {
    try_fma(lhs, rhs, acc).or_panic()
}

impl HighMlir for Fma {
    fn name(&self) -> &str {
        "vector.fma"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.lhs.clone(), self.rhs.clone(), self.acc.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.lhs.ty()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.fma {} : {}", p.operands(0..3), p.operand_type(0))
    }
}

/// `vector.from_elements`: a fixed-length vector of the elements in row-major order.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FromElements {
    pub elements: Vec<Val>,
    pub ty: Ty,
}

/// `vector.from_elements`; literal elements take the element type.
#[track_caller]
pub fn try_from_elements(elements: impl Operands, ty: impl Into<Ty>) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.from_elements";
    let ty = ty.into();
    let v = as_vector(name, &ty)?;
    let elements = elements.into_vals(&|_| Some(v.element.clone())).map_err(|e| TypeError::new(name, e.message))?;
    let n = num_elements(&v);
    let ok = !v.is_scalable() && elements.len() as u64 == n && types(name, &elements)?.iter().all(|t| *t == v.element);
    expect(name, ok, || format!("`{ty}` needs {n} elements of type `{}`", v.element))?;
    Val::new_typed(FromElements { elements, ty })
}

/// `vector.from_elements`; literal elements take the element type.
#[track_caller]
pub fn from_elements(elements: impl Operands, ty: impl Into<Ty>) -> Val<VectorType> {
    try_from_elements(elements, ty).or_panic()
}

impl HighMlir for FromElements {
    fn name(&self) -> &str {
        "vector.from_elements"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.elements.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.from_elements {} : {}", p.operands(0..self.elements.len()), self.ty)
    }
}

/// `vector.to_elements`: one result per element, in row-major order.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ToElements {
    pub source: Val,
}

/// `vector.to_elements` of a fixed-length vector: its elements in row-major order.
#[track_caller]
pub fn try_to_elements(source: impl Into<Val>) -> Result<Tuple, TypeError> {
    let name = "vector.to_elements";
    let source = source.into();
    let (ty, v) = vector_ty(name, &source)?;
    expect(name, !v.is_scalable(), || format!("`{ty}` is scalable"))?;
    Val::all_results(ToElements { source })
}

/// `vector.to_elements` of a fixed-length vector: its elements in row-major order.
#[track_caller]
pub fn to_elements(source: impl Into<Val>) -> Tuple {
    try_to_elements(source).or_panic()
}

impl HighMlir for ToElements {
    fn name(&self) -> &str {
        "vector.to_elements"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        let v = self.source.ty().as_vector().unwrap().clone();
        vec![v.element.clone(); num_elements(&v) as usize]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.to_elements {} : {}", p.operand(0), p.operand_type(0))
    }
}

/// `vector.shuffle`: picks leading slices of `v1` (`0..n`) and `v2` (`n..`) by `mask`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Shuffle {
    pub v1: Val,
    pub v2: Val,
    pub mask: Vec<i64>,
}

/// `vector.shuffle` of fixed-length vectors with the same trailing dimensions.
#[track_caller]
pub fn try_shuffle(v1: impl Into<Val>, v2: impl Into<Val>, mask: impl IntoIterator<Item = i64>) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.shuffle";
    let (v1, v2, mask): (Val, Val, Vec<i64>) = (v1.into(), v2.into(), mask.into_iter().collect());
    let ((t1, a), (t2, b)) = (vector_ty(name, &v1)?, vector_ty(name, &v2)?);
    let lead = |v: &VectorType| v.dims().first().copied().unwrap_or(1) as i64;
    let total = lead(&a) + lead(&b);
    let ok = a.rank() == b.rank()
        && a.trailing(1) == b.trailing(1)
        && !a.is_scalable()
        && !b.is_scalable()
        && !mask.is_empty()
        && mask.iter().all(|&m| m == -1 || (0..total).contains(&m));
    expect(name, ok, || format!("cannot shuffle `{t1}` and `{t2}` with {}", i64_list(&mask)))?;
    Val::new_typed(Shuffle { v1, v2, mask })
}

/// `vector.shuffle` of fixed-length vectors with the same trailing dimensions.
#[track_caller]
pub fn shuffle(v1: impl Into<Val>, v2: impl Into<Val>, mask: impl IntoIterator<Item = i64>) -> Val<VectorType> {
    try_shuffle(v1, v2, mask).or_panic()
}

impl HighMlir for Shuffle {
    fn name(&self) -> &str {
        "vector.shuffle"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.v1.clone(), self.v2.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        let v = self.v1.ty().as_vector().unwrap().clone();
        let rest = v.dims().into_iter().skip(1);
        vec![VectorType::new(std::iter::once(self.mask.len() as u64).chain(rest), v.element).into()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("mask".into(), Attr::i64_array(self.mask.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (a, b) = (p.operand(0), p.operand(1));
        let types = TypeList(&p.operand_types(0..2)).to_string();
        write!(p, "vector.shuffle {a}, {b} {} : {types}", i64_list(&self.mask))
    }
}

/// The vector with its last dimension multiplied by `num` and divided by `den`.
fn resize_last(v: &VectorType, num: u64, den: u64) -> Ty {
    let mut dims: Vec<(u64, bool)> = v.dims().into_iter().zip(v.scalable_dims().iter().copied()).collect();
    match dims.last_mut() {
        Some((d, _)) => *d = *d * num / den,
        None => dims.push((num, false)),
    }
    VectorType::scalable(dims, v.element.clone()).into()
}

/// `vector.interleave`: the elements of `lhs` and `rhs` alternately, along the last dimension.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Interleave {
    pub lhs: Val,
    pub rhs: Val,
}

/// `vector.interleave` of two vectors of one type.
#[track_caller]
pub fn try_interleave(lhs: impl Into<Val>, rhs: impl Into<Val>) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.interleave";
    let (lhs, rhs) = (lhs.into(), rhs.into());
    let (ty, _) = vector_ty(name, &lhs)?;
    let rt = value_ty(name, &rhs)?;
    expect(name, rt == ty, || format!("operand types differ: `{ty}` and `{rt}`"))?;
    Val::new_typed(Interleave { lhs, rhs })
}

/// `vector.interleave` of two vectors of one type.
#[track_caller]
pub fn interleave(lhs: impl Into<Val>, rhs: impl Into<Val>) -> Val<VectorType> {
    try_interleave(lhs, rhs).or_panic()
}

impl HighMlir for Interleave {
    fn name(&self) -> &str {
        "vector.interleave"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.lhs.clone(), self.rhs.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![resize_last(self.lhs.ty().as_vector().unwrap(), 2, 1)]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let result = self.result_types().remove(0);
        write!(p, "vector.interleave {}, {} : {} -> {result}", p.operand(0), p.operand(1), p.operand_type(0))
    }
}

/// `vector.deinterleave`: the even and odd elements along the last dimension.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Deinterleave {
    pub source: Val,
}

/// `vector.deinterleave` of a vector with an even last dimension; returns
/// the even and the odd elements.
#[track_caller]
pub fn try_deinterleave(source: impl Into<Val>) -> Result<(Val<VectorType>, Val<VectorType>), TypeError> {
    let name = "vector.deinterleave";
    let source = source.into();
    let (ty, v) = vector_ty(name, &source)?;
    expect(name, v.dims().last().is_some_and(|d| d % 2 == 0), || format!("`{ty}` has no even last dimension"))?;
    let [first, second] = Val::all_results(Deinterleave { source })?.try_into().unwrap();
    Ok((first.downcast().unwrap(), second.downcast().unwrap()))
}

/// `vector.deinterleave` of a vector with an even last dimension; returns
/// the even and the odd elements.
#[track_caller]
pub fn deinterleave(source: impl Into<Val>) -> (Val<VectorType>, Val<VectorType>) {
    try_deinterleave(source).or_panic()
}

impl HighMlir for Deinterleave {
    fn name(&self) -> &str {
        "vector.deinterleave"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        let half = resize_last(self.source.ty().as_vector().unwrap(), 1, 2);
        vec![half.clone(), half]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let result = self.result_types().remove(0);
        write!(p, "vector.deinterleave {} : {} -> {result}", p.operand(0), p.operand_type(0))
    }
}

/// `vector.transpose`: dimension `i` of the result is dimension `permutation[i]`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Transpose {
    pub vector: Val,
    pub permutation: Vec<i64>,
}

/// `vector.transpose`
#[track_caller]
pub fn try_transpose(vector: impl Into<Val>, permutation: impl IntoIterator<Item = i64>) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.transpose";
    let (vector, permutation): (Val, Vec<i64>) = (vector.into(), permutation.into_iter().collect());
    let (ty, v) = vector_ty(name, &vector)?;
    let mut sorted = permutation.clone();
    sorted.sort();
    expect(name, sorted == (0..v.rank() as i64).collect::<Vec<_>>(), || {
        format!("{} is not a permutation of the dimensions of `{ty}`", i64_list(&permutation))
    })?;
    Val::new_typed(Transpose { vector, permutation })
}

/// `vector.transpose`
#[track_caller]
pub fn transpose(vector: impl Into<Val>, permutation: impl IntoIterator<Item = i64>) -> Val<VectorType> {
    try_transpose(vector, permutation).or_panic()
}

impl HighMlir for Transpose {
    fn name(&self) -> &str {
        "vector.transpose"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.vector.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        let v = self.vector.ty().as_vector().unwrap().clone();
        let (dims, scalable) = (v.dims(), v.scalable_dims());
        let permuted = self.permutation.iter().map(|&d| (dims[d as usize], scalable[d as usize]));
        vec![VectorType::scalable(permuted, v.element.clone()).into()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("permutation".into(), Attr::i64_array(self.permutation.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let result = self.result_types().remove(0);
        let perm = i64_list(&self.permutation);
        write!(p, "vector.transpose {}, {perm} : {} to {result}", p.operand(0), p.operand_type(0))
    }
}

/// `vector.reduction` of all elements, optionally into an accumulator.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Reduction {
    pub kind: CombiningKind,
    pub vector: Val,
    pub acc: Option<Val>,
    pub fastmath: FastMath,
}

/// `vector.reduction` of all elements of `vector` (and `acc`) to a scalar.
#[track_caller]
pub fn try_reduction(
    kind: CombiningKind,
    vector: impl Into<Val>,
    acc: Option<Val>,
    fastmath: impl Into<FastMath>,
) -> Result<Val, TypeError> {
    let name = "vector.reduction";
    let vector = vector.into();
    let (_, v) = vector_ty(name, &vector)?;
    check_kind(name, kind, &v.element)?;
    if let Some(a) = &acc {
        let t = value_ty(name, a)?;
        expect(name, t == v.element, || format!("the accumulator must be a `{}`, not `{t}`", v.element))?;
    }
    Val::single(Reduction { kind, vector, acc, fastmath: fastmath.into() })
}

/// `vector.reduction` of all elements of `vector` (and `acc`) to a scalar.
#[track_caller]
pub fn reduction(kind: CombiningKind, vector: impl Into<Val>, acc: Option<Val>, fastmath: impl Into<FastMath>) -> Val {
    try_reduction(kind, vector, acc, fastmath).or_panic()
}

impl HighMlir for Reduction {
    fn name(&self) -> &str {
        "vector.reduction"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.vector.clone()).chain(self.acc.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.vector.ty().element().clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = vec![kind_attr(self.kind)];
        attrs.extend(self.fastmath.attributes());
        attrs
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.reduction <{}>, {}", self.kind.keyword(), p.operand(0))?;
        if self.acc.is_some() {
            write!(p, ", {}", p.operand(1))?;
        }
        let result = self.result_types().remove(0);
        write!(p, "{} : {} into {result}", self.fastmath.suffix(), p.operand_type(0))
    }
}

/// `vector.multi_reduction` of the dimensions `dims` into `acc`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MultiReduction {
    pub kind: CombiningKind,
    pub source: Val,
    pub acc: Val,
    pub dims: Vec<i64>,
}

/// `vector.multi_reduction`; `acc` has the type of `source` without `dims`.
#[track_caller]
pub fn try_multi_reduction(
    kind: CombiningKind,
    source: impl Into<Val>,
    acc: impl Into<Val>,
    dims: impl IntoIterator<Item = i64>,
) -> Result<Val, TypeError> {
    let name = "vector.multi_reduction";
    let (source, acc, dims): (Val, Val, Vec<i64>) = (source.into(), acc.into(), dims.into_iter().collect());
    let (ty, v) = vector_ty(name, &source)?;
    check_kind(name, kind, &v.element)?;
    let kept: Vec<(u64, bool)> = (0..v.rank())
        .filter(|d| !dims.contains(&(*d as i64)))
        .map(|d| (v.dims()[d], v.scalable_dims()[d]))
        .collect();
    let expected: Ty = if kept.is_empty() { v.element.clone() } else { VectorType::scalable(kept, v.element.clone()).into() };
    let acc_ty = value_ty(name, &acc)?;
    let ok = dims.iter().all(|&d| (0..v.rank() as i64).contains(&d)) && acc_ty == expected;
    expect(name, ok, || format!("reducing {} of `{ty}` gives `{expected}`, not `{acc_ty}`", i64_list(&dims)))?;
    Val::single(MultiReduction { kind, source, acc, dims })
}

/// `vector.multi_reduction`; `acc` has the type of `source` without `dims`.
#[track_caller]
pub fn multi_reduction(kind: CombiningKind, source: impl Into<Val>, acc: impl Into<Val>, dims: impl IntoIterator<Item = i64>) -> Val {
    try_multi_reduction(kind, source, acc, dims).or_panic()
}

impl HighMlir for MultiReduction {
    fn name(&self) -> &str {
        "vector.multi_reduction"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.acc.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.acc.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![kind_attr(self.kind), ("reduction_dims".into(), Attr::i64_array(self.dims.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (k, s, a) = (self.kind.keyword(), p.operand(0), p.operand(1));
        let (st, at) = (p.operand_type(0), p.operand_type(1));
        write!(p, "vector.multi_reduction <{k}>, {s}, {a} {} : {st} to {at}", i64_list(&self.dims))
    }
}

/// `vector.scan`: prefix reductions along `reduction_dim`, and the total.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Scan {
    pub kind: CombiningKind,
    pub source: Val,
    pub initial_value: Val,
    pub reduction_dim: i64,
    pub inclusive: bool,
}

/// `vector.scan`; `initial_value` has the type of `source` without
/// `reduction_dim`. Returns the scan and the accumulated value.
#[track_caller]
pub fn try_scan(
    kind: CombiningKind,
    source: impl Into<Val>,
    initial_value: impl Into<Val>,
    reduction_dim: i64,
    inclusive: bool,
) -> Result<(Val<VectorType>, Val), TypeError> {
    let name = "vector.scan";
    let (source, initial_value) = (source.into(), initial_value.into());
    let (ty, v) = vector_ty(name, &source)?;
    check_kind(name, kind, &v.element)?;
    let (it, i) = vector_ty(name, &initial_value)?;
    let ok = (0..v.rank() as i64).contains(&reduction_dim) && i.rank() + 1 == v.rank() && i.element == v.element;
    expect(name, ok, || format!("cannot scan dimension {reduction_dim} of `{ty}` from `{it}`"))?;
    let [first, second] = Val::all_results(Scan { kind, source, initial_value, reduction_dim, inclusive })?.try_into().unwrap();
    Ok((first.downcast().unwrap(), second))
}

/// `vector.scan`; `initial_value` has the type of `source` without
/// `reduction_dim`. Returns the scan and the accumulated value.
#[track_caller]
pub fn scan(
    kind: CombiningKind,
    source: impl Into<Val>,
    initial_value: impl Into<Val>,
    reduction_dim: i64,
    inclusive: bool,
) -> (Val<VectorType>, Val) {
    try_scan(kind, source, initial_value, reduction_dim, inclusive).or_panic()
}

impl HighMlir for Scan {
    fn name(&self) -> &str {
        "vector.scan"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.initial_value.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.source.ty(), self.initial_value.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![
            kind_attr(self.kind),
            ("reduction_dim".into(), Attr::Int(self.reduction_dim, I64.into())),
            ("inclusive".into(), Attr::Bool(self.inclusive)),
        ]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (k, s, i) = (self.kind.keyword(), p.operand(0), p.operand(1));
        let (st, it) = (p.operand_type(0), p.operand_type(1));
        let attrs = format!("{{inclusive = {}, reduction_dim = {} : i64}}", self.inclusive, self.reduction_dim);
        write!(p, "vector.scan <{k}>, {s}, {i} {attrs} : {st}, {it}")
    }
}

/// `vector.outerproduct` of two 1-D vectors (or a vector and a scalar, an AXPY).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct OuterProduct {
    pub lhs: Val,
    pub rhs: Val,
    pub acc: Option<Val>,
    pub kind: CombiningKind,
}

/// `vector.outerproduct`, combining with `acc` by `kind`.
#[track_caller]
pub fn try_outerproduct(
    lhs: impl Into<Val>,
    rhs: impl Into<Val>,
    acc: Option<Val>,
    kind: CombiningKind,
) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.outerproduct";
    let (lhs, rhs) = (lhs.into(), rhs.into());
    let (lt, l) = vector_ty(name, &lhs)?;
    let rt = value_ty(name, &rhs)?;
    let ok = l.rank() == 1 && rt.element() == &l.element && rt.as_vector().is_none_or(|r| r.rank() == 1);
    expect(name, ok, || format!("cannot take the outer product of `{lt}` and `{rt}`"))?;
    check_kind(name, kind, &l.element)?;
    let op = OuterProduct { lhs, rhs, acc, kind };
    let result = op.result_types().remove(0);
    if let Some(a) = &op.acc {
        let at = value_ty(name, a)?;
        expect(name, at == result, || format!("the accumulator must be a `{result}`, not `{at}`"))?;
    }
    Val::new_typed(op)
}

/// `vector.outerproduct`, combining with `acc` by `kind`.
#[track_caller]
pub fn outerproduct(lhs: impl Into<Val>, rhs: impl Into<Val>, acc: Option<Val>, kind: CombiningKind) -> Val<VectorType> {
    try_outerproduct(lhs, rhs, acc, kind).or_panic()
}

impl HighMlir for OuterProduct {
    fn name(&self) -> &str {
        "vector.outerproduct"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.lhs.clone(), self.rhs.clone()].into_iter().chain(self.acc.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        let l = self.lhs.ty().as_vector().unwrap().clone();
        let mut dims: Vec<(u64, bool)> = vec![(l.dims()[0], l.scalable_dims()[0])];
        if let Some(r) = self.rhs.ty().as_vector() {
            dims.push((r.dims()[0], r.scalable_dims()[0]));
        }
        vec![VectorType::scalable(dims, l.element.clone()).into()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![kind_attr(self.kind)]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = 2 + self.acc.is_some() as usize;
        let types = TypeList(&p.operand_types(0..2)).to_string();
        write!(p, "vector.outerproduct {} {{kind = #vector.kind<{}>}} : {types}", p.operands(0..n), self.kind.keyword())
    }
}

/// `vector.contract`: a generalized matrix product; `indexing_maps` map the
/// iteration space to `lhs`, `rhs` and `acc`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Contract {
    pub lhs: Val,
    pub rhs: Val,
    pub acc: Val,
    pub indexing_maps: [AffineMap; 3],
    pub iterator_types: Vec<IteratorType>,
    pub kind: CombiningKind,
    pub fastmath: FastMath,
}

/// `vector.contract` of `lhs` and `rhs` into `acc`.
pub fn try_contract(
    lhs: impl Into<Val>,
    rhs: impl Into<Val>,
    acc: impl Into<Val>,
    indexing_maps: [AffineMap; 3],
    iterator_types: impl IntoIterator<Item = IteratorType>,
    kind: CombiningKind,
    fastmath: impl Into<FastMath>,
) -> Result<Val, TypeError> {
    let name = "vector.contract";
    let (lhs, rhs, acc) = (lhs.into(), rhs.into(), acc.into());
    let iterator_types: Vec<IteratorType> = iterator_types.into_iter().collect();
    let ((_, l), (_, r), at) = (vector_ty(name, &lhs)?, vector_ty(name, &rhs)?, value_ty(name, &acc)?);
    let ranks = [l.rank(), r.rank(), at.as_vector().map_or(0, VectorType::rank)];
    let ok = indexing_maps.iter().zip(ranks).all(|(m, rank)| m.dims as usize == iterator_types.len() && m.results.len() == rank);
    expect(name, ok, || "the indexing maps do not match the operands and iterators".into())?;
    check_kind(name, kind, at.element())?;
    Val::single(Contract { lhs, rhs, acc, indexing_maps, iterator_types, kind, fastmath: fastmath.into() })
}

/// `vector.contract` of `lhs` and `rhs` into `acc`.
#[track_caller]
pub fn contract(
    lhs: impl Into<Val>,
    rhs: impl Into<Val>,
    acc: impl Into<Val>,
    indexing_maps: [AffineMap; 3],
    iterator_types: impl IntoIterator<Item = IteratorType>,
    kind: CombiningKind,
    fastmath: impl Into<FastMath>,
) -> Val {
    try_contract(lhs, rhs, acc, indexing_maps, iterator_types, kind, fastmath).or_panic()
}

impl Contract {
    fn maps(&self) -> Attr {
        Attr::List(self.indexing_maps.iter().cloned().map(Attr::AffineMap).collect())
    }
}

impl HighMlir for Contract {
    fn name(&self) -> &str {
        "vector.contract"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.lhs.clone(), self.rhs.clone(), self.acc.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.acc.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let iterators = self
            .iterator_types
            .iter()
            .map(|t| Attr::Raw(format!("#vector.iterator_type<{}>", t.keyword())))
            .collect();
        let mut attrs = vec![
            ("indexing_maps".into(), self.maps()),
            ("iterator_types".into(), Attr::List(iterators)),
            kind_attr(self.kind),
        ];
        attrs.extend(self.fastmath.attributes());
        attrs
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let iterators: Vec<Attr> = self.iterator_types.iter().map(|t| Attr::str(t.keyword())).collect();
        write!(p, "vector.contract {{indexing_maps = {}, iterator_types = {}", self.maps(), Attr::List(iterators))?;
        write!(p, ", kind = #vector.kind<{}>", self.kind.keyword())?;
        if let Some((_, f)) = self.fastmath.attributes().pop() {
            write!(p, ", fastmath = {f}")?;
        }
        let (lt, rt, at) = (p.operand_type(0), p.operand_type(1), p.operand_type(2));
        write!(p, "}} {} : {lt}, {rt} into {at}", p.operands(0..3))
    }
}

/// `vector.scalable.extract` of a fixed-length vector at a multiple of its length.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ScalableExtract {
    pub source: Val,
    pub pos: i64,
    pub ty: Ty,
}

/// `vector.scalable.extract` of a 1-D vector `ty` from a 1-D scalable vector.
#[track_caller]
pub fn try_scalable_extract(source: impl Into<Val>, pos: i64, ty: impl Into<Ty>) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.scalable.extract";
    let (source, ty) = (source.into(), ty.into());
    let (st, s) = vector_ty(name, &source)?;
    let r = as_vector(name, &ty)?;
    let ok = s.is_scalable() && !r.is_scalable() && s.element == r.element && s.rank() == 1 && r.rank() == 1
        && pos % r.dims()[0] as i64 == 0;
    expect(name, ok, || format!("cannot extract `{ty}` at {pos} from `{st}`"))?;
    Val::new_typed(ScalableExtract { source, pos, ty })
}

/// `vector.scalable.extract` of a 1-D vector `ty` from a 1-D scalable vector.
#[track_caller]
pub fn scalable_extract(source: impl Into<Val>, pos: i64, ty: impl Into<Ty>) -> Val<VectorType> {
    try_scalable_extract(source, pos, ty).or_panic()
}

impl HighMlir for ScalableExtract {
    fn name(&self) -> &str {
        "vector.scalable.extract"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("pos".into(), Attr::Int(self.pos, I64.into()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.scalable.extract {}[{}] : {} from {}", p.operand(0), self.pos, self.ty, p.operand_type(0))
    }
}

/// `vector.scalable.insert` of a fixed-length vector at a multiple of its length.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ScalableInsert {
    pub value: Val,
    pub dest: Val,
    pub pos: i64,
}

/// `vector.scalable.insert` of a 1-D vector into a 1-D scalable vector.
#[track_caller]
pub fn try_scalable_insert(value: impl Into<Val>, dest: impl Into<Val>, pos: i64) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.scalable.insert";
    let (value, dest) = (value.into(), dest.into());
    let ((vt, v), (dt, d)) = (vector_ty(name, &value)?, vector_ty(name, &dest)?);
    let ok = d.is_scalable() && !v.is_scalable() && v.element == d.element && v.rank() == 1 && d.rank() == 1
        && pos % v.dims()[0] as i64 == 0;
    expect(name, ok, || format!("cannot insert `{vt}` at {pos} into `{dt}`"))?;
    Val::new_typed(ScalableInsert { value, dest, pos })
}

/// `vector.scalable.insert` of a 1-D vector into a 1-D scalable vector.
#[track_caller]
pub fn scalable_insert(value: impl Into<Val>, dest: impl Into<Val>, pos: i64) -> Val<VectorType> {
    try_scalable_insert(value, dest, pos).or_panic()
}

impl HighMlir for ScalableInsert {
    fn name(&self) -> &str {
        "vector.scalable.insert"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.value.clone(), self.dest.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.dest.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("pos".into(), Attr::Int(self.pos, I64.into()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (v, d, vt, dt) = (p.operand(0), p.operand(1), p.operand_type(0), p.operand_type(1));
        write!(p, "vector.scalable.insert {v}, {d}[{}] : {vt} into {dt}", self.pos)
    }
}

/// A memref (or tensor) and indices into it, as accessed by vector loads and stores.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Base {
    pub base: Val,
    pub indices: Vec<Val>,
}

impl Base {
    /// `tensor` allows a ranked tensor base.
    fn new(op: &str, base: Val, idx: impl Operands, tensor: bool) -> Result<(Base, Ty), TypeError> {
        let idx = indices(op, idx)?;
        let ty = value_ty(op, &base)?;
        let rank = match (ty.as_memref(), ty.downcast_ref::<TensorType>()) {
            (Some(m), _) => m.rank(),
            (_, Some(t)) if tensor => t.shape.len(),
            _ => return Err(TypeError::new(op, format!("expected a memref{}, not `{ty}`", if tensor { " or tensor" } else { "" }))),
        };
        check_rank(op, &ty, rank, idx.len())?;
        Ok((Base { base, indices: idx }, ty))
    }

    fn is_memref(&self) -> bool {
        self.base.ty().as_memref().is_some()
    }

    fn values(&self) -> impl Iterator<Item = Val> + '_ {
        std::iter::once(self.base.clone()).chain(self.indices.iter().cloned())
    }

    /// `%m[%i, ..]` for operands from `start`.
    fn format(&self, p: &OpPrinter<'_>, start: usize) -> String {
        format!("{}[{}]", p.operand(start), p.operands(start + 1..start + 1 + self.indices.len()))
    }
}

/// `vector.load` of a vector from a memref.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Load {
    pub base: Base,
    pub ty: Ty,
    pub access: Access,
}

/// `vector.load` of a vector `ty` from `base[indices]`.
#[track_caller]
pub fn try_load(base: impl Into<Val>, indices: impl Operands, ty: impl Into<Ty>, access: impl Into<Access>) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.load";
    let (ty, access) = (ty.into(), access.into());
    let (base, base_ty) = Base::new(name, base.into(), indices, false)?;
    let v = as_vector(name, &ty)?;
    expect(name, &v.element == base_ty.element() || ty == *base_ty.element(), || {
        format!("cannot load `{ty}` from `{base_ty}`")
    })?;
    access.check(name)?;
    Val::new_typed(Load { base, ty, access })
}

/// `vector.load` of a vector `ty` from `base[indices]`.
#[track_caller]
pub fn load(base: impl Into<Val>, indices: impl Operands, ty: impl Into<Ty>, access: impl Into<Access>) -> Val<VectorType> {
    try_load(base, indices, ty, access).or_panic()
}

impl HighMlir for Load {
    fn name(&self) -> &str {
        "vector.load"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.base.values())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.access.attributes()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.load {}", self.base.format(p, 0))?;
        p.attr_dict(&self.attributes())?;
        write!(p, " : {}, {}", p.operand_type(0), self.ty)
    }
}

/// `vector.store` of a vector into a memref.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Store {
    pub value: Val,
    pub base: Base,
    pub access: Access,
}

/// `vector.store` of `value` into `base[indices]`.
#[track_caller]
pub fn try_store(value: impl Into<Val>, base: impl Into<Val>, indices: impl Operands, access: impl Into<Access>) -> Result<(), TypeError> {
    let name = "vector.store";
    let (value, access) = (value.into(), access.into());
    let (base, base_ty) = Base::new(name, base.into(), indices, false)?;
    let (vt, v) = vector_ty(name, &value)?;
    expect(name, &v.element == base_ty.element() || vt == *base_ty.element(), || {
        format!("cannot store `{vt}` into `{base_ty}`")
    })?;
    access.check(name)?;
    Val::unit(Store { value, base, access })
}

/// `vector.store` of `value` into `base[indices]`.
#[track_caller]
pub fn store(value: impl Into<Val>, base: impl Into<Val>, indices: impl Operands, access: impl Into<Access>) {
    try_store(value, base, indices, access).or_panic()
}

impl HighMlir for Store {
    fn name(&self) -> &str {
        "vector.store"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.value.clone()).chain(self.base.values()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.access.attributes()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        write!(p, "vector.store {}, {}", p.operand(0), self.base.format(p, 1))?;
        p.attr_dict(&self.attributes())?;
        write!(p, " : {}, {}", p.operand_type(1), p.operand_type(0))
    }
}

/// Checks a 1-D vector access: `mask` (of `i1`) and `value` have the same shape
/// and `value` has the element type of the base.
fn check_masked(op: &str, base_ty: &Ty, mask: &Val, value: &Val) -> Result<(), TypeError> {
    let (mt, m) = vector_ty(op, mask)?;
    let (vt, v) = vector_ty(op, value)?;
    let ok = m.element == I1 && m.dims() == v.dims() && &v.element == base_ty.element();
    expect(op, ok, || format!("`{mt}` does not mask `{vt}` of `{base_ty}`"))
}

/// Masked memory accesses: `$f(base, indices, mask, value, alignment)`;
/// loads return a vector of the type of `value` (the pass-through).
macro_rules! masked_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident, $value:ident, $load:literal, $ret:ty, $make:ident;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub base: Base,
            pub mask: Val,
            pub $value: Val,
            pub alignment: Option<u64>,
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(
            base: impl Into<Val>,
            indices: impl Operands,
            mask: impl Into<Val>,
            $value: impl Into<Val>,
            alignment: Option<u64>,
        ) -> Result<$ret, TypeError> {
            let name = concat!("vector.", stringify!($f));
            let (base, base_ty) = Base::new(name, base.into(), indices, false)?;
            let (mask, $value) = (mask.into(), $value.into());
            check_masked(name, &base_ty, &mask, &$value)?;
            check_alignment(name, alignment)?;
            Val::$make($op { base, mask, $value, alignment })
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f(base: impl Into<Val>, indices: impl Operands, mask: impl Into<Val>, $value: impl Into<Val>, alignment: Option<u64>) -> $ret {
            $try_f(base, indices, mask, $value, alignment).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("vector.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(self.base.values().chain([self.mask.clone(), self.$value.clone()]))
            }

            fn result_types(&self) -> Vec<Ty> {
                if $load { vec![self.$value.ty()] } else { vec![] }
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                alignment_attr(self.alignment)
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let n = 1 + self.base.indices.len();
                write!(p, "{} {}, {}, {}", self.name(), self.base.format(p, 0), p.operand(n), p.operand(n + 1))?;
                p.attr_dict(&self.attributes())?;
                let types = TypeList(&[p.operand_type(0), p.operand_type(n), p.operand_type(n + 1)]).to_string();
                write!(p, " : {types}")?;
                if $load {
                    write!(p, " into {}", p.operand_type(n + 1))?;
                }
                Ok(())
            }
        }
    )*};
}

masked_ops! {
    /// `vector.maskedload`: the masked elements from memory, the others from `pass_thru`.
    MaskedLoad = maskedload / try_maskedload, pass_thru, true, Val, single;
    /// `vector.maskedstore`: stores the masked elements of `value`.
    MaskedStore = maskedstore / try_maskedstore, value, false, (), unit;
    /// `vector.expandload`: consecutive elements from memory into the masked lanes.
    ExpandLoad = expandload / try_expandload, pass_thru, true, Val, single;
    /// `vector.compressstore`: the masked elements of `value` to consecutive memory.
    CompressStore = compressstore / try_compressstore, value, false, (), unit;
}

/// Gather and scatter: `base[offsets]` plus a vector of `indices` per lane.
macro_rules! indexed_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident, $value:ident, $load:literal, $ret:ty, $make:ident;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub base: Base,
            pub indices: Val,
            pub mask: Val,
            pub $value: Val,
            pub alignment: Option<u64>,
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(
            base: impl Into<Val>,
            offsets: impl Operands,
            indices: impl Into<Val>,
            mask: impl Into<Val>,
            $value: impl Into<Val>,
            alignment: Option<u64>,
        ) -> Result<$ret, TypeError> {
            let name = concat!("vector.", stringify!($f));
            let (base, base_ty) = Base::new(name, base.into(), offsets, !$load)?;
            let (indices, mask, $value) = (indices.into(), mask.into(), $value.into());
            check_masked(name, &base_ty, &mask, &$value)?;
            let (it, i) = vector_ty(name, &indices)?;
            let int = i.element.is_index() || i.element.int_width().is_some();
            let same = Some(i.dims()) == $value.ty().as_vector().map(VectorType::dims);
            expect(name, int && same, || format!("`{it}` are not integer indices for each lane"))?;
            check_alignment(name, alignment)?;
            Val::$make($op { base, indices, mask, $value, alignment })
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f(
            base: impl Into<Val>,
            offsets: impl Operands,
            indices: impl Into<Val>,
            mask: impl Into<Val>,
            $value: impl Into<Val>,
            alignment: Option<u64>,
        ) -> $ret {
            $try_f(base, offsets, indices, mask, $value, alignment).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("vector.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(self.base.values().chain([self.indices.clone(), self.mask.clone(), self.$value.clone()]))
            }

            fn result_types(&self) -> Vec<Ty> {
                match () {
                    _ if $load => vec![self.$value.ty()],
                    _ if !self.base.is_memref() => vec![self.base.base.ty()],
                    _ => vec![],
                }
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                alignment_attr(self.alignment)
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let n = 1 + self.base.indices.len();
                let (i, m, v) = (p.operand(n), p.operand(n + 1), p.operand(n + 2));
                write!(p, "{} {} [{i}], {m}, {v}", self.name(), self.base.format(p, 0))?;
                p.attr_dict(&self.attributes())?;
                let types: Vec<Ty> = [0, n, n + 1, n + 2].into_iter().map(|k| p.operand_type(k)).collect();
                write!(p, " : {}", TypeList(&types))?;
                match self.result_types().pop() {
                    Some(r) if $load => write!(p, " into {r}"),
                    Some(r) => write!(p, " -> {r}"),
                    None => Ok(()),
                }
            }
        }
    )*};
}

indexed_ops! {
    /// `vector.gather` of the masked lanes from `base[offsets + indices]`, the others from `pass_thru`.
    Gather = gather / try_gather, pass_thru, true, Val, single;
    /// `vector.scatter` of the masked lanes of `value` to `base[offsets + indices]`;
    /// returns the new tensor for a tensor base.
    Scatter = scatter / try_scatter, value, false, Option<Val>, optional;
}

/// Optional operands and attributes of `vector.transfer_read`/`transfer_write`.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Transfer {
    /// From the base dimensions to the vector dimensions; the minor identity by default.
    pub permutation_map: Option<AffineMap>,
    /// Lanes to access (a vector of `i1`).
    pub mask: Option<Val>,
    /// Per vector dimension, whether the access is known to be in bounds; none by default.
    pub in_bounds: Option<Vec<bool>>,
}

impl From<()> for Transfer {
    fn from(_: ()) -> Transfer {
        Transfer::default()
    }
}

/// The attributes of a transfer between `base_rank` dimensions and `vector`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TransferAttrs {
    pub permutation_map: AffineMap,
    pub in_bounds: Vec<bool>,
}

impl TransferAttrs {
    fn new(op: &str, opts: &Transfer, base_rank: usize, vector: &VectorType) -> Result<TransferAttrs, TypeError> {
        let permutation_map = opts
            .permutation_map
            .clone()
            .unwrap_or_else(|| AffineMap::minor_identity(base_rank as u32, vector.rank().min(base_rank) as u32));
        let in_bounds = opts.in_bounds.clone().unwrap_or_else(|| vec![false; vector.rank()]);
        let ok = permutation_map.dims as usize == base_rank
            && permutation_map.results.len() == vector.rank()
            && in_bounds.len() == vector.rank();
        expect(op, ok, || format!("`{permutation_map}` does not map {base_rank} dimensions to `{vector}`"))?;
        if let Some(m) = &opts.mask {
            check_mask_type(op, &value_ty(op, m)?)?;
        }
        Ok(TransferAttrs { permutation_map, in_bounds })
    }

    fn attributes(&self, segments: (String, Attr)) -> Vec<(String, Attr)> {
        vec![
            ("in_bounds".into(), Attr::List(self.in_bounds.iter().map(|&b| Attr::Bool(b)).collect())),
            ("permutation_map".into(), Attr::AffineMap(self.permutation_map.clone())),
            segments,
        ]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let in_bounds = Attr::List(self.in_bounds.iter().map(|&b| Attr::Bool(b)).collect());
        write!(p, " {{in_bounds = {in_bounds}, permutation_map = affine_map<{}>}}", self.permutation_map)
    }
}

fn base_rank(ty: &Ty) -> usize {
    ty.shape().map_or(0, <[Option<u64>]>::len)
}

/// `vector.transfer_read`: a vector from a memref or tensor, padding out of bounds lanes.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TransferRead {
    pub base: Base,
    pub padding: Val,
    pub mask: Option<Val>,
    pub attrs: TransferAttrs,
    pub ty: Ty,
}

/// `vector.transfer_read` of a vector `ty` from `base[indices]`; a literal
/// `padding` takes the element type.
#[track_caller]
pub fn try_transfer_read(
    base: impl Into<Val>,
    indices: impl Operands,
    padding: impl Operand,
    ty: impl Into<Ty>,
    opts: impl Into<Transfer>,
) -> Result<Val<VectorType>, TypeError> {
    let name = "vector.transfer_read";
    let (ty, opts) = (ty.into(), opts.into());
    let (base, base_ty) = Base::new(name, base.into(), indices, true)?;
    let v = as_vector(name, &ty)?;
    let padding = padding.into_val(base_ty.element()).map_err(|e| TypeError::new(name, e.message))?;
    let pt = value_ty(name, &padding)?;
    expect(name, &pt == base_ty.element() && &v.element == base_ty.element(), || {
        format!("cannot read `{ty}` padded with `{pt}` from `{base_ty}`")
    })?;
    let attrs = TransferAttrs::new(name, &opts, base_rank(&base_ty), &v)?;
    let op = TransferRead { base, padding, mask: opts.mask, attrs, ty };
    Val::new_typed(op)
}

/// `vector.transfer_read` of a vector `ty` from `base[indices]`; a literal
/// `padding` takes the element type.
#[track_caller]
pub fn transfer_read(
    base: impl Into<Val>,
    indices: impl Operands,
    padding: impl Operand,
    ty: impl Into<Ty>,
    opts: impl Into<Transfer>,
) -> Val<VectorType> {
    try_transfer_read(base, indices, padding, ty, opts).or_panic()
}

impl HighMlir for TransferRead {
    fn name(&self) -> &str {
        "vector.transfer_read"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.base.values().chain(std::iter::once(self.padding.clone())).chain(self.mask.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.attrs.attributes(Attr::segments([1, self.base.indices.len(), 1, self.mask.is_some() as usize]))
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = 1 + self.base.indices.len();
        write!(p, "vector.transfer_read {}, {}", self.base.format(p, 0), p.operand(n))?;
        if self.mask.is_some() {
            write!(p, ", {}", p.operand(n + 1))?;
        }
        self.attrs.print(p)?;
        write!(p, " : {}, {}", p.operand_type(0), self.ty)
    }
}

/// `vector.transfer_write`: a vector into a memref, or a tensor (returning the new tensor).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TransferWrite {
    pub value: Val,
    pub base: Base,
    pub mask: Option<Val>,
    pub attrs: TransferAttrs,
}

/// `vector.transfer_write` of `value` into `base[indices]`; returns the new
/// tensor for a tensor base.
#[track_caller]
pub fn try_transfer_write(
    value: impl Into<Val>,
    base: impl Into<Val>,
    indices: impl Operands,
    opts: impl Into<Transfer>,
) -> Result<Option<Val>, TypeError> {
    let name = "vector.transfer_write";
    let (value, opts) = (value.into(), opts.into());
    let (base, base_ty) = Base::new(name, base.into(), indices, true)?;
    let (vt, v) = vector_ty(name, &value)?;
    expect(name, &v.element == base_ty.element(), || format!("cannot write `{vt}` into `{base_ty}`"))?;
    let attrs = TransferAttrs::new(name, &opts, base_rank(&base_ty), &v)?;
    Val::optional(TransferWrite { value, base, mask: opts.mask, attrs })
}

/// `vector.transfer_write` of `value` into `base[indices]`; returns the new
/// tensor for a tensor base.
#[track_caller]
pub fn transfer_write(value: impl Into<Val>, base: impl Into<Val>, indices: impl Operands, opts: impl Into<Transfer>) -> Option<Val> {
    try_transfer_write(value, base, indices, opts).or_panic()
}

impl HighMlir for TransferWrite {
    fn name(&self) -> &str {
        "vector.transfer_write"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.value.clone()).chain(self.base.values()).chain(self.mask.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        if self.base.is_memref() { vec![] } else { vec![self.base.base.ty()] }
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.attrs.attributes(Attr::segments([1, 1, self.base.indices.len(), self.mask.is_some() as usize]))
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.transfer_write {}, {}", p.operand(0), self.base.format(p, 1))?;
        if self.mask.is_some() {
            write!(p, ", {}", p.operand(2 + self.base.indices.len()))?;
        }
        self.attrs.print(p)?;
        write!(p, " : {}, {}", p.operand_type(0), p.operand_type(1))
    }
}

/// `vector.mask`: applies `mask` to the one maskable operation of its region,
/// whose results it yields (masked-off lanes take `passthru`).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Mask {
    pub mask: Val,
    pub passthru: Option<Val>,
    pub body: BuiltRegion,
}

/// `vector.mask`; `body()` builds one maskable operation (its operands
/// must be built outside) and returns its results.
#[track_caller]
pub fn try_mask(mask: impl Into<Val>, passthru: Option<Val>, body: impl FnOnce() -> Tuple) -> Result<Tuple, TypeError> {
    let name = "vector.mask";
    let mask = mask.into();
    check_mask_type(name, &value_ty(name, &mask)?)?;
    let mut builder = ClosureBuilder::new(false);
    let region = builder.region(&[], |_| body());
    builder.try_finish()?;
    let results = types(name, &region.exprs)?;
    if let Some(pt) = &passthru {
        let t = value_ty(name, pt)?;
        expect(name, results.len() == 1 && results[0] == t, || format!("the passthru `{t}` does not match the results"))?;
    }
    Val::all_results(Mask { mask, passthru, body: region })
}

/// `vector.mask`; `body()` builds one maskable operation (its operands
/// must be built outside) and returns its results.
#[track_caller]
pub fn mask(mask: impl Into<Val>, passthru: Option<Val>, body: impl FnOnce() -> Tuple) -> Tuple {
    try_mask(mask, passthru, body).or_panic()
}

impl HighMlir for Mask {
    fn name(&self) -> &str {
        "vector.mask"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.mask.clone()).chain(self.passthru.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        self.body.exprs.iter().map(|v| v.ty()).collect()
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "vector.mask {}", p.operand(0))?;
        if self.passthru.is_some() {
            write!(p, ", {}", p.operand(1))?;
        }
        p.write_str(" ")?;
        p.region(0, false)?;
        write!(p, " : {}", p.operand_type(0))?;
        let results = self.result_types();
        if !results.is_empty() {
            write!(p, " -> {}", TypeList(&results))?;
        }
        Ok(())
    }
}

impl Closure for Mask {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }


    fn terminator(&self, _: usize) -> Option<&str> {
        Some("vector.yield")
    }
}

/// `vector.print` of a value, a string or punctuation; a side effect.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Print {
    pub source: Option<Val>,
    pub punctuation: Punctuation,
    pub string: Option<String>,
}

/// `vector.print` of `source` followed by `punctuation`.
#[track_caller]
pub fn try_print(source: impl Into<Val>, punctuation: impl Into<Punctuation>) -> Result<(), TypeError> {
    let source = source.into();
    value_ty("vector.print", &source)?;
    Val::unit(Print { source: Some(source), punctuation: punctuation.into(), string: None })
}

/// `vector.print` of `source` followed by `punctuation`.
#[track_caller]
pub fn print(source: impl Into<Val>, punctuation: impl Into<Punctuation>) {
    try_print(source, punctuation).or_panic()
}

/// `vector.print` of `punctuation` alone.
#[track_caller]
pub fn print_punctuation(punctuation: Punctuation) {
    Val::unit(Print { source: None, punctuation, string: None }).or_panic()
}

/// `vector.print str "..."`
#[track_caller]
pub fn print_str(string: impl Into<String>) {
    Val::unit(Print { source: None, punctuation: Punctuation::NewLine, string: Some(string.into()) }).or_panic()
}

impl HighMlir for Print {
    fn name(&self) -> &str {
        "vector.print"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.source.clone().into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs =
            vec![("punctuation".into(), Attr::Raw(format!("#vector.punctuation<{}>", self.punctuation.keyword())))];
        if let Some(s) = &self.string {
            attrs.push(("stringLiteral".into(), Attr::str(s)));
        }
        attrs
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.write_str("vector.print")?;
        if self.source.is_some() {
            write!(p, " {} : {}", p.operand(0), p.operand_type(0))?;
        }
        if let Some(s) = &self.string {
            write!(p, " str {}", Attr::str(s))?;
        }
        if self.punctuation != Punctuation::NewLine {
            write!(p, " punctuation <{}>", self.punctuation.keyword())?;
        }
        Ok(())
    }
}
