//! `tensor` dialect: creating, reading and reshaping immutable tensors.
//!
//! Each operation is built by the function of its name (`tensor::empty`,
//! `tensor::extract_slice`, ...), with a `try_` variant returning a
//! [`TypeError`]. "Writes" (`insert`, `insert_slice`, ...) return a new
//! tensor. Result types are inferred as MLIR infers them; the `_with_type`
//! constructors take an explicit one (e.g. for rank-reducing slices).
//! Tensor values also have the common operations as methods.

use std::fmt::{self, Write};

use crate::{
    Attr, BuiltRegion, Closure, ClosureBuilder, Edges, HighMlir, IndexType, OpPrinter, TensorType, Ty, TypeError, Val,
    arith::{Operand, Operands},
    dialect::util::{check_rank, expect, indices, tensor_ty, types, value_ty},
    memref::{Reassociation, Window, check_reassociation, fmt_reassociation, reassociation_attr},
    mixed::{MixedList, MixedValues},
    types::{INDEX, TypeList},
    val::OrPanic,
};

/// Methods of tensor values, on `Val<TensorType>`.
impl Val<TensorType> {
    /// `tensor.extract` of an element.
    #[track_caller]
    pub fn extract(&self, indices: impl Operands) -> Val {
        extract(self, indices)
    }

    /// `tensor.insert` of an element: the new tensor. A literal takes the element type.
    #[track_caller]
    pub fn insert(&self, scalar: impl Operand, indices: impl Operands) -> Val<TensorType> {
        insert(scalar, self, indices)
    }

    /// `tensor.dim`
    #[track_caller]
    pub fn dim(&self, index: impl Operand) -> Val<IndexType> {
        dim(self, index)
    }

    /// `tensor.rank`
    #[track_caller]
    pub fn rank(&self) -> Val<IndexType> {
        rank(self)
    }

    /// `tensor.extract_slice`; literals are static.
    #[track_caller]
    pub fn extract_slice(&self, offsets: impl MixedList, sizes: impl MixedList, strides: impl MixedList) -> Val<TensorType> {
        extract_slice(self, offsets, sizes, strides)
    }

    /// `tensor.insert_slice` of this tensor into `dest`: the new `dest`.
    #[track_caller]
    pub fn insert_slice_into(
        &self,
        dest: impl Into<Val>,
        offsets: impl MixedList,
        sizes: impl MixedList,
        strides: impl MixedList,
    ) -> Val<TensorType> {
        insert_slice(self, dest, offsets, sizes, strides)
    }

    /// `tensor.expand_shape`; literal sizes are static.
    #[track_caller]
    pub fn expand_shape(&self, reassociation: Reassociation, output_shape: impl MixedList) -> Val<TensorType> {
        expand_shape(self, reassociation, output_shape)
    }

    /// `tensor.collapse_shape`
    #[track_caller]
    pub fn collapse_shape(&self, reassociation: Reassociation) -> Val<TensorType> {
        collapse_shape(self, reassociation)
    }

    /// `tensor.pad` with `value`; literals are static.
    #[track_caller]
    pub fn pad(&self, low: impl MixedList, high: impl MixedList, value: impl Operand) -> Val<TensorType> {
        pad(self, low, high, value)
    }
}

fn tensor_of(shape: Vec<Option<u64>>, element: Ty) -> Ty {
    TensorType { shape, element }.into()
}

/// One `index` per dynamic dimension of `ty`.
fn check_dynamic_sizes(op: &str, ty: &Ty, sizes: &[Val]) -> Result<TensorType, TypeError> {
    let Some(t) = ty.downcast_ref::<TensorType>().cloned() else {
        return Err(TypeError::new(op, format!("expected a ranked tensor type, not `{ty}`")));
    };
    let dynamic = t.shape.iter().filter(|d| d.is_none()).count();
    expect(op, sizes.len() == dynamic, || format!("`{ty}` needs {dynamic} sizes, got {}", sizes.len()))?;
    Ok(t)
}

/// `tensor.empty`: a tensor of unspecified contents.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Empty {
    pub ty: Ty,
    /// One `index` per dynamic dimension.
    pub sizes: Vec<Val>,
}

/// `tensor.empty` of type `ty`, with one size per dynamic dimension.
#[track_caller]
pub fn try_empty(ty: impl Into<Ty>, sizes: impl Operands) -> Result<Val<TensorType>, TypeError> {
    let (ty, sizes) = (ty.into(), indices("tensor.empty", sizes)?);
    check_dynamic_sizes("tensor.empty", &ty, &sizes)?;
    Val::new_typed(Empty { ty, sizes })
}

/// `tensor.empty` of type `ty`, with one size per dynamic dimension.
#[track_caller]
pub fn empty(ty: impl Into<Ty>, sizes: impl Operands) -> Val<TensorType> {
    try_empty(ty, sizes).or_panic()
}

impl HighMlir for Empty {
    fn name(&self) -> &str {
        "tensor.empty"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.sizes.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "tensor.empty({}) : {}", p.operands(0..self.sizes.len()), self.ty)
    }
}

/// Whether two shapes agree wherever both are static.
fn compatible(a: &[Option<u64>], b: &[Option<u64>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.is_none() || y.is_none() || x == y)
}

macro_rules! cast_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident, $check:expr;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub source: Val,
            pub ty: Ty,
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(source: impl Into<Val>, ty: impl Into<Ty>) -> Result<Val<TensorType>, TypeError> {
            let name = concat!("tensor.", stringify!($f));
            let (source, ty) = (source.into(), ty.into());
            let (src_ty, src) = tensor_ty(name, &source)?;
            let dst = ty.downcast_ref::<TensorType>();
            let check: fn(&TensorType, &TensorType) -> bool = $check;
            expect(name, dst.is_some_and(|d| check(&src, d)), || format!("cannot cast `{src_ty}` to `{ty}`"))?;
            Val::new_typed($op { source, ty })
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f(source: impl Into<Val>, ty: impl Into<Ty>) -> Val<TensorType> {
            $try_f(source, ty).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("tensor.", stringify!($f))
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

cast_ops! {
    /// `tensor.cast` to a type of a compatible shape (e.g. making a dimension dynamic or static).
    Cast = cast / try_cast, |s, d| s.element == d.element && compatible(&s.shape, &d.shape);
    /// `tensor.bitcast` to elements of the same bit width.
    Bitcast = bitcast / try_bitcast, |s, d| {
        compatible(&s.shape, &d.shape) && s.element.bit_width().is_some() && s.element.bit_width() == d.element.bit_width()
    };
}

/// `tensor.dim`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Dim {
    pub source: Val,
    pub index: Val,
}

/// `tensor.dim`; a literal `index` becomes an `index` constant.
#[track_caller]
pub fn try_dim(source: impl Into<Val>, index: impl Operand) -> Result<Val<IndexType>, TypeError> {
    let name = "tensor.dim";
    let source = source.into();
    let (ty, t) = tensor_ty(name, &source)?;
    expect(name, !t.shape.is_empty(), || format!("`{ty}` has no dimensions"))?;
    let index = indices(name, index)?.remove(0);
    Val::new_typed(Dim { source, index })
}

/// `tensor.dim`; a literal `index` becomes an `index` constant.
#[track_caller]
pub fn dim(source: impl Into<Val>, index: impl Operand) -> Val<IndexType> {
    try_dim(source, index).or_panic()
}

impl HighMlir for Dim {
    fn name(&self) -> &str {
        "tensor.dim"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.index.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![INDEX.into()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "tensor.dim {}, {} : {}", p.operand(0), p.operand(1), p.operand_type(0))
    }
}

/// `tensor.rank`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Rank {
    pub source: Val,
}

/// `tensor.rank`
#[track_caller]
pub fn try_rank(source: impl Into<Val>) -> Result<Val<IndexType>, TypeError> {
    let source = source.into();
    tensor_ty("tensor.rank", &source)?;
    Val::new_typed(Rank { source })
}

/// `tensor.rank`
#[track_caller]
pub fn rank(source: impl Into<Val>) -> Val<IndexType> {
    try_rank(source).or_panic()
}

impl HighMlir for Rank {
    fn name(&self) -> &str {
        "tensor.rank"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![INDEX.into()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "tensor.rank {} : {}", p.operand(0), p.operand_type(0))
    }
}

/// `tensor.extract` of an element.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Extract {
    pub tensor: Val,
    pub indices: Vec<Val>,
}

/// `tensor.extract` of the element at `indices`.
#[track_caller]
pub fn try_extract(tensor: impl Into<Val>, indices: impl Operands) -> Result<Val, TypeError> {
    let name = "tensor.extract";
    let tensor = tensor.into();
    let (ty, t) = tensor_ty(name, &tensor)?;
    let indices = self::indices(name, indices)?;
    check_rank(name, &ty, t.shape.len(), indices.len())?;
    Val::single(Extract { tensor, indices })
}

/// `tensor.extract` of the element at `indices`.
#[track_caller]
pub fn extract(tensor: impl Into<Val>, indices: impl Operands) -> Val {
    try_extract(tensor, indices).or_panic()
}

impl HighMlir for Extract {
    fn name(&self) -> &str {
        "tensor.extract"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.tensor.clone()).chain(self.indices.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.tensor.ty().element().clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = self.indices.len();
        write!(p, "tensor.extract {}[{}] : {}", p.operand(0), p.operands(1..=n), p.operand_type(0))
    }
}

/// `tensor.insert` of an element.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Insert {
    pub scalar: Val,
    pub dest: Val,
    pub indices: Vec<Val>,
}

/// `tensor.insert` of `scalar` at `indices` of `dest`: the new tensor. A
/// literal takes the element type.
#[track_caller]
pub fn try_insert(scalar: impl Operand, dest: impl Into<Val>, indices: impl Operands) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.insert";
    let dest = dest.into();
    let (ty, t) = tensor_ty(name, &dest)?;
    let scalar = scalar.into_val(&t.element)?;
    let s = value_ty(name, &scalar)?;
    expect(name, s == t.element, || format!("cannot insert a `{s}` into `{ty}`"))?;
    let indices = self::indices(name, indices)?;
    check_rank(name, &ty, t.shape.len(), indices.len())?;
    Val::new_typed(Insert { scalar, dest, indices })
}

/// `tensor.insert` of `scalar` at `indices` of `dest`: the new tensor.
#[track_caller]
pub fn insert(scalar: impl Operand, dest: impl Into<Val>, indices: impl Operands) -> Val<TensorType> {
    try_insert(scalar, dest, indices).or_panic()
}

impl HighMlir for Insert {
    fn name(&self) -> &str {
        "tensor.insert"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.scalar.clone(), self.dest.clone()].into_iter().chain(self.indices.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.dest.ty()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = self.indices.len();
        write!(p, "tensor.insert {} into {}[{}] : {}", p.operand(0), p.operand(1), p.operands(2..2 + n), p.operand_type(1))
    }
}

/// `tensor.from_elements`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FromElements {
    pub elements: Vec<Val>,
    pub ty: Ty,
}

/// `tensor.from_elements` of a static tensor type, in row-major order;
/// literals take the element type.
#[track_caller]
pub fn try_from_elements(elements: impl Operands, ty: impl Into<Ty>) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.from_elements";
    let ty = ty.into();
    let t = check_dynamic_sizes(name, &ty, &[])?;
    let elements = elements.into_vals(&|_| Some(t.element.clone()))?;
    let n: u64 = t.shape.iter().map(|d| d.unwrap()).product();
    expect(name, elements.len() as u64 == n, || format!("`{ty}` has {n} elements, got {}", elements.len()))?;
    let found = types(name, &elements)?;
    expect(name, found.iter().all(|e| *e == t.element), || format!("elements ({}) for `{ty}`", TypeList(&found)))?;
    Val::new_typed(FromElements { elements, ty })
}

/// `tensor.from_elements` of a static tensor type, in row-major order.
#[track_caller]
pub fn from_elements(elements: impl Operands, ty: impl Into<Ty>) -> Val<TensorType> {
    try_from_elements(elements, ty).or_panic()
}

impl HighMlir for FromElements {
    fn name(&self) -> &str {
        "tensor.from_elements"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.elements.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "tensor.from_elements {} : {}", p.operands(0..self.elements.len()), self.ty)
    }
}

/// `tensor.splat`: a tensor filled with one value.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Splat {
    pub input: Val,
    pub sizes: Vec<Val>,
    pub ty: Ty,
}

/// `tensor.splat` of `input` into `ty`, with one size per dynamic
/// dimension; a literal takes the element type.
#[track_caller]
pub fn try_splat(input: impl Operand, ty: impl Into<Ty>, sizes: impl Operands) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.splat";
    let (ty, sizes) = (ty.into(), indices(name, sizes)?);
    let t = check_dynamic_sizes(name, &ty, &sizes)?;
    let input = input.into_val(&t.element)?;
    let i = value_ty(name, &input)?;
    expect(name, i == t.element, || format!("cannot splat a `{i}` into `{ty}`"))?;
    Val::new_typed(Splat { input, sizes, ty })
}

/// `tensor.splat` of `input` into `ty`, with one size per dynamic dimension.
#[track_caller]
pub fn splat(input: impl Operand, ty: impl Into<Ty>, sizes: impl Operands) -> Val<TensorType> {
    try_splat(input, ty, sizes).or_panic()
}

impl HighMlir for Splat {
    fn name(&self) -> &str {
        "tensor.splat"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.input.clone()).chain(self.sizes.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "tensor.splat {}", p.operand(0))?;
        if !self.sizes.is_empty() {
            write!(p, "[{}]", p.operands(1..=self.sizes.len()))?;
        }
        write!(p, " : {}", self.ty)
    }
}

/// `tensor.extract_slice`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ExtractSlice {
    pub source: Val,
    pub window: Window,
    pub ty: Ty,
}

/// `tensor.extract_slice`; the result has the shape of the sizes (literals are static).
#[track_caller]
pub fn try_extract_slice(
    source: impl Into<Val>,
    offsets: impl MixedList,
    sizes: impl MixedList,
    strides: impl MixedList,
) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.extract_slice";
    let source = source.into();
    let (_, t) = tensor_ty(name, &source)?;
    let window = Window::new(name, offsets, sizes, strides)?;
    let ty = tensor_of(window.static_shape(), t.element);
    try_extract_slice_with_type(source, window, ty)
}

/// `tensor.extract_slice` with an explicit result type, e.g. dropping unit dimensions.
#[track_caller]
pub fn try_extract_slice_with_type(source: impl Into<Val>, window: Window, ty: Ty) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.extract_slice";
    let source = source.into();
    let (src_ty, t) = tensor_ty(name, &source)?;
    check_window(name, &src_ty, t.shape.len(), &window)?;
    let ok = ty.downcast_ref::<TensorType>().is_some_and(|d| d.element == t.element && d.shape.len() <= t.shape.len());
    expect(name, ok, || format!("cannot take a `{ty}` slice of `{src_ty}`"))?;
    Val::new_typed(ExtractSlice { source, window, ty })
}

/// `tensor.extract_slice`
#[track_caller]
pub fn extract_slice(source: impl Into<Val>, offsets: impl MixedList, sizes: impl MixedList, strides: impl MixedList) -> Val<TensorType> {
    try_extract_slice(source, offsets, sizes, strides).or_panic()
}

fn check_window(op: &str, ty: &Ty, rank: usize, w: &Window) -> Result<(), TypeError> {
    expect(op, [&w.offsets, &w.sizes, &w.strides].iter().all(|l| l.len() == rank), || {
        format!("needs {rank} offsets, sizes and strides for `{ty}`")
    })
}

impl HighMlir for ExtractSlice {
    fn name(&self) -> &str {
        "tensor.extract_slice"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()).chain(self.window.dynamic()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.window.attributes_after(1)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let [o, s, t] = self.window.format_from(p, 1);
        write!(p, "tensor.extract_slice {}{o} {s} {t} : {} to {}", p.operand(0), p.operand_type(0), self.ty)
    }
}

/// `tensor.insert_slice`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct InsertSlice {
    pub source: Val,
    pub dest: Val,
    pub window: Window,
}

/// `tensor.insert_slice` of `source` into a window of `dest`: the new
/// `dest`. Literals are static; `source` may drop unit dimensions.
#[track_caller]
pub fn try_insert_slice(
    source: impl Into<Val>,
    dest: impl Into<Val>,
    offsets: impl MixedList,
    sizes: impl MixedList,
    strides: impl MixedList,
) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.insert_slice";
    let (source, dest) = (source.into(), dest.into());
    let (src_ty, s) = tensor_ty(name, &source)?;
    let (dst_ty, d) = tensor_ty(name, &dest)?;
    let window = Window::new(name, offsets, sizes, strides)?;
    check_window(name, &dst_ty, d.shape.len(), &window)?;
    let ok = s.element == d.element && s.shape.len() <= d.shape.len();
    expect(name, ok, || format!("cannot insert `{src_ty}` into `{dst_ty}`"))?;
    Val::new_typed(InsertSlice { source, dest, window })
}

/// `tensor.insert_slice` of `source` into a window of `dest`: the new `dest`.
#[track_caller]
pub fn insert_slice(
    source: impl Into<Val>,
    dest: impl Into<Val>,
    offsets: impl MixedList,
    sizes: impl MixedList,
    strides: impl MixedList,
) -> Val<TensorType> {
    try_insert_slice(source, dest, offsets, sizes, strides).or_panic()
}

impl HighMlir for InsertSlice {
    fn name(&self) -> &str {
        "tensor.insert_slice"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.dest.clone()].into_iter().chain(self.window.dynamic()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.dest.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.window.attributes_after(2)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let [o, s, t] = self.window.format_from(p, 2);
        let (src, dst) = (p.operand(0), p.operand(1));
        write!(p, "tensor.insert_slice {src} into {dst}{o} {s} {t} : {} into {}", p.operand_type(0), p.operand_type(1))
    }
}

/// `tensor.expand_shape`: splits each source dimension into a group of dimensions.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ExpandShape {
    pub source: Val,
    pub reassociation: Reassociation,
    pub output_shape: MixedValues,
    pub ty: Ty,
}

/// `tensor.expand_shape`; the result has the static entries of `output_shape`.
#[track_caller]
pub fn try_expand_shape(
    source: impl Into<Val>,
    reassociation: Reassociation,
    output_shape: impl MixedList,
) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.expand_shape";
    let source = source.into();
    let (src_ty, t) = tensor_ty(name, &source)?;
    let output_shape = MixedValues::new(name, output_shape)?;
    let shape: Vec<Option<u64>> = output_shape.static_values().iter().map(|d| d.map(|d| d as u64)).collect();
    check_reassociation(name, &reassociation, shape.len())?;
    expect(name, reassociation.len() == t.shape.len(), || format!("cannot expand `{src_ty}` by {}", fmt_reassociation(&reassociation)))?;
    for (g, group) in reassociation.iter().enumerate() {
        let size = group.iter().try_fold(1, |acc, &d| shape[d as usize].map(|n| acc * n));
        expect(name, size == t.shape[g], || {
            format!("dimension {g} of `{src_ty}` does not match the output shape {:?}", output_shape.static_values())
        })?;
    }
    let ty = tensor_of(shape, t.element);
    Val::new_typed(ExpandShape { source, reassociation, output_shape, ty })
}

/// `tensor.expand_shape`
#[track_caller]
pub fn expand_shape(source: impl Into<Val>, reassociation: Reassociation, output_shape: impl MixedList) -> Val<TensorType> {
    try_expand_shape(source, reassociation, output_shape).or_panic()
}

impl HighMlir for ExpandShape {
    fn name(&self) -> &str {
        "tensor.expand_shape"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()).chain(self.output_shape.dynamic.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![
            ("reassociation".into(), reassociation_attr(&self.reassociation)),
            ("static_output_shape".into(), Attr::i64_array(self.output_shape.statics.clone())),
        ]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let shape = self.output_shape.format(|k| p.operand(1 + k));
        let r = fmt_reassociation(&self.reassociation);
        write!(p, "tensor.expand_shape {} {r} output_shape {shape} : {} into {}", p.operand(0), p.operand_type(0), self.ty)
    }
}

/// `tensor.collapse_shape`: merges each group of dimensions.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CollapseShape {
    pub source: Val,
    pub reassociation: Reassociation,
    pub ty: Ty,
}

/// `tensor.collapse_shape`
#[track_caller]
pub fn try_collapse_shape(source: impl Into<Val>, reassociation: Reassociation) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.collapse_shape";
    let source = source.into();
    let (_, t) = tensor_ty(name, &source)?;
    check_reassociation(name, &reassociation, t.shape.len())?;
    let shape = reassociation
        .iter()
        .map(|g| g.iter().map(|&d| t.shape[d as usize]).try_fold(1, |acc, d| d.map(|d| acc * d)))
        .collect();
    let ty = tensor_of(shape, t.element);
    Val::new_typed(CollapseShape { source, reassociation, ty })
}

/// `tensor.collapse_shape`
#[track_caller]
pub fn collapse_shape(source: impl Into<Val>, reassociation: Reassociation) -> Val<TensorType> {
    try_collapse_shape(source, reassociation).or_panic()
}

impl HighMlir for CollapseShape {
    fn name(&self) -> &str {
        "tensor.collapse_shape"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("reassociation".into(), reassociation_attr(&self.reassociation))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let r = fmt_reassociation(&self.reassociation);
        write!(p, "tensor.collapse_shape {} {r} : {} into {}", p.operand(0), p.operand_type(0), self.ty)
    }
}

/// `tensor.reshape` to the shape stored in a 1-D tensor.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Reshape {
    pub source: Val,
    pub shape: Val,
    pub ty: Ty,
}

/// `tensor.reshape` of `source` to `ty`, the shape stored in `shape` (a 1-D
/// tensor of integers or `index` with one entry per dimension of `ty`).
#[track_caller]
pub fn try_reshape(source: impl Into<Val>, shape: impl Into<Val>, ty: impl Into<Ty>) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.reshape";
    let (source, shape, ty) = (source.into(), shape.into(), ty.into());
    let (src_ty, s) = tensor_ty(name, &source)?;
    let (shape_ty, sh) = tensor_ty(name, &shape)?;
    let ok_shape = sh.shape.len() == 1 && (sh.element.is_index() || sh.element.int_width().is_some());
    expect(name, ok_shape, || format!("the shape must be a 1-D tensor of integers, not `{shape_ty}`"))?;
    let ok = ty.downcast_ref::<TensorType>().is_some_and(|d| {
        d.element == s.element && sh.shape[0].is_none_or(|n| n as usize == d.shape.len())
    });
    expect(name, ok, || format!("cannot reshape `{src_ty}` to `{ty}` with `{shape_ty}`"))?;
    Val::new_typed(Reshape { source, shape, ty })
}

/// `tensor.reshape`
#[track_caller]
pub fn reshape(source: impl Into<Val>, shape: impl Into<Val>, ty: impl Into<Ty>) -> Val<TensorType> {
    try_reshape(source, shape, ty).or_panic()
}

impl HighMlir for Reshape {
    fn name(&self) -> &str {
        "tensor.reshape"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.shape.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let types = TypeList(&p.operand_types(0..2)).to_string();
        write!(p, "tensor.reshape {}({}) : ({types}) -> {}", p.operand(0), p.operand(1), self.ty)
    }
}

/// `tensor.concat` along a dimension.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Concat {
    pub dim: u64,
    pub inputs: Vec<Val>,
    pub ty: Ty,
}

/// `tensor.concat` of `inputs` along dimension `dim`.
#[track_caller]
pub fn try_concat(dim: u64, inputs: impl IntoIterator<Item = impl Into<Val>>) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.concat";
    let inputs: Vec<Val> = inputs.into_iter().map(Into::into).collect();
    expect(name, !inputs.is_empty(), || "needs at least one input".into())?;
    let tensors = inputs.iter().map(|v| tensor_ty(name, v).map(|(_, t)| t)).collect::<Result<Vec<_>, _>>()?;
    let first = &tensors[0];
    let rank = first.shape.len();
    expect(name, (dim as usize) < rank, || format!("no dimension {dim} in rank {rank}"))?;
    let mut shape = first.shape.clone();
    for t in &tensors[1..] {
        expect(name, t.element == first.element && t.shape.len() == rank, || "inputs of different ranks or elements".into())?;
        for (d, (a, b)) in shape.iter_mut().zip(&t.shape).enumerate() {
            *a = if d == dim as usize { a.zip(*b).map(|(a, b)| a + b) } else { a.or(*b) };
        }
    }
    let ty = tensor_of(shape, first.element.clone());
    Val::new_typed(Concat { dim, inputs, ty })
}

/// `tensor.concat` of `inputs` along dimension `dim`.
#[track_caller]
pub fn concat(dim: u64, inputs: impl IntoIterator<Item = impl Into<Val>>) -> Val<TensorType> {
    try_concat(dim, inputs).or_panic()
}

impl HighMlir for Concat {
    fn name(&self) -> &str {
        "tensor.concat"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.inputs.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("dim".into(), Attr::Int(self.dim as i64, crate::I64.into()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = self.inputs.len();
        let types = TypeList(&p.operand_types(0..n)).to_string();
        write!(p, "tensor.concat dim({}) {} : ({types}) -> {}", self.dim, p.operands(0..n), self.ty)
    }
}

/// The result shape of `gather`: the batch dimensions of `indices`, then
/// the source shape with the gathered dimensions of size 1.
fn gathered_shape(source: &TensorType, indices: &TensorType, dims: &[i64]) -> Vec<Option<u64>> {
    let batch = &indices.shape[..indices.shape.len().saturating_sub(1)];
    let slice = source.shape.iter().enumerate().map(|(d, s)| if dims.contains(&(d as i64)) { Some(1) } else { *s });
    batch.iter().copied().chain(slice).collect()
}

fn check_gather_dims(op: &str, source: &TensorType, indices: &TensorType, dims: &[i64]) -> Result<(), TypeError> {
    let sorted = dims.windows(2).all(|w| w[0] < w[1]) && dims.iter().all(|&d| d >= 0 && (d as usize) < source.shape.len());
    let last = indices.shape.last().copied().flatten();
    let ok = indices.element.is_index() || indices.element.int_width().is_some();
    expect(op, sorted && ok && last.is_none_or(|n| n as usize == dims.len()), || {
        format!("dimensions {dims:?} do not match the indices `{}`", Ty::from(indices.clone()))
    })
}

/// `tensor.gather`: slices of `source` at coordinates `indices`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Gather {
    pub source: Val,
    pub indices: Val,
    pub gather_dims: Vec<i64>,
    pub unique: bool,
    pub ty: Ty,
}

/// `tensor.gather` along the (increasing) `gather_dims`; the last dimension
/// of `indices` holds the coordinates. The result keeps the gathered
/// dimensions, of size 1.
#[track_caller]
pub fn try_gather(source: impl Into<Val>, indices: impl Into<Val>, gather_dims: impl IntoIterator<Item = i64>, unique: bool) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.gather";
    let (source, indices) = (source.into(), indices.into());
    let gather_dims: Vec<i64> = gather_dims.into_iter().collect();
    let (_, s) = tensor_ty(name, &source)?;
    let (_, i) = tensor_ty(name, &indices)?;
    check_gather_dims(name, &s, &i, &gather_dims)?;
    let ty = tensor_of(gathered_shape(&s, &i, &gather_dims), s.element.clone());
    Val::new_typed(Gather { source, indices, gather_dims, unique, ty })
}

/// `tensor.gather`
#[track_caller]
pub fn gather(source: impl Into<Val>, indices: impl Into<Val>, gather_dims: impl IntoIterator<Item = i64>, unique: bool) -> Val<TensorType> {
    try_gather(source, indices, gather_dims, unique).or_panic()
}

fn fmt_dims(dims: &[i64]) -> String {
    format!("[{}]", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
}

impl HighMlir for Gather {
    fn name(&self) -> &str {
        "tensor.gather"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.indices.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = vec![("gather_dims".into(), Attr::i64_array(self.gather_dims.clone()))];
        if self.unique {
            attrs.push(("unique".into(), Attr::Unit));
        }
        attrs
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let types = TypeList(&p.operand_types(0..2)).to_string();
        let unique = if self.unique { " unique" } else { "" };
        let dims = fmt_dims(&self.gather_dims);
        write!(p, "tensor.gather {}[{}] gather_dims({dims}){unique} : ({types}) -> {}", p.operand(0), p.operand(1), self.ty)
    }
}

/// `tensor.scatter`: writes slices of `source` into `dest` at coordinates `indices`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Scatter {
    pub source: Val,
    pub dest: Val,
    pub indices: Val,
    pub scatter_dims: Vec<i64>,
    pub unique: bool,
}

/// `tensor.scatter` of `source` (shaped as `gather` returns) into `dest`:
/// the new `dest`. MLIR requires `unique`.
#[track_caller]
pub fn try_scatter(
    source: impl Into<Val>,
    dest: impl Into<Val>,
    indices: impl Into<Val>,
    scatter_dims: impl IntoIterator<Item = i64>,
    unique: bool,
) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.scatter";
    let (source, dest, indices) = (source.into(), dest.into(), indices.into());
    let scatter_dims: Vec<i64> = scatter_dims.into_iter().collect();
    let (src_ty, s) = tensor_ty(name, &source)?;
    let (_, d) = tensor_ty(name, &dest)?;
    let (_, i) = tensor_ty(name, &indices)?;
    check_gather_dims(name, &d, &i, &scatter_dims)?;
    let expected = gathered_shape(&d, &i, &scatter_dims);
    expect(name, s.element == d.element && compatible(&s.shape, &expected), || {
        format!("the source `{src_ty}` does not have shape {expected:?}")
    })?;
    Val::new_typed(Scatter { source, dest, indices, scatter_dims, unique })
}

/// `tensor.scatter`
#[track_caller]
pub fn scatter(
    source: impl Into<Val>,
    dest: impl Into<Val>,
    indices: impl Into<Val>,
    scatter_dims: impl IntoIterator<Item = i64>,
    unique: bool,
) -> Val<TensorType> {
    try_scatter(source, dest, indices, scatter_dims, unique).or_panic()
}

impl HighMlir for Scatter {
    fn name(&self) -> &str {
        "tensor.scatter"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.dest.clone(), self.indices.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.dest.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = vec![("scatter_dims".into(), Attr::i64_array(self.scatter_dims.clone()))];
        if self.unique {
            attrs.push(("unique".into(), Attr::Unit));
        }
        attrs
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let types = TypeList(&p.operand_types(0..3)).to_string();
        let unique = if self.unique { " unique" } else { "" };
        let dims = fmt_dims(&self.scatter_dims);
        let (s, d, i) = (p.operand(0), p.operand(1), p.operand(2));
        write!(p, "tensor.scatter {s} into {d}[{i}] scatter_dims({dims}){unique} : ({types}) -> {}", self.dest.ty())
    }
}

/// Builds a region of one `index` parameter per dimension yielding one element.
fn element_region(op: &str, rank: usize, element: &Ty, body: impl FnOnce(&[Val<IndexType>]) -> Val) -> Result<BuiltRegion, TypeError> {
    let params = vec![Ty::from(INDEX); rank];
    let mut builder = ClosureBuilder::new(false);
    let region = builder.region(&params, |args| {
        let args: Vec<Val<IndexType>> = args.iter().map(|a| a.downcast().unwrap()).collect();
        body(&args).into()
    });
    builder.try_finish()?;
    let yielded = types(op, &region.exprs)?;
    expect(op, yielded == [element.clone()], || format!("the body yields ({}), not a `{element}`", TypeList(&yielded)))?;
    Ok(region)
}

/// `tensor.generate`: a tensor whose elements a region computes from their indices.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Generate {
    pub ty: Ty,
    pub sizes: Vec<Val>,
    pub body: BuiltRegion,
}

/// `tensor.generate` of type `ty` (one size per dynamic dimension), whose
/// element at `indices` is `body(indices)`.
#[track_caller]
pub fn try_generate(ty: impl Into<Ty>, sizes: impl Operands, body: impl FnOnce(&[Val<IndexType>]) -> Val) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.generate";
    let (ty, sizes) = (ty.into(), indices(name, sizes)?);
    let t = check_dynamic_sizes(name, &ty, &sizes)?;
    let body = element_region(name, t.shape.len(), &t.element, body)?;
    Val::new_typed(Generate { ty, sizes, body })
}

/// `tensor.generate`
#[track_caller]
pub fn generate(ty: impl Into<Ty>, sizes: impl Operands, body: impl FnOnce(&[Val<IndexType>]) -> Val) -> Val<TensorType> {
    try_generate(ty, sizes, body).or_panic()
}

impl HighMlir for Generate {
    fn name(&self) -> &str {
        "tensor.generate"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.sizes.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "tensor.generate {} ", p.operands(0..self.sizes.len()))?;
        p.region(0, true)?;
        write!(p, " : {}", self.ty)
    }
}

impl Closure for Generate {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("tensor.yield")
    }
}

/// `tensor.pad`: the source surrounded by padding.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Pad {
    pub source: Val,
    pub low: MixedValues,
    pub high: MixedValues,
    pub nofold: bool,
    pub ty: Ty,
    pub body: BuiltRegion,
}

/// `tensor.pad` by `low` and `high` elements per dimension (literals are
/// static), the padding at `indices` being `body(indices)`.
#[track_caller]
pub fn try_pad_with(
    source: impl Into<Val>,
    low: impl MixedList,
    high: impl MixedList,
    nofold: bool,
    body: impl FnOnce(&[Val<IndexType>]) -> Val,
) -> Result<Val<TensorType>, TypeError> {
    let name = "tensor.pad";
    let source = source.into();
    let (ty, t) = tensor_ty(name, &source)?;
    let (low, high) = (MixedValues::new(name, low)?, MixedValues::new(name, high)?);
    let rank = t.shape.len();
    expect(name, low.len() == rank && high.len() == rank, || format!("needs {rank} low and high paddings for `{ty}`"))?;
    let shape = (t.shape.iter().zip(low.static_values()).zip(high.static_values()))
        .map(|((d, l), h)| Some(d.as_ref()? + l? as u64 + h? as u64))
        .collect();
    let body = element_region(name, rank, &t.element, body)?;
    let ty = tensor_of(shape, t.element.clone());
    Val::new_typed(Pad { source, low, high, nofold, ty, body })
}

/// `tensor.pad` with `value` (a literal takes the element type).
#[track_caller]
pub fn try_pad(source: impl Into<Val>, low: impl MixedList, high: impl MixedList, value: impl Operand) -> Result<Val<TensorType>, TypeError> {
    let source = source.into();
    let (_, t) = tensor_ty("tensor.pad", &source)?;
    let value = value.into_val(&t.element)?;
    try_pad_with(source, low, high, false, |_| value)
}

/// `tensor.pad` with `value` (a literal takes the element type).
#[track_caller]
pub fn pad(source: impl Into<Val>, low: impl MixedList, high: impl MixedList, value: impl Operand) -> Val<TensorType> {
    try_pad(source, low, high, value).or_panic()
}

impl HighMlir for Pad {
    fn name(&self) -> &str {
        "tensor.pad"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()).chain(self.low.dynamic.iter().cloned()).chain(self.high.dynamic.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = vec![
            Attr::segments([1, self.low.dynamic.len(), self.high.dynamic.len()]),
            ("static_low".into(), Attr::i64_array(self.low.statics.clone())),
            ("static_high".into(), Attr::i64_array(self.high.statics.clone())),
        ];
        if self.nofold {
            attrs.push(("nofold".into(), Attr::Unit));
        }
        attrs
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let nl = self.low.dynamic.len();
        let low = self.low.format(|k| p.operand(1 + k));
        let high = self.high.format(|k| p.operand(1 + nl + k));
        let nofold = if self.nofold { " nofold" } else { "" };
        write!(p, "tensor.pad {}{nofold} low{low} high{high} ", p.operand(0))?;
        p.region(0, true)?;
        write!(p, " : {} to {}", p.operand_type(0), self.ty)
    }
}

impl Closure for Pad {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("tensor.yield")
    }
}
