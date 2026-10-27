//! `memref` dialect.
//!
//! Loads, stores and other accesses are emitted in the order they are
//! built, like every operation, so they keep the order of the Rust code.
//!
//! Result types are inferred as MLIR infers them; the `try_with_type`
//! constructors take an explicit one (e.g. for rank-reducing subviews).

use std::fmt::{self, Write};

use crate::{
    Attr, BuiltRegion, Closure, ClosureBuilder, Edges, HighMlir, IndexType, Layout,
    MemRefType, OpPrinter, Ty, TypeError, UnrankedMemRefType, Val,
    arith::{Operand, Operands, predicates},
    attr::{AffineMap, Symbol},
    dialect::util::{check_rank, expect, indices, memref_ty, types, value_ty},
    mixed::{MixedList, MixedValues},
    types::{I32, I64, INDEX, TypeList},
    val::{AsVal, OrPanic, Tuple},
};

/// Methods of memref values, on `Val<MemRefType>`.
macro_rules! memref_methods {
    ($($t:ty),*) => {$(
        impl $t {
            /// `memref.load` from this memref.
            #[track_caller]
            pub fn load(&self, indices: impl Operands) -> Val {
                try_load(self.as_val(), indices, ()).or_panic()
            }

            /// `memref.store` into this memref; a literal value takes its element type.
            #[track_caller]
            pub fn store(&self, value: impl Operand, indices: impl Operands) {
                try_store(value, self.as_val(), indices, ()).or_panic()
            }

            /// `memref.dealloc` of this memref.
            #[track_caller]
            pub fn dealloc(&self) {
                try_dealloc(self.as_val()).or_panic()
            }

            /// `memref.dim` of this memref.
            #[track_caller]
            pub fn dim(&self, index: impl Operand) -> Val<IndexType> {
                try_dim(self.as_val(), index).or_panic()
            }

            /// `memref.subview` of this memref; literals are static.
            #[track_caller]
            pub fn subview(&self, offsets: impl MixedList, sizes: impl MixedList, strides: impl MixedList) -> Val<MemRefType> {
                try_subview(self.as_val(), offsets, sizes, strides).or_panic()
            }

            /// `memref.cast` to a compatible type.
            #[track_caller]
            pub fn cast(&self, ty: impl Into<Ty>) -> Val {
                cast(self, ty)
            }

            /// `memref.memory_space_cast` to the same type in another memory space.
            #[track_caller]
            pub fn memory_space_cast(&self, memory_space: Option<Attr>) -> Val {
                memory_space_cast(self, memory_space)
            }

            /// `memref.copy` of this memref into `target`.
            #[track_caller]
            pub fn copy_to(&self, target: impl Into<Val>) {
                copy(self, target)
            }

            /// `memref.rank`
            #[track_caller]
            pub fn rank(&self) -> Val<IndexType> {
                rank(self)
            }

            /// `memref.extract_aligned_pointer_as_index`
            #[track_caller]
            pub fn extract_aligned_pointer_as_index(&self) -> Val<IndexType> {
                extract_aligned_pointer_as_index(self)
            }

            /// `memref.extract_strided_metadata`: base buffer, offset, sizes and strides.
            #[track_caller]
            pub fn extract_strided_metadata(&self) -> StridedMetadata {
                extract_strided_metadata(self)
            }

            /// `memref.assume_alignment`
            #[track_caller]
            pub fn assume_alignment(&self, alignment: u32) -> Val<MemRefType> {
                assume_alignment(self, alignment)
            }

            /// `memref.reinterpret_cast`; literals are static.
            #[track_caller]
            pub fn reinterpret_cast(&self, offset: impl MixedList, sizes: impl MixedList, strides: impl MixedList) -> Val<MemRefType> {
                reinterpret_cast(self, offset, sizes, strides)
            }

            /// `memref.reshape` to the shape stored in `shape`.
            #[track_caller]
            pub fn reshape(&self, shape: impl Into<Val>, ty: impl Into<Ty>) -> Val {
                reshape(self, shape, ty)
            }

            /// `memref.expand_shape`; literal sizes are static.
            #[track_caller]
            pub fn expand_shape(&self, reassociation: Reassociation, output_shape: impl MixedList) -> Val<MemRefType> {
                expand_shape(self, reassociation, output_shape)
            }

            /// `memref.collapse_shape`
            #[track_caller]
            pub fn collapse_shape(&self, reassociation: Reassociation) -> Val<MemRefType> {
                collapse_shape(self, reassociation)
            }

            /// `memref.transpose`: dimension `i` of the result is dimension `permutation[i]`.
            #[track_caller]
            pub fn transpose(&self, permutation: impl IntoIterator<Item = u32>) -> Val<MemRefType> {
                transpose(self, permutation)
            }

            /// `memref.view` of this 1-D `i8` buffer as `ty`.
            #[track_caller]
            pub fn view(&self, byte_shift: impl Operand, sizes: impl Operands, ty: impl Into<Ty>) -> Val<MemRefType> {
                view(self, byte_shift, sizes, ty)
            }

            /// `memref.realloc`; this memref must not be used afterwards.
            #[track_caller]
            pub fn realloc(&self, ty: impl Into<Ty>, size: Option<Val>) -> Val<MemRefType> {
                realloc(self, ty, size, None)
            }

            /// `memref.atomic_rmw` into an element, returning its previous value.
            #[track_caller]
            pub fn atomic_rmw(&self, kind: AtomicRMWKind, value: impl Operand, indices: impl Operands) -> Val {
                atomic_rmw(kind, value, self, indices)
            }

            /// `memref.generic_atomic_rmw`: replaces an element by `body(current)`.
            #[track_caller]
            pub fn generic_atomic_rmw(&self, indices: impl Operands, body: impl FnOnce(Val) -> Val) -> Val {
                generic_atomic_rmw(self, indices, body)
            }

            /// `memref.prefetch` of an element for reading or writing.
            #[track_caller]
            pub fn prefetch(&self, indices: impl Operands, is_write: bool, locality: u32, is_data_cache: bool) {
                prefetch(self, indices, is_write, locality, is_data_cache)
            }
        }
    )*};
}

memref_methods!(Val<MemRefType>);

pub(crate) fn check_alignment(op: &str, alignment: Option<u64>) -> Result<(), TypeError> {
    expect(op, alignment.is_none_or(u64::is_power_of_two), || {
        format!("alignment must be a power of two, not {}", alignment.unwrap())
    })
}

pub(crate) fn alignment_attr(alignment: Option<u64>) -> Vec<(String, Attr)> {
    alignment.map(|a| ("alignment".into(), Attr::Int(a as i64, I64.into()))).into_iter().collect()
}

/// The memory space of a ranked or unranked memref type.
fn memory_space(ty: &Ty) -> Option<Option<Attr>> {
    match (ty.as_memref(), ty.downcast_ref::<UnrankedMemRefType>()) {
        (Some(m), _) => Some(m.memory_space.clone()),
        (_, Some(u)) => Some(u.memory_space.clone()),
        _ => None,
    }
}

/// Optional attributes of loads and stores.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Access {
    /// The access is not expected to be reused soon.
    pub nontemporal: bool,
    /// Alignment in bytes, a power of two.
    pub alignment: Option<u64>,
    /// For loads only: the memory does not change while the memref is live.
    pub invariant: bool,
}

impl From<()> for Access {
    fn from(_: ()) -> Access {
        Access::default()
    }
}

impl Access {
    pub(crate) fn check(&self, op: &str) -> Result<(), TypeError> {
        expect(op, !self.invariant || op.ends_with("load"), || "only loads can be invariant".into())?;
        check_alignment(op, self.alignment)
    }

    pub(crate) fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = alignment_attr(self.alignment);
        if self.nontemporal {
            attrs.push(("nontemporal".into(), Attr::Bool(true)));
        }
        if self.invariant {
            attrs.push(("invariant".into(), Attr::Bool(true)));
        }
        attrs
    }
}

/// Optional operands and attributes of `memref.alloc` and `memref.alloca`.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct AllocOpts {
    /// Values of the symbols of an affine map layout.
    pub symbols: Vec<Val>,
    /// Alignment in bytes, a power of two.
    pub alignment: Option<u64>,
}

impl From<()> for AllocOpts {
    fn from(_: ()) -> AllocOpts {
        AllocOpts::default()
    }
}

macro_rules! alloc_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub ty: Ty,
            /// One `index` size per dynamic dimension.
            pub sizes: Vec<Val>,
            /// Values of the symbols of an affine map layout.
            pub symbols: Vec<Val>,
            pub alignment: Option<u64>,
        }

        $(#[$doc])*
        /// Takes one size per dynamic dimension.
        #[track_caller]
        pub fn $try_f(ty: impl Into<Ty>, sizes: impl Operands, opts: impl Into<AllocOpts>) -> Result<Val<MemRefType>, TypeError> {
            let name = concat!("memref.", stringify!($f));
            let (ty, AllocOpts { symbols, alignment }) = (ty.into(), opts.into());
            let sizes = indices(name, sizes)?;
            for s in &symbols {
                expect(name, value_ty(name, s)?.is_index(), || "symbols must be `index`".into())?;
            }
            let Some(m) = ty.as_memref() else {
                return Err(TypeError::new(name, format!("expected a memref type, not `{ty}`")));
            };
            let dynamic = m.shape.iter().filter(|d| d.is_none()).count();
            expect(name, sizes.len() == dynamic, || format!("`{ty}` needs {dynamic} sizes, got {}", sizes.len()))?;
            let needed = match &m.layout {
                Layout::Map(map) => map.symbols as usize,
                _ => 0,
            };
            expect(name, symbols.len() == needed, || {
                format!("`{ty}` needs {needed} symbol operands, got {}", symbols.len())
            })?;
            check_alignment(name, alignment)?;
            Val::new_typed($op { ty, sizes, symbols, alignment })
        }

        $(#[$doc])*
        /// Takes one size per dynamic dimension.
        #[track_caller]
        pub fn $f(ty: impl Into<Ty>, sizes: impl Operands, opts: impl Into<AllocOpts>) -> Val<MemRefType> {
            $try_f(ty, sizes, opts).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("memref.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(self.sizes.iter().chain(&self.symbols).cloned())
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![self.ty.clone()]
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                let mut attrs = vec![Attr::segments([self.sizes.len(), self.symbols.len()])];
                attrs.extend(alignment_attr(self.alignment));
                attrs
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let n = self.sizes.len();
                write!(p, "{}({})", concat!("memref.", stringify!($f)), p.operands(0..n))?;
                if !self.symbols.is_empty() {
                    write!(p, "[{}]", p.operands(n..n + self.symbols.len()))?;
                }
                p.attr_dict(&alignment_attr(self.alignment))?;
                write!(p, " : {}", self.ty)
            }
        }
    )*};
}

alloc_ops! {
    /// `memref.alloc`; one `index` size per dynamic dimension.
    Alloc = alloc / try_alloc;
    /// `memref.alloca`: stack allocation, freed when the enclosing
    /// function or [`AllocaScope`] returns.
    Alloca = alloca / try_alloca;
}

/// `memref.realloc` of a 1-D memref; the source must not be used afterwards.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Realloc {
    pub source: Val,
    pub ty: Ty,
    /// The size, if the result is dynamic.
    pub size: Option<Val>,
    pub alignment: Option<u64>,
}

/// `memref.realloc` of `source` to `ty`, with a `size` if `ty` is dynamic.
#[track_caller]
pub fn try_realloc(
    source: impl Into<Val>,
    ty: impl Into<Ty>,
    size: Option<Val>,
    alignment: Option<u64>,
) -> Result<Val<MemRefType>, TypeError> {
    let name = "memref.realloc";
    let (source, ty) = (source.into(), ty.into());
    let (src_ty, src) = memref_ty(name, &source)?;
    let Some(dst) = ty.as_memref() else {
        return Err(TypeError::new(name, format!("expected a memref type, not `{ty}`")));
    };
    let one_d = |m: &MemRefType| m.rank() == 1 && m.layout == Layout::Identity;
    expect(name, one_d(&src) && one_d(dst) && src.element == dst.element && src.memory_space == dst.memory_space, || {
        format!("cannot reallocate `{src_ty}` to `{ty}`")
    })?;
    let size = match size {
        Some(s) => Some(indices(name, s)?.remove(0)),
        None => None,
    };
    expect(name, size.is_some() == dst.shape[0].is_none(), || format!("a size is needed exactly when `{ty}` is dynamic"))?;
    check_alignment(name, alignment)?;
    Val::new_typed(Realloc { source, ty, size, alignment })
}

/// `memref.realloc` of `source` to `ty`, with a `size` if `ty` is dynamic.
#[track_caller]
pub fn realloc(source: impl Into<Val>, ty: impl Into<Ty>, size: Option<Val>, alignment: Option<u64>) -> Val<MemRefType> {
    try_realloc(source, ty, size, alignment).or_panic()
}

impl HighMlir for Realloc {
    fn name(&self) -> &str {
        "memref.realloc"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()).chain(self.size.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        alignment_attr(self.alignment)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "memref.realloc {}", p.operand(0))?;
        if self.size.is_some() {
            write!(p, "({})", p.operand(1))?;
        }
        p.attr_dict(&self.attributes())?;
        write!(p, " : {} to {}", p.operand_type(0), self.ty)
    }
}

/// `memref.load`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Load {
    pub memref: Val,
    pub indices: Vec<Val>,
    pub access: Access,
}

/// `memref.load`; literal indices become `index` constants.
#[track_caller]
pub fn try_load(memref: impl Into<Val>, indices: impl Operands, access: impl Into<Access>) -> Result<Val, TypeError> {
    let name = "memref.load";
    let (memref, access) = (memref.into(), access.into());
    let indices = self::indices(name, indices)?;
    let (ty, m) = memref_ty(name, &memref)?;
    check_rank(name, &ty, m.rank(), indices.len())?;
    access.check(name)?;
    Val::single(Load { memref, indices, access })
}

/// `memref.load`; literal indices become `index` constants.
#[track_caller]
pub fn load(memref: impl Into<Val>, indices: impl Operands, access: impl Into<Access>) -> Val {
    try_load(memref, indices, access).or_panic()
}

impl HighMlir for Load {
    fn name(&self) -> &str {
        "memref.load"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.memref.clone()).chain(self.indices.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.memref.ty().element().clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.access.attributes()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = self.indices.len();
        write!(p, "memref.load {}[{}]", p.operand(0), p.operands(1..=n))?;
        p.attr_dict(&self.attributes())?;
        write!(p, " : {}", p.operand_type(0))
    }
}

/// `memref.store`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Store {
    pub value: Val,
    pub memref: Val,
    pub indices: Vec<Val>,
    pub access: Access,
}

/// `memref.store`; a literal value takes the element type.
#[track_caller]
pub fn try_store(
    value: impl Operand,
    memref: impl Into<Val>,
    indices: impl Operands,
    access: impl Into<Access>,
) -> Result<(), TypeError> {
    let name = "memref.store";
    let (memref, access) = (memref.into(), access.into());
    let indices = self::indices(name, indices)?;
    let (ty, m) = memref_ty(name, &memref)?;
    check_rank(name, &ty, m.rank(), indices.len())?;
    let value = value.into_val(&m.element).map_err(|e| TypeError::new(name, e.message))?;
    let value_ty = value_ty(name, &value)?;
    expect(name, value_ty == m.element, || format!("cannot store `{value_ty}` into `{ty}`"))?;
    access.check(name)?;
    Val::unit(Store { value, memref, indices, access })
}

/// `memref.store`; a literal value takes the element type.
#[track_caller]
pub fn store(value: impl Operand, memref: impl Into<Val>, indices: impl Operands, access: impl Into<Access>) {
    try_store(value, memref, indices, access).or_panic()
}

impl HighMlir for Store {
    fn name(&self) -> &str {
        "memref.store"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.value.clone(), self.memref.clone()].into_iter().chain(self.indices.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.access.attributes()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let n = self.indices.len();
        write!(p, "memref.store {}, {}[{}]", p.operand(0), p.operand(1), p.operands(2..2 + n))?;
        p.attr_dict(&self.attributes())?;
        write!(p, " : {}", p.operand_type(1))
    }
}

/// `memref.dealloc`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Dealloc {
    pub memref: Val,
}

/// `memref.dealloc`
#[track_caller]
pub fn try_dealloc(memref: impl Into<Val>) -> Result<(), TypeError> {
    let memref = memref.into();
    let ty = value_ty("memref.dealloc", &memref)?;
    expect("memref.dealloc", ty.is_memref_like(), || format!("expected a memref, not `{ty}`"))?;
    Val::unit(Dealloc { memref })
}

/// `memref.dealloc`
#[track_caller]
pub fn dealloc(memref: impl Into<Val>) {
    try_dealloc(memref).or_panic()
}

impl HighMlir for Dealloc {
    fn name(&self) -> &str {
        "memref.dealloc"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.memref.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        write!(p, "memref.dealloc {} : {}", p.operand(0), p.operand_type(0))
    }
}

/// `memref.copy` between memrefs of the same shape and element type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Copy {
    pub source: Val,
    pub target: Val,
}

/// `memref.copy`
#[track_caller]
pub fn try_copy(source: impl Into<Val>, target: impl Into<Val>) -> Result<(), TypeError> {
    let name = "memref.copy";
    let (source, target) = (source.into(), target.into());
    let (s, t) = (value_ty(name, &source)?, value_ty(name, &target)?);
    let shapes = match (s.shape(), t.shape()) {
        (Some(a), Some(b)) => compatible_shapes(a, b),
        _ => true,
    };
    expect(name, s.is_memref_like() && t.is_memref_like() && s.element() == t.element() && shapes, || {
        format!("cannot copy `{s}` to `{t}`")
    })?;
    Val::unit(Copy { source, target })
}

/// `memref.copy`
#[track_caller]
pub fn copy(source: impl Into<Val>, target: impl Into<Val>) {
    try_copy(source, target).or_panic()
}

impl HighMlir for Copy {
    fn name(&self) -> &str {
        "memref.copy"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.target.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let (s, t) = (p.operand(0), p.operand(1));
        write!(p, "memref.copy {s}, {t} : {} to {}", p.operand_type(0), p.operand_type(1))
    }
}

/// Dimensions of ranked types, compatible when equal or either is dynamic.
fn compatible_shapes(a: &[Option<u64>], b: &[Option<u64>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.is_none() || y.is_none() || x == y)
}

/// `memref.cast` to a compatible type: between ranked and unranked, or
/// between static and dynamic dimensions, strides and offset.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Cast {
    pub source: Val,
    pub ty: Ty,
}

/// `memref.cast`
#[track_caller]
pub fn try_cast(source: impl Into<Val>, ty: impl Into<Ty>) -> Result<Val, TypeError> {
    let name = "memref.cast";
    let (source, ty) = (source.into(), ty.into());
    let src = value_ty(name, &source)?;
    let (s, d) = (memory_space(&src), memory_space(&ty));
    let shapes = match (src.as_memref(), ty.as_memref()) {
        (Some(a), Some(b)) => compatible_shapes(&a.shape, &b.shape),
        (None, None) => false,
        _ => true,
    };
    expect(name, s.is_some() && s == d && src.element() == ty.element() && shapes, || {
        format!("cannot cast `{src}` to `{ty}`")
    })?;
    Val::single(Cast { source, ty })
}

/// `memref.cast`
#[track_caller]
pub fn cast(source: impl Into<Val>, ty: impl Into<Ty>) -> Val {
    try_cast(source, ty).or_panic()
}

/// `$op %source : type to type` operations whose result is a view of the
/// source, with their `HighMlir` boilerplate.
macro_rules! view_cast_impl {
    ($op:ident = $name:literal) => {
        impl HighMlir for $op {
            fn name(&self) -> &str {
                $name
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(std::iter::once(self.source.clone()))
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![self.ty.clone()]
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                write!(p, "{} {} : {} to {}", $name, p.operand(0), p.operand_type(0), self.ty)
            }
        }
    };
}

view_cast_impl!(Cast = "memref.cast");

/// `memref.memory_space_cast` to the same type in another memory space.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MemorySpaceCast {
    pub source: Val,
    pub ty: Ty,
}

/// `memref.memory_space_cast`
#[track_caller]
pub fn try_memory_space_cast(source: impl Into<Val>, memory_space: Option<Attr>) -> Result<Val, TypeError> {
    let name = "memref.memory_space_cast";
    let source = source.into();
    let src = value_ty(name, &source)?;
    let ty: Ty = match (src.as_memref(), src.downcast_ref::<UnrankedMemRefType>()) {
        (Some(m), _) => m.clone().with_memory_space(memory_space).into(),
        (_, Some(u)) => UnrankedMemRefType { memory_space, ..u.clone() }.into(),
        _ => return Err(TypeError::new(name, format!("expected a memref, not `{src}`"))),
    };
    Val::single(MemorySpaceCast { source, ty })
}

/// `memref.memory_space_cast`
#[track_caller]
pub fn memory_space_cast(source: impl Into<Val>, memory_space: Option<Attr>) -> Val {
    try_memory_space_cast(source, memory_space).or_panic()
}

view_cast_impl!(MemorySpaceCast = "memref.memory_space_cast");

/// `memref.assume_alignment`: the same memref, known to be aligned.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct AssumeAlignment {
    pub memref: Val,
    pub alignment: u32,
}

/// `memref.assume_alignment`
#[track_caller]
pub fn try_assume_alignment(memref: impl Into<Val>, alignment: u32) -> Result<Val<MemRefType>, TypeError> {
    let name = "memref.assume_alignment";
    let memref = memref.into();
    memref_ty(name, &memref)?;
    check_alignment(name, Some(alignment as u64))?;
    Val::new_typed(AssumeAlignment { memref, alignment })
}

/// `memref.assume_alignment`
#[track_caller]
pub fn assume_alignment(memref: impl Into<Val>, alignment: u32) -> Val<MemRefType> {
    try_assume_alignment(memref, alignment).or_panic()
}

impl HighMlir for AssumeAlignment {
    fn name(&self) -> &str {
        "memref.assume_alignment"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.memref.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.memref.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("alignment".into(), Attr::Int(self.alignment as i64, I32.into()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "memref.assume_alignment {}, {} : {}", p.operand(0), self.alignment, p.operand_type(0))
    }
}

/// `memref.distinct_objects`: the same memrefs, known not to alias.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct DistinctObjects {
    pub memrefs: Vec<Val>,
}

/// `memref.distinct_objects`
#[track_caller]
pub fn try_distinct_objects(memrefs: impl IntoIterator<Item = impl Into<Val>>) -> Result<Tuple, TypeError> {
    let name = "memref.distinct_objects";
    let memrefs: Vec<Val> = memrefs.into_iter().map(Into::into).collect();
    for m in &memrefs {
        memref_ty(name, m)?;
    }
    expect(name, !memrefs.is_empty(), || "needs at least one memref".into())?;
    Val::all_results(DistinctObjects { memrefs })
}

/// `memref.distinct_objects`: the same memrefs, known not to alias.
#[track_caller]
pub fn distinct_objects(memrefs: impl IntoIterator<Item = impl Into<Val>>) -> Tuple {
    try_distinct_objects(memrefs).or_panic()
}

impl HighMlir for DistinctObjects {
    fn name(&self) -> &str {
        "memref.distinct_objects"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.memrefs.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        self.memrefs.iter().map(|m| m.ty()).collect()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = self.memrefs.len();
        write!(p, "memref.distinct_objects {} : {}", p.operands(0..n), TypeList(&p.operand_types(0..n)))
    }
}

/// `memref.dim`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Dim {
    pub source: Val,
    pub index: Val,
}

/// A literal `index` becomes an `index` constant.
#[track_caller]
pub fn try_dim(source: impl Into<Val>, index: impl Operand) -> Result<Val<IndexType>, TypeError> {
    let name = "memref.dim";
    let source = source.into();
    let ty = value_ty(name, &source)?;
    expect(name, ty.is_memref_like() && ty.shape() != Some(&[]), || format!("expected a non 0-d memref, not `{ty}`"))?;
    let index = indices(name, index)?.remove(0);
    Val::new_typed(Dim { source, index })
}

/// `memref.dim`; a literal `index` becomes an `index` constant.
#[track_caller]
pub fn dim(source: impl Into<Val>, index: impl Operand) -> Val<IndexType> {
    try_dim(source, index).or_panic()
}

impl HighMlir for Dim {
    fn name(&self) -> &str {
        "memref.dim"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.index.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![INDEX.into()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "memref.dim {}, {} : {}", p.operand(0), p.operand(1), p.operand_type(0))
    }
}

/// `$op %source : type` operations returning an `index`.
macro_rules! index_query_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident, $suffix:literal;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub source: Val,
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(source: impl Into<Val>) -> Result<Val<IndexType>, TypeError> {
            let name = concat!("memref.", stringify!($f));
            let source = source.into();
            let ty = value_ty(name, &source)?;
            expect(name, ty.is_memref_like(), || format!("expected a memref, not `{ty}`"))?;
            Val::new_typed($op { source })
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f(source: impl Into<Val>) -> Val<IndexType> {
            $try_f(source).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("memref.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(std::iter::once(self.source.clone()))
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![INDEX.into()]
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                write!(p, "{} {} : {}{}", concat!("memref.", stringify!($f)), p.operand(0), p.operand_type(0), $suffix)
            }
        }
    )*};
}

index_query_ops! {
    /// `memref.rank`
    Rank = rank / try_rank, "";
    /// `memref.extract_aligned_pointer_as_index`
    ExtractAlignedPointerAsIndex = extract_aligned_pointer_as_index / try_extract_aligned_pointer_as_index, " -> index";
}

/// The base buffer, offset, sizes and strides of a memref.
pub type StridedMetadata = (Val<MemRefType>, Val<IndexType>, Vec<Val<IndexType>>, Vec<Val<IndexType>>);

/// `memref.extract_strided_metadata`: the base buffer (a 0-d view of the
/// source), offset, sizes and strides.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ExtractStridedMetadata {
    pub source: Val,
}

/// `memref.extract_strided_metadata`
#[track_caller]
pub fn try_extract_strided_metadata(source: impl Into<Val>) -> Result<StridedMetadata, TypeError> {
    let name = "memref.extract_strided_metadata";
    let source = source.into();
    let (ty, m) = memref_ty(name, &source)?;
    expect(name, m.strides_and_offset().is_some(), || format!("`{ty}` is not strided"))?;
    let mut results = Val::all_results(ExtractStridedMetadata { source })?.0.into_iter();
    let base = results.next().unwrap().downcast().unwrap();
    let mut index = || results.next().unwrap().downcast::<IndexType>().unwrap();
    let offset = index();
    let sizes = (0..m.rank()).map(|_| index()).collect();
    let strides = (0..m.rank()).map(|_| index()).collect();
    Ok((base, offset, sizes, strides))
}

/// `memref.extract_strided_metadata`
#[track_caller]
pub fn extract_strided_metadata(source: impl Into<Val>) -> StridedMetadata {
    try_extract_strided_metadata(source).or_panic()
}

impl HighMlir for ExtractStridedMetadata {
    fn name(&self) -> &str {
        "memref.extract_strided_metadata"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        let m = self.source.ty().as_memref().unwrap().clone();
        let base = MemRefType::new([], m.element.clone()).with_memory_space(m.memory_space.clone());
        let index = || -> Ty { INDEX.into() };
        std::iter::once(base.into()).chain((0..1 + 2 * m.rank()).map(|_| index())).collect()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let results = self.result_types();
        write!(p, "memref.extract_strided_metadata {} : {} -> {}", p.operand(0), p.operand_type(0), TypeList(&results))
    }
}

/// `memref.reshape` to the shape stored in a 1-D memref.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Reshape {
    pub source: Val,
    pub shape: Val,
    pub ty: Ty,
}

/// `memref.reshape`
#[track_caller]
pub fn try_reshape(source: impl Into<Val>, shape: impl Into<Val>, ty: impl Into<Ty>) -> Result<Val, TypeError> {
    let name = "memref.reshape";
    let (source, shape, ty) = (source.into(), shape.into(), ty.into());
    let src = value_ty(name, &source)?;
    let (shape_ty, s) = memref_ty(name, &shape)?;
    let ok_shape = s.rank() == 1 && (s.element.is_index() || s.element.int_width().is_some());
    expect(name, ok_shape, || format!("the shape must be a 1-D memref of integers, not `{shape_ty}`"))?;
    let ranks = match ty.as_memref() {
        Some(m) => s.shape[0].is_none_or(|n| n as usize == m.rank()),
        None => s.shape[0].is_none(),
    };
    let ok = src.is_memref_like() && ty.is_memref_like() && src.element() == ty.element();
    expect(name, ok && ranks, || format!("cannot reshape `{src}` to `{ty}` with `{shape_ty}`"))?;
    Val::single(Reshape { source, shape, ty })
}

/// `memref.reshape`
#[track_caller]
pub fn reshape(source: impl Into<Val>, shape: impl Into<Val>, ty: impl Into<Ty>) -> Val {
    try_reshape(source, shape, ty).or_panic()
}

impl HighMlir for Reshape {
    fn name(&self) -> &str {
        "memref.reshape"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.shape.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (s, shape) = (p.operand(0), p.operand(1));
        let types = TypeList(&p.operand_types(0..2)).to_string();
        write!(p, "memref.reshape {s}({shape}) : ({types}) -> {}", self.ty)
    }
}

/// Groups of consecutive dimensions, e.g. `[[0, 1], [2]]`.
pub type Reassociation = Vec<Vec<u32>>;

pub(crate) fn reassociation_attr(r: &Reassociation) -> Attr {
    Attr::List(r.iter().map(|g| Attr::i64_list(g.iter().map(|&d| d as i64))).collect())
}

pub(crate) fn fmt_reassociation(r: &Reassociation) -> String {
    let groups: Vec<String> =
        r.iter().map(|g| format!("[{}]", g.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))).collect();
    format!("[{}]", groups.join(", "))
}

/// Checks that `r` partitions `0..rank` into consecutive, non-empty groups.
pub(crate) fn check_reassociation(op: &str, r: &Reassociation, rank: usize) -> Result<(), TypeError> {
    let flat: Vec<u32> = r.iter().flatten().copied().collect();
    let ok = r.iter().all(|g| !g.is_empty()) && flat == (0..rank as u32).collect::<Vec<_>>();
    expect(op, ok, || format!("{} is not a reassociation of {rank} dimensions", fmt_reassociation(r)))
}

fn mul(a: Option<i64>, b: Option<i64>) -> Option<i64> {
    a.zip(b).map(|(a, b)| a * b)
}

/// `memref.expand_shape`: splits each source dimension into a group of dimensions.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ExpandShape {
    pub source: Val,
    pub reassociation: Reassociation,
    pub output_shape: MixedValues,
    pub ty: Ty,
}

/// Infers the result type from `output_shape` (literals are static).
#[track_caller]
pub fn try_expand_shape(
    source: impl Into<Val>,
    reassociation: Reassociation,
    output_shape: impl MixedList,
) -> Result<Val<MemRefType>, TypeError> {
    let name = "memref.expand_shape";
    let source = source.into();
    let (src_ty, m) = memref_ty(name, &source)?;
    let output_shape = MixedValues::new(name, output_shape)?;
    let shape: Vec<Option<u64>> = output_shape.static_values().iter().map(|d| d.map(|d| d as u64)).collect();
    check_reassociation(name, &reassociation, shape.len())?;
    let layout = match m.strides_and_offset() {
        _ if m.layout == Layout::Identity => Layout::Identity,
        Some((strides, offset)) => {
            let mut out = vec![None; shape.len()];
            for (g, group) in reassociation.iter().enumerate() {
                let mut stride = strides.get(g).copied().flatten();
                for &d in group.iter().rev() {
                    out[d as usize] = stride;
                    stride = mul(stride, shape[d as usize].map(|n| n as i64));
                }
            }
            Layout::Strided { strides: out, offset }
        }
        None => return Err(TypeError::new(name, format!("`{src_ty}` is not strided"))),
    };
    let ty = MemRefType { shape, layout, ..m }.into();
    try_expand_shape_with_type(source, reassociation, output_shape, ty)
}

/// `memref.expand_shape` with an explicit result type.
#[track_caller]
pub fn try_expand_shape_with_type(
    source: impl Into<Val>,
    reassociation: Reassociation,
    output_shape: MixedValues,
    ty: Ty,
) -> Result<Val<MemRefType>, TypeError> {
    let name = "memref.expand_shape";
    let source = source.into();
    let (src_ty, m) = memref_ty(name, &source)?;
    let Some(dst) = ty.as_memref() else {
        return Err(TypeError::new(name, format!("expected a memref type, not `{ty}`")));
    };
    expect(name, reassociation.len() == m.rank() && dst.rank() == output_shape.len() && dst.element == m.element, || {
        format!("cannot expand `{src_ty}` to `{ty}`")
    })?;
    check_reassociation(name, &reassociation, dst.rank())?;
    for (g, group) in reassociation.iter().enumerate() {
        let size = group.iter().try_fold(1, |acc, &d| dst.shape[d as usize].map(|n| acc * n));
        expect(name, size == m.shape[g], || format!("dimension {g} of `{src_ty}` does not match `{ty}`"))?;
    }
    Val::new_typed(ExpandShape { source, reassociation, output_shape, ty })
}

/// `memref.expand_shape`
#[track_caller]
pub fn expand_shape(source: impl Into<Val>, reassociation: Reassociation, output_shape: impl MixedList) -> Val<MemRefType> {
    try_expand_shape(source, reassociation, output_shape).or_panic()
}

impl HighMlir for ExpandShape {
    fn name(&self) -> &str {
        "memref.expand_shape"
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
        write!(p, "memref.expand_shape {} {r} output_shape {shape} : {} into {}", p.operand(0), p.operand_type(0), self.ty)
    }
}

/// `memref.collapse_shape`: merges each group of contiguous dimensions.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CollapseShape {
    pub source: Val,
    pub reassociation: Reassociation,
    pub ty: Ty,
}

/// `memref.collapse_shape`
#[track_caller]
pub fn try_collapse_shape(source: impl Into<Val>, reassociation: Reassociation) -> Result<Val<MemRefType>, TypeError> {
    let name = "memref.collapse_shape";
    let source = source.into();
    let (src_ty, m) = memref_ty(name, &source)?;
    check_reassociation(name, &reassociation, m.rank())?;
    let shape: Vec<Option<u64>> = reassociation
        .iter()
        .map(|g| g.iter().map(|&d| m.shape[d as usize]).try_fold(1, |acc, d| d.map(|d| acc * d)))
        .collect();
    let layout = match m.strides_and_offset() {
        _ if m.layout == Layout::Identity => Layout::Identity,
        Some((strides, offset)) => {
            let strides = reassociation.iter().map(|g| strides[*g.last().unwrap() as usize]).collect();
            Layout::Strided { strides, offset }
        }
        None => return Err(TypeError::new(name, format!("`{src_ty}` is not strided"))),
    };
    let ty = MemRefType { shape, layout, ..m }.into();
    try_collapse_shape_with_type(source, reassociation, ty)
}

/// `memref.collapse_shape` with an explicit result type.
#[track_caller]
pub fn try_collapse_shape_with_type(source: impl Into<Val>, reassociation: Reassociation, ty: Ty) -> Result<Val<MemRefType>, TypeError> {
    let name = "memref.collapse_shape";
    let source = source.into();
    let (src_ty, m) = memref_ty(name, &source)?;
    check_reassociation(name, &reassociation, m.rank())?;
    let ok = ty.as_memref().is_some_and(|d| d.rank() == reassociation.len() && d.element == m.element);
    expect(name, ok, || format!("cannot collapse `{src_ty}` to `{ty}`"))?;
    Val::new_typed(CollapseShape { source, reassociation, ty })
}

/// `memref.collapse_shape`
#[track_caller]
pub fn collapse_shape(source: impl Into<Val>, reassociation: Reassociation) -> Val<MemRefType> {
    try_collapse_shape(source, reassociation).or_panic()
}

impl HighMlir for CollapseShape {
    fn name(&self) -> &str {
        "memref.collapse_shape"
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
        write!(p, "memref.collapse_shape {} {r} : {} into {}", p.operand(0), p.operand_type(0), self.ty)
    }
}

/// Offsets, sizes and strides of `memref.subview` and `memref.reinterpret_cast`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Window {
    pub offsets: MixedValues,
    pub sizes: MixedValues,
    pub strides: MixedValues,
}

impl Window {
    pub(crate) fn new(op: &str, offsets: impl MixedList, sizes: impl MixedList, strides: impl MixedList) -> Result<Window, TypeError> {
        let offsets = MixedValues::new(op, offsets)?;
        let sizes = MixedValues::new(op, sizes)?;
        let strides = MixedValues::new(op, strides)?;
        Ok(Window { offsets, sizes, strides })
    }

    pub(crate) fn dynamic(&self) -> impl Iterator<Item = Val> + '_ {
        self.offsets.dynamic.iter().chain(&self.sizes.dynamic).chain(&self.strides.dynamic).cloned()
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.attributes_after(1)
    }

    /// The attributes, after `leading` single operands (e.g. a source and a destination).
    pub(crate) fn attributes_after(&self, leading: usize) -> Vec<(String, Attr)> {
        let (o, s, t) = (&self.offsets, &self.sizes, &self.strides);
        let segments = std::iter::repeat_n(1, leading).chain([o.dynamic.len(), s.dynamic.len(), t.dynamic.len()]);
        vec![
            Attr::segments(segments),
            ("static_offsets".into(), Attr::i64_array(o.statics.clone())),
            ("static_sizes".into(), Attr::i64_array(s.statics.clone())),
            ("static_strides".into(), Attr::i64_array(t.statics.clone())),
        ]
    }

    /// The three lists, with the dynamic entries named after operands `1..`.
    fn format(&self, p: &OpPrinter<'_>) -> [String; 3] {
        self.format_from(p, 1)
    }

    /// The three lists, with the dynamic entries named after operands `first..`.
    pub(crate) fn format_from(&self, p: &OpPrinter<'_>, first: usize) -> [String; 3] {
        let (no, ns) = (self.offsets.dynamic.len(), self.sizes.dynamic.len());
        [
            self.offsets.format(|k| p.operand(first + k)),
            self.sizes.format(|k| p.operand(first + no + k)),
            self.strides.format(|k| p.operand(first + no + ns + k)),
        ]
    }

    pub(crate) fn static_shape(&self) -> Vec<Option<u64>> {
        self.sizes.static_values().iter().map(|d| d.map(|d| d as u64)).collect()
    }
}

/// `memref.subview`: a strided window of the source.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SubView {
    pub source: Val,
    pub window: Window,
    pub ty: Ty,
}

/// Infers the strided result type; literals are static.
#[track_caller]
pub fn try_subview(
    source: impl Into<Val>,
    offsets: impl MixedList,
    sizes: impl MixedList,
    strides: impl MixedList,
) -> Result<Val<MemRefType>, TypeError> {
    let name = "memref.subview";
    let source = source.into();
    let (src_ty, m) = memref_ty(name, &source)?;
    let window = Window::new(name, offsets, sizes, strides)?;
    let Some((src_strides, src_offset)) = m.strides_and_offset() else {
        return Err(TypeError::new(name, format!("`{src_ty}` is not strided")));
    };
    expect(name, [&window.offsets, &window.sizes, &window.strides].iter().all(|l| l.len() == m.rank()), || {
        format!("needs {} offsets, sizes and strides for `{src_ty}`", m.rank())
    })?;
    let strides = src_strides.iter().zip(window.strides.static_values()).map(|(&a, b)| mul(a, b)).collect();
    let offset = src_strides
        .iter()
        .zip(window.offsets.static_values())
        .try_fold(src_offset, |acc, (&s, o)| Some(acc.zip(mul(s, o)).map(|(a, b)| a + b)))
        .flatten();
    let ty = MemRefType { shape: window.static_shape(), layout: Layout::Strided { strides, offset }, ..m }.into();
    try_subview_with_type(source, window, ty)
}

/// With an explicit result type, e.g. dropping unit dimensions.
#[track_caller]
pub fn try_subview_with_type(source: impl Into<Val>, window: Window, ty: Ty) -> Result<Val<MemRefType>, TypeError> {
    let name = "memref.subview";
    let source = source.into();
    let (src_ty, m) = memref_ty(name, &source)?;
    let ok = ty.as_memref().is_some_and(|d| d.element == m.element && d.rank() <= m.rank());
    expect(name, ok && window.sizes.len() == m.rank(), || format!("cannot take a `{ty}` subview of `{src_ty}`"))?;
    Val::new_typed(SubView { source, window, ty })
}

/// `memref.subview`
#[track_caller]
pub fn subview(source: impl Into<Val>, offsets: impl MixedList, sizes: impl MixedList, strides: impl MixedList) -> Val<MemRefType> {
    try_subview(source, offsets, sizes, strides).or_panic()
}

impl HighMlir for SubView {
    fn name(&self) -> &str {
        "memref.subview"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()).chain(self.window.dynamic()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.window.attributes()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let [o, s, t] = self.window.format(p);
        write!(p, "memref.subview {}{o} {s} {t} : {} to {}", p.operand(0), p.operand_type(0), self.ty)
    }
}

/// `memref.reinterpret_cast`: the source's buffer with a new offset, sizes and strides.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ReinterpretCast {
    pub source: Val,
    pub window: Window,
    pub ty: Ty,
}

/// `offset` is one value; literals are static.
#[track_caller]
pub fn try_reinterpret_cast(
    source: impl Into<Val>,
    offset: impl MixedList,
    sizes: impl MixedList,
    strides: impl MixedList,
) -> Result<Val<MemRefType>, TypeError> {
    let name = "memref.reinterpret_cast";
    let source = source.into();
    let src = value_ty(name, &source)?;
    let window = Window::new(name, offset, sizes, strides)?;
    expect(name, src.is_memref_like() && window.offsets.len() == 1 && window.sizes.len() == window.strides.len(), || {
        format!("needs a memref, one offset and as many sizes as strides, not `{src}`")
    })?;
    let offset = window.offsets.static_values()[0];
    let layout = Layout::Strided { strides: window.strides.static_values(), offset };
    let ty = MemRefType::new(window.static_shape(), src.element().clone())
        .with_layout(layout)
        .with_memory_space(memory_space(&src).flatten())
        .into();
    Val::new_typed(ReinterpretCast { source, window, ty })
}

/// `memref.reinterpret_cast`
#[track_caller]
pub fn reinterpret_cast(
    source: impl Into<Val>,
    offset: impl MixedList,
    sizes: impl MixedList,
    strides: impl MixedList,
) -> Val<MemRefType> {
    try_reinterpret_cast(source, offset, sizes, strides).or_panic()
}

impl HighMlir for ReinterpretCast {
    fn name(&self) -> &str {
        "memref.reinterpret_cast"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()).chain(self.window.dynamic()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.window.attributes()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let [o, s, t] = self.window.format(p);
        let (src, src_ty) = (p.operand(0), p.operand_type(0));
        write!(p, "memref.reinterpret_cast {src} to offset: {o}, sizes: {s}, strides: {t} : {src_ty} to {}", self.ty)
    }
}

/// `memref.transpose`: dimension `i` of the result is dimension
/// `permutation[i]` of the source.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Transpose {
    pub source: Val,
    pub permutation: Vec<u32>,
    pub ty: Ty,
}

/// `memref.transpose`
#[track_caller]
pub fn try_transpose(source: impl Into<Val>, permutation: impl IntoIterator<Item = u32>) -> Result<Val<MemRefType>, TypeError> {
    let name = "memref.transpose";
    let source = source.into();
    let permutation: Vec<u32> = permutation.into_iter().collect();
    let (src_ty, m) = memref_ty(name, &source)?;
    let mut sorted = permutation.clone();
    sorted.sort();
    expect(name, sorted == (0..m.rank() as u32).collect::<Vec<_>>(), || {
        format!("{permutation:?} is not a permutation of the dimensions of `{src_ty}`")
    })?;
    let Some((strides, offset)) = m.strides_and_offset() else {
        return Err(TypeError::new(name, format!("`{src_ty}` is not strided")));
    };
    let shape = permutation.iter().map(|&d| m.shape[d as usize]).collect();
    let strides = permutation.iter().map(|&d| strides[d as usize]).collect();
    let ty = MemRefType { shape, layout: Layout::Strided { strides, offset }, ..m }.into();
    Val::new_typed(Transpose { source, permutation, ty })
}

/// `memref.transpose`
#[track_caller]
pub fn transpose(source: impl Into<Val>, permutation: impl IntoIterator<Item = u32>) -> Val<MemRefType> {
    try_transpose(source, permutation).or_panic()
}

impl HighMlir for Transpose {
    fn name(&self) -> &str {
        "memref.transpose"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.source.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("permutation".into(), Attr::AffineMap(AffineMap::permutation(&self.permutation)))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let map = AffineMap::permutation(&self.permutation);
        write!(p, "memref.transpose {} {map} : {} to {}", p.operand(0), p.operand_type(0), self.ty)
    }
}

/// `memref.view`: a memref of any element type over a 1-D `i8` buffer.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct View {
    pub source: Val,
    pub byte_shift: Val,
    /// One size per dynamic dimension of the result.
    pub sizes: Vec<Val>,
    pub ty: Ty,
}

/// `memref.view`
#[track_caller]
pub fn try_view(
    source: impl Into<Val>,
    byte_shift: impl Operand,
    sizes: impl Operands,
    ty: impl Into<Ty>,
) -> Result<Val<MemRefType>, TypeError> {
    let name = "memref.view";
    let (source, ty) = (source.into(), ty.into());
    let (src_ty, m) = memref_ty(name, &source)?;
    let byte_shift = indices(name, byte_shift)?.remove(0);
    let sizes = indices(name, sizes)?;
    expect(name, m.rank() == 1 && m.element.int_width() == Some(8) && m.layout == Layout::Identity, || {
        format!("the source must be a contiguous 1-D `i8` memref, not `{src_ty}`")
    })?;
    let Some(dst) = ty.as_memref() else {
        return Err(TypeError::new(name, format!("expected a memref type, not `{ty}`")));
    };
    let dynamic = dst.shape.iter().filter(|d| d.is_none()).count();
    expect(name, sizes.len() == dynamic && dst.memory_space == m.memory_space, || {
        format!("cannot view `{src_ty}` as `{ty}` with {} sizes", sizes.len())
    })?;
    Val::new_typed(View { source, byte_shift, sizes, ty })
}

/// `memref.view`
#[track_caller]
pub fn view(source: impl Into<Val>, byte_shift: impl Operand, sizes: impl Operands, ty: impl Into<Ty>) -> Val<MemRefType> {
    try_view(source, byte_shift, sizes, ty).or_panic()
}

impl HighMlir for View {
    fn name(&self) -> &str {
        "memref.view"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.byte_shift.clone()].into_iter().chain(self.sizes.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let sizes = p.operands(2..2 + self.sizes.len());
        write!(p, "memref.view {}[{}][{sizes}] : {} to {}", p.operand(0), p.operand(1), p.operand_type(0), self.ty)
    }
}

/// `memref.prefetch`: a hint.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Prefetch {
    pub memref: Val,
    pub indices: Vec<Val>,
    pub is_write: bool,
    /// From 0 (no locality) to 3 (keep in cache).
    pub locality: u32,
    pub is_data_cache: bool,
}

/// `memref.prefetch`
#[track_caller]
pub fn try_prefetch(
    memref: impl Into<Val>,
    indices: impl Operands,
    is_write: bool,
    locality: u32,
    is_data_cache: bool,
) -> Result<(), TypeError> {
    let name = "memref.prefetch";
    let memref = memref.into();
    let indices = self::indices(name, indices)?;
    let (ty, m) = memref_ty(name, &memref)?;
    check_rank(name, &ty, m.rank(), indices.len())?;
    expect(name, locality <= 3, || format!("locality must be at most 3, not {locality}"))?;
    Val::unit(Prefetch { memref, indices, is_write, locality, is_data_cache })
}

/// `memref.prefetch`
#[track_caller]
pub fn prefetch(memref: impl Into<Val>, indices: impl Operands, is_write: bool, locality: u32, is_data_cache: bool) {
    try_prefetch(memref, indices, is_write, locality, is_data_cache).or_panic()
}

impl HighMlir for Prefetch {
    fn name(&self) -> &str {
        "memref.prefetch"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.memref.clone()).chain(self.indices.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![
            ("isWrite".into(), Attr::Bool(self.is_write)),
            ("localityHint".into(), Attr::Int(self.locality as i64, I32.into())),
            ("isDataCache".into(), Attr::Bool(self.is_data_cache)),
        ]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let n = self.indices.len();
        let rw = if self.is_write { "write" } else { "read" };
        let cache = if self.is_data_cache { "data" } else { "instr" };
        let (m, i, ty) = (p.operand(0), p.operands(1..=n), p.operand_type(0));
        write!(p, "memref.prefetch {m}[{i}], {rw}, locality<{}>, {cache} : {ty}", self.locality)
    }
}

predicates! {
    /// `memref.atomic_rmw` kind, in MLIR's numbering.
    AtomicRMWKind {
        AddF = "addf", AddI = "addi", AndI = "andi", Assign = "assign", MaximumF = "maximumf",
        MaxNumF = "maxnumf", MaxS = "maxs", MaxU = "maxu", MinimumF = "minimumf", MinNumF = "minnumf",
        MinS = "mins", MinU = "minu", MulF = "mulf", MulI = "muli", OrI = "ori", XOrI = "xori",
    }
}

/// `memref.atomic_rmw`: atomically combines `value` into an element and returns its previous value.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct AtomicRMW {
    pub kind: AtomicRMWKind,
    pub value: Val,
    pub memref: Val,
    pub indices: Vec<Val>,
}

/// A literal value takes the element type.
#[track_caller]
pub fn try_atomic_rmw(
    kind: AtomicRMWKind,
    value: impl Operand,
    memref: impl Into<Val>,
    indices: impl Operands,
) -> Result<Val, TypeError> {
    let name = "memref.atomic_rmw";
    let memref = memref.into();
    let indices = self::indices(name, indices)?;
    let (ty, m) = memref_ty(name, &memref)?;
    check_rank(name, &ty, m.rank(), indices.len())?;
    let value = value.into_val(&m.element).map_err(|e| TypeError::new(name, e.message))?;
    let value_ty = value_ty(name, &value)?;
    use AtomicRMWKind as K;
    let float = matches!(kind, K::AddF | K::MaximumF | K::MaxNumF | K::MinimumF | K::MinNumF | K::MulF);
    let domain = kind == K::Assign || if float { value_ty.float_kind().is_some() } else { value_ty.int_width().is_some() };
    expect(name, value_ty == m.element && domain, || {
        format!("cannot `{}` `{value_ty}` into `{ty}`", kind.keyword())
    })?;
    Val::single(AtomicRMW { kind, value, memref, indices })
}

/// `memref.atomic_rmw`; a literal value takes the element type.
#[track_caller]
pub fn atomic_rmw(kind: AtomicRMWKind, value: impl Operand, memref: impl Into<Val>, indices: impl Operands) -> Val {
    try_atomic_rmw(kind, value, memref, indices).or_panic()
}

impl HighMlir for AtomicRMW {
    fn name(&self) -> &str {
        "memref.atomic_rmw"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.value.clone(), self.memref.clone()].into_iter().chain(self.indices.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.value.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("kind".into(), Attr::Int(self.kind as i64, I64.into()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = self.indices.len();
        let (v, m, i) = (p.operand(0), p.operand(1), p.operands(2..2 + n));
        let (vt, mt) = (p.operand_type(0), p.operand_type(1));
        write!(p, "memref.atomic_rmw {} {v}, {m}[{i}] : ({vt}, {mt}) -> {vt}", self.kind.keyword())
    }
}

macro_rules! closure_boilerplate {
    () => {

    };
}

/// `memref.generic_atomic_rmw`: atomically replaces an element by what the
/// body computes from its current value, and returns the previous value.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct GenericAtomicRMW {
    pub memref: Val,
    pub indices: Vec<Val>,
    pub body: BuiltRegion,
}

/// `body(current)` returns the new value.
#[track_caller]
pub fn try_generic_atomic_rmw(
    memref: impl Into<Val>,
    indices: impl Operands,
    body: impl FnOnce(Val) -> Val,
) -> Result<Val, TypeError> {
    let name = "memref.generic_atomic_rmw";
    let memref = memref.into();
    let indices = self::indices(name, indices)?;
    let (ty, m) = memref_ty(name, &memref)?;
    check_rank(name, &ty, m.rank(), indices.len())?;
    let mut builder = ClosureBuilder::new(false);
    let region = builder.region(std::slice::from_ref(&m.element), |p| body(p[0].clone()).into());
    builder.try_finish()?;
    let yielded = types(name, &region.exprs)?;
    expect(name, yielded == [m.element.clone()], || {
        format!("the body yields ({}) for `{ty}`", TypeList(&yielded))
    })?;
    Val::single(GenericAtomicRMW { memref, indices, body: region })
}

/// `memref.generic_atomic_rmw`
#[track_caller]
pub fn generic_atomic_rmw(memref: impl Into<Val>, indices: impl Operands, body: impl FnOnce(Val) -> Val) -> Val {
    try_generic_atomic_rmw(memref, indices, body).or_panic()
}

impl HighMlir for GenericAtomicRMW {
    fn name(&self) -> &str {
        "memref.generic_atomic_rmw"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.memref.clone()).chain(self.indices.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.memref.ty().element().clone()]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = self.indices.len();
        write!(p, "memref.generic_atomic_rmw {}[{}] : {} ", p.operand(0), p.operands(1..=n), p.operand_type(0))?;
        p.region(0, true)
    }
}

impl Closure for GenericAtomicRMW {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    closure_boilerplate!();

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("memref.atomic_yield")
    }
}

/// `memref.alloca_scope`: allocas inside are freed when the scope ends.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct AllocaScope {
    pub body: BuiltRegion,
}

/// `body()` returns the results of the scope.
#[track_caller]
pub fn try_alloca_scope(body: impl FnOnce() -> Tuple) -> Result<Tuple, TypeError> {
    let mut builder = ClosureBuilder::new(false);
    let region = builder.region(&[], |_| body());
    builder.try_finish()?;
    types("memref.alloca_scope", &region.exprs)?;
    Val::all_results(AllocaScope { body: region })
}

/// `memref.alloca_scope`
#[track_caller]
pub fn alloca_scope(body: impl FnOnce() -> Tuple) -> Tuple {
    try_alloca_scope(body).or_panic()
}

impl HighMlir for AllocaScope {
    fn name(&self) -> &str {
        "memref.alloca_scope"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        self.body.exprs.iter().map(|v| v.ty()).collect()
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        p.write_str("memref.alloca_scope ")?;
        let results = self.result_types();
        if !results.is_empty() {
            write!(p, "-> ({}) ", TypeList(&results))?;
        }
        p.region(0, false)
    }
}

impl Closure for AllocaScope {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    closure_boilerplate!();

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("memref.alloca_scope.return")
    }
}

/// A memref and indices into it, as accessed by DMA operations.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Indexed {
    pub memref: Val,
    pub indices: Vec<Val>,
}

impl Indexed {
    fn new(op: &str, memref: Val, idx: impl Operands) -> Result<Indexed, TypeError> {
        let idx = indices(op, idx)?;
        let (ty, m) = memref_ty(op, &memref)?;
        check_rank(op, &ty, m.rank(), idx.len())?;
        Ok(Indexed { memref, indices: idx })
    }

    fn values(&self) -> impl Iterator<Item = Val> + '_ {
        std::iter::once(self.memref.clone()).chain(self.indices.iter().cloned())
    }

    /// `%m[%i, ..]` for operands from `start`.
    fn format(&self, p: &OpPrinter<'_>, start: usize) -> String {
        format!("{}[{}]", p.operand(start), p.operands(start + 1..start + 1 + self.indices.len()))
    }
}

/// `memref.dma_start`: starts copying `num_elements` from `source` to
/// `target`, signalling completion on `tag`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct DmaStart {
    pub source: Indexed,
    pub target: Indexed,
    pub num_elements: Val,
    pub tag: Indexed,
    /// Stride and number of elements per stride, for a strided transfer.
    pub stride: Option<(Val, Val)>,
}

/// `memref.dma_start`
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn try_dma_start(
    source: impl Into<Val>,
    source_indices: impl Operands,
    target: impl Into<Val>,
    target_indices: impl Operands,
    num_elements: impl Operand,
    tag: impl Into<Val>,
    tag_indices: impl Operands,
    stride: Option<(Val, Val)>,
) -> Result<(), TypeError> {
    let name = "memref.dma_start";
    let source = Indexed::new(name, source.into(), source_indices)?;
    let target = Indexed::new(name, target.into(), target_indices)?;
    let tag = Indexed::new(name, tag.into(), tag_indices)?;
    let num_elements = indices(name, num_elements)?.remove(0);
    let stride = match stride {
        Some((s, n)) => Some((indices(name, s)?.remove(0), indices(name, n)?.remove(0))),
        None => None,
    };
    Val::unit(DmaStart { source, target, num_elements, tag, stride })
}

/// `memref.dma_start`
#[allow(clippy::too_many_arguments)]
/// `memref.dma_start`
#[track_caller]
pub fn dma_start(
    source: impl Into<Val>,
    source_indices: impl Operands,
    target: impl Into<Val>,
    target_indices: impl Operands,
    num_elements: impl Operand,
    tag: impl Into<Val>,
    tag_indices: impl Operands,
    stride: Option<(Val, Val)>,
) {
    try_dma_start(source, source_indices, target, target_indices, num_elements, tag, tag_indices, stride).or_panic()
}

impl HighMlir for DmaStart {
    fn name(&self) -> &str {
        "memref.dma_start"
    }

    fn inputs(&self) -> Edges<'_> {
        let stride = self.stride.iter().flat_map(|(s, n)| [s.clone(), n.clone()]);
        Box::new(
            self.source
                .values()
                .chain(self.target.values())
                .chain(std::iter::once(self.num_elements.clone()))
                .chain(self.tag.values())
                .chain(stride),
        )
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let t = 1 + self.source.indices.len();
        let n = t + 1 + self.target.indices.len();
        let tag = n + 1;
        let (src, dst, tag_str) = (self.source.format(p, 0), self.target.format(p, t), self.tag.format(p, tag));
        write!(p, "memref.dma_start {src}, {dst}, {}, {tag_str}", p.operand(n))?;
        if self.stride.is_some() {
            let s = tag + 1 + self.tag.indices.len();
            write!(p, ", {}, {}", p.operand(s), p.operand(s + 1))?;
        }
        let types = [p.operand_type(0), p.operand_type(t), p.operand_type(tag)];
        write!(p, " : {}", TypeList(&types))
    }
}

/// `memref.dma_wait` for the completion of `num_elements` on `tag`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct DmaWait {
    pub tag: Indexed,
    pub num_elements: Val,
}

/// `memref.dma_wait`
#[track_caller]
pub fn try_dma_wait(tag: impl Into<Val>, tag_indices: impl Operands, num_elements: impl Operand) -> Result<(), TypeError> {
    let name = "memref.dma_wait";
    let tag = Indexed::new(name, tag.into(), tag_indices)?;
    let num_elements = indices(name, num_elements)?.remove(0);
    Val::unit(DmaWait { tag, num_elements })
}

/// `memref.dma_wait`
#[track_caller]
pub fn dma_wait(tag: impl Into<Val>, tag_indices: impl Operands, num_elements: impl Operand) {
    try_dma_wait(tag, tag_indices, num_elements).or_panic()
}

impl HighMlir for DmaWait {
    fn name(&self) -> &str {
        "memref.dma_wait"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.tag.values().chain(std::iter::once(self.num_elements.clone())))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let n = 1 + self.tag.indices.len();
        write!(p, "memref.dma_wait {}, {} : {}", self.tag.format(p, 0), p.operand(n), p.operand_type(0))
    }
}

/// `memref.global` named `name`, private and uninitialized; adjust it with
/// the [`Global`] builder methods, then `build()` it.
pub fn global(name: impl Into<String>, ty: impl Into<Ty>) -> Global {
    Global::new(name, ty)
}

/// The initial value of a [`Global`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum GlobalInit {
    /// An external declaration.
    Declaration,
    /// Allocated but not initialized.
    Uninitialized,
    /// An elements attribute of the global's shape, e.g. `Attr::Splat` or `Attr::Dense`.
    Value(Attr),
}

/// `memref.global`, a module-level operation; [`get_global`] reads its address.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Global {
    pub name: String,
    pub ty: Ty,
    pub private: bool,
    pub constant: bool,
    pub init: GlobalInit,
    pub alignment: Option<u64>,
}

impl Global {
    /// A private, uninitialized global; adjust it with the builder methods,
    /// then [`build`](Self::build) it.
    pub fn new(name: impl Into<String>, ty: impl Into<Ty>) -> Global {
        Global {
            name: name.into(),
            ty: ty.into(),
            private: true,
            constant: false,
            init: GlobalInit::Uninitialized,
            alignment: None,
        }
    }

    /// Public rather than private.
    pub fn public(self) -> Global {
        Global { private: false, ..self }
    }

    /// Read-only.
    pub fn constant(self) -> Global {
        Global { constant: true, ..self }
    }

    /// An external declaration, without initial value.
    pub fn declaration(self) -> Global {
        Global { init: GlobalInit::Declaration, ..self }
    }

    /// Alignment in bytes, a power of two.
    pub fn alignment(self, alignment: u64) -> Global {
        Global { alignment: Some(alignment), ..self }
    }

    /// Initial elements, an attribute of the global's tensor type (see [`tensor_type`](Self::tensor_type)).
    pub fn init(self, value: Attr) -> Global {
        Global { init: GlobalInit::Value(value), ..self }
    }

    /// Integer elements in row-major order (one value is a splat).
    pub fn ints(self, values: impl IntoIterator<Item = i64>) -> Global {
        let e = self.ty.element().clone();
        let values: Vec<Attr> = values.into_iter().map(|v| Attr::int(v, e.clone())).collect();
        self.elements(values)
    }

    /// Float elements in row-major order (one value is a splat).
    pub fn floats(self, values: impl IntoIterator<Item = f64>) -> Global {
        let e = self.ty.element().clone();
        let values: Vec<Attr> = values.into_iter().map(|v| Attr::float(v, e.clone())).collect();
        self.elements(values)
    }

    fn elements(self, mut values: Vec<Attr>) -> Global {
        let ty = self.tensor_type();
        let value = if values.len() == 1 { Attr::Splat(Box::new(values.remove(0)), ty) } else { Attr::Dense(values, ty) };
        self.init(value)
    }

    /// The tensor type of the initial value: the memref's shape and element type.
    pub fn tensor_type(&self) -> Ty {
        Ty::tensor(self.ty.shape().unwrap_or(&[]).iter().copied(), self.ty.element().clone())
    }

    /// Builds the `memref.global` operation.
    #[track_caller]
    pub fn try_build(self) -> Result<Val, TypeError> {
        let name = "memref.global";
        let static_shape = self.ty.as_memref().is_some_and(|m| m.shape.iter().all(Option::is_some));
        expect(name, static_shape, || format!("`{}` must be a statically shaped memref", self.ty))?;
        check_alignment(name, self.alignment)?;
        if let GlobalInit::Value(Attr::Dense(vs, _)) = &self.init {
            let n: u64 = self.ty.shape().unwrap().iter().map(|d| d.unwrap()).product();
            expect(name, vs.len() as u64 == n, || format!("`{}` has {n} elements, got {}", self.ty, vs.len()))?;
        }
        Val::single(self)
    }

    /// Builds the `memref.global` operation.
    #[track_caller]
    pub fn build(self) -> Val {
        self.try_build().or_panic()
    }
}

/// `memref.get_global` of `global`, a [`Global`] operation. Accesses are
/// ordered along the returned value, so use one per function.
#[track_caller]
pub fn try_get_global(global: &Val) -> Result<Val<MemRefType>, TypeError> {
    let Some(g) = global.op_as::<Global>() else {
        return Err(TypeError::new("memref.get_global", format!("`{}` is not a `memref.global`", global.name())));
    };
    let op = GetGlobal { name: g.name.clone(), ty: g.ty.clone() };
    Ok(Val::single(op)?.downcast().unwrap())
}

/// `memref.get_global`
#[track_caller]
pub fn get_global(global: &Val) -> Val<MemRefType> {
    try_get_global(global).or_panic()
}

impl HighMlir for Global {
    fn name(&self) -> &str {
        "memref.global"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = vec![("sym_name".into(), Attr::str(&self.name)), ("type".into(), Attr::Type(self.ty.clone()))];
        if self.private {
            attrs.push(("sym_visibility".into(), Attr::str("private")));
        }
        if self.constant {
            attrs.push(("constant".into(), Attr::Unit));
        }
        match &self.init {
            GlobalInit::Declaration => {}
            GlobalInit::Uninitialized => attrs.push(("initial_value".into(), Attr::Unit)),
            GlobalInit::Value(v) => attrs.push(("initial_value".into(), v.clone())),
        }
        attrs.extend(alignment_attr(self.alignment));
        attrs
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.write_str("memref.global ")?;
        if self.private {
            p.write_str("\"private\" ")?;
        }
        if self.constant {
            p.write_str("constant ")?;
        }
        write!(p, "{} : {}", Symbol(&self.name), self.ty)?;
        match &self.init {
            GlobalInit::Declaration => {}
            GlobalInit::Uninitialized => p.write_str(" = uninitialized")?,
            GlobalInit::Value(v) => write!(p, " = {}", v.untyped())?,
        }
        p.attr_dict(&alignment_attr(self.alignment))
    }
}

/// `memref.get_global`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct GetGlobal {
    pub name: String,
    pub ty: Ty,
}

impl HighMlir for GetGlobal {
    fn name(&self) -> &str {
        "memref.get_global"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("name".into(), Attr::Symbol(self.name.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "memref.get_global {} : {}", Symbol(&self.name), self.ty)
    }
}
