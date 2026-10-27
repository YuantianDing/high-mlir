//! MLIR types behind the typed pointer [`Ty`].
//!
//! `Ty<T>` points at a concrete type (e.g. `Ty<IntType>`), and the default
//! `Ty = Ty<dyn MlirType>` is the dynamic form. A dialect adds its own type by
//! implementing [`MlirType`] (and `Display` with the type's textual MLIR form):
//!
//! ```
//! use high_mlir::{I64, MlirType, Ty};
//!
//! #[derive(Debug, PartialEq, Eq, Hash)]
//! struct HashSet { key: Ty }
//!
//! impl std::fmt::Display for HashSet {
//!     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//!         write!(f, "!collection.hashset<{}>", self.key)
//!     }
//! }
//!
//! impl MlirType for HashSet {}
//!
//! let set: Ty<HashSet> = Ty::new(HashSet { key: I64.into() });
//! let erased: Ty = set.clone().into();
//! assert_eq!(erased.to_string(), "!collection.hashset<i64>");
//! assert_eq!(erased.downcast::<HashSet>(), Some(set));
//! ```

use std::{
    any::Any,
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
};

use std::sync::Arc;

use dyn_eq::DynEq;
use dyn_hash::DynHash;

use crate::{Attr, attr::AffineMap};

/// Behaviour shared by every MLIR type.
///
/// `Display` must produce the type's textual MLIR form; it is also how the
/// type is handed to melior (`melior::ir::Type::parse`). Every method has a
/// default, so an opaque dialect type needs an empty `impl`.
pub trait MlirType: fmt::Debug + fmt::Display + DynEq + DynHash + Any + Send + Sync + 'static {
    /// Shape and element type, for shaped (tensor/memref-like) types.
    /// `None` in the shape is a dynamic (`?`) dimension.
    fn shaped(&self) -> Option<(&[Option<u64>], &Ty)> {
        None
    }

    /// Rebuilds a shaped type with a different element type.
    fn with_element(&self, _element: Ty) -> Option<Ty> {
        None
    }

    /// Bit width of a fixed-width scalar.
    fn bit_width(&self) -> Option<u32> {
        None
    }
}

dyn_eq::eq_trait_object!(MlirType);
dyn_hash::hash_trait_object!(MlirType);

/// Typed pointer to an MLIR type; compares and hashes structurally.
pub struct Ty<T: ?Sized = dyn MlirType>(Arc<T>);
impl<T: MlirType> Ty<T> {
    /// A type of static type `T`.
    pub fn new(ty: T) -> Ty<T> {
        Ty(Arc::new(ty))
    }

    /// Forgets the static type; same as `.into()`.
    pub fn erase(self) -> Ty {
        Ty(self.0)
    }
}

impl Ty {
    /// This type with the static type `T`, if it has it.
    pub fn downcast<T: MlirType>(&self) -> Option<Ty<T>> {
        let any: Arc<dyn Any + Send + Sync> = self.0.clone();
        any.downcast().ok().map(Ty)
    }

    /// This type as a `T`, if it is one.
    pub fn downcast_ref<T: MlirType>(&self) -> Option<&T> {
        (&*self.0 as &dyn Any).downcast_ref()
    }

    /// Whether this type is a `T`.
    pub fn is<T: MlirType>(&self) -> bool {
        self.downcast_ref::<T>().is_some()
    }

    /// See also the [`tensor!`](macro@crate::tensor) macro.
    pub fn tensor(shape: impl IntoIterator<Item = Option<u64>>, element: impl Into<Ty>) -> Ty {
        Ty::new(TensorType { shape: shape.into_iter().collect(), element: element.into() }).erase()
    }

    /// See also the [`memref!`](macro@crate::memref) macro.
    pub fn memref(shape: impl IntoIterator<Item = Option<u64>>, element: impl Into<Ty>) -> Ty {
        MemRefType::new(shape, element).into()
    }

    /// `memref<*xT>`
    pub fn unranked_memref(element: impl Into<Ty>) -> Ty {
        Ty::new(UnrankedMemRefType { element: element.into(), memory_space: None }).erase()
    }

    /// A fixed-length vector; see also the [`vector!`](macro@crate::vector) macro.
    pub fn vector(shape: impl IntoIterator<Item = u64>, element: impl Into<Ty>) -> Ty {
        VectorType::new(shape, element).into()
    }

    /// A function type `(inputs) -> results`.
    pub fn function(
        inputs: impl IntoIterator<Item = Ty>,
        results: impl IntoIterator<Item = Ty>,
    ) -> Ty {
        Ty::new(FunctionType {
            inputs: inputs.into_iter().collect(),
            results: results.into_iter().collect(),
        })
        .erase()
    }

    /// The shape of a shaped type; `None` in it is a dynamic dimension.
    pub fn shape(&self) -> Option<&[Option<u64>]> {
        self.shaped().map(|(s, _)| s)
    }

    /// The element type of a shaped type (or unranked memref), or the type itself otherwise.
    pub fn element(&self) -> &Ty {
        match self.downcast_ref::<UnrankedMemRefType>() {
            Some(u) => &u.element,
            None => self.shaped().map_or(self, |(_, e)| e),
        }
    }

    /// The same shape (if any) with a different element type.
    pub fn with_element_ty(&self, element: Ty) -> Ty {
        match self.shaped() {
            Some(_) => self.with_element(element).expect("shaped type without `with_element`"),
            None => element,
        }
    }

    /// The width if this is a signless integer `iN`.
    pub fn int_width(&self) -> Option<u32> {
        self.downcast_ref::<IntType>().map(|t| t.0)
    }

    /// Whether this is `index`.
    pub fn is_index(&self) -> bool {
        self.is::<IndexType>()
    }

    /// The float type, if this is one.
    pub fn float_kind(&self) -> Option<FloatType> {
        self.downcast_ref::<FloatType>().copied()
    }

    /// Integers, `index` and floats.
    pub fn is_scalar(&self) -> bool {
        self.is::<IntType>() || self.is_index() || self.is::<FloatType>()
    }

    /// This type as a ranked memref type, if it is one.
    pub fn as_memref(&self) -> Option<&MemRefType> {
        self.downcast_ref()
    }

    /// This type as a vector type, if it is one.
    pub fn as_vector(&self) -> Option<&VectorType> {
        self.downcast_ref()
    }

    /// Ranked or unranked memref.
    pub fn is_memref_like(&self) -> bool {
        self.is::<MemRefType>() || self.is::<UnrankedMemRefType>()
    }

    /// Integer or index scalars and tensors/vectors of them (what `arith` integer ops accept).
    pub fn is_int_like(&self) -> bool {
        self.is_elementwise() && (self.element().is::<IntType>() || self.element().is_index())
    }

    /// Float scalars and tensors/vectors of floats.
    pub fn is_float_like(&self) -> bool {
        self.is_elementwise() && self.element().is::<FloatType>()
    }

    /// `i1` or a tensor/vector of `i1`.
    pub fn is_bool_like(&self) -> bool {
        self.is_elementwise() && self.element().int_width() == Some(1)
    }

    /// Scalars, tensors and vectors, i.e. what `arith` operates on.
    fn is_elementwise(&self) -> bool {
        self.is_scalar() || self.is::<TensorType>() || self.is::<VectorType>()
    }
}

impl<T: MlirType> From<Ty<T>> for Ty {
    fn from(ty: Ty<T>) -> Ty {
        ty.erase()
    }
}

impl<T: MlirType> From<T> for Ty {
    fn from(ty: T) -> Ty {
        Ty::new(ty).erase()
    }
}

impl From<&Ty> for Ty {
    fn from(ty: &Ty) -> Ty {
        ty.clone()
    }
}

impl<T: MlirType + PartialEq> PartialEq<T> for Ty {
    fn eq(&self, other: &T) -> bool {
        self.downcast_ref::<T>() == Some(other)
    }
}

/// `i1`
pub const I1: IntType = IntType(1);
/// `i8`
pub const I8: IntType = IntType(8);
/// `i16`
pub const I16: IntType = IntType(16);
/// `i32`
pub const I32: IntType = IntType(32);
/// `i64`
pub const I64: IntType = IntType(64);
/// `index`
pub const INDEX: IndexType = IndexType;
/// `f16`
pub const F16: FloatType = FloatType::F16;
/// `bf16`
pub const BF16: FloatType = FloatType::BF16;
/// `f32`
pub const F32: FloatType = FloatType::F32;
/// `f64`
pub const F64: FloatType = FloatType::F64;

impl<T: ?Sized> Clone for Ty<T> {
    fn clone(&self) -> Self {
        Ty(self.0.clone())
    }
}

impl<T: ?Sized> Deref for Ty<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Ty<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}

impl<T: ?Sized + Eq> Eq for Ty<T> {}

impl<T: ?Sized + Hash> Hash for Ty<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<T: ?Sized + fmt::Display> fmt::Debug for Ty<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ty({})", self.0)
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Ty<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// `none`: the type of an operation without outputs, as a [`Val`](crate::Val).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NoneType;

impl MlirType for NoneType {}

impl fmt::Display for NoneType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("none")
    }
}

/// Signless integer `iN`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IntType(pub u32);

impl MlirType for IntType {
    fn bit_width(&self) -> Option<u32> {
        Some(self.0)
    }
}

impl fmt::Display for IntType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "i{}", self.0)
    }
}

/// `index`: a target-dependent integer for sizes and indices.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IndexType;

impl MlirType for IndexType {
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("index")
    }
}

/// A builtin float type, named as in MLIR (`F8E4M3FN` is `f8E4M3FN`).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatType {
    /// `f16`
    F16,
    /// `bf16`
    BF16,
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// `tf32`
    TF32,
    /// `f80`
    F80,
    /// `f128`
    F128,
    /// `f8E5M2`
    F8E5M2,
    /// `f8E4M3`
    F8E4M3,
    /// `f8E4M3FN`
    F8E4M3FN,
    /// `f8E5M2FNUZ`
    F8E5M2FNUZ,
    /// `f8E4M3FNUZ`
    F8E4M3FNUZ,
    /// `f8E4M3B11FNUZ`
    F8E4M3B11FNUZ,
    /// `f8E3M4`
    F8E3M4,
    /// `f8E8M0FNU`
    F8E8M0FNU,
    /// `f6E2M3FN`
    F6E2M3FN,
    /// `f6E3M2FN`
    F6E3M2FN,
    /// `f4E2M1FN`
    F4E2M1FN,
}

impl FloatType {
    fn info(self) -> (&'static str, u32) {
        use FloatType::*;
        match self {
            F16 => ("f16", 16),
            BF16 => ("bf16", 16),
            F32 => ("f32", 32),
            F64 => ("f64", 64),
            TF32 => ("tf32", 19),
            F80 => ("f80", 80),
            F128 => ("f128", 128),
            F8E5M2 => ("f8E5M2", 8),
            F8E4M3 => ("f8E4M3", 8),
            F8E4M3FN => ("f8E4M3FN", 8),
            F8E5M2FNUZ => ("f8E5M2FNUZ", 8),
            F8E4M3FNUZ => ("f8E4M3FNUZ", 8),
            F8E4M3B11FNUZ => ("f8E4M3B11FNUZ", 8),
            F8E3M4 => ("f8E3M4", 8),
            F8E8M0FNU => ("f8E8M0FNU", 8),
            F6E2M3FN => ("f6E2M3FN", 6),
            F6E3M2FN => ("f6E3M2FN", 6),
            F4E2M1FN => ("f4E2M1FN", 4),
        }
    }
}

impl MlirType for FloatType {
    fn bit_width(&self) -> Option<u32> {
        Some(self.info().1)
    }
}

impl fmt::Display for FloatType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.info().0)
    }
}

/// `complex<T>` of a float type `T`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ComplexType {
    /// The type of the real and imaginary parts.
    pub element: Ty,
}

impl ComplexType {
    /// `complex<element>`
    pub fn new(element: impl Into<Ty>) -> ComplexType {
        ComplexType { element: element.into() }
    }
}

impl MlirType for ComplexType {
    fn bit_width(&self) -> Option<u32> {
        self.element.bit_width().map(|w| 2 * w)
    }
}

impl fmt::Display for ComplexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "complex<{}>", self.element)
    }
}

/// A ranked tensor `tensor<4x?xT>`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TensorType {
    /// The dimensions; `None` is dynamic (`?`).
    pub shape: Vec<Option<u64>>,
    /// The element type.
    pub element: Ty,
}

impl MlirType for TensorType {
    fn shaped(&self) -> Option<(&[Option<u64>], &Ty)> {
        Some((&self.shape, &self.element))
    }

    fn with_element(&self, element: Ty) -> Option<Ty> {
        Some(Ty::new(TensorType { shape: self.shape.clone(), element }).erase())
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_shaped(f, "tensor", &self.shape, &self.element)
    }
}

/// A ranked memref, with an optional layout and memory space.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MemRefType {
    /// The dimensions; `None` is dynamic (`?`).
    pub shape: Vec<Option<u64>>,
    /// The element type.
    pub element: Ty,
    /// How indices map to memory.
    pub layout: Layout,
    /// The memory space, if not the default.
    pub memory_space: Option<Attr>,
}

/// The layout of a memref. `None` in strides or offset is dynamic (`?`).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Layout {
    /// Row-major and contiguous; not printed.
    Identity,
    /// `strided<[strides], offset: offset>`
    Strided { strides: Vec<Option<i64>>, offset: Option<i64> },
    /// An affine map from indices to the linear position.
    Map(AffineMap),
}

impl MemRefType {
    /// A memref of the identity layout in the default memory space.
    pub fn new(shape: impl IntoIterator<Item = Option<u64>>, element: impl Into<Ty>) -> MemRefType {
        MemRefType { shape: shape.into_iter().collect(), element: element.into(), layout: Layout::Identity, memory_space: None }
    }

    /// The same memref type with another layout.
    pub fn with_layout(self, layout: Layout) -> MemRefType {
        MemRefType { layout, ..self }
    }

    /// `strided<[strides], offset: offset>`
    pub fn with_strides(self, strides: impl IntoIterator<Item = Option<i64>>, offset: Option<i64>) -> MemRefType {
        self.with_layout(Layout::Strided { strides: strides.into_iter().collect(), offset })
    }

    /// The same memref type in another memory space.
    pub fn with_memory_space(self, memory_space: Option<Attr>) -> MemRefType {
        MemRefType { memory_space, ..self }
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Strides and offset, computed row-major for the identity layout;
    /// `None` for an affine map layout.
    pub fn strides_and_offset(&self) -> Option<(Vec<Option<i64>>, Option<i64>)> {
        match &self.layout {
            Layout::Identity => {
                let mut strides = vec![Some(1); self.rank()];
                for i in (0..self.rank().saturating_sub(1)).rev() {
                    strides[i] = strides[i + 1].zip(self.shape[i + 1]).map(|(s, d)| s * d as i64);
                }
                Some((strides, Some(0)))
            }
            Layout::Strided { strides, offset } => Some((strides.clone(), *offset)),
            Layout::Map(_) => None,
        }
    }
}

impl MlirType for MemRefType {
    fn shaped(&self) -> Option<(&[Option<u64>], &Ty)> {
        Some((&self.shape, &self.element))
    }

    fn with_element(&self, element: Ty) -> Option<Ty> {
        Some(MemRefType { element, ..self.clone() }.into())
    }
}

fn fmt_dynamic(f: &mut fmt::Formatter<'_>, v: Option<i64>) -> fmt::Result {
    match v {
        Some(v) => write!(f, "{v}"),
        None => f.write_str("?"),
    }
}

fn fmt_memory_space(f: &mut fmt::Formatter<'_>, space: &Option<Attr>) -> fmt::Result {
    match space {
        Some(Attr::Int(v, _)) => write!(f, ", {v}"),
        Some(a) => write!(f, ", {a}"),
        None => Ok(()),
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Layout::Identity => Ok(()),
            Layout::Strided { strides, offset } => {
                f.write_str("strided<[")?;
                for (i, s) in strides.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    fmt_dynamic(f, *s)?;
                }
                f.write_str("]")?;
                if *offset != Some(0) {
                    f.write_str(", offset: ")?;
                    fmt_dynamic(f, *offset)?;
                }
                f.write_str(">")
            }
            Layout::Map(m) => write!(f, "affine_map<{m}>"),
        }
    }
}

impl fmt::Display for MemRefType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_shaped_open(f, "memref", &self.shape, &self.element)?;
        if self.layout != Layout::Identity {
            write!(f, ", {}", self.layout)?;
        }
        fmt_memory_space(f, &self.memory_space)?;
        f.write_str(">")
    }
}

/// `memref<*xT>`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct UnrankedMemRefType {
    /// The element type.
    pub element: Ty,
    /// The memory space, if not the default.
    pub memory_space: Option<Attr>,
}

impl MlirType for UnrankedMemRefType {}

impl fmt::Display for UnrankedMemRefType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "memref<*x{}", self.element)?;
        fmt_memory_space(f, &self.memory_space)?;
        f.write_str(">")
    }
}

/// `vector<4x[8]xf32>`: dimensions are static, and scalable ones (in
/// brackets) are multiples of the runtime `vscale`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VectorType {
    shape: Vec<Option<u64>>,
    scalable: Vec<bool>,
    /// The element type.
    pub element: Ty,
}

impl VectorType {
    /// A fixed-length vector type.
    pub fn new(shape: impl IntoIterator<Item = u64>, element: impl Into<Ty>) -> VectorType {
        VectorType::scalable(shape.into_iter().map(|d| (d, false)), element)
    }

    /// Dimensions with whether each is scalable.
    pub fn scalable(dims: impl IntoIterator<Item = (u64, bool)>, element: impl Into<Ty>) -> VectorType {
        let (shape, scalable): (Vec<_>, Vec<_>) = dims.into_iter().map(|(d, s)| (Some(d), s)).unzip();
        VectorType { shape, scalable, element: element.into() }
    }

    /// The dimensions (multiples of `vscale` for scalable ones).
    pub fn dims(&self) -> Vec<u64> {
        self.shape.iter().map(|d| d.unwrap()).collect()
    }

    /// Whether each dimension is scalable.
    pub fn scalable_dims(&self) -> &[bool] {
        &self.scalable
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Whether some dimension is scalable.
    pub fn is_scalable(&self) -> bool {
        self.scalable.iter().any(|&s| s)
    }

    /// The vector of the trailing dimensions from `from` on, or the element type if none remain.
    pub fn trailing(&self, from: usize) -> Ty {
        if from >= self.rank() {
            return self.element.clone();
        }
        let dims = self.dims()[from..].iter().copied().zip(self.scalable[from..].iter().copied()).collect::<Vec<_>>();
        VectorType::scalable(dims, self.element.clone()).into()
    }
}

impl MlirType for VectorType {
    fn shaped(&self) -> Option<(&[Option<u64>], &Ty)> {
        Some((&self.shape, &self.element))
    }

    fn with_element(&self, element: Ty) -> Option<Ty> {
        Some(VectorType { element, ..self.clone() }.into())
    }
}

impl fmt::Display for VectorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("vector<")?;
        for (d, s) in self.shape.iter().zip(&self.scalable) {
            let d = d.unwrap();
            if *s { write!(f, "[{d}]x")? } else { write!(f, "{d}x")? }
        }
        write!(f, "{}>", self.element)
    }
}

/// A function type `(inputs) -> results`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FunctionType {
    /// Parameter types.
    pub inputs: Vec<Ty>,
    /// Result types.
    pub results: Vec<Ty>,
}

impl MlirType for FunctionType {}

impl fmt::Display for FunctionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) -> {}", TypeList(&self.inputs), ResultTypes(&self.results))
    }
}

fn fmt_shaped(f: &mut fmt::Formatter<'_>, name: &str, shape: &[Option<u64>], e: &Ty) -> fmt::Result {
    fmt_shaped_open(f, name, shape, e)?;
    f.write_str(">")
}

/// `name<4x?xT`, without the closing `>`.
fn fmt_shaped_open(f: &mut fmt::Formatter<'_>, name: &str, shape: &[Option<u64>], e: &Ty) -> fmt::Result {
    write!(f, "{name}<")?;
    for d in shape {
        match d {
            Some(n) => write!(f, "{n}x")?,
            None => write!(f, "?x")?,
        }
    }
    write!(f, "{e}")
}

pub(crate) struct TypeList<'a>(pub &'a [Ty]);

impl fmt::Display for TypeList<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, t) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{t}")?;
        }
        Ok(())
    }
}

/// Function results as MLIR prints them: `i32`, `(i32, i32)` or `()`.
pub(crate) struct ResultTypes<'a>(pub &'a [Ty]);

impl fmt::Display for ResultTypes<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            [t] if !t.is::<FunctionType>() => write!(f, "{t}"),
            ts => write!(f, "({})", TypeList(ts)),
        }
    }
}
