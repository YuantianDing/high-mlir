//! The core: [`HighMlir`] operations, and the values [`Val`] of their results,
//! typed by their MLIR type.
//!
//! Operations are recorded in build order. A thread-local stack keeps one
//! frame per closure under construction (a [`ClosureBuilder`]); each frame
//! owns the operations built in its current region, as `Arc<Node>`. A
//! finished region is owned by the closure operation that has it, and the
//! outermost one by the [`Root`] that [`builtin::module`](crate::builtin::module)
//! returns. A [`Val`] only has a weak pointer to its operation.
//!
//! An operation can use the values of its own region and of the regions
//! enclosing it, up to the nearest isolated closure (e.g. `func.func`);
//! building it checks so. The IR is the regions, in build order.

use std::{
    any::Any,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::Deref,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use dyn_clone::DynClone;
use dyn_eq::DynEq;
use dyn_hash::DynHash;

use spin::Mutex;

use crate::{Attr, MlirType, NoneType, OpPrinter, Ty, annotations};

/// Values an operation refers to.
pub type Edges<'a> = Box<dyn Iterator<Item = Val> + 'a>;

/// An MLIR operation.
///
/// Equality and hashing (`DynEq`/`DynHash`) are shallow: operands are
/// [`Val`]s, which compare by identity.
pub trait HighMlir: fmt::Debug + DynClone + DynEq + DynHash + Any + Send + Sync + 'static {
    /// Fully qualified operation name, e.g. `arith.addi`.
    fn name(&self) -> &str;

    /// The operands, in order.
    fn inputs(&self) -> Edges<'_>;

    /// Types of the results, in order.
    fn result_types(&self) -> Vec<Ty>;

    /// The inherent attributes, as the generic form and melior take them.
    fn attributes(&self) -> Vec<(String, Attr)> {
        Vec::new()
    }

    /// `Some` if this operation has regions.
    fn as_closure(&self) -> Option<&dyn Closure> {
        None
    }

    /// Prints the operation; the default is MLIR's generic form.
    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.generic()
    }
}

dyn_clone::clone_trait_object!(HighMlir);
dyn_eq::eq_trait_object!(HighMlir);
dyn_hash::hash_trait_object!(HighMlir);

/// An operation with regions: `scf.for`, `scf.if`, `func.func`, ... Build
/// its regions with a [`ClosureBuilder`].
pub trait Closure: HighMlir {
    /// Number of regions.
    fn num_regions(&self) -> usize {
        1
    }

    /// Region `r`.
    fn region(&self, r: usize) -> &BuiltRegion;

    /// The terminator (e.g. `scf.yield`) that region `r` ends with, applied
    /// to the values it yields; `None` if its last operation is the
    /// terminator (e.g. `scf.condition`) or it has none.
    fn terminator(&self, r: usize) -> Option<&str>;

    /// Region `r` has no block at all (e.g. the body of an external function).
    fn is_empty(&self, _r: usize) -> bool {
        false
    }
}

/// Identifies a region under construction, so that operations know where
/// they may be used.
pub type RegionId = u64;

fn fresh_region_id() -> RegionId {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// An operation in its shared allocation, in the region it was built in.
/// Dropping it forgets the operation's annotations (see [`crate::annotations`]).
pub struct Node<T: ?Sized = dyn HighMlir> {
    region: RegionId,
    op: T,
}

impl<T: ?Sized> Node<T> {
    /// The address keying the operation's annotations.
    pub(crate) fn key(&self) -> u64 {
        self as *const Node<T> as *const () as usize as u64
    }

    /// The region the operation was built in.
    pub fn region(&self) -> RegionId {
        self.region
    }
}

impl Node {
    /// Builds the operation in the region under construction: checks that
    /// its operands are visible there and records it, in order. Fails
    /// outside of any closure (a module included); [`Val::new`] makes
    /// values of the results.
    #[track_caller]
    pub fn build(op: impl HighMlir) -> Result<Arc<Node>, TypeError> {
        let node = frames::record(op)?;
        annotations::record_rust_location(&node);
        Ok(node)
    }
}

impl<T: ?Sized> Deref for Node<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.op
    }
}

impl<T: ?Sized> Drop for Node<T> {
    fn drop(&mut self) {
        annotations::forget(self.key());
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.op.fmt(f)
    }
}

/// A node outside of any region, e.g. a terminator made when printing.
pub(crate) fn detached(op: impl HighMlir) -> Arc<Node> {
    Arc::new(Node { region: fresh_region_id(), op })
}

/// The outermost operation of an IR, e.g. a `builtin.module`, which owns
/// everything built in it. It prints as textual MLIR and converts to melior.
pub struct Root {
    node: Arc<Node>,
}

impl Root {
    /// An operation outside of any closure; `op` must be a closure whose
    /// regions were built with a [`ClosureBuilder`].
    #[track_caller]
    pub(crate) fn new(op: impl HighMlir) -> Root {
        let node: Arc<Node> = Arc::new(Node { region: fresh_region_id(), op });
        annotations::record_rust_location(&node);
        Root { node }
    }

    /// The operation.
    pub fn op(&self) -> &Arc<Node> {
        &self.node
    }

    /// The operation as a value (of type `none`).
    pub fn val(&self) -> Val {
        Val::new(&self.node, 0)
    }
}

/// A shared reference to an operation of type `O`; see [`Val::op_as`].
pub struct OpRef<O> {
    node: Arc<Node>,
    marker: PhantomData<fn() -> O>,
}

impl<O: HighMlir> Deref for OpRef<O> {
    type Target = O;

    fn deref(&self) -> &O {
        (&**self.node as &dyn Any).downcast_ref().expect("checked when made")
    }
}

/// Output `port` of an operation, typed by its MLIR type; compares and
/// hashes by identity.
///
/// `Val<T>` keeps the concrete type of the value (e.g. `Val<MemRefType>`),
/// and the default `Val = Val<dyn MlirType>` is the dynamic form, so that
/// methods and operators can be provided for the values of a given type.
/// An operation without results (e.g. `func.func`) is a value of type
/// [`NoneType`]. A value has a weak pointer to its operation, which the
/// region it was built in owns.
pub struct Val<T: ?Sized = dyn MlirType> {
    ty: Ty<T>,
    inner: Weak<Node>,
    port: u32,
}

/// Conversion between any `Val<T>` and the dynamic [`Val`].
pub trait AsVal: Sized {
    /// This value with the dynamic type.
    fn as_val(&self) -> Val;

    /// `v` with this static type, if it has it.
    fn from_val(v: Val) -> Option<Self>;

    /// Whether a value of type `ty` has this static type.
    fn accepts(ty: &Ty) -> bool;
}

impl<T: MlirType> AsVal for Val<T> {
    fn as_val(&self) -> Val {
        Val { ty: self.ty.clone().erase(), inner: self.inner.clone(), port: self.port }
    }

    fn from_val(v: Val) -> Option<Val<T>> {
        v.downcast()
    }

    fn accepts(ty: &Ty) -> bool {
        ty.is::<T>()
    }
}


impl AsVal for Val {
    fn as_val(&self) -> Val {
        self.clone()
    }

    fn from_val(v: Val) -> Option<Val> {
        Some(v)
    }

    fn accepts(_: &Ty) -> bool {
        true
    }
}

impl Val {
    /// Result `port` of the operation `node`. Port 0 of an operation without
    /// results is the operation itself, a [`NoneType`] value. Panics if the
    /// operation has no such result.
    #[track_caller]
    pub fn new(node: &Arc<Node>, port: u32) -> Val {
        let mut results = node.result_types();
        let ty = match results.len() {
            0 if port == 0 => NoneType.into(),
            n if (port as usize) < n => results.swap_remove(port as usize),
            _ => panic!("`{}` has no result #{port}", node.name()),
        };
        Val { ty, inner: Arc::downgrade(node), port }
    }

    /// Builds the operation and returns its only result (or the operation
    /// itself if it has none).
    #[track_caller]
    pub(crate) fn single(op: impl HighMlir) -> Result<Val, TypeError> {
        Ok(Val::new(&Node::build(op)?, 0))
    }

    /// Builds the operation, whose first result is known to have type `T`.
    #[track_caller]
    pub(crate) fn new_typed<T: MlirType>(op: impl HighMlir) -> Result<Val<T>, TypeError> {
        let v = Val::single(op)?;
        let ty = v.ty;
        let ty = ty.downcast().unwrap_or_else(|| panic!("an operation has a result of type `{ty}`"));
        Ok(Val { ty, inner: v.inner, port: v.port })
    }

    /// Builds the operation and returns all its results.
    #[track_caller]
    pub(crate) fn all_results(op: impl HighMlir) -> Result<Tuple, TypeError> {
        let node = Node::build(op)?;
        let n = node.result_types().len() as u32;
        Ok(Tuple((0..n).map(|port| Val::new(&node, port)).collect()))
    }

    /// Builds the operation and returns its result, if it has one.
    #[track_caller]
    pub(crate) fn optional(op: impl HighMlir) -> Result<Option<Val>, TypeError> {
        let node = Node::build(op)?;
        Ok((!node.result_types().is_empty()).then(|| Val::new(&node, 0)))
    }

    /// Builds an operation without results.
    #[track_caller]
    pub(crate) fn unit(op: impl HighMlir) -> Result<(), TypeError> {
        Node::build(op).map(drop)
    }

    /// This value with the static type `T`, if it has it.
    pub fn downcast<T: MlirType>(&self) -> Option<Val<T>> {
        let ty = self.ty.downcast()?;
        Some(Val { ty, inner: self.inner.clone(), port: self.port })
    }

    /// Whether the value has type `T`.
    pub fn is<T: MlirType>(&self) -> bool {
        self.ty.is::<T>()
    }
}

impl<T: MlirType> Val<T> {
    /// Forgets the static type; same as `.into()`.
    pub fn erase(self) -> Val {
        Val { ty: self.ty.erase(), inner: self.inner, port: self.port }
    }
}

impl<T: ?Sized> Val<T> {
    /// The result number of this value.
    pub fn port(&self) -> u32 {
        self.port
    }

    /// The operation; panics if it was dropped (with the module it was built in).
    #[track_caller]
    pub fn op(&self) -> Arc<Node> {
        self.try_op().expect("the operation of a value was dropped, with the module it was built in")
    }

    /// The operation, unless it was dropped.
    pub fn try_op(&self) -> Option<Arc<Node>> {
        self.inner.upgrade()
    }

    /// The operation, if it is an `O`.
    pub fn op_as<O: HighMlir>(&self) -> Option<OpRef<O>> {
        let node = self.try_op()?;
        (&**node as &dyn Any).is::<O>().then_some(OpRef { node, marker: PhantomData })
    }

    /// Whether the operation is an `O`.
    pub fn is_op<O: HighMlir>(&self) -> bool {
        self.op_as::<O>().is_some()
    }

    /// The name of the operation (`<dropped>` if it was dropped).
    pub fn name(&self) -> String {
        self.try_op().map_or("<dropped>".into(), |op| op.name().to_string())
    }

    /// The type of this value, statically a `T`.
    pub fn static_ty(&self) -> &Ty<T> {
        &self.ty
    }

    pub(crate) fn key(&self) -> *const () {
        Weak::as_ptr(&self.inner) as *const ()
    }

    /// Number of results of the operation.
    #[track_caller]
    pub fn num_results(&self) -> u32 {
        self.op().result_types().len() as u32
    }

    /// Whether this is a result (rather than an operation without results).
    pub fn is_result(&self) -> bool
    where
        Ty<T>: Into<Ty>,
    {
        !self.ty().is::<NoneType>()
    }

    /// Type of this value.
    pub fn ty(&self) -> Ty
    where
        Ty<T>: Into<Ty>,
    {
        self.ty.clone().into()
    }

    /// Type of this value, `None` for an operation without results.
    pub fn try_ty(&self) -> Option<Ty>
    where
        Ty<T>: Into<Ty>,
    {
        self.is_result().then(|| self.ty())
    }
}

/// `v` with the static type `T`, for an operation whose result has the
/// type of its first operand.
pub(crate) fn retype<T: ?Sized>(op: &str, v: Val) -> Result<Val<T>, TypeError>
where
    Val<T>: AsVal,
{
    let ty = v.ty();
    Val::<T>::from_val(v).ok_or_else(|| TypeError::new(op, format!("built a value of type `{ty}`, not of the operand type")))
}

impl<T: MlirType> From<Val<T>> for Val {
    #[track_caller]
    fn from(v: Val<T>) -> Val {
        v.erase()
    }
}

impl<T: MlirType> From<&Val<T>> for Val {
    #[track_caller]
    fn from(v: &Val<T>) -> Val {
        v.as_val()
    }
}

impl From<&Val> for Val {
    #[track_caller]
    fn from(v: &Val) -> Val {
        v.clone()
    }
}

impl<T: ?Sized> Clone for Val<T> {
    fn clone(&self) -> Self {
        Val { ty: self.ty.clone(), inner: self.inner.clone(), port: self.port }
    }
}

impl<T: ?Sized, U: ?Sized> PartialEq<Val<U>> for Val<T> {
    fn eq(&self, other: &Val<U>) -> bool {
        self.key() == other.key() && self.port == other.port
    }
}

impl<T: ?Sized> Eq for Val<T> {}

impl<T: ?Sized> Hash for Val<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
        self.port.hash(state);
    }
}

impl<T: ?Sized> fmt::Debug for Val<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Val({}#{} @{:p})", self.name(), self.port, self.key())
    }
}

/// A parameter of a region (a block argument).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Param {
    pub ty: Ty,
}

impl HighMlir for Param {
    fn name(&self) -> &str {
        "<param>"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }
}

/// Values yielded or returned together, e.g. by a closure body.
///
/// Anything convertible into a [`Val`] converts into a one-value tuple, so a
/// body can end with `x.into()`; `().into()` yields nothing, and Rust tuples
/// such as `(x, 1).into()` yield several values.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Tuple(pub Vec<Val>);

impl Deref for Tuple {
    type Target = [Val];

    fn deref(&self) -> &[Val] {
        &self.0
    }
}

impl IntoIterator for Tuple {
    type Item = Val;
    type IntoIter = std::vec::IntoIter<Val>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<V: Into<Val>> From<V> for Tuple {
    fn from(v: V) -> Tuple {
        Tuple(vec![v.into()])
    }
}

impl<const N: usize> TryFrom<Tuple> for [Val; N] {
    type Error = Tuple;

    /// The values, if there are exactly `N`.
    fn try_from(t: Tuple) -> Result<[Val; N], Tuple> {
        <[Val; N]>::try_from(t.0).map_err(Tuple)
    }
}

impl From<Vec<Val>> for Tuple {
    fn from(vs: Vec<Val>) -> Tuple {
        Tuple(vs)
    }
}

impl<const N: usize> From<[Val; N]> for Tuple {
    fn from(vs: [Val; N]) -> Tuple {
        Tuple(vs.into())
    }
}

macro_rules! tuple_from {
    ($(($($t:ident),*))*) => {$(
        impl<$($t: Into<Val>),*> From<($($t,)*)> for Tuple {
            #[allow(non_snake_case)]
            fn from(($($t,)*): ($($t,)*)) -> Tuple {
                Tuple(vec![$($t.into()),*])
            }
        }
    )*};
}

tuple_from! { () (A, B) (A, B, C) (A, B, C, D) (A, B, C, D, E) (A, B, C, D, E, F) }

/// Marker for closures over a runtime number of values: `FnOnce(&[Val]) -> Tuple`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Dyn;

/// A Rust closure building a region body: it takes one value per region
/// parameter, `FnOnce(Val<A>, Val<B>, ..) -> Tuple` (marker `(Val<A>, Val<B>, ..)`,
/// where each `Val<T>` may be the dynamic `Val`), or all of them as a slice,
/// `FnOnce(&[Val]) -> Tuple` (marker [`Dyn`]).
pub trait MlirClosure<Args> {
    /// Number of values taken, or `None` for any number.
    const ARITY: Option<usize>;

    /// Whether the parameters can have these types.
    fn accepts(types: &[Ty]) -> bool;

    /// Calls the closure on the region parameters.
    fn call(self, args: &[Val]) -> Tuple;
}

impl<F: FnOnce(&[Val]) -> Tuple> MlirClosure<Dyn> for F {
    const ARITY: Option<usize> = None;

    fn accepts(_: &[Ty]) -> bool {
        true
    }

    fn call(self, args: &[Val]) -> Tuple {
        self(args)
    }
}

/// Expands to the type `Val` whatever its input; counts values in macros.
#[doc(hidden)]
#[macro_export]
macro_rules! __val {
    ($($x:tt)*) => {
        $crate::Val
    };
}

/// Invokes `$m!(extra..; a b ..)` for every supported closure arity (0 to 8).
macro_rules! for_each_arity {
    ($m:ident $($extra:tt)*) => {
        $m!($($extra)*;);
        $m!($($extra)*; a);
        $m!($($extra)*; a b);
        $m!($($extra)*; a b c);
        $m!($($extra)*; a b c d);
        $m!($($extra)*; a b c d e);
        $m!($($extra)*; a b c d e f);
        $m!($($extra)*; a b c d e f g);
        $m!($($extra)*; a b c d e f g h);
    };
}

pub(crate) use for_each_arity;

macro_rules! closure_arity {
    (; $($a:ident)*) => {
        #[allow(non_camel_case_types)]
        impl<F, $($a: ?Sized),*> MlirClosure<($(Val<$a>,)*)> for F
        where
            F: FnOnce($(Val<$a>),*) -> Tuple,
            $(Val<$a>: AsVal,)*
        {
            const ARITY: Option<usize> = Some(<[&str]>::len(&[$(stringify!($a)),*]));

            #[allow(unused_variables)]
            fn accepts(types: &[Ty]) -> bool {
                let [$($a),*] = types else { return false };
                true $(&& <Val<$a>>::accepts($a))*
            }

            #[allow(unused_variables)]
            fn call(self, args: &[Val]) -> Tuple {
                let [$($a),*] = args else {
                    panic!("closure takes {:?} values, got {}", Self::ARITY, args.len());
                };
                self($(<Val<$a>>::from_val($a.clone()).expect("the parameter types were checked")),*)
            }
        }
    };
}

for_each_arity!(closure_arity);

/// Checks that a closure of marker `A` can take values of `types`.
pub(crate) fn check_params<A, C: MlirClosure<A>>(op: &str, types: &[Ty]) -> Result<(), TypeError> {
    match C::ARITY {
        Some(arity) if arity != types.len() => {
            Err(TypeError::new(op, format!("the body takes {arity} values but the region binds {}", types.len())))
        }
        _ if !C::accepts(types) => Err(TypeError::new(
            op,
            format!("the body does not take values of types ({})", crate::types::TypeList(types)),
        )),
        _ => Ok(()),
    }
}

/// A type with the static MLIR type `Mlir` (`I32` has `IntType`, a
/// `Ty<MemRefType>` has `MemRefType`, a dynamic `Ty` has `dyn MlirType`).
pub trait IntoTy {
    /// The static MLIR type.
    type Mlir: ?Sized;

    /// This type as a dynamic [`Ty`].
    fn into_ty(self) -> Ty;
}

impl<T: MlirType> IntoTy for T {
    type Mlir = T;

    fn into_ty(self) -> Ty {
        Ty::new(self).erase()
    }
}

impl<T: MlirType> IntoTy for Ty<T> {
    type Mlir = T;

    fn into_ty(self) -> Ty {
        self.erase()
    }
}

impl IntoTy for Ty {
    type Mlir = dyn MlirType;

    fn into_ty(self) -> Ty {
        self
    }
}

/// A value with the static type `Static`: a `Val<T>`, or a Rust literal
/// (an `arith.constant` of `i32`, `i64`, `index` (`usize`), `f32`, `f64` or `i1` (`bool`)).
pub trait IntoVal {
    /// The static MLIR type of the value.
    type Static: ?Sized;

    /// This value as a dynamic [`Val`].
    fn into_val(self) -> Val;
}

impl<T: ?Sized> IntoVal for Val<T>
where
    Val<T>: AsVal,
{
    type Static = T;

    fn into_val(self) -> Val {
        self.as_val()
    }
}

impl<T: ?Sized> IntoVal for &Val<T>
where
    Val<T>: AsVal,
{
    type Static = T;

    fn into_val(self) -> Val {
        self.as_val()
    }
}

macro_rules! literal_into_val {
    ($($t:ty => $mlir:ty;)*) => {$(
        impl IntoVal for $t {
            type Static = $mlir;

            #[track_caller]
            fn into_val(self) -> Val {
                self.into()
            }
        }
    )*};
}

literal_into_val! {
    i8 => crate::IntType;
    i16 => crate::IntType;
    i32 => crate::IntType;
    i64 => crate::IntType;
    usize => crate::IndexType;
    f32 => crate::FloatType;
    f64 => crate::FloatType;
    bool => crate::IntType;
}

/// Values of static shape `Self` made from a [`Tuple`]: `()`, a `Val<A>`,
/// or a Rust tuple of `Val`s.
pub trait FromTuple: Sized {
    /// The values, if they have the static types.
    fn from_tuple(t: Tuple) -> Option<Self>;
}

impl FromTuple for Tuple {
    fn from_tuple(t: Tuple) -> Option<Tuple> {
        Some(t)
    }
}

impl FromTuple for () {
    fn from_tuple(t: Tuple) -> Option<()> {
        t.0.is_empty().then_some(())
    }
}

impl<A: ?Sized> FromTuple for Val<A>
where
    Val<A>: AsVal,
{
    fn from_tuple(t: Tuple) -> Option<Val<A>> {
        let [a] = <[Val; 1]>::try_from(t.0).ok()?;
        Val::<A>::from_val(a)
    }
}

/// Typed values of several types, as the types `Self` describe.
pub trait IntoTys {
    /// The closure marker: `(Val<A>, Val<B>, ..)`.
    type Vals;
    /// The results shape: `()`, `Val<A>` or `(Val<A>, Val<B>, ..)`.
    type Results: FromTuple;

    /// The types, in order.
    fn into_tys(self) -> Vec<Ty>;
}

/// Values of several static types, e.g. initial values of a loop.
pub trait IntoVals {
    /// The closure marker: `(Val<A>, Val<B>, ..)`.
    type Vals;
    /// The marker with a leading `Val<I>`, e.g. for an induction variable.
    type WithFirst<I: ?Sized>;
    /// The results shape: `()`, `Val<A>` or `(Val<A>, Val<B>, ..)`.
    type Results: FromTuple;

    /// The values, in order.
    fn into_vals(self) -> Vec<Val>;
}

impl IntoTys for () {
    type Vals = ();
    type Results = ();

    fn into_tys(self) -> Vec<Ty> {
        vec![]
    }
}

impl IntoVals for () {
    type Vals = ();
    type WithFirst<I: ?Sized> = (Val<I>,);
    type Results = ();

    fn into_vals(self) -> Vec<Val> {
        vec![]
    }
}

impl<A: IntoTy> IntoTys for (A,)
where
    Val<A::Mlir>: AsVal,
{
    type Vals = (Val<A::Mlir>,);
    type Results = Val<A::Mlir>;

    fn into_tys(self) -> Vec<Ty> {
        vec![self.0.into_ty()]
    }
}

impl<A: IntoVal> IntoVals for (A,)
where
    Val<A::Static>: AsVal,
{
    type Vals = (Val<A::Static>,);
    type WithFirst<I: ?Sized> = (Val<I>, Val<A::Static>);
    type Results = Val<A::Static>;

    #[track_caller]
    fn into_vals(self) -> Vec<Val> {
        vec![self.0.into_val()]
    }
}

macro_rules! typed_tuples {
    ($(($($t:ident $i:tt),*))*) => {$(
        #[allow(non_camel_case_types)]
        impl<$($t: IntoTy),*> IntoTys for ($($t,)*)
        where
            $(Val<$t::Mlir>: AsVal,)*
        {
            type Vals = ($(Val<$t::Mlir>,)*);
            type Results = ($(Val<$t::Mlir>,)*);

            fn into_tys(self) -> Vec<Ty> {
                vec![$(self.$i.into_ty()),*]
            }
        }

        #[allow(non_camel_case_types)]
        impl<$($t: IntoVal),*> IntoVals for ($($t,)*)
        where
            $(Val<$t::Static>: AsVal,)*
        {
            type Vals = ($(Val<$t::Static>,)*);
            type WithFirst<I: ?Sized> = (Val<I>, $(Val<$t::Static>,)*);
            type Results = ($(Val<$t::Static>,)*);

            #[track_caller]
            fn into_vals(self) -> Vec<Val> {
                vec![$(self.$i.into_val()),*]
            }
        }

        #[allow(non_camel_case_types)]
        impl<$($t: ?Sized),*> FromTuple for ($(Val<$t>,)*)
        where
            $(Val<$t>: AsVal,)*
        {
            #[allow(unused_variables, non_snake_case)]
            fn from_tuple(t: Tuple) -> Option<Self> {
                let [$($t),*] = <[Val; [$($i),*].len()]>::try_from(t.0).ok()?;
                Some(($(Val::<$t>::from_val($t)?,)*))
            }
        }
    )*};
}

typed_tuples! {
    (a 0, b 1)
    (a 0, b 1, c 2)
    (a 0, b 1, c 2, d 3)
    (a 0, b 1, c 2, d 3, e 4)
    (a 0, b 1, c 2, d 3, e 4, f 5)
    (a 0, b 1, c 2, d 3, e 4, f 5, g 6)
    (a 0, b 1, c 2, d 3, e 4, f 5, g 6, h 7)
}

/// One region built by [`ClosureBuilder::region`]: its parameters, its
/// operations in build order and the values it yields.
#[derive(Clone)]
pub struct BuiltRegion {
    pub params: Vec<Arc<Node>>,
    pub ops: Vec<Arc<Node>>,
    pub exprs: Vec<Val>,
}

impl BuiltRegion {
    /// A region without parameters, operations or values (e.g. the body of
    /// an external function).
    pub fn empty() -> BuiltRegion {
        BuiltRegion { params: vec![], ops: vec![], exprs: vec![] }
    }

    /// The parameters, as values.
    pub fn param_vals(&self) -> Vec<Val> {
        self.params.iter().map(|p| Val::new(p, 0)).collect()
    }

    /// The types of the yielded values.
    pub fn types(&self) -> Vec<Ty> {
        self.exprs.iter().map(|v| v.ty()).collect()
    }
}

/// Regions compare and hash by the identity of their operations.
impl PartialEq for BuiltRegion {
    fn eq(&self, other: &BuiltRegion) -> bool {
        let same = |a: &[Arc<Node>], b: &[Arc<Node>]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| Arc::ptr_eq(x, y));
        same(&self.params, &other.params) && same(&self.ops, &other.ops) && self.exprs == other.exprs
    }
}

impl Eq for BuiltRegion {}

impl Hash for BuiltRegion {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for n in self.params.iter().chain(&self.ops) {
            n.key().hash(state);
        }
        self.exprs.hash(state);
    }
}

impl fmt::Debug for BuiltRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BuiltRegion({} params, {} ops, {} yielded)", self.params.len(), self.ops.len(), self.exprs.len())
    }
}

/// Builds the regions of a closure operation.
///
/// While it is alive, its frame is on the thread-local stack: operations
/// built are recorded in the region under construction. Build the closure
/// operation after [`finish`](Self::finish), in the enclosing region.
pub struct ClosureBuilder {
    depth: usize,
    finished: bool,
}

impl ClosureBuilder {
    /// Starts a closure; an `isolated` one cannot use values from outside.
    pub fn new(isolated: bool) -> ClosureBuilder {
        ClosureBuilder { depth: frames::push(isolated), finished: false }
    }

    /// Builds the next region: binds parameters of `param_types`, records
    /// the operations `body` builds and yields the values it returns.
    pub fn region(&mut self, param_types: &[Ty], body: impl FnOnce(&[Val]) -> Tuple) -> BuiltRegion {
        let region = frames::begin_region(self.depth);
        let params: Vec<Arc<Node>> =
            param_types.iter().map(|ty| Arc::new(Node { region, op: Param { ty: ty.clone() } }) as Arc<Node>).collect();
        let vals: Vec<Val> = params.iter().map(|p| Val::new(p, 0)).collect();
        let exprs = body(&vals).0;
        let ops = frames::end_region(self.depth, &exprs);
        BuiltRegion { params, ops, exprs }
    }

    /// Ends the closure; fails if a region used or yielded a value that is
    /// not visible in it.
    pub fn try_finish(mut self) -> Result<(), TypeError> {
        self.finished = true;
        frames::pop(self.depth)
    }

    /// Ends the closure; panics if a region used or yielded a value that is
    /// not visible in it.
    #[track_caller]
    pub fn finish(self) {
        self.try_finish().or_panic()
    }
}

impl Drop for ClosureBuilder {
    fn drop(&mut self) {
        if !self.finished {
            let _ = frames::pop(self.depth);
        }
    }
}

/// The thread-local stack of closures under construction.
mod frames {
    use super::*;

    struct Frame {
        isolated: bool,
        /// The region under construction.
        region: RegionId,
        /// Its operations, in build order.
        ops: Vec<Arc<Node>>,
        /// The first error of yielded values.
        error: Option<TypeError>,
    }

    thread_local! {
        static FRAMES: Mutex<Vec<Frame>> = const { Mutex::new(Vec::new()) };
    }

    pub(super) fn push(isolated: bool) -> usize {
        FRAMES.with(|f| {
            let f = &mut *f.lock();
            f.push(Frame { isolated, region: fresh_region_id(), ops: vec![], error: None });
            f.len() - 1
        })
    }

    pub(super) fn begin_region(depth: usize) -> RegionId {
        FRAMES.with(|f| {
            let frame = &mut f.lock()[depth];
            frame.region = fresh_region_id();
            frame.region
        })
    }

    /// Whether operations of `region` are visible in the region under
    /// construction: in it or in an enclosing one, up to an isolated closure.
    fn visible(frames: &[Frame], region: RegionId) -> bool {
        for frame in frames.iter().rev() {
            if frame.region == region {
                return true;
            }
            if frame.isolated {
                return false;
            }
        }
        false
    }

    /// The error if `v` cannot be used where `frames` build.
    fn check(frames: &[Frame], user: &str, v: &Val) -> Result<(), TypeError> {
        let Some(node) = v.try_op() else {
            return Err(TypeError::new(user, "uses a value whose operation was dropped"));
        };
        if visible(frames, node.region) {
            Ok(())
        } else {
            let message = format!(
                "uses a value of `{}` that is not visible here (built in a finished region, or outside an isolated closure)",
                node.name()
            );
            Err(TypeError::new(user, message))
        }
    }

    /// Checks the operands of `op` and records it in the region under construction.
    pub(super) fn record(op: impl HighMlir) -> Result<Arc<Node>, TypeError> {
        FRAMES.with(|f| {
            let f = &mut *f.lock();
            let Some(top) = f.last() else {
                return Err(TypeError::new(op.name(), "built outside of a module or closure"));
            };
            let region = top.region;
            for v in op.inputs() {
                check(f, op.name(), &v)?;
            }
            let node: Arc<Node> = Arc::new(Node { region, op });
            f.last_mut().unwrap().ops.push(node.clone());
            Ok(node)
        })
    }

    /// Ends the region under construction; returns its operations.
    pub(super) fn end_region(depth: usize, exprs: &[Val]) -> Vec<Arc<Node>> {
        FRAMES.with(|f| {
            let f = &mut *f.lock();
            if f[depth].error.is_none() {
                let error = exprs.iter().find_map(|v| check(&f[..=depth], "yield", v).err());
                f[depth].error = error;
            }
            std::mem::take(&mut f[depth].ops)
        })
    }

    /// Pops the closure; returns the first error of its yielded values.
    pub(super) fn pop(depth: usize) -> Result<(), TypeError> {
        let frame = FRAMES.with(|f| {
            let f = &mut *f.lock();
            assert_eq!(f.len(), depth + 1, "closures must be finished innermost first");
            f.pop().unwrap()
        });
        frame.error.map_or(Ok(()), Err)
    }
}

/// An operation was built from ill-typed operands.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TypeError {
    pub op: String,
    pub message: String,
}

impl TypeError {
    /// A type error in operation `op`.
    pub fn new(op: &str, message: impl Into<String>) -> TypeError {
        TypeError { op: op.to_string(), message: message.into() }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type error in `{}`: {}", self.op, self.message)
    }
}

impl std::error::Error for TypeError {}

/// Panics with the type error at the caller's location.
pub(crate) trait OrPanic<T> {
    fn or_panic(self) -> T;
}

impl<T> OrPanic<T> for Result<T, TypeError> {
    #[track_caller]
    fn or_panic(self) -> T {
        match self {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        }
    }
}
