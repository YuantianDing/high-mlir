//! Build [MLIR](https://mlir.llvm.org) as typed Rust expressions.
//!
//! Every operation is an expression whose value has a static MLIR type
//! ([`Val<T>`](Val)): `x + 1` on a `Val<IntType>` builds `arith.addi` and
//! returns a `Val<IntType>`. Operands are type-checked as the IR is built,
//! operations are emitted in the order they are built, and regions
//! (modules, functions, loop bodies, ...) are Rust closures. The module
//! prints as textual MLIR through `Display`, or converts to
//! [melior](https://crates.io/crates/melior) IR with `Root::to_melior`
//! (with the default `melior` feature).
//!
//! ```
//! use high_mlir::{F32, INDEX, builtin, memref, scf, sig};
//!
//! let module = builtin::module(|| {
//!     sig!(fn sum(memref![F32; ?], INDEX) -> F32).define(|m, n| {
//!         let r = scf!{for 0 to n step 1 iter_args(0f32)}.then(|i, acc| (acc + m.load(i)).into());
//!         r.into()
//!     });
//! });
//! let text = module.to_string();
//! assert!(text.contains("scf.for"));
//! ```
//!
//! # Where to start
//!
//! - Functions: [`sig!`] and [`func::Signature`]; modules: [`builtin::module`].
//! - Operations: one function per operation, named after it, in [`arith`],
//!   [`mod@memref`], [`mod@scf`] (with the [`scf!`] macro), [`mod@vector`],
//!   [`mod@tensor`], [`math`], [`index`], [`complex`], [`ub`],
//!   [`bufferization`], [`affine`], [`linalg`] and [`gpu`]; memref, vector,
//!   tensor, complex and scalar values also have common ones as methods.
//! - Types: [`types`] (including [`ComplexType`]), the constants [`I32`], [`F32`], [`INDEX`], … and the
//!   [`memref!`], [`tensor!`] and [`vector!`] macros.
//! - Attributes: [`Attr`], [`AffineMap`] and [`IntegerSet`]; discardable attributes and
//!   source locations: [`annotations`].
//! - New operations and types: [`HighMlir`], [`Closure`] and [`MlirType`].
//!
//! # Model
//!
//! - **Values.** A [`Val<T>`](Val) is an output of an operation, of MLIR type
//!   `T`; the default `Val = Val<dyn MlirType>` is dynamically typed and
//!   [`Val::downcast`] recovers a static type. Functions return one value, a
//!   Rust tuple for a fixed number of results, a [`Tuple`] for a variable
//!   number, or `()`.
//! - **Regions.** A thread-local stack holds the regions being built: a
//!   closure such as [`builtin::module`], a function body or a loop body
//!   pushes one, and every operation built while it is open is appended to
//!   it, in order, whether or not its results are used. The region owns its
//!   operations; a [`Val`] holds a weak pointer to its operation and its
//!   result number, so a value outlives its module only as a dangling
//!   reference. Building an operation with no region open is an error, as
//!   is using a value of a region that is not open (or, across an isolated
//!   region such as a function, of an enclosing one).
//! - **Roots.** [`builtin::module`] returns the [`Root`] owning the IR.
//!   [`ClosureBuilder`] builds the regions of new closure operations.
//!
//! # Features
//!
//! - `melior` (default): conversion to melior IR (`Root::to_melior`,
//!   `context`). It links against an LLVM/MLIR 23 installation through
//!   `mlir-sys`; see the README. Without it, the crate is pure Rust: IR is
//!   built, type-checked and printed as text.

#![cfg_attr(docsrs, feature(doc_cfg))]

/// MLIR attributes and affine maps.
pub mod attr;
#[cfg(feature = "melior")]
mod convert;
/// The dialects; see the README's roadmap for their coverage.
pub mod dialect;
pub mod mixed;
mod print;
mod lower;
pub mod types;
pub mod val;
pub mod annotations;

pub use attr::{AffineExpr, AffineMap, Attr, IntegerSet};
#[cfg(feature = "melior")]
#[cfg_attr(docsrs, doc(cfg(feature = "melior")))]
pub use convert::context;
/// The melior version this crate converts to.
#[cfg(feature = "melior")]
#[cfg_attr(docsrs, doc(cfg(feature = "melior")))]
pub use melior;
pub use dialect::{affine, arith, bufferization, builtin, complex, func, gpu, index, linalg, math, memref, scf, tensor, ub, vector};
pub use print::OpPrinter;
pub use lower::Error;
pub use mixed::{IntoMixed, Mixed, MixedList};
pub use types::{
    BF16, ComplexType, F16, F32, F64, FloatType, FunctionType, I1, I8, I16, I32, I64, INDEX, IndexType, IntType, Layout,
    MemRefType, MlirType, NoneType, TensorType, Ty, UnrankedMemRefType, VectorType,
};
pub use val::{
    AsVal, BuiltRegion, Closure, ClosureBuilder, Dyn, Edges, FromTuple, HighMlir, IntoTy, IntoTys, IntoVal, IntoVals,
    MlirClosure, Node, OpRef, Param, RegionId, Root, Tuple, TypeError, Val,
};

/// A memref type: `memref![F32; ?, 4]` is `memref<?x4xf32>`, `memref![F32]` is
/// `memref<f32>` and `memref![F32; *]` is the unranked `memref<*xf32>`.
#[macro_export]
macro_rules! memref {
    ($element:expr; *) => {
        $crate::Ty::new($crate::UnrankedMemRefType { element: $crate::Ty::from($element), memory_space: None })
    };
    ($element:expr $(; $($dim:tt),* $(,)?)?) => {
        $crate::Ty::new($crate::MemRefType::new([$($($crate::__dim!($dim)),*)?], $element))
    };
}

/// A tensor type: `tensor![I32; 4, ?]` is `tensor<4x?xi32>`.
#[macro_export]
macro_rules! tensor {
    ($element:expr $(; $($dim:tt),* $(,)?)?) => {
        $crate::Ty::new($crate::TensorType {
            shape: ::std::vec![$($($crate::__dim!($dim)),*)?],
            element: $crate::Ty::from($element),
        })
    };
}

/// A vector type: `vector![F32; 4, [8]]` is `vector<4x[8]xf32>` (bracketed
/// dimensions are scalable), `vector![F32]` is the 0-d `vector<f32>`.
#[macro_export]
macro_rules! vector {
    ($element:expr $(; $($dim:tt),* $(,)?)?) => {
        $crate::Ty::new($crate::VectorType::scalable([$($($crate::__vdim!($dim)),*)?], $element))
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __vdim {
    ([$n:expr]) => {
        (($n) as u64, true)
    };
    ($n:expr) => {
        (($n) as u64, false)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __dim {
    (?) => {
        ::core::option::Option::None
    };
    ($n:expr) => {
        ::core::option::Option::Some(($n) as u64)
    };
}

/// A function signature: `sig!(fn add(I32, I32) -> I32)`, `sig!(fn swap(F32, F32) -> (F32, F32))`
/// or `sig!(fn fill(memref![F32; ?]))`. Its `define` takes a closure with
/// one value per parameter, of the static type of the parameter type (e.g.
/// `Val<IntType>` for `I32`); its `call` returns the results the same way.
#[macro_export]
macro_rules! sig {
    (fn $name:ident ($($input:expr),* $(,)?) -> ($($output:expr),* $(,)?)) => {
        $crate::func::Signature::typed(stringify!($name), ($($input,)*), ($($output,)*))
    };
    (fn $name:ident ($($input:expr),* $(,)?) -> $output:expr) => {
        $crate::func::Signature::typed(stringify!($name), ($($input,)*), ($output,))
    };
    (fn $name:ident ($($input:expr),* $(,)?)) => {
        $crate::func::Signature::typed(stringify!($name), ($($input,)*), ())
    };
}

/// Structured control flow:
///
/// - `scf!{for lb to ub step s iter_args(a, b)}.then(|iv, a, b| ...)`;
///   bounds are single tokens (wrap expressions in parentheses) and are borrowed;
///   `.unsigned()` before `.then` compares them as unsigned;
/// - `scf!{if cond}.then(|| ...).else_(|| ...)`, or `.end()` without else;
/// - `scf!{parallel (lb, ..) to (ub, ..) step (s, ..)}.then(|i, ..| ...)`, and with
///   reductions `scf!{parallel (..) to (..) step (..) init(a, ..)}.then_reduce(|i, ..| ..., |k, x, y| ...)`;
/// - `scf!{while (a, ..)}.cond(|a, ..| (cond, a, ..).into()).body(|a, ..| ...)`;
/// - `scf!{forall (ub, ..)}.then(|i, ..| ...)` or
///   `scf!{forall (lb, ..) to (ub, ..) step (s, ..)}.then(...)`, where literal
///   bounds are static.
///
/// Closure parameters are typed: induction variables of `parallel` and
/// `forall` are `Val<IndexType>`, that of `for` has the static type of the
/// upper bound (`index` for a literal), and loop-carried values have the
/// static types of their initial values (`0f32` is a `Val<FloatType>`).
/// The results of `for`, `while` and `parallel` have the same types: `()`,
/// one `Val<T>` or a Rust tuple; those of `if` are a [`Tuple`].
///
/// Bodies return a [`Tuple`]: end them with `x.into()`, `(x, y).into()` or `().into()`.
/// See also [`scf::index_switch`] and [`scf::execute_region`].
#[macro_export]
macro_rules! scf {
    (for $lower:tt to $upper:tt step $step:tt $(iter_args($($init:expr),* $(,)?))?) => {
        $crate::scf::For::build(&$lower, &$upper, &$step, ($($($init,)*)?))
    };
    (if $condition:expr) => {
        $crate::scf::If::build($condition)
    };
    (parallel ($($lower:expr),+ $(,)?) to ($($upper:expr),+ $(,)?) step ($($step:expr),+ $(,)?)
        $(init($($init:expr),* $(,)?))?) => {
        $crate::scf::Parallel::build::<($($crate::__index!($lower),)+)>(($(&$lower,)+), ($(&$upper,)+), ($(&$step,)+))
            .init(($($($init,)*)?))
    };
    (while ($($init:expr),* $(,)?)) => {
        $crate::scf::While::build(($($init,)*))
    };
    (forall ($($lower:expr),+ $(,)?) to ($($upper:expr),+ $(,)?) step ($($step:expr),+ $(,)?)) => {
        $crate::scf::Forall::build::<($($crate::__index!($lower),)+)>(($(&$lower,)+), ($(&$upper,)+), ($(&$step,)+))
    };
    (forall ($($upper:expr),+ $(,)?)) => {
        $crate::scf::Forall::build_upper::<($($crate::__index!($upper),)+)>(($(&$upper,)+))
    };
}

/// Expands to `Val<IndexType>` whatever its input: the type of an induction variable.
#[doc(hidden)]
#[macro_export]
macro_rules! __index {
    ($($x:tt)*) => {
        $crate::Val<$crate::IndexType>
    };
}

/// The examples of the README, compiled and run as doctests.
#[cfg(all(doctest, feature = "melior"))]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
