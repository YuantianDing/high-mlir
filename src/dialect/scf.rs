//! `scf` dialect: structured control flow as closures.
//!
//! Build these with the [`scf!`](crate::scf!) macro:
//!
//! ```ignore
//! scf!{for 0 to n step 1 iter_args(0i64)}.then(|iv, acc| (acc + iv.cast(I64)).into())
//! scf!{if c}.then(|| x.into()).else_(|| y.into())
//! scf!{parallel (0, 0) to (n, m) step (1, 1)}.then(|i, j| { ...; ().into() })
//! scf!{parallel (0) to (n) step (1) init(0f32)}.then_reduce(|i| m.load(i).into(), |_, a, b| a + b)
//! scf!{while (x0)}.cond(|x| (x.lt(10), x).into()).body(|x| (x * 2).into())
//! scf!{forall (n, 8)}.then(|i, j| { ...; ().into() })
//! ```
//!
//! and `scf.index_switch` / `scf.execute_region` with [`index_switch`] and
//! [`execute_region`].

use std::{
    fmt::{self, Write},
    marker::PhantomData,
};

use crate::{
    Attr, BuiltRegion, Closure, ClosureBuilder, Edges, HighMlir, IndexType, OpPrinter, OpRef, Ty,
    TypeError, Val, annotations,
    arith::{Operand, Operands},
    dialect::util::{expect, indices, types, value_ty},
    mixed::{MixedList, MixedValues},
    types::{I1, INDEX, TypeList},
    val::{AsVal, Dyn, FromTuple, IntoVals, MlirClosure, OrPanic, Tuple, check_params, for_each_arity},
};

/// The terminator built from the values of a region body, located at
/// `caller`; a type error is stashed into `error`.
#[track_caller]
fn terminated(
    error: &mut Option<TypeError>,
    caller: &std::panic::Location<'_>,
    terminator: impl FnOnce(Vec<Val>) -> Result<Val, TypeError>,
    values: Tuple,
) -> Tuple {
    match terminator(values.0) {
        Ok(t) => {
            // Built in a closure: located at the caller of the loop builder.
            annotations::record_rust_location_at(&t.op(), caller);
            Tuple(vec![t])
        }
        Err(e) => {
            *error = Some(e);
            Tuple::default()
        }
    }
}

/// The three values, or the first error.
fn all3<T>(a: Result<T, TypeError>, b: Result<T, TypeError>, c: Result<T, TypeError>) -> Result<[T; 3], TypeError> {
    Ok([a?, b?, c?])
}

/// `scf.for` with loop-carried values.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct For {
    pub lower: Val,
    pub upper: Val,
    pub step: Val,
    pub inits: Vec<Val>,
    /// Compare the induction variable as unsigned.
    pub unsigned: bool,
    pub body: BuiltRegion,
}

/// The static type of a loop bound: `T` for a `Val<T>`, `index` for a literal.
pub trait Bound {
    /// The static type.
    type Ty: ?Sized;
}

impl<T: ?Sized> Bound for Val<T> {
    type Ty = T;
}

impl<T: Bound + ?Sized> Bound for &T {
    type Ty = T::Ty;
}

macro_rules! literal_bounds {
    ($($t:ty),*) => {$(
        impl Bound for $t {
            type Ty = IndexType;
        }
    )*};
}

literal_bounds!(i32, i64, isize, u32, u64, usize);

/// `scf!{for lb to ub step s iter_args(...)}`, waiting for its body.
///
/// `A` is the marker of the body: `(Val<I>, Val<T>, ..)` for the induction
/// variable (of the static type of the upper bound) and each loop-carried
/// value (of the static type of its initial value), or [`Dyn`]. `R` is the
/// shape of the results: `()`, `Val<T>`, a Rust tuple or a [`Tuple`].
#[must_use = "a loop is built by `.then(body)`"]
pub struct ForBuilder<A = Dyn, R = Tuple> {
    bounds: Result<[Val; 3], TypeError>,
    inits: Vec<Val>,
    unsigned: bool,
    marker: PhantomData<fn() -> (A, R)>,
}

impl For {
    /// Bounds may be literals; they take the type of the bound that is a
    /// value, or `index`.
    #[track_caller]
    pub fn build<U: Operand + Bound, I: IntoVals>(
        lower: impl Operand,
        upper: U,
        step: impl Operand,
        inits: I,
    ) -> ForBuilder<I::WithFirst<U::Ty>, I::Results> {
        let like = lower.ty_hint().or_else(|| upper.ty_hint()).or_else(|| step.ty_hint()).unwrap_or(INDEX.into());
        let bounds = all3(lower.into_val(&like), upper.into_val(&like), step.into_val(&like));
        ForBuilder { bounds, inits: inits.into_vals(), unsigned: false, marker: PhantomData }
    }
}

impl<A, R: FromTuple> ForBuilder<A, R> {
    /// Compares the induction variable with the upper bound as unsigned.
    #[track_caller]
    pub fn unsigned(self) -> ForBuilder<A, R> {
        ForBuilder { unsigned: true, ..self }
    }

    /// `body(iv, iter_args..)` returns the next values of the loop-carried
    /// values, which are also the loop's results.
    #[track_caller]
    pub fn try_then_with<C: MlirClosure<A>>(self, body: C) -> Result<R, TypeError> {
        let name = "scf.for";
        let [lower, upper, step] = self.bounds?;
        let iv_ty = value_ty(name, &lower)?;
        if !(iv_ty.is_index() || iv_ty.int_width().is_some()) {
            return Err(TypeError::new(name, format!("bounds must be index or integer, not `{iv_ty}`")));
        }
        for b in [&upper, &step] {
            let t = value_ty(name, b)?;
            if t != iv_ty {
                return Err(TypeError::new(name, format!("bound types differ: `{iv_ty}` and `{t}`")));
            }
        }
        let init_types = types(name, &self.inits)?;
        let param_types: Vec<Ty> = std::iter::once(iv_ty).chain(init_types.iter().cloned()).collect();
        check_params::<A, C>(name, &param_types)?;
        let mut builder = ClosureBuilder::new(false);
        let region = builder.region(&param_types, |p| body.call(p));
        builder.try_finish()?;
        let yielded = types(name, &region.exprs)?;
        if yielded != init_types {
            return Err(TypeError::new(
                name,
                format!("body yields ({}) but iter_args are ({})", TypeList(&yielded), TypeList(&init_types)),
            ));
        }
        let (inits, unsigned) = (self.inits, self.unsigned);
        typed_results(name, Val::all_results(For { lower, upper, step, inits, unsigned, body: region })?)
    }
}

/// The results of an operation in the shape `R`.
fn typed_results<R: FromTuple>(op: &str, results: Tuple) -> Result<R, TypeError> {
    let types = TypeList(&results.iter().map(|v| v.ty()).collect::<Vec<_>>()).to_string();
    R::from_tuple(results).ok_or_else(|| TypeError::new(op, format!("results ({types}) of unexpected static types")))
}

macro_rules! dyn_methods {
    ($builder:ident, $doc:literal) => {
        impl<R: FromTuple> $builder<Dyn, R> {
            #[doc = $doc]
            #[track_caller]
            pub fn try_then(self, body: impl FnOnce(&[Val]) -> Tuple) -> Result<R, TypeError> {
                self.try_then_with(body)
            }

            #[doc = $doc]
            #[track_caller]
            pub fn then(self, body: impl FnOnce(&[Val]) -> Tuple) -> R {
                self.try_then_with(body).or_panic()
            }
        }
    };
}

/// `then`/`try_then` taking one typed value per region parameter, so
/// closure parameter types are inferred.
macro_rules! arity_methods {
    ($builder:ident, $doc:literal; $($a:ident)*) => {
        #[allow(non_camel_case_types)]
        impl<R: FromTuple, $($a: ?Sized),*> $builder<($(Val<$a>,)*), R>
        where
            $(Val<$a>: AsVal,)*
        {
            #[doc = $doc]
            #[track_caller]
            pub fn try_then(self, body: impl FnOnce($(Val<$a>),*) -> Tuple) -> Result<R, TypeError> {
                self.try_then_with(body)
            }

            #[doc = $doc]
            #[track_caller]
            pub fn then(self, body: impl FnOnce($(Val<$a>),*) -> Tuple) -> R {
                self.try_then_with(body).or_panic()
            }
        }
    };
}

dyn_methods!(ForBuilder, "Builds the loop; `body(iv, iter_args..)` yields the next loop-carried values.");
for_each_arity!(arity_methods ForBuilder, "Builds the loop; `body(iv, iter_args..)` yields the next loop-carried values.");

impl HighMlir for For {
    fn name(&self) -> &str {
        "scf.for"
    }

    fn inputs(&self) -> Edges<'_> {
        let bounds = [self.lower.clone(), self.upper.clone(), self.step.clone()];
        Box::new(bounds.into_iter().chain(self.inits.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        self.inits.iter().map(|v| v.ty()).collect()
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        if self.unsigned { vec![("unsignedCmp".into(), Attr::Unit)] } else { vec![] }
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let iv = p.region_arg(0, 0);
        let unsigned = if self.unsigned { "unsigned " } else { "" };
        write!(p, "scf.for {unsigned}{iv} = {} to {} step {}", p.operand(0), p.operand(1), p.operand(2))?;
        if !self.inits.is_empty() {
            let iter_args: Vec<String> =
                (0..self.inits.len()).map(|i| format!("{} = {}", p.region_arg(0, i + 1), p.operand(i + 3))).collect();
            let types = p.operand_types(3..3 + self.inits.len());
            write!(p, " iter_args({}) -> ({})", iter_args.join(", "), TypeList(&types))?;
        }
        let iv_ty = p.operand_type(0);
        if !iv_ty.is_index() {
            write!(p, " : {iv_ty}")?;
        }
        p.write_str(" ")?;
        p.region(0, false)
    }
}

impl Closure for For {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("scf.yield")
    }
}

/// `scf.if`; both branches yield values of the same types.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct If {
    pub condition: Val,
    pub then_branch: BuiltRegion,
    pub else_branch: BuiltRegion,
}

/// `scf!{if cond}`, waiting for its branches.
#[must_use = "a branch is built by `.then(..).else_(..)` or `.then(..).end()`"]
pub struct IfBuilder {
    condition: Val,
}

/// `scf!{if cond}.then(..)`, waiting for its else branch.
#[must_use = "a branch is built by `.else_(..)` or `.end()`"]
pub struct IfThen<'a> {
    condition: Val,
    then_branch: Box<dyn FnOnce() -> Tuple + 'a>,
}

impl If {
    /// `scf.if` on an `i1` `condition`; see [`scf!`](crate::scf!).
    #[track_caller]
    pub fn build(condition: impl Into<Val>) -> IfBuilder {
        IfBuilder { condition: condition.into() }
    }
}

impl IfBuilder {
    /// The branch run when the condition holds.
    #[track_caller]
    pub fn then<'a>(self, then_branch: impl FnOnce() -> Tuple + 'a) -> IfThen<'a> {
        IfThen { condition: self.condition, then_branch: Box::new(then_branch) }
    }
}

impl IfThen<'_> {
    /// The else branch, yielding values of the types the then branch yields.
    #[track_caller]
    pub fn try_else_(self, else_branch: impl FnOnce() -> Tuple) -> Result<Tuple, TypeError> {
        let name = "scf.if";
        let condition = self.condition;
        let c = value_ty(name, &condition)?;
        if c != I1 {
            return Err(TypeError::new(name, format!("condition must be `i1`, not `{c}`")));
        }
        let mut builder = ClosureBuilder::new(false);
        let then_branch = builder.region(&[], |_| (self.then_branch)());
        let else_branch = builder.region(&[], |_| else_branch());
        builder.try_finish()?;
        let (t, e) = (types(name, &then_branch.exprs)?, types(name, &else_branch.exprs)?);
        if t != e {
            return Err(TypeError::new(name, format!("then yields ({}) but else yields ({})", TypeList(&t), TypeList(&e))));
        }
        Val::all_results(If { condition, then_branch, else_branch })
    }

    /// The else branch; `else` is a keyword, hence the underscore.
    #[track_caller]
    pub fn else_(self, else_branch: impl FnOnce() -> Tuple) -> Tuple {
        self.try_else_(else_branch).or_panic()
    }

    /// No else branch: the then branch must not yield values.
    #[track_caller]
    pub fn try_end(self) -> Result<Tuple, TypeError> {
        self.try_else_(Tuple::default)
    }

    /// No else branch: the then branch must not yield values.
    #[track_caller]
    pub fn end(self) -> Tuple {
        self.try_end().or_panic()
    }
}

impl If {
    fn branch(&self, region: usize) -> &BuiltRegion {
        if region == 0 { &self.then_branch } else { &self.else_branch }
    }
}

impl HighMlir for If {
    fn name(&self) -> &str {
        "scf.if"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.condition.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        self.then_branch.exprs.iter().map(|v| v.ty()).collect()
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "scf.if {}", p.operand(0))?;
        let results = self.result_types();
        if !results.is_empty() {
            write!(p, " -> ({})", TypeList(&results))?;
        }
        p.write_str(" ")?;
        p.region(0, false)?;
        p.write_str(" else ")?;
        p.region(1, false)
    }
}

impl Closure for If {
    fn region(&self, r: usize) -> &BuiltRegion {
        self.branch(r)
    }

    fn num_regions(&self) -> usize {
        2
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("scf.yield")
    }
}

/// `scf.parallel`: a parallel loop nest, whose iterations may produce values
/// combined by its [`Reduce`] terminator.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Parallel {
    pub lowers: Vec<Val>,
    pub uppers: Vec<Val>,
    pub steps: Vec<Val>,
    /// Initial values of the reductions, which are the results.
    pub inits: Vec<Val>,
    /// Yields the [`Reduce`] terminator.
    pub body: BuiltRegion,
}

/// `scf!{parallel (..) to (..) step (..) init(..)}`, waiting for its body.
///
/// `A` is the marker of the body, `(Val<IndexType>, ..)` for each induction
/// variable, or [`Dyn`]; `R` is the shape of the results of the reductions.
#[must_use = "a parallel loop is built by `.then(body)` or `.then_reduce(body, combine)`"]
pub struct ParallelBuilder<A = Dyn, R = ()> {
    bounds: Result<[Vec<Val>; 3], TypeError>,
    inits: Vec<Val>,
    marker: PhantomData<fn() -> (A, R)>,
}

impl Parallel {
    /// Bounds are `index` values; literals become `index` constants.
    #[track_caller]
    pub fn build<A>(lowers: impl Operands, uppers: impl Operands, steps: impl Operands) -> ParallelBuilder<A> {
        let index = |_| Some(INDEX.into());
        let bounds = all3(lowers.into_vals(&index), uppers.into_vals(&index), steps.into_vals(&index));
        ParallelBuilder { bounds, inits: vec![], marker: PhantomData }
    }
}

impl<A, R: FromTuple> ParallelBuilder<A, R> {
    /// The initial values of reductions, whose static types the results
    /// have; see [`then_reduce`](Self::try_then_reduce_with).
    #[track_caller]
    pub fn init<I: IntoVals>(self, inits: I) -> ParallelBuilder<A, I::Results> {
        ParallelBuilder { bounds: self.bounds, inits: inits.into_vals(), marker: PhantomData }
    }

    /// `body(ivs..)` performs the effects of one iteration and yields nothing.
    #[track_caller]
    pub fn try_then_with<C: MlirClosure<A>>(self, body: C) -> Result<R, TypeError> {
        expect("scf.parallel", self.inits.is_empty(), || "reductions need `then_reduce`".into())?;
        self.try_then_reduce_with(body, |_, _, _| unreachable!())
    }

    /// `body(ivs..)` yields one value per initial value; `combine(k, a, b)`
    /// combines two values of reduction `k`.
    #[track_caller]
    pub fn try_then_reduce_with<C: MlirClosure<A>>(
        self,
        body: C,
        combine: impl FnMut(usize, Val, Val) -> Val,
    ) -> Result<R, TypeError> {
        let caller = std::panic::Location::caller();
        let name = "scf.parallel";
        let [lowers, uppers, steps] = self.bounds?;
        check_params::<A, C>(name, &vec![INDEX.into(); lowers.len()])?;
        if lowers.is_empty() || lowers.len() != uppers.len() || lowers.len() != steps.len() {
            return Err(TypeError::new(name, "needs the same positive number of lower bounds, upper bounds and steps"));
        }
        for v in lowers.iter().chain(&uppers).chain(&steps) {
            let ty = value_ty(name, v)?;
            if !ty.is_index() {
                return Err(TypeError::new(name, format!("bounds must be `index`, not `{ty}`")));
            }
        }
        let init_types = types(name, &self.inits)?;
        let mut builder = ClosureBuilder::new(false);
        let mut error = None;
        let region = builder.region(&vec![INDEX.into(); lowers.len()], |ivs| {
            let values = body.call(ivs);
            terminated(&mut error, caller, |vs| Reduce::try_new(vs, &init_types, combine), values)
        });
        builder.try_finish()?;
        if let Some(e) = error {
            return Err(e);
        }
        let inits = self.inits;
        typed_results(name, Val::all_results(Parallel { lowers, uppers, steps, inits, body: region })?)
    }
}

dyn_methods!(ParallelBuilder, "Builds the loop; `body(ivs..)` performs one iteration and yields `()`.");
for_each_arity!(arity_methods ParallelBuilder, "Builds the loop; `body(ivs..)` performs one iteration and yields `()`.");

/// `then_reduce`/`try_then_reduce` for each arity.
macro_rules! reduce_methods {
    ($body:ty) => {
        /// Builds the loop; `body(ivs..)` yields one value per reduction and
        /// `combine(k, a, b)` combines two values of reduction `k`.
        #[track_caller]
        pub fn try_then_reduce(
            self,
            body: $body,
            combine: impl FnMut(usize, Val, Val) -> Val,
        ) -> Result<R, TypeError> {
            self.try_then_reduce_with(body, combine)
        }

        /// Builds the loop; `body(ivs..)` yields one value per reduction and
        /// `combine(k, a, b)` combines two values of reduction `k`.
        #[track_caller]
        pub fn then_reduce(self, body: $body, combine: impl FnMut(usize, Val, Val) -> Val) -> R {
            self.try_then_reduce_with(body, combine).or_panic()
        }
    };
    (; $($a:ident)*) => {
        #[allow(non_camel_case_types)]
        impl<R: FromTuple, $($a: ?Sized),*> ParallelBuilder<($(Val<$a>,)*), R>
        where
            $(Val<$a>: AsVal,)*
        {
            reduce_methods!(impl FnOnce($(Val<$a>),*) -> Tuple);
        }
    };
}

impl<R: FromTuple> ParallelBuilder<Dyn, R> {
    reduce_methods!(impl FnOnce(&[Val]) -> Tuple);
}

for_each_arity!(reduce_methods);

impl HighMlir for Parallel {
    fn name(&self) -> &str {
        "scf.parallel"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.lowers.iter().chain(&self.uppers).chain(&self.steps).chain(&self.inits).cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        self.inits.iter().map(|v| v.ty()).collect()
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let n = self.lowers.len();
        vec![Attr::segments([n, n, n, self.inits.len()])]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = self.lowers.len();
        let ivs: Vec<String> = (0..n).map(|i| p.region_arg(0, i)).collect();
        let (l, u, s) = (p.operands(0..n), p.operands(n..2 * n), p.operands(2 * n..3 * n));
        write!(p, "scf.parallel ({}) = ({l}) to ({u}) step ({s}) ", ivs.join(", "))?;
        if !self.inits.is_empty() {
            let range = 3 * n..3 * n + self.inits.len();
            let types = p.operand_types(range.clone());
            write!(p, "init ({}) -> ({}) ", p.operands(range), TypeList(&types))?;
        }
        p.region(0, false)
    }
}

impl Closure for Parallel {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        None
    }
}

/// `scf.reduce`, the terminator of [`Parallel`]: region `k` combines two
/// values of operand `k` (and ends with `scf.reduce.return`).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Reduce {
    pub operands: Vec<Val>,
    pub reductions: Vec<BuiltRegion>,
}

impl Reduce {
    /// `combine(k, a, b)` builds region `k`; `operands` must have `types`.
    #[track_caller]
    pub(crate) fn try_new(
        operands: Vec<Val>,
        types: &[Ty],
        mut combine: impl FnMut(usize, Val, Val) -> Val,
    ) -> Result<Val, TypeError> {
        let name = "scf.reduce";
        let actual = self::types(name, &operands)?;
        expect(name, actual == types, || {
            format!("the body yields ({}) but the reductions are ({})", TypeList(&actual), TypeList(types))
        })?;
        let mut builder = ClosureBuilder::new(false);
        let reductions: Vec<BuiltRegion> = types
            .iter()
            .enumerate()
            .map(|(k, ty)| builder.region(&[ty.clone(), ty.clone()], |p| combine(k, p[0].clone(), p[1].clone()).into()))
            .collect();
        builder.try_finish()?;
        for (r, ty) in reductions.iter().zip(types) {
            let yielded = self::types(name, &r.exprs)?;
            expect(name, yielded == [ty.clone()], || {
                format!("a reduction of `{ty}` yields ({})", TypeList(&yielded))
            })?;
        }
        Val::single(Reduce { operands, reductions })
    }
}

impl HighMlir for Reduce {
    fn name(&self) -> &str {
        "scf.reduce"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.operands.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.write_str("scf.reduce")?;
        let n = self.operands.len();
        if n > 0 {
            write!(p, "({} : {})", p.operands(0..n), TypeList(&p.operand_types(0..n)))?;
        }
        for r in 0..n {
            p.write_str(if r == 0 { " " } else { ", " })?;
            p.region(r, true)?;
        }
        Ok(())
    }
}

impl Closure for Reduce {
    fn region(&self, r: usize) -> &BuiltRegion {
        &self.reductions[r]
    }

    fn num_regions(&self) -> usize {
        self.reductions.len()
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("scf.reduce.return")
    }
}

/// `scf.while`: the `before` region computes a condition and the values
/// forwarded to the `after` region (or returned when it is false); the
/// `after` region yields the next values of the loop-carried values.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct While {
    pub inits: Vec<Val>,
    /// Yields the [`Condition`] terminator.
    pub before: BuiltRegion,
    pub after: BuiltRegion,
}

/// `scf!{while (inits..)}`, waiting for its condition.
///
/// `A` is the marker of the condition and body: `(Val<T>, ..)` with one
/// value per loop-carried value (of the static type of its initial value),
/// or [`Dyn`]; `R` is the shape of the results, which `body` forwards.
#[must_use = "a loop is built by `.cond(..).body(..)`"]
pub struct WhileBuilder<A = Dyn, R = Tuple> {
    inits: Vec<Val>,
    marker: PhantomData<fn() -> (A, R)>,
}

/// A region body taking the region parameters.
type Body<'a> = Box<dyn FnOnce(&[Val]) -> Tuple + 'a>;

/// `scf!{while (..)}.cond(..)`, waiting for its body.
#[must_use = "a loop is built by `.body(..)`"]
pub struct WhileCond<'a, A = Dyn, R = Tuple> {
    inits: Vec<Val>,
    cond: Body<'a>,
    error: Option<TypeError>,
    marker: PhantomData<fn() -> (A, R)>,
}

impl While {
    /// `scf.while` with loop-carried values starting at `inits`; see [`scf!`](crate::scf!).
    #[track_caller]
    pub fn build<I: IntoVals>(inits: I) -> WhileBuilder<I::Vals, I::Results> {
        WhileBuilder { inits: inits.into_vals(), marker: PhantomData }
    }
}

impl<A, R> WhileBuilder<A, R> {
    /// `cond(args..)` yields the condition followed by the forwarded values.
    #[track_caller]
    pub fn cond_with<'a, C: MlirClosure<A> + 'a>(self, cond: C) -> WhileCond<'a, A, R> {
        let name = "scf.while";
        let error = types(name, &self.inits).and_then(|t| check_params::<A, C>(name, &t)).err();
        WhileCond { inits: self.inits, cond: Box::new(|args| cond.call(args)), error, marker: PhantomData }
    }
}

impl<'a, A, R> WhileCond<'a, A, R> {
    /// `body(forwarded..)` yields the next loop-carried values; the results
    /// have the shape `S`.
    #[track_caller]
    pub fn try_body_with<B, C: MlirClosure<B>, S: FromTuple>(self, body: C) -> Result<S, TypeError> {
        let caller = std::panic::Location::caller();
        let name = "scf.while";
        if let Some(e) = self.error {
            return Err(e);
        }
        let init_types = types(name, &self.inits)?;
        let mut builder = ClosureBuilder::new(false);
        let mut error = None;
        let cond = self.cond;
        let before = builder.region(&init_types, |args| {
            let values = cond(args);
            terminated(&mut error, caller, Condition::try_new, values)
        });
        if let Some(e) = error {
            return Err(e);
        }
        let forwarded = before.exprs[0].op_as::<Condition>().unwrap().forwarded_types();
        check_params::<B, C>(name, &forwarded)?;
        let after = builder.region(&forwarded, |p| body.call(p));
        builder.try_finish()?;
        let yielded = types(name, &after.exprs)?;
        expect(name, yielded == init_types, || {
            format!("the body yields ({}) but the loop carries ({})", TypeList(&yielded), TypeList(&init_types))
        })?;
        typed_results(name, Val::all_results(While { inits: self.inits, before, after })?)
    }

    /// `body(&forwarded)`, for any number of forwarded values.
    #[track_caller]
    pub fn try_body_dyn(self, body: impl FnOnce(&[Val]) -> Tuple) -> Result<Tuple, TypeError> {
        self.try_body_with(body)
    }

    /// `body(&forwarded)`, for any number of forwarded values.
    #[track_caller]
    pub fn body_dyn(self, body: impl FnOnce(&[Val]) -> Tuple) -> Tuple {
        self.try_body_with(body).or_panic()
    }
}

/// `cond`/`body` taking one value per loop-carried (and forwarded) value.
macro_rules! while_methods {
    ([$($gen:tt)*] [$($bounds:tt)*] $marker:ty, [$($f:tt)*]) => {
        #[allow(non_camel_case_types)]
        impl<R: FromTuple, $($gen)*> WhileBuilder<$marker, R>
        where
            $($bounds)*
        {
            /// `cond(args..)` yields the condition followed by the forwarded values.
            #[track_caller]
            pub fn cond<'a>(self, cond: impl $($f)* + 'a) -> WhileCond<'a, $marker, R> {
                self.cond_with(cond)
            }
        }

        #[allow(non_camel_case_types)]
        impl<R: FromTuple, $($gen)*> WhileCond<'_, $marker, R>
        where
            $($bounds)*
        {
            /// `body(forwarded..)` yields the next loop-carried values; it
            /// takes as many values, of the same types, as the loop carries.
            #[track_caller]
            pub fn try_body(self, body: impl $($f)*) -> Result<R, TypeError> {
                self.try_body_with(body)
            }

            /// `body(forwarded..)` yields the next loop-carried values; it
            /// takes as many values, of the same types, as the loop carries.
            #[track_caller]
            pub fn body(self, body: impl $($f)*) -> R {
                self.try_body_with(body).or_panic()
            }
        }
    };
    (; $($a:ident)*) => {
        while_methods!([$($a: ?Sized),*] [$(Val<$a>: AsVal,)*] ($(Val<$a>,)*), [FnOnce($(Val<$a>),*) -> Tuple]);
    };
}

while_methods!([] [] Dyn, [FnOnce(&[Val]) -> Tuple]);
for_each_arity!(while_methods);

impl While {
    fn condition(&self) -> OpRef<Condition> {
        self.before.exprs[0].op_as().unwrap()
    }
}

impl HighMlir for While {
    fn name(&self) -> &str {
        "scf.while"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.inits.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        self.condition().forwarded_types()
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        p.write_str("scf.while ")?;
        let n = self.inits.len();
        if n > 0 {
            let args: Vec<String> = (0..n).map(|i| format!("{} = {}", p.region_arg(0, i), p.operand(i))).collect();
            write!(p, "({}) ", args.join(", "))?;
        }
        let (inits, results) = (p.operand_types(0..n), self.result_types());
        write!(p, ": ({}) -> ({}) ", TypeList(&inits), TypeList(&results))?;
        p.region(0, false)?;
        p.write_str(" do ")?;
        p.region(1, true)
    }
}

impl Closure for While {
    fn region(&self, r: usize) -> &BuiltRegion {
        if r == 0 { &self.before } else { &self.after }
    }

    fn num_regions(&self) -> usize {
        2
    }

    fn terminator(&self, region: usize) -> Option<&str> {
        (region == 1).then_some("scf.yield")
    }
}

/// `scf.condition`, the terminator of the `before` region of [`While`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Condition {
    pub condition: Val,
    pub forwarded: Vec<Val>,
}

impl Condition {
    /// `values` are the `i1` condition followed by the forwarded values.
    #[track_caller]
    pub(crate) fn try_new(mut values: Vec<Val>) -> Result<Val, TypeError> {
        let name = "scf.condition";
        expect(name, !values.is_empty(), || "needs a condition".into())?;
        let condition = values.remove(0);
        let c = value_ty(name, &condition)?;
        expect(name, c == I1, || format!("condition must be `i1`, not `{c}`"))?;
        types(name, &values)?;
        Val::single(Condition { condition, forwarded: values })
    }

    fn forwarded_types(&self) -> Vec<Ty> {
        self.forwarded.iter().map(|v| v.ty()).collect()
    }
}

impl HighMlir for Condition {
    fn name(&self) -> &str {
        "scf.condition"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.condition.clone()).chain(self.forwarded.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        write!(p, "scf.condition({})", p.operand(0))?;
        let n = self.forwarded.len();
        if n > 0 {
            write!(p, " {} : {}", p.operands(1..=n), TypeList(&p.operand_types(1..=n)))?;
        }
        Ok(())
    }
}

/// `scf.execute_region`: runs its region once and returns what it yields.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ExecuteRegion {
    /// Keep the region from being inlined.
    pub no_inline: bool,
    pub body: BuiltRegion,
}

/// `scf.execute_region`; `body()` returns the results.
#[track_caller]
pub fn try_execute_region(no_inline: bool, body: impl FnOnce() -> Tuple) -> Result<Tuple, TypeError> {
    let mut builder = ClosureBuilder::new(false);
    let region = builder.region(&[], |_| body());
    builder.try_finish()?;
    types("scf.execute_region", &region.exprs)?;
    Val::all_results(ExecuteRegion { no_inline, body: region })
}

/// `scf.execute_region`; `body()` returns the results.
#[track_caller]
pub fn execute_region(no_inline: bool, body: impl FnOnce() -> Tuple) -> Tuple {
    try_execute_region(no_inline, body).or_panic()
}

impl HighMlir for ExecuteRegion {
    fn name(&self) -> &str {
        "scf.execute_region"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        self.body.exprs.iter().map(|v| v.ty()).collect()
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        if self.no_inline { vec![("no_inline".into(), Attr::Unit)] } else { vec![] }
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        p.write_str("scf.execute_region ")?;
        let results = self.result_types();
        if !results.is_empty() {
            write!(p, "-> ({}) ", TypeList(&results))?;
        }
        if self.no_inline {
            p.write_str("no_inline ")?;
        }
        p.region(0, false)
    }
}

impl Closure for ExecuteRegion {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("scf.yield")
    }
}

/// `scf.index_switch`: runs the region of the case equal to `arg`, or the
/// default region; all of them yield values of the same types.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct IndexSwitch {
    pub arg: Val,
    pub cases: Vec<i64>,
    pub default: BuiltRegion,
    pub case_regions: Vec<BuiltRegion>,
}

/// `index_switch(arg).case(..)..`, waiting for its default region.
#[must_use = "a switch is built by `.default(..)`"]
pub struct IndexSwitchBuilder<'a> {
    arg: Val,
    cases: Vec<(i64, Box<dyn FnOnce() -> Tuple + 'a>)>,
}

/// `scf.index_switch` on `arg`, built by `.case(..)` and `.default(..)`;
/// a literal `arg` becomes an `index` constant.
#[track_caller]
pub fn index_switch<'a>(arg: impl Operand) -> IndexSwitchBuilder<'a> {
    let arg = arg.into_val(&INDEX.into()).expect("a switch argument");
    IndexSwitchBuilder { arg, cases: vec![] }
}

impl IndexSwitch {
    fn switch_region(&self, r: usize) -> &BuiltRegion {
        if r == 0 { &self.default } else { &self.case_regions[r - 1] }
    }
}

impl<'a> IndexSwitchBuilder<'a> {
    /// The region run when the argument is `value`.
    #[track_caller]
    pub fn case(mut self, value: i64, body: impl FnOnce() -> Tuple + 'a) -> IndexSwitchBuilder<'a> {
        self.cases.push((value, Box::new(body)));
        self
    }

    /// The default region, run when no case matches.
    #[track_caller]
    pub fn try_default(self, body: impl FnOnce() -> Tuple) -> Result<Tuple, TypeError> {
        let name = "scf.index_switch";
        let arg = indices(name, self.arg)?.remove(0);
        let cases: Vec<i64> = self.cases.iter().map(|(v, _)| *v).collect();
        let mut sorted = cases.clone();
        sorted.sort();
        sorted.dedup();
        expect(name, sorted.len() == cases.len(), || format!("duplicate cases in {cases:?}"))?;
        let mut builder = ClosureBuilder::new(false);
        let default = builder.region(&[], |_| body());
        let case_regions: Vec<BuiltRegion> = self.cases.into_iter().map(|(_, f)| builder.region(&[], |_| f())).collect();
        builder.try_finish()?;
        let expected = types(name, &default.exprs)?;
        for (c, r) in cases.iter().zip(&case_regions) {
            let yielded = types(name, &r.exprs)?;
            expect(name, yielded == expected, || {
                format!("case {c} yields ({}) but default yields ({})", TypeList(&yielded), TypeList(&expected))
            })?;
        }
        Val::all_results(IndexSwitch { arg, cases, default, case_regions })
    }

    /// The default region, run when no case matches.
    #[track_caller]
    pub fn default(self, body: impl FnOnce() -> Tuple) -> Tuple {
        self.try_default(body).or_panic()
    }
}

impl HighMlir for IndexSwitch {
    fn name(&self) -> &str {
        "scf.index_switch"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.arg.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        self.default.exprs.iter().map(|v| v.ty()).collect()
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("cases".into(), Attr::i64_array(self.cases.clone()))]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "scf.index_switch {}", p.operand(0))?;
        let results = self.result_types();
        if !results.is_empty() {
            write!(p, " -> {}", TypeList(&results))?;
        }
        for (i, c) in self.cases.iter().enumerate() {
            write!(p, "\ncase {c} ")?;
            p.region(i + 1, false)?;
        }
        p.write_str("\ndefault ")?;
        p.region(0, false)
    }
}

impl Closure for IndexSwitch {
    fn region(&self, r: usize) -> &BuiltRegion {
        self.switch_region(r)
    }

    fn num_regions(&self) -> usize {
        1 + self.cases.len()
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("scf.yield")
    }
}

/// `scf.forall` without shared outputs: a parallel loop nest run for its
/// side effects. Bounds and steps are static (literals) or `index` values.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Forall {
    pub lowers: MixedValues,
    pub uppers: MixedValues,
    pub steps: MixedValues,
    /// E.g. `[#gpu.thread<x>]`.
    pub mapping: Option<Attr>,
    /// Yields the [`InParallel`] terminator.
    pub body: BuiltRegion,
}

/// `scf!{forall (..)}`, waiting for its body.
///
/// `A` is the marker of the body, `(Val<IndexType>, ..)` for each induction
/// variable, or [`Dyn`]. The loop has no results (`R = ()`).
#[must_use = "a parallel loop is built by `.then(body)`"]
pub struct ForallBuilder<A = Dyn, R = ()> {
    bounds: Result<[MixedValues; 3], TypeError>,
    mapping: Option<Attr>,
    marker: PhantomData<fn() -> (A, R)>,
}

impl Forall {
    /// `scf.forall` from `lowers` to `uppers` by `steps`; see [`scf!`](crate::scf!).
    #[track_caller]
    pub fn build<A>(lowers: impl MixedList, uppers: impl MixedList, steps: impl MixedList) -> ForallBuilder<A> {
        let name = "scf.forall";
        let bounds = (|| {
            Ok([MixedValues::new(name, lowers)?, MixedValues::new(name, uppers)?, MixedValues::new(name, steps)?])
        })();
        ForallBuilder { bounds, mapping: None, marker: PhantomData }
    }

    /// From 0 to `uppers` with step 1.
    #[track_caller]
    pub fn build_upper<A>(uppers: impl MixedList) -> ForallBuilder<A> {
        let uppers = MixedValues::new("scf.forall", uppers);
        let bounds = uppers.map(|u| {
            let n = u.len();
            let fixed = |v| MixedValues { dynamic: vec![], statics: vec![v; n] };
            [fixed(0), u, fixed(1)]
        });
        ForallBuilder { bounds, mapping: None, marker: PhantomData }
    }

    fn is_normalized(&self) -> bool {
        self.lowers.dynamic.is_empty()
            && self.steps.dynamic.is_empty()
            && self.lowers.statics.iter().all(|&l| l == 0)
            && self.steps.statics.iter().all(|&s| s == 1)
    }
}

impl<A, R: FromTuple> ForallBuilder<A, R> {
    /// The mapping of the dimensions to processors, e.g. `[#gpu.thread<x>]`.
    #[track_caller]
    pub fn mapping(self, mapping: Attr) -> ForallBuilder<A, R> {
        ForallBuilder { mapping: Some(mapping), ..self }
    }

    /// `body(ivs..)` performs the effects of one iteration and yields nothing.
    #[track_caller]
    pub fn try_then_with<C: MlirClosure<A>>(self, body: C) -> Result<R, TypeError> {
        let caller = std::panic::Location::caller();
        let name = "scf.forall";
        let [lowers, uppers, steps] = self.bounds?;
        let n = lowers.len();
        expect(name, n > 0 && uppers.len() == n && steps.len() == n, || {
            "needs the same positive number of lower bounds, upper bounds and steps".into()
        })?;
        check_params::<A, C>(name, &vec![INDEX.into(); n])?;
        let mut builder = ClosureBuilder::new(false);
        let mut error = None;
        let region = builder.region(&vec![INDEX.into(); n], |ivs| {
            let values = body.call(ivs);
            let terminator = |vs: Vec<Val>| {
                expect(name, vs.is_empty(), || "the body must yield nothing".into())?;
                InParallel::build()
            };
            terminated(&mut error, caller, terminator, values)
        });
        builder.try_finish()?;
        if let Some(e) = error {
            return Err(e);
        }
        let mapping = self.mapping;
        typed_results(name, Val::all_results(Forall { lowers, uppers, steps, mapping, body: region })?)
    }
}

dyn_methods!(ForallBuilder, "Builds the loop; `body(ivs..)` performs one iteration and yields `()`.");
for_each_arity!(arity_methods ForallBuilder, "Builds the loop; `body(ivs..)` performs one iteration and yields `()`.");

/// `(a, 4, b)` from a mixed list printed as `[a, 4, b]`.
fn parenthesized(list: String) -> String {
    format!("({})", &list[1..list.len() - 1])
}

impl HighMlir for Forall {
    fn name(&self) -> &str {
        "scf.forall"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.lowers.dynamic.iter().chain(&self.uppers.dynamic).chain(&self.steps.dynamic).cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let (l, u, s) = (&self.lowers, &self.uppers, &self.steps);
        let mut attrs = vec![
            Attr::segments([l.dynamic.len(), u.dynamic.len(), s.dynamic.len(), 0]),
            ("staticLowerBound".into(), Attr::i64_array(l.statics.clone())),
            ("staticUpperBound".into(), Attr::i64_array(u.statics.clone())),
            ("staticStep".into(), Attr::i64_array(s.statics.clone())),
        ];
        if let Some(m) = &self.mapping {
            attrs.push(("mapping".into(), m.clone()));
        }
        attrs
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let n = self.lowers.len();
        let ivs: Vec<String> = (0..n).map(|i| p.region_arg(0, i)).collect();
        let (nl, nu) = (self.lowers.dynamic.len(), self.uppers.dynamic.len());
        let uppers = parenthesized(self.uppers.format(|k| p.operand(nl + k)));
        write!(p, "scf.forall ({}) ", ivs.join(", "))?;
        if self.is_normalized() {
            write!(p, "in {uppers} ")?;
        } else {
            let lowers = parenthesized(self.lowers.format(|k| p.operand(k)));
            let steps = parenthesized(self.steps.format(|k| p.operand(nl + nu + k)));
            write!(p, "= {lowers} to {uppers} step {steps} ")?;
        }
        p.region(0, false)?;
        if let Some(m) = &self.mapping {
            write!(p, " {{mapping = {m}}}")?;
        }
        Ok(())
    }
}

impl Closure for Forall {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        None
    }
}

/// `scf.forall.in_parallel`, the (empty) terminator of [`Forall`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct InParallel {
    pub body: BuiltRegion,
}

impl InParallel {
    #[track_caller]
    pub(crate) fn build() -> Result<Val, TypeError> {
        let mut builder = ClosureBuilder::new(false);
        let body = builder.region(&[], |_| Tuple::default());
        builder.finish();
        Val::single(InParallel { body })
    }
}

impl HighMlir for InParallel {
    fn name(&self) -> &str {
        "scf.forall.in_parallel"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.write_str("scf.forall.in_parallel ")?;
        p.region(0, false)
    }
}

impl Closure for InParallel {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        None
    }
}
