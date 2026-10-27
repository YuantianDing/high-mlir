//! `ub` dialect: undefined behavior.

use std::fmt::{self, Write};

use crate::{
    Edges, HighMlir, IntoTy, OpPrinter, Ty, TypeError, Val,
    val::{AsVal, OrPanic, retype},
};

/// `ub.poison`: a poison value of any type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Poison {
    pub ty: Ty,
}

/// `ub.poison` of type `ty`.
#[track_caller]
pub fn try_poison<U: IntoTy>(ty: U) -> Result<Val<U::Mlir>, TypeError>
where
    Val<U::Mlir>: AsVal,
{
    retype("ub.poison", Val::single(Poison { ty: ty.into_ty() })?)
}

/// `ub.poison` of type `ty`.
#[track_caller]
pub fn poison<U: IntoTy>(ty: U) -> Val<U::Mlir>
where
    Val<U::Mlir>: AsVal,
{
    try_poison(ty).or_panic()
}

impl HighMlir for Poison {
    fn name(&self) -> &str {
        "ub.poison"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "ub.poison : {}", self.ty)
    }
}

/// `ub.unreachable`: control never reaches this point. It is a terminator,
/// so it can end a region that has no other terminator to build.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Unreachable;

/// `ub.unreachable`
#[track_caller]
pub fn try_unreachable() -> Result<(), TypeError> {
    Val::unit(Unreachable)
}

/// `ub.unreachable`
#[track_caller]
pub fn unreachable() {
    try_unreachable().or_panic()
}

impl HighMlir for Unreachable {
    fn name(&self) -> &str {
        "ub.unreachable"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.write_str("ub.unreachable")
    }
}
