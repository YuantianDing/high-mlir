//! `index` dialect: integer arithmetic on `index` values, of the target's
//! pointer width.
//!
//! Operations take [`IndexOperand`]s: `index` values, or integer literals,
//! which become `index.constant`s.

use std::fmt::{self, Write};

use crate::{
    Attr, Edges, HighMlir, IndexType, IntType, OpPrinter, Ty, TypeError, Val,
    arith::predicates,
    dialect::util::{expect, value_ty},
    types::{I1, INDEX},
    val::{AsVal, OrPanic, retype},
};

/// An `index` operand: a value, or an integer literal (an `index.constant`).
pub trait IndexOperand {
    /// This operand as a value, checked to be an `index`.
    #[track_caller]
    fn into_index(self, op: &str) -> Result<Val, TypeError>;
}

impl<T: ?Sized> IndexOperand for Val<T>
where
    Val<T>: AsVal,
{
    fn into_index(self, op: &str) -> Result<Val, TypeError> {
        let v = self.as_val();
        let ty = value_ty(op, &v)?;
        expect(op, ty.is_index(), || format!("expected an `index`, not `{ty}`"))?;
        Ok(v)
    }
}

impl<T: ?Sized> IndexOperand for &Val<T>
where
    Val<T>: AsVal,
{
    fn into_index(self, op: &str) -> Result<Val, TypeError> {
        self.as_val().into_index(op)
    }
}

macro_rules! literal_operands {
    ($($t:ty),*) => {$(
        impl IndexOperand for $t {
            #[track_caller]
            fn into_index(self, _: &str) -> Result<Val, TypeError> {
                Ok(try_constant(self as i64)?.erase())
            }
        }
    )*};
}

literal_operands!(i32, i64, isize, u32, u64, usize);

/// `index.constant`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Constant {
    pub value: i64,
}

/// `index.constant`
#[track_caller]
pub fn try_constant(value: i64) -> Result<Val<IndexType>, TypeError> {
    Val::new_typed(Constant { value })
}

/// `index.constant`
#[track_caller]
pub fn constant(value: i64) -> Val<IndexType> {
    try_constant(value).or_panic()
}

impl HighMlir for Constant {
    fn name(&self) -> &str {
        "index.constant"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![INDEX.into()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("value".into(), Attr::Int(self.value, INDEX.into()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "index.constant {}", self.value)
    }
}

/// `index.bool.constant`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct BoolConstant {
    pub value: bool,
}

/// `index.bool.constant`
#[track_caller]
pub fn try_bool_constant(value: bool) -> Result<Val<IntType>, TypeError> {
    Val::new_typed(BoolConstant { value })
}

/// `index.bool.constant`
#[track_caller]
pub fn bool_constant(value: bool) -> Val<IntType> {
    try_bool_constant(value).or_panic()
}

impl HighMlir for BoolConstant {
    fn name(&self) -> &str {
        "index.bool.constant"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![I1.into()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("value".into(), Attr::Bool(self.value))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "index.bool.constant {}", self.value)
    }
}

macro_rules! binary_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident;)*) => {$(
        #[doc = concat!("`index.", stringify!($f), "`")]
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub lhs: Val,
            pub rhs: Val,
        }

        #[doc = concat!("`index.", stringify!($f), "`")]
        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(lhs: impl IndexOperand, rhs: impl IndexOperand) -> Result<Val<IndexType>, TypeError> {
            let name = concat!("index.", stringify!($f));
            let (lhs, rhs) = (lhs.into_index(name)?, rhs.into_index(name)?);
            Val::new_typed($op { lhs, rhs })
        }

        #[doc = concat!("`index.", stringify!($f), "`")]
        $(#[$doc])*
        #[track_caller]
        pub fn $f(lhs: impl IndexOperand, rhs: impl IndexOperand) -> Val<IndexType> {
            $try_f(lhs, rhs).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("index.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new([self.lhs.clone(), self.rhs.clone()].into_iter())
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![INDEX.into()]
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                write!(p, "{} {}", self.name(), p.operands(0..2))
            }
        }
    )*};
}

binary_ops! {
    Add = add / try_add;
    Sub = sub / try_sub;
    Mul = mul / try_mul;
    /// Signed division, rounding towards zero.
    DivS = divs / try_divs;
    /// Unsigned division.
    DivU = divu / try_divu;
    /// Signed division, rounding towards positive infinity.
    CeilDivS = ceildivs / try_ceildivs;
    /// Unsigned division, rounding up.
    CeilDivU = ceildivu / try_ceildivu;
    /// Signed division, rounding towards negative infinity.
    FloorDivS = floordivs / try_floordivs;
    /// Signed remainder.
    RemS = rems / try_rems;
    /// Unsigned remainder.
    RemU = remu / try_remu;
    MaxS = maxs / try_maxs;
    MaxU = maxu / try_maxu;
    MinS = mins / try_mins;
    MinU = minu / try_minu;
    Shl = shl / try_shl;
    /// Arithmetic right shift.
    ShrS = shrs / try_shrs;
    /// Logical right shift.
    ShrU = shru / try_shru;
    And = and / try_and;
    Or = or / try_or;
    XOr = xor / try_xor;
}

predicates! {
    /// `index.cmp` predicate, in MLIR's numbering.
    IndexCmpPredicate {
        Eq = "eq", Ne = "ne", Slt = "slt", Sle = "sle", Sgt = "sgt", Sge = "sge",
        Ult = "ult", Ule = "ule", Ugt = "ugt", Uge = "uge",
    }
}

/// `index.cmp`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Cmp {
    pub predicate: IndexCmpPredicate,
    pub lhs: Val,
    pub rhs: Val,
}

/// `index.cmp`: an `i1`.
#[track_caller]
pub fn try_cmp(predicate: IndexCmpPredicate, lhs: impl IndexOperand, rhs: impl IndexOperand) -> Result<Val<IntType>, TypeError> {
    let (lhs, rhs) = (lhs.into_index("index.cmp")?, rhs.into_index("index.cmp")?);
    Val::new_typed(Cmp { predicate, lhs, rhs })
}

/// `index.cmp`: an `i1`.
#[track_caller]
pub fn cmp(predicate: IndexCmpPredicate, lhs: impl IndexOperand, rhs: impl IndexOperand) -> Val<IntType> {
    try_cmp(predicate, lhs, rhs).or_panic()
}

impl HighMlir for Cmp {
    fn name(&self) -> &str {
        "index.cmp"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.lhs.clone(), self.rhs.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![I1.into()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("pred".into(), Attr::Raw(format!("#index<cmp_predicate {}>", self.predicate.keyword())))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "index.cmp {}({})", self.predicate.keyword(), p.operands(0..2))
    }
}

macro_rules! cast_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub input: Val,
            pub ty: Ty,
        }

        $(#[$doc])*
        /// One of the input and `ty` is `index`, the other an integer.
        #[track_caller]
        pub fn $try_f<U: crate::IntoTy>(input: impl Into<Val>, ty: U) -> Result<Val<U::Mlir>, TypeError>
        where
            Val<U::Mlir>: AsVal,
        {
            let name = concat!("index.", stringify!($f));
            let (input, ty) = (input.into(), ty.into_ty());
            let src = value_ty(name, &input)?;
            let ok = (src.is_index() && ty.is::<IntType>()) || (src.is::<IntType>() && ty.is_index());
            expect(name, ok, || format!("cannot cast `{src}` to `{ty}`"))?;
            retype(name, Val::single($op { input, ty })?)
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f<U: crate::IntoTy>(input: impl Into<Val>, ty: U) -> Val<U::Mlir>
        where
            Val<U::Mlir>: AsVal,
        {
            $try_f(input, ty).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("index.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(std::iter::once(self.input.clone()))
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
    /// `index.casts`: signed cast between `index` and an integer type.
    CastS = casts / try_casts;
    /// `index.castu`: unsigned cast between `index` and an integer type.
    CastU = castu / try_castu;
}

/// `index.sizeof`: the bit width of `index`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SizeOf;

/// `index.sizeof`: the bit width of `index`.
#[track_caller]
pub fn try_sizeof() -> Result<Val<IndexType>, TypeError> {
    Val::new_typed(SizeOf)
}

/// `index.sizeof`: the bit width of `index`.
#[track_caller]
pub fn sizeof() -> Val<IndexType> {
    try_sizeof().or_panic()
}

impl HighMlir for SizeOf {
    fn name(&self) -> &str {
        "index.sizeof"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![INDEX.into()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        p.write_str("index.sizeof")
    }
}
