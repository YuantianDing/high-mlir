//! `math` dialect: elementwise functions on integers and floats (and on
//! tensors and vectors of them).
//!
//! Each operation is built by the function of its name (`math::sqrt`,
//! `math::powf`, ...), with a `try_` variant returning a [`TypeError`]. The
//! result has the static type of the first operand; a literal operand takes
//! the type of the others. Float operations take [`FastMath`] flags (`()` for
//! none). Scalars also have them as methods: `x.sqrt()`, `x.powf(2.0)`, ...

use std::fmt::{self, Write};

use crate::{
    Attr, Edges, FloatType, HighMlir, IntType, OpPrinter, Ty, TypeError, Val,
    arith::{FastMath, FloatFlags, OpFlags, Operand, operands},
    dialect::util::{expect, value_ty},
    types::I1,
    val::{AsVal, OrPanic, retype},
};

fn check_float(op: &str, ty: &Ty) -> Result<(), TypeError> {
    expect(op, ty.is_float_like(), || format!("expected floats, not `{ty}`"))
}

fn check_int(op: &str, ty: &Ty) -> Result<(), TypeError> {
    expect(op, ty.is_int_like(), || format!("expected integers or `index`, not `{ty}`"))
}

fn check_same(op: &str, vals: &[&Val]) -> Result<Ty, TypeError> {
    let ty = value_ty(op, vals[0])?;
    for v in &vals[1..] {
        let t = value_ty(op, v)?;
        expect(op, t == ty, || format!("operand types differ: `{ty}` and `{t}`"))?;
    }
    Ok(ty)
}

/// The type of a value among three operands, for literals.
#[track_caller]
fn three(op: &str, a: impl Operand, b: impl Operand, c: impl Operand) -> Result<[Val; 3], TypeError> {
    let like = a
        .ty_hint()
        .or_else(|| b.ty_hint())
        .or_else(|| c.ty_hint())
        .ok_or_else(|| TypeError::new(op, "cannot infer the type of three literals"))?;
    Ok([a.into_val(&like)?, b.into_val(&like)?, c.into_val(&like)?])
}

macro_rules! unary_ops {
    ($check:ident, $flags:ty; $($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident;)*) => {$(
        #[doc = concat!("`math.", stringify!($f), "`")]
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub operand: Val,
            pub flags: $flags,
        }

        #[doc = concat!("`math.", stringify!($f), "`")]
        $(#[$doc])*
        #[track_caller]
        pub fn $try_f<T: ?Sized>(operand: &Val<T>, flags: impl Into<$flags>) -> Result<Val<T>, TypeError>
        where
            Val<T>: AsVal,
        {
            let name = concat!("math.", stringify!($f));
            let operand = operand.as_val();
            $check(name, &value_ty(name, &operand)?)?;
            retype(name, Val::single($op { operand, flags: flags.into() })?)
        }

        #[doc = concat!("`math.", stringify!($f), "`")]
        $(#[$doc])*
        #[track_caller]
        pub fn $f<T: ?Sized>(operand: &Val<T>, flags: impl Into<$flags>) -> Val<T>
        where
            Val<T>: AsVal,
        {
            $try_f(operand, flags).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("math.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(std::iter::once(self.operand.clone()))
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![self.operand.ty()]
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                self.flags.attributes()
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let (x, ty) = (p.operand(0), p.operand_type(0));
                write!(p, "{} {x}{} : {ty}", self.name(), self.flags.suffix())
            }
        }
    )*};
}

unary_ops! { check_float, FastMath;
    AbsF = absf / try_absf;
    Acos = acos / try_acos;
    Acosh = acosh / try_acosh;
    Asin = asin / try_asin;
    Asinh = asinh / try_asinh;
    Atan = atan / try_atan;
    Atanh = atanh / try_atanh;
    Cbrt = cbrt / try_cbrt;
    Ceil = ceil / try_ceil;
    Cos = cos / try_cos;
    Cosh = cosh / try_cosh;
    Erf = erf / try_erf;
    Erfc = erfc / try_erfc;
    Exp = exp / try_exp;
    Exp2 = exp2 / try_exp2;
    ExpM1 = expm1 / try_expm1;
    Floor = floor / try_floor;
    Log = log / try_log;
    Log10 = log10 / try_log10;
    Log1p = log1p / try_log1p;
    Log2 = log2 / try_log2;
    Round = round / try_round;
    RoundEven = roundeven / try_roundeven;
    Rsqrt = rsqrt / try_rsqrt;
    Sin = sin / try_sin;
    Sinh = sinh / try_sinh;
    Sqrt = sqrt / try_sqrt;
    Tan = tan / try_tan;
    Tanh = tanh / try_tanh;
    Trunc = trunc / try_trunc;
}

unary_ops! { check_int, ();
    AbsI = absi / try_absi;
    /// Count of leading zero bits.
    Ctlz = ctlz / try_ctlz;
    /// Count of trailing zero bits.
    Cttz = cttz / try_cttz;
    /// Count of set bits.
    CtPop = ctpop / try_ctpop;
}

macro_rules! classification_ops {
    ($($op:ident = $f:ident / $try_f:ident;)*) => {$(
        #[doc = concat!("`math.", stringify!($f), "`: an `i1` per element.")]
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub operand: Val,
            pub flags: FastMath,
        }

        #[doc = concat!("`math.", stringify!($f), "`: an `i1` (or a tensor or vector of `i1`) per element.")]
        #[track_caller]
        pub fn $try_f(operand: impl Into<Val>, flags: impl Into<FastMath>) -> Result<Val, TypeError> {
            let name = concat!("math.", stringify!($f));
            let operand = operand.into();
            check_float(name, &value_ty(name, &operand)?)?;
            Val::single($op { operand, flags: flags.into() })
        }

        #[doc = concat!("`math.", stringify!($f), "`: an `i1` (or a tensor or vector of `i1`) per element.")]
        #[track_caller]
        pub fn $f(operand: impl Into<Val>, flags: impl Into<FastMath>) -> Val {
            $try_f(operand, flags).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("math.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(std::iter::once(self.operand.clone()))
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![self.operand.ty().with_element_ty(I1.into())]
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                self.flags.attributes()
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let (x, ty) = (p.operand(0), p.operand_type(0));
                write!(p, "{} {x}", self.name())?;
                p.attr_dict(&self.flags.attributes())?;
                write!(p, " : {ty}")
            }
        }
    )*};
}

classification_ops! {
    IsFinite = isfinite / try_isfinite;
    IsInf = isinf / try_isinf;
    IsNaN = isnan / try_isnan;
    IsNormal = isnormal / try_isnormal;
}

macro_rules! binary_ops {
    ($check:ident, $flags:ty; $($op:ident = $f:ident / $try_f:ident;)*) => {$(
        #[doc = concat!("`math.", stringify!($f), "`")]
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub lhs: Val,
            pub rhs: Val,
            pub flags: $flags,
        }

        #[doc = concat!("`math.", stringify!($f), "`; a literal operand takes the type of the other.")]
        #[track_caller]
        pub fn $try_f<L: Operand>(lhs: L, rhs: impl Operand, flags: impl Into<$flags>) -> Result<Val<L::Static>, TypeError>
        where
            Val<L::Static>: AsVal,
        {
            let name = concat!("math.", stringify!($f));
            let (lhs, rhs) = operands(name, lhs, rhs)?;
            $check(name, &check_same(name, &[&lhs, &rhs])?)?;
            retype(name, Val::single($op { lhs, rhs, flags: flags.into() })?)
        }

        #[doc = concat!("`math.", stringify!($f), "`; a literal operand takes the type of the other.")]
        #[track_caller]
        pub fn $f<L: Operand>(lhs: L, rhs: impl Operand, flags: impl Into<$flags>) -> Val<L::Static>
        where
            Val<L::Static>: AsVal,
        {
            $try_f(lhs, rhs, flags).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("math.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new([self.lhs.clone(), self.rhs.clone()].into_iter())
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![self.lhs.ty()]
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                self.flags.attributes()
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let (lhs, rhs, ty) = (p.operand(0), p.operand(1), p.operand_type(0));
                write!(p, "{} {lhs}, {rhs}{} : {ty}", self.name(), self.flags.suffix())
            }
        }
    )*};
}

binary_ops! { check_float, FastMath;
    Atan2 = atan2 / try_atan2;
    CopySign = copysign / try_copysign;
    PowF = powf / try_powf;
}

binary_ops! { check_int, ();
    IPowI = ipowi / try_ipowi;
}

/// `math.fpowi`: a float raised to an integer power.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FPowI {
    pub lhs: Val,
    pub rhs: Val,
    pub flags: FastMath,
}

/// `math.fpowi`: `lhs` (floats) raised to `rhs` (integers of the same shape).
#[track_caller]
pub fn try_fpowi<T: ?Sized>(lhs: &Val<T>, rhs: impl Into<Val>, flags: impl Into<FastMath>) -> Result<Val<T>, TypeError>
where
    Val<T>: AsVal,
{
    let name = "math.fpowi";
    let (lhs, rhs) = (lhs.as_val(), rhs.into());
    let (l, r) = (value_ty(name, &lhs)?, value_ty(name, &rhs)?);
    check_float(name, &l)?;
    expect(name, r.is_int_like() && l.with_element_ty(r.element().clone()) == r, || {
        format!("the exponent `{r}` is not integers of the shape of `{l}`")
    })?;
    retype(name, Val::single(FPowI { lhs, rhs, flags: flags.into() })?)
}

/// `math.fpowi`
#[track_caller]
pub fn fpowi<T: ?Sized>(lhs: &Val<T>, rhs: impl Into<Val>, flags: impl Into<FastMath>) -> Val<T>
where
    Val<T>: AsVal,
{
    try_fpowi(lhs, rhs, flags).or_panic()
}

impl HighMlir for FPowI {
    fn name(&self) -> &str {
        "math.fpowi"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.lhs.clone(), self.rhs.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.lhs.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.flags.attributes()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (lhs, rhs, lt, rt) = (p.operand(0), p.operand(1), p.operand_type(0), p.operand_type(1));
        write!(p, "math.fpowi {lhs}, {rhs}{} : {lt}, {rt}", self.flags.suffix())
    }
}

/// `math.fma`: `a * b + c`, rounded once.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Fma {
    pub a: Val,
    pub b: Val,
    pub c: Val,
    pub flags: FloatFlags,
}

/// `math.fma`: `a * b + c`; literals take the type of the other operands.
#[track_caller]
pub fn try_fma<A: Operand>(a: A, b: impl Operand, c: impl Operand, flags: impl Into<FloatFlags>) -> Result<Val<A::Static>, TypeError>
where
    Val<A::Static>: AsVal,
{
    let name = "math.fma";
    let [a, b, c] = three(name, a, b, c)?;
    check_float(name, &check_same(name, &[&a, &b, &c])?)?;
    retype(name, Val::single(Fma { a, b, c, flags: flags.into() })?)
}

/// `math.fma`: `a * b + c`; literals take the type of the other operands.
#[track_caller]
pub fn fma<A: Operand>(a: A, b: impl Operand, c: impl Operand, flags: impl Into<FloatFlags>) -> Val<A::Static>
where
    Val<A::Static>: AsVal,
{
    try_fma(a, b, c, flags).or_panic()
}

impl HighMlir for Fma {
    fn name(&self) -> &str {
        "math.fma"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.a.clone(), self.b.clone(), self.c.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.a.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.flags.attributes()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let ty = p.operand_type(0);
        write!(p, "math.fma {}{} : {ty}", p.operands(0..3), self.flags.suffix())
    }
}

/// `math.clampf`: `value` clamped to `[min, max]`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ClampF {
    pub value: Val,
    pub min: Val,
    pub max: Val,
    pub flags: FastMath,
}

/// `math.clampf`: `value` clamped to `[min, max]`; literals take the type of the other operands.
#[track_caller]
pub fn try_clampf<V: Operand>(value: V, min: impl Operand, max: impl Operand, flags: impl Into<FastMath>) -> Result<Val<V::Static>, TypeError>
where
    Val<V::Static>: AsVal,
{
    let name = "math.clampf";
    let [value, min, max] = three(name, value, min, max)?;
    check_float(name, &check_same(name, &[&value, &min, &max])?)?;
    retype(name, Val::single(ClampF { value, min, max, flags: flags.into() })?)
}

/// `math.clampf`: `value` clamped to `[min, max]`; literals take the type of the other operands.
#[track_caller]
pub fn clampf<V: Operand>(value: V, min: impl Operand, max: impl Operand, flags: impl Into<FastMath>) -> Val<V::Static>
where
    Val<V::Static>: AsVal,
{
    try_clampf(value, min, max, flags).or_panic()
}

impl HighMlir for ClampF {
    fn name(&self) -> &str {
        "math.clampf"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.value.clone(), self.min.clone(), self.max.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.value.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.flags.attributes()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (v, lo, hi, ty) = (p.operand(0), p.operand(1), p.operand(2), p.operand_type(0));
        write!(p, "math.clampf {v} to [{lo}, {hi}]{} : {ty}", self.flags.suffix())
    }
}

/// `math.sincos`: the sine and the cosine.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SinCos {
    pub operand: Val,
    pub flags: FastMath,
}

/// `math.sincos`: the sine and the cosine.
#[track_caller]
pub fn try_sincos<T: ?Sized>(operand: &Val<T>, flags: impl Into<FastMath>) -> Result<(Val<T>, Val<T>), TypeError>
where
    Val<T>: AsVal,
{
    let name = "math.sincos";
    let operand = operand.as_val();
    check_float(name, &value_ty(name, &operand)?)?;
    let [sin, cos] = Val::all_results(SinCos { operand, flags: flags.into() })?.try_into().unwrap();
    Ok((retype(name, sin)?, retype(name, cos)?))
}

/// `math.sincos`: the sine and the cosine.
#[track_caller]
pub fn sincos<T: ?Sized>(operand: &Val<T>, flags: impl Into<FastMath>) -> (Val<T>, Val<T>)
where
    Val<T>: AsVal,
{
    try_sincos(operand, flags).or_panic()
}

impl HighMlir for SinCos {
    fn name(&self) -> &str {
        "math.sincos"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.operand.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.operand.ty(), self.operand.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        self.flags.attributes()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (x, ty) = (p.operand(0), p.operand_type(0));
        write!(p, "math.sincos {x}{} : {ty}", self.flags.suffix())
    }
}

/// Methods for unary float functions, without fast-math flags.
macro_rules! float_methods {
    ($($f:ident)*) => {$(
        #[doc = concat!("`math.", stringify!($f), "`")]
        #[track_caller]
        pub fn $f(&self) -> Val<FloatType> {
            $f(self, ())
        }
    )*};
}

/// `math` functions as methods of float scalars, without fast-math flags.
impl Val<FloatType> {
    float_methods! {
        absf acos acosh asin asinh atan atanh cbrt ceil cos cosh erf erfc exp exp2 expm1 floor
        log log10 log1p log2 round roundeven rsqrt sin sinh sqrt tan tanh trunc
    }

    /// `math.powf`
    #[track_caller]
    pub fn powf(&self, exponent: impl Operand) -> Val<FloatType> {
        powf(self, exponent, ())
    }

    /// `math.atan2` of `self / other`.
    #[track_caller]
    pub fn atan2(&self, other: impl Operand) -> Val<FloatType> {
        atan2(self, other, ())
    }

    /// `math.copysign`
    #[track_caller]
    pub fn copysign(&self, sign: impl Operand) -> Val<FloatType> {
        copysign(self, sign, ())
    }

    /// `math.fpowi`
    #[track_caller]
    pub fn fpowi(&self, exponent: impl Into<Val>) -> Val<FloatType> {
        fpowi(self, exponent, ())
    }

    /// `math.fma`: `self * b + c`.
    #[track_caller]
    pub fn fma(&self, b: impl Operand, c: impl Operand) -> Val<FloatType> {
        fma(self, b, c, ())
    }

    /// `math.clampf` to `[min, max]`.
    #[track_caller]
    pub fn clampf(&self, min: impl Operand, max: impl Operand) -> Val<FloatType> {
        clampf(self, min, max, ())
    }

    /// `math.sincos`
    #[track_caller]
    pub fn sincos(&self) -> (Val<FloatType>, Val<FloatType>) {
        sincos(self, ())
    }

    /// `math.isnan`
    #[track_caller]
    pub fn isnan(&self) -> Val<IntType> {
        isnan(self, ()).downcast().unwrap()
    }

    /// `math.isinf`
    #[track_caller]
    pub fn isinf(&self) -> Val<IntType> {
        isinf(self, ()).downcast().unwrap()
    }

    /// `math.isfinite`
    #[track_caller]
    pub fn isfinite(&self) -> Val<IntType> {
        isfinite(self, ()).downcast().unwrap()
    }
}

/// `math` functions as methods of integer scalars.
impl Val<IntType> {
    /// `math.absi`
    #[track_caller]
    pub fn absi(&self) -> Val<IntType> {
        absi(self, ())
    }

    /// `math.ctlz`
    #[track_caller]
    pub fn ctlz(&self) -> Val<IntType> {
        ctlz(self, ())
    }

    /// `math.cttz`
    #[track_caller]
    pub fn cttz(&self) -> Val<IntType> {
        cttz(self, ())
    }

    /// `math.ctpop`
    #[track_caller]
    pub fn ctpop(&self) -> Val<IntType> {
        ctpop(self, ())
    }

    /// `math.ipowi`
    #[track_caller]
    pub fn ipowi(&self, exponent: impl Operand) -> Val<IntType> {
        ipowi(self, exponent, ())
    }
}
