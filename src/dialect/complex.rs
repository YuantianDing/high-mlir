//! `complex` dialect: arithmetic on [`ComplexType`] values.
//!
//! Each operation is built by the function of its name (`complex::add`,
//! `complex::abs`, ...), with a `try_` variant returning a [`TypeError`].
//! Operations of floats take [`FastMath`] flags (`()` for none).
//! `Val<ComplexType>` has the operators `+`, `-`, `*`, `/` and unary `-`, and
//! methods such as `re`, `im`, `abs` and `conj`.

use std::fmt::{self, Write};

use crate::{
    Attr, ComplexType, Edges, FloatType, HighMlir, IntType, OpPrinter, Ty, TypeError, Val,
    attr::Float,
    arith::{FastMath, OpFlags, Operand, operands},
    dialect::util::{expect, value_ty},
    types::I1,
    val::{OrPanic, retype},
};

/// The complex type of a value.
fn complex_ty(op: &str, v: &Val) -> Result<ComplexType, TypeError> {
    let ty = value_ty(op, v)?;
    ty.downcast_ref::<ComplexType>().cloned().ok_or_else(|| TypeError::new(op, format!("expected a complex, not `{ty}`")))
}

fn check_same(op: &str, lhs: &Val, rhs: &Val) -> Result<(), TypeError> {
    let (l, r) = (complex_ty(op, lhs)?, complex_ty(op, rhs)?);
    expect(op, l == r, || format!("operand types differ: `{l}` and `{r}`"))
}

macro_rules! binary_ops {
    ($($op:ident = $f:ident / $try_f:ident;)*) => {$(
        #[doc = concat!("`complex.", stringify!($f), "`")]
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub lhs: Val,
            pub rhs: Val,
            pub flags: FastMath,
        }

        #[doc = concat!("`complex.", stringify!($f), "`")]
        #[track_caller]
        pub fn $try_f(lhs: impl Into<Val>, rhs: impl Into<Val>, flags: impl Into<FastMath>) -> Result<Val<ComplexType>, TypeError> {
            let name = concat!("complex.", stringify!($f));
            let (lhs, rhs) = (lhs.into(), rhs.into());
            check_same(name, &lhs, &rhs)?;
            Val::new_typed($op { lhs, rhs, flags: flags.into() })
        }

        #[doc = concat!("`complex.", stringify!($f), "`")]
        #[track_caller]
        pub fn $f(lhs: impl Into<Val>, rhs: impl Into<Val>, flags: impl Into<FastMath>) -> Val<ComplexType> {
            $try_f(lhs, rhs, flags).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("complex.", stringify!($f))
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
                let ty = p.operand_type(0);
                write!(p, "{} {}{} : {ty}", self.name(), p.operands(0..2), self.flags.suffix())
            }
        }
    )*};
}

binary_ops! {
    Add = add / try_add;
    Sub = sub / try_sub;
    Mul = mul / try_mul;
    Div = div / try_div;
    Pow = pow / try_pow;
    Atan2 = atan2 / try_atan2;
}

/// Unary operations; `$part` ones return the float element type.
macro_rules! unary_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident, $res:ident;)*) => {$(
        #[doc = concat!("`complex.", stringify!($f), "`")]
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub operand: Val,
            pub flags: FastMath,
        }

        #[doc = concat!("`complex.", stringify!($f), "`")]
        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(operand: impl Into<Val>, flags: impl Into<FastMath>) -> Result<Val<$res>, TypeError> {
            let name = concat!("complex.", stringify!($f));
            let operand = operand.into();
            complex_ty(name, &operand)?;
            Val::new_typed($op { operand, flags: flags.into() })
        }

        #[doc = concat!("`complex.", stringify!($f), "`")]
        $(#[$doc])*
        #[track_caller]
        pub fn $f(operand: impl Into<Val>, flags: impl Into<FastMath>) -> Val<$res> {
            $try_f(operand, flags).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("complex.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(std::iter::once(self.operand.clone()))
            }

            fn result_types(&self) -> Vec<Ty> {
                unary_ops!(@result $res, self.operand.ty())
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
    (@result ComplexType, $ty:expr) => {
        vec![$ty]
    };
    (@result FloatType, $ty:expr) => {
        vec![$ty.downcast_ref::<ComplexType>().expect("checked when built").element.clone()]
    };
}

unary_ops! {
    /// The modulus.
    Abs = abs / try_abs, FloatType;
    /// The argument.
    Angle = angle / try_angle, FloatType;
    /// The real part.
    Re = re / try_re, FloatType;
    /// The imaginary part.
    Im = im / try_im, FloatType;
    /// The conjugate.
    Conj = conj / try_conj, ComplexType;
    Cos = cos / try_cos, ComplexType;
    Exp = exp / try_exp, ComplexType;
    Expm1 = expm1 / try_expm1, ComplexType;
    Log = log / try_log, ComplexType;
    Log1p = log1p / try_log1p, ComplexType;
    Neg = neg / try_neg, ComplexType;
    Rsqrt = rsqrt / try_rsqrt, ComplexType;
    Sign = sign / try_sign, ComplexType;
    Sin = sin / try_sin, ComplexType;
    Sqrt = sqrt / try_sqrt, ComplexType;
    Tan = tan / try_tan, ComplexType;
    Tanh = tanh / try_tanh, ComplexType;
}

/// `complex.create` from the real and imaginary parts.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Create {
    pub real: Val,
    pub imaginary: Val,
}

/// `complex.create` from the real and imaginary parts (floats of one type;
/// a literal takes the type of the other).
#[track_caller]
pub fn try_create(real: impl Operand, imaginary: impl Operand) -> Result<Val<ComplexType>, TypeError> {
    let name = "complex.create";
    let (real, imaginary) = operands(name, real, imaginary)?;
    let (r, i) = (value_ty(name, &real)?, value_ty(name, &imaginary)?);
    expect(name, r.is::<FloatType>() && r == i, || format!("parts of types `{r}` and `{i}`"))?;
    Val::new_typed(Create { real, imaginary })
}

/// `complex.create` from the real and imaginary parts.
#[track_caller]
pub fn create(real: impl Operand, imaginary: impl Operand) -> Val<ComplexType> {
    try_create(real, imaginary).or_panic()
}

impl HighMlir for Create {
    fn name(&self) -> &str {
        "complex.create"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.real.clone(), self.imaginary.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![ComplexType::new(self.real.ty()).into()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "complex.create {} : {}", p.operands(0..2), self.result_types()[0])
    }
}

/// `complex.constant`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Constant {
    pub real: Float,
    pub imaginary: Float,
    pub ty: Ty,
}

impl Constant {
    fn value(&self) -> Attr {
        let e = self.ty.downcast_ref::<ComplexType>().expect("checked when built").element.clone();
        Attr::List(vec![Attr::float(self.real.0, e.clone()), Attr::float(self.imaginary.0, e)])
    }
}

/// `complex.constant` of type `complex<element>`.
#[track_caller]
pub fn try_constant(real: f64, imaginary: f64, element: FloatType) -> Result<Val<ComplexType>, TypeError> {
    let ty = ComplexType::new(element).into();
    Val::new_typed(Constant { real: Float(real), imaginary: Float(imaginary), ty })
}

/// `complex.constant` of type `complex<element>`.
#[track_caller]
pub fn constant(real: f64, imaginary: f64, element: FloatType) -> Val<ComplexType> {
    try_constant(real, imaginary, element).or_panic()
}

impl HighMlir for Constant {
    fn name(&self) -> &str {
        "complex.constant"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("value".into(), self.value())]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "complex.constant {} : {}", self.value(), self.ty)
    }
}

macro_rules! compare_ops {
    ($($op:ident = $f:ident / $try_f:ident;)*) => {$(
        #[doc = concat!("`complex.", stringify!($f), "`: an `i1`.")]
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub lhs: Val,
            pub rhs: Val,
        }

        #[doc = concat!("`complex.", stringify!($f), "`: an `i1`.")]
        #[track_caller]
        pub fn $try_f(lhs: impl Into<Val>, rhs: impl Into<Val>) -> Result<Val<IntType>, TypeError> {
            let name = concat!("complex.", stringify!($f));
            let (lhs, rhs) = (lhs.into(), rhs.into());
            check_same(name, &lhs, &rhs)?;
            Val::new_typed($op { lhs, rhs })
        }

        #[doc = concat!("`complex.", stringify!($f), "`: an `i1`.")]
        #[track_caller]
        pub fn $f(lhs: impl Into<Val>, rhs: impl Into<Val>) -> Val<IntType> {
            $try_f(lhs, rhs).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("complex.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new([self.lhs.clone(), self.rhs.clone()].into_iter())
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![I1.into()]
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                write!(p, "{} {} : {}", self.name(), p.operands(0..2), p.operand_type(0))
            }
        }
    )*};
}

compare_ops! {
    Eq = eq / try_eq;
    Neq = neq / try_neq;
}

/// `complex.powi`: a complex raised to an integer power.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PowI {
    pub lhs: Val,
    pub rhs: Val,
    pub flags: FastMath,
}

/// `complex.powi`: `lhs` raised to the integer `rhs`.
#[track_caller]
pub fn try_powi(lhs: impl Into<Val>, rhs: impl Into<Val>, flags: impl Into<FastMath>) -> Result<Val<ComplexType>, TypeError> {
    let name = "complex.powi";
    let (lhs, rhs) = (lhs.into(), rhs.into());
    complex_ty(name, &lhs)?;
    let r = value_ty(name, &rhs)?;
    expect(name, r.is::<IntType>(), || format!("the exponent must be an integer, not `{r}`"))?;
    Val::new_typed(PowI { lhs, rhs, flags: flags.into() })
}

/// `complex.powi`
#[track_caller]
pub fn powi(lhs: impl Into<Val>, rhs: impl Into<Val>, flags: impl Into<FastMath>) -> Val<ComplexType> {
    try_powi(lhs, rhs, flags).or_panic()
}

impl HighMlir for PowI {
    fn name(&self) -> &str {
        "complex.powi"
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
        let (lt, rt) = (p.operand_type(0), p.operand_type(1));
        write!(p, "complex.powi {}{} : {lt}, {rt}", p.operands(0..2), self.flags.suffix())
    }
}

/// `complex.bitcast` between a complex and a scalar of the same width.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Bitcast {
    pub operand: Val,
    pub ty: Ty,
}

/// `complex.bitcast` between a complex and an integer or float of the same width.
#[track_caller]
pub fn try_bitcast<U: crate::IntoTy>(operand: impl Into<Val>, ty: U) -> Result<Val<U::Mlir>, TypeError>
where
    Val<U::Mlir>: crate::AsVal,
{
    let name = "complex.bitcast";
    let (operand, ty) = (operand.into(), ty.into_ty());
    let src = value_ty(name, &operand)?;
    let complex = src.is::<ComplexType>() || ty.is::<ComplexType>();
    expect(name, complex && src.bit_width().is_some() && src.bit_width() == ty.bit_width(), || {
        format!("cannot bitcast `{src}` to `{ty}`")
    })?;
    retype(name, Val::single(Bitcast { operand, ty })?)
}

/// `complex.bitcast`
#[track_caller]
pub fn bitcast<U: crate::IntoTy>(operand: impl Into<Val>, ty: U) -> Val<U::Mlir>
where
    Val<U::Mlir>: crate::AsVal,
{
    try_bitcast(operand, ty).or_panic()
}

impl HighMlir for Bitcast {
    fn name(&self) -> &str {
        "complex.bitcast"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.operand.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "complex.bitcast {} : {} to {}", p.operand(0), p.operand_type(0), self.ty)
    }
}

macro_rules! operators {
    ($($trait:ident :: $method:ident => $f:ident;)*) => {$(
        impl<R: Into<Val>> std::ops::$trait<R> for Val<ComplexType> {
            type Output = Val<ComplexType>;

            #[track_caller]
            fn $method(self, rhs: R) -> Val<ComplexType> {
                $f(self, rhs, ())
            }
        }

        impl<R: Into<Val>> std::ops::$trait<R> for &Val<ComplexType> {
            type Output = Val<ComplexType>;

            #[track_caller]
            fn $method(self, rhs: R) -> Val<ComplexType> {
                $f(self, rhs, ())
            }
        }
    )*};
}

operators! {
    Add::add => add;
    Sub::sub => sub;
    Mul::mul => mul;
    Div::div => div;
}

impl std::ops::Neg for Val<ComplexType> {
    type Output = Val<ComplexType>;

    #[track_caller]
    fn neg(self) -> Val<ComplexType> {
        neg(self, ())
    }
}

impl std::ops::Neg for &Val<ComplexType> {
    type Output = Val<ComplexType>;

    #[track_caller]
    fn neg(self) -> Val<ComplexType> {
        neg(self, ())
    }
}

/// Methods of complex values, without fast-math flags.
impl Val<ComplexType> {
    /// `complex.re`
    #[track_caller]
    pub fn re(&self) -> Val<FloatType> {
        re(self, ())
    }

    /// `complex.im`
    #[track_caller]
    pub fn im(&self) -> Val<FloatType> {
        im(self, ())
    }

    /// `complex.abs`
    #[track_caller]
    pub fn abs(&self) -> Val<FloatType> {
        abs(self, ())
    }

    /// `complex.angle`
    #[track_caller]
    pub fn angle(&self) -> Val<FloatType> {
        angle(self, ())
    }

    /// `complex.conj`
    #[track_caller]
    pub fn conj(&self) -> Val<ComplexType> {
        conj(self, ())
    }

    /// `complex.eq`
    #[track_caller]
    pub fn equal(&self, other: impl Into<Val>) -> Val<IntType> {
        eq(self, other)
    }

    /// `complex.neq`
    #[track_caller]
    pub fn not_equal(&self, other: impl Into<Val>) -> Val<IntType> {
        neq(self, other)
    }
}
