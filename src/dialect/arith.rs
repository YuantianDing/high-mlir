//! `arith` dialect.
//!
//! Each operation is a node type (`Val`, `Val`, ...) whose
//! `try_new` type-checks its operands and whose `new` panics on a type error.
//! The free functions (`add`, `lt`, `cast`, ...) pick the right operation
//! from the operand types and are what the `Val` operators use.

use std::{
    fmt::{self, Write},
    hash::Hash,
};

use crate::{
    Attr, FloatType, HighMlir, IndexType, IntType, MlirType, OpPrinter, Ty, TypeError, Val,
    types::{F32, F64, I1, I8, I16, I32, I64, INDEX, TensorType, TypeList, VectorType},
    val::{AsVal, IntoTy, OrPanic, retype},
};

type Edges<'a> = Box<dyn Iterator<Item = Val> + 'a>;

/// An operand: a value, or a Rust literal that becomes an `arith.constant`
/// of the type of the other operand.
pub trait Operand {
    /// The static type of this operand: `T` for a `Val<T>`, the dynamic
    /// type for a literal. Operations of one type (e.g. `addf`) return a
    /// value of the static type of their first operand.
    type Static: ?Sized;

    /// The type of this operand if it is already a value.
    fn ty_hint(&self) -> Option<Ty>;

    /// This operand as a value; a literal becomes a constant of type `like`.
    #[track_caller]
    fn into_val(self, like: &Ty) -> Result<Val, TypeError>;
}

impl<T: ?Sized> Operand for Val<T>
where
    Val<T>: AsVal,
{
    type Static = T;

    fn ty_hint(&self) -> Option<Ty> {
        self.as_val().try_ty()
    }

    #[track_caller]
    fn into_val(self, _: &Ty) -> Result<Val, TypeError> {
        Ok(self.as_val())
    }
}

impl<T: Operand + Clone> Operand for &T {
    type Static = T::Static;

    fn ty_hint(&self) -> Option<Ty> {
        (**self).ty_hint()
    }

    #[track_caller]
    fn into_val(self, like: &Ty) -> Result<Val, TypeError> {
        self.clone().into_val(like)
    }
}

macro_rules! int_operands {
    ($($t:ty),*) => {$(
        impl Operand for $t {
            type Static = dyn MlirType;

            fn ty_hint(&self) -> Option<Ty> {
                None
            }

            #[track_caller]
            fn into_val(self, like: &Ty) -> Result<Val, TypeError> {
                if like.is_float_like() {
                    Ok(Constant::try_float(self as f64, like.clone())?)
                } else {
                    Ok(Constant::try_int(self as i64, like.clone())?)
                }
            }
        }
    )*};
}

int_operands!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

macro_rules! float_operands {
    ($($t:ty),*) => {$(
        impl Operand for $t {
            type Static = dyn MlirType;

            fn ty_hint(&self) -> Option<Ty> {
                None
            }

            #[track_caller]
            fn into_val(self, like: &Ty) -> Result<Val, TypeError> {
                Constant::try_float(self as f64, like.clone())
            }
        }
    )*};
}

float_operands!(f32, f64);

impl Operand for bool {
    type Static = dyn MlirType;

    fn ty_hint(&self) -> Option<Ty> {
        None
    }

    #[track_caller]
    fn into_val(self, like: &Ty) -> Result<Val, TypeError> {
        if !like.is_bool_like() {
            return Err(TypeError::new("arith.constant", format!("`{self}` is not a `{like}`")));
        }
        Constant::try_int(self as i64, like.clone())
    }
}

macro_rules! literal_vals {
    ($($t:ty => $ty:expr, $make:ident, $mlir:ty;)*) => {$(
        #[doc = concat!("A constant of the literal's type, so `x.into()` builds `arith.constant`.")]
        impl From<$t> for Val {
            #[track_caller]
            fn from(value: $t) -> Val {
                Constant::$make(value as _, $ty).erase()
            }
        }

        #[doc = concat!("A constant of the literal's type, so `x.into()` builds `arith.constant`.")]
        impl From<$t> for Val<$mlir> {
            #[track_caller]
            fn from(value: $t) -> Val<$mlir> {
                Constant::$make(value as _, $ty)
            }
        }
    )*};
}

literal_vals! {
    i8 => I8, int, IntType;
    i16 => I16, int, IntType;
    i32 => I32, int, IntType;
    i64 => I64, int, IntType;
    usize => INDEX, int, IndexType;
    f32 => F32, float, FloatType;
    f64 => F64, float, FloatType;
    bool => I1, int, IntType;
}

/// Several operands: one [`Operand`], an array, a `Vec` or a Rust tuple of them.
/// Literals take the type expected at their position.
pub trait Operands {
    /// The operands as values; literal `i` takes the type `expected(i)`.
    #[track_caller]
    fn into_vals(self, expected: &dyn Fn(usize) -> Option<Ty>) -> Result<Vec<Val>, TypeError>;
}

#[track_caller]
fn typed(op: impl Operand, i: usize, expected: &dyn Fn(usize) -> Option<Ty>) -> Result<Val, TypeError> {
    let ty = op
        .ty_hint()
        .or_else(|| expected(i))
        .ok_or_else(|| TypeError::new("operand", format!("cannot infer the type of literal #{i}")))?;
    op.into_val(&ty)
}

impl<T: Operand> Operands for T {
    #[track_caller]
    fn into_vals(self, expected: &dyn Fn(usize) -> Option<Ty>) -> Result<Vec<Val>, TypeError> {
        Ok(vec![typed(self, 0, expected)?])
    }
}

impl<T: Operand, const N: usize> Operands for [T; N] {
    #[track_caller]
    fn into_vals(self, expected: &dyn Fn(usize) -> Option<Ty>) -> Result<Vec<Val>, TypeError> {
        Vec::from(self).into_vals(expected)
    }
}

impl<T: Operand> Operands for Vec<T> {
    #[track_caller]
    fn into_vals(self, expected: &dyn Fn(usize) -> Option<Ty>) -> Result<Vec<Val>, TypeError> {
        // A loop rather than a closure, to keep the caller's location.
        let mut vals = Vec::with_capacity(self.len());
        for (i, op) in self.into_iter().enumerate() {
            vals.push(typed(op, i, expected)?);
        }
        Ok(vals)
    }
}

macro_rules! tuple_operands {
    ($(($($t:ident $i:tt),*))*) => {$(
        impl<$($t: Operand),*> Operands for ($($t,)*) {
            #[allow(unused_variables)]
            #[track_caller]
            fn into_vals(self, expected: &dyn Fn(usize) -> Option<Ty>) -> Result<Vec<Val>, TypeError> {
                Ok(vec![$(typed(self.$i, $i, expected)?),*])
            }
        }
    )*};
}

tuple_operands! {
    ()
    (A 0)
    (A 0, B 1)
    (A 0, B 1, C 2)
    (A 0, B 1, C 2, D 3)
    (A 0, B 1, C 2, D 3, E 4)
    (A 0, B 1, C 2, D 3, E 4, F 5)
}

/// Converts two operands, using the type of whichever is a value for literals.
#[track_caller]
pub(crate) fn operands(op: &str, lhs: impl Operand, rhs: impl Operand) -> Result<(Val, Val), TypeError> {
    let like = lhs
        .ty_hint()
        .or_else(|| rhs.ty_hint())
        .ok_or_else(|| TypeError::new(op, "cannot infer the type of two literals"))?;
    Ok((lhs.into_val(&like)?, rhs.into_val(&like)?))
}

fn value_ty(op: &str, v: &Val) -> Result<Ty, TypeError> {
    v.try_ty()
        .ok_or_else(|| TypeError::new(op, format!("`{}` has no result #{}", v.name(), v.port())))
}

/// `arith.constant`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Constant {
    pub value: Attr,
    pub ty: Ty,
}

impl Constant {
    /// Integer constant of an integer/index type, or splat of a static tensor/vector of them.
    #[track_caller]
    pub fn try_int<U: IntoTy>(value: i64, ty: U) -> Result<Val<U::Mlir>, TypeError>
    where
        Val<U::Mlir>: AsVal,
    {
        let ty = ty.into_ty();
        if !ty.is_int_like() {
            return Err(TypeError::new("arith.constant", format!("`{value}` is not a `{ty}`")));
        }
        let value = Constant::splat(Attr::int(value, ty.element().clone()), &ty)?;
        retype("arith.constant", Val::single(Constant { value, ty })?)
    }

    /// Float constant of a float type, or splat of a static tensor/vector of floats.
    #[track_caller]
    pub fn try_float<U: IntoTy>(value: f64, ty: U) -> Result<Val<U::Mlir>, TypeError>
    where
        Val<U::Mlir>: AsVal,
    {
        let ty = ty.into_ty();
        if !ty.is_float_like() {
            return Err(TypeError::new("arith.constant", format!("`{value}` is not a `{ty}`")));
        }
        let value = Constant::splat(Attr::float(value, ty.element().clone()), &ty)?;
        retype("arith.constant", Val::single(Constant { value, ty })?)
    }

    /// Integer elements of a static tensor or vector type, in row-major order.
    #[track_caller]
    pub fn try_ints<U: IntoTy>(values: impl IntoIterator<Item = i64>, ty: U) -> Result<Val<U::Mlir>, TypeError>
    where
        Val<U::Mlir>: AsVal,
    {
        let ty = ty.into_ty();
        if !ty.is_int_like() {
            return Err(TypeError::new("arith.constant", format!("integers are not a `{ty}`")));
        }
        let values = values.into_iter().map(|v| Attr::int(v, ty.element().clone())).collect();
        retype("arith.constant", Constant::try_dense(values, ty)?)
    }

    /// Float elements of a static tensor or vector type, in row-major order.
    #[track_caller]
    pub fn try_floats<U: IntoTy>(values: impl IntoIterator<Item = f64>, ty: U) -> Result<Val<U::Mlir>, TypeError>
    where
        Val<U::Mlir>: AsVal,
    {
        let ty = ty.into_ty();
        if !ty.is_float_like() {
            return Err(TypeError::new("arith.constant", format!("floats are not a `{ty}`")));
        }
        let values = values.into_iter().map(|v| Attr::float(v, ty.element().clone())).collect();
        retype("arith.constant", Constant::try_dense(values, ty)?)
    }

    #[track_caller]
    fn try_dense(values: Vec<Attr>, ty: Ty) -> Result<Val, TypeError> {
        let Some(shape) = ty.shape().and_then(|s| s.iter().copied().collect::<Option<Vec<u64>>>()) else {
            return Err(TypeError::new("arith.constant", format!("`{ty}` is not a statically shaped type")));
        };
        let n: u64 = shape.iter().product();
        if values.len() as u64 != n {
            return Err(TypeError::new("arith.constant", format!("`{ty}` has {n} elements, got {}", values.len())));
        }
        let value = Attr::Dense(values, ty.clone());
        Val::single(Constant { value, ty })
    }

    /// Integer elements of a static tensor or vector type, in row-major order.
    #[track_caller]
    pub fn ints<U: IntoTy>(values: impl IntoIterator<Item = i64>, ty: U) -> Val<U::Mlir>
    where
        Val<U::Mlir>: AsVal,
    {
        Constant::try_ints(values, ty).or_panic()
    }

    /// Float elements of a static tensor or vector type, in row-major order.
    #[track_caller]
    pub fn floats<U: IntoTy>(values: impl IntoIterator<Item = f64>, ty: U) -> Val<U::Mlir>
    where
        Val<U::Mlir>: AsVal,
    {
        Constant::try_floats(values, ty).or_panic()
    }

    fn splat(scalar: Attr, ty: &Ty) -> Result<Attr, TypeError> {
        match ty.shape() {
            None => Ok(scalar),
            Some(shape) if shape.iter().all(Option::is_some) => Ok(Attr::Splat(Box::new(scalar), ty.clone())),
            Some(_) => Err(TypeError::new("arith.constant", format!("`{ty}` has a dynamic shape"))),
        }
    }

    /// Integer constant of an integer/index type, or splat of a static tensor/vector of them.
    #[track_caller]
    pub fn int<U: IntoTy>(value: i64, ty: U) -> Val<U::Mlir>
    where
        Val<U::Mlir>: AsVal,
    {
        Constant::try_int(value, ty).or_panic()
    }

    /// Float constant of a float type, or splat of a static tensor/vector of floats.
    #[track_caller]
    pub fn float<U: IntoTy>(value: f64, ty: U) -> Val<U::Mlir>
    where
        Val<U::Mlir>: AsVal,
    {
        Constant::try_float(value, ty).or_panic()
    }

    /// An `index` constant.
    #[track_caller]
    pub fn index(value: i64) -> Val<IndexType> {
        Constant::int(value, INDEX)
    }

    /// An `i1` constant.
    #[track_caller]
    pub fn bool(value: bool) -> Val<IntType> {
        Constant::int(value as i64, I1)
    }
}

/// `arith.constant` of `value`, an attribute of type `ty` (see also the
/// [`Constant`] constructors for numbers).
#[track_caller]
pub fn constant<U: IntoTy>(value: Attr, ty: U) -> Val<U::Mlir>
where
    Val<U::Mlir>: AsVal,
{
    retype("arith.constant", Val::single(Constant { value, ty: ty.into_ty() }).or_panic()).or_panic()
}

impl HighMlir for Constant {
    fn name(&self) -> &str {
        "arith.constant"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("value".into(), self.value.clone())]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "arith.constant {}", self.value)
    }
}

/// Optional attributes of an operation, printed after its operands. They
/// convert from `()`, which means none.
pub trait OpFlags: Clone + Eq + Hash + fmt::Debug + Default + 'static {
    /// The attributes; none for the default flags.
    fn attributes(&self) -> Vec<(String, Attr)>;

    /// The keywords printed after the operands, each with a leading space.
    fn suffix(&self) -> String;
}

/// No flags.
impl OpFlags for () {
    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![]
    }

    fn suffix(&self) -> String {
        String::new()
    }
}

macro_rules! bit_flags {
    ($(#[$doc:meta])* $name:ident($attr:literal, $mnemonic:literal, $sep:literal) {
        $($flag:ident = $bit:literal, $kw:literal;)*
    } $(all = $all:literal)?) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
        pub struct $name(pub u32);

        impl $name {
            /// No flags.
            pub const NONE: $name = $name(0);
            $(#[doc = concat!("`", $kw, "`")] pub const $flag: $name = $name(1 << $bit);)*

            /// Whether all the flags of `other` are set.
            pub fn contains(self, other: $name) -> bool {
                self.0 & other.0 == other.0
            }

            const ALL: $name = $name(0 $(| (1 << $bit))*);

            /// The keywords inside `<..>`, e.g. `nsw, nuw`.
            fn keywords(self) -> String {
                let all: Option<&str> = None $(.or(Some($all)))?;
                if let Some(all) = all && self == $name::ALL {
                    return all.to_string();
                }
                let names: Vec<&str> = [$(($name::$flag, $kw)),*]
                    .into_iter()
                    .filter(|(f, _)| self.contains(*f))
                    .map(|(_, kw)| kw)
                    .collect();
                names.join($sep)
            }
        }

        impl From<()> for $name {
            fn from(_: ()) -> $name {
                $name::NONE
            }
        }

        impl std::ops::BitOr for $name {
            type Output = $name;

            fn bitor(self, rhs: $name) -> $name {
                $name(self.0 | rhs.0)
            }
        }

        impl OpFlags for $name {
            fn attributes(&self) -> Vec<(String, Attr)> {
                if self.0 == 0 {
                    return vec![];
                }
                vec![($attr.into(), Attr::Raw(format!("#arith.{}<{}>", $mnemonic, self.keywords())))]
            }

            fn suffix(&self) -> String {
                if self.0 == 0 { String::new() } else { format!(" {}<{}>", $mnemonic, self.keywords()) }
            }
        }
    };
}

bit_flags! {
    /// `#arith.overflow<..>`: no signed / unsigned wrap.
    Overflow("overflowFlags", "overflow", ", ") {
        NSW = 0, "nsw";
        NUW = 1, "nuw";
    }
}

bit_flags! {
    /// `#arith.fastmath<..>`
    FastMath("fastmath", "fastmath", ",") {
        REASSOC = 0, "reassoc";
        NNAN = 1, "nnan";
        NINF = 2, "ninf";
        NSZ = 3, "nsz";
        ARCP = 4, "arcp";
        CONTRACT = 5, "contract";
        AFN = 6, "afn";
    } all = "fast"
}

impl FastMath {
    /// All the flags (`fast`).
    pub const FAST: FastMath = FastMath::ALL;
}

macro_rules! unit_flag {
    ($(#[$doc:meta])* $name:ident($attr:literal, $kw:literal)) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
        pub struct $name(pub bool);

        impl From<()> for $name {
            fn from(_: ()) -> $name {
                $name(false)
            }
        }

        impl OpFlags for $name {
            fn attributes(&self) -> Vec<(String, Attr)> {
                if self.0 { vec![($attr.into(), Attr::Unit)] } else { vec![] }
            }

            fn suffix(&self) -> String {
                if self.0 { concat!(" ", $kw).to_string() } else { String::new() }
            }
        }
    };
}

unit_flag! {
    /// `exact`: the division has no remainder / the shift drops no set bit.
    Exact("isExact", "exact")
}

unit_flag! {
    /// `nneg`: the operand is non-negative.
    NonNeg("nonNeg", "nneg")
}

predicates! {
    /// Floating point rounding mode, in MLIR's numbering.
    RoundingMode {
        ToNearestEven = "to_nearest_even", Downward = "downward", Upward = "upward",
        TowardZero = "toward_zero", ToNearestAway = "to_nearest_away",
    }
}

/// A rounding mode and fast-math flags.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FloatFlags {
    /// The rounding mode, if not the default.
    pub rounding: Option<RoundingMode>,
    /// Fast-math flags.
    pub fastmath: FastMath,
}

impl From<()> for FloatFlags {
    fn from(_: ()) -> FloatFlags {
        FloatFlags::default()
    }
}

impl From<FastMath> for FloatFlags {
    fn from(fastmath: FastMath) -> FloatFlags {
        FloatFlags { rounding: None, fastmath }
    }
}

impl From<RoundingMode> for FloatFlags {
    fn from(rounding: RoundingMode) -> FloatFlags {
        FloatFlags { rounding: Some(rounding), fastmath: FastMath::NONE }
    }
}

impl OpFlags for FloatFlags {
    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = self.fastmath.attributes();
        if let Some(r) = self.rounding {
            attrs.push(("roundingmode".into(), Attr::Int(r as i64, I32.into())));
        }
        attrs
    }

    fn suffix(&self) -> String {
        let rounding = self.rounding.map_or(String::new(), |r| format!(" {}", r.keyword()));
        rounding + &self.fastmath.suffix()
    }
}

#[derive(Clone, Copy)]
enum Domain {
    Int,
    Float,
}

fn check_domain(op: &str, ty: &Ty, domain: Domain) -> Result<(), TypeError> {
    let ok = match domain {
        Domain::Int => ty.is_int_like(),
        Domain::Float => ty.is_float_like(),
    };
    if !ok {
        return Err(TypeError::new(op, format!("unsupported operand type `{ty}`")));
    }
    Ok(())
}

fn check_binary(op: &str, lhs: &Val, rhs: &Val, domain: Domain) -> Result<(), TypeError> {
    let (l, r) = (value_ty(op, lhs)?, value_ty(op, rhs)?);
    if l != r {
        return Err(TypeError::new(op, format!("operand types differ: `{l}` and `{r}`")));
    }
    check_domain(op, &l, domain)
}

/// The flags type of an operation, `()` if it has none.
macro_rules! flags_ty {
    () => {
        ()
    };
    ($flags:ty) => {
        $flags
    };
}

macro_rules! binary_ops {
    ($($op:ident = $f:ident / $try_f:ident, $domain:ident $(, $flags:ty)?;)*) => {$(
        #[doc = concat!("`arith.", stringify!($f), "`")]
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub lhs: Val,
            pub rhs: Val,
            pub flags: flags_ty!($($flags)?),
        }

        impl $op {
            #[track_caller]
            pub(crate) fn try_new(lhs: Val, rhs: Val, flags: flags_ty!($($flags)?)) -> Result<Val, TypeError> {
                check_binary(concat!("arith.", stringify!($f)), &lhs, &rhs, Domain::$domain)?;
                Val::single($op { lhs, rhs, flags })
            }
        }

        #[doc = concat!("`arith.", stringify!($f), "`; a literal operand takes the type of the other.")]
        #[track_caller]
        pub fn $try_f<L: Operand>(lhs: L, rhs: impl Operand $(, flags: impl Into<$flags>)?) -> Result<Val<L::Static>, TypeError>
        where
            Val<L::Static>: AsVal,
        {
            let name = concat!("arith.", stringify!($f));
            let (lhs, rhs) = operands(name, lhs, rhs)?;
            retype(name, $op::try_new(lhs, rhs, ($(Into::<$flags>::into(flags))?))?)
        }

        #[doc = concat!("`arith.", stringify!($f), "`; a literal operand takes the type of the other.")]
        #[track_caller]
        pub fn $f<L: Operand>(lhs: L, rhs: impl Operand $(, flags: impl Into<$flags>)?) -> Val<L::Static>
        where
            Val<L::Static>: AsVal,
        {
            $try_f(lhs, rhs $(, Into::<$flags>::into(flags))?).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("arith.", stringify!($f))
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

binary_ops! {
    AddI = addi / try_addi, Int, Overflow;
    SubI = subi / try_subi, Int, Overflow;
    MulI = muli / try_muli, Int, Overflow;
    DivSI = divsi / try_divsi, Int, Exact;
    DivUI = divui / try_divui, Int, Exact;
    CeilDivSI = ceildivsi / try_ceildivsi, Int;
    CeilDivUI = ceildivui / try_ceildivui, Int;
    FloorDivSI = floordivsi / try_floordivsi, Int;
    RemSI = remsi / try_remsi, Int;
    RemUI = remui / try_remui, Int;
    AndI = andi / try_andi, Int;
    OrI = ori / try_ori, Int;
    XOrI = xori / try_xori, Int;
    ShLI = shli / try_shli, Int, Overflow;
    ShRSI = shrsi / try_shrsi, Int, Exact;
    ShRUI = shrui / try_shrui, Int, Exact;
    MaxSI = maxsi / try_maxsi, Int;
    MinSI = minsi / try_minsi, Int;
    MaxUI = maxui / try_maxui, Int;
    MinUI = minui / try_minui, Int;
    AddF = addf / try_addf, Float, FloatFlags;
    SubF = subf / try_subf, Float, FloatFlags;
    MulF = mulf / try_mulf, Float, FloatFlags;
    DivF = divf / try_divf, Float, FloatFlags;
    RemF = remf / try_remf, Float, FastMath;
    MaximumF = maximumf / try_maximumf, Float, FastMath;
    MinimumF = minimumf / try_minimumf, Float, FastMath;
    MaxNumF = maxnumf / try_maxnumf, Float, FastMath;
    MinNumF = minnumf / try_minnumf, Float, FastMath;
}

macro_rules! extended_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident, $second:ident;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub lhs: Val,
            pub rhs: Val,
        }

        $(#[$doc])*
        /// A literal operand takes the type of the other.
        #[track_caller]
        pub fn $try_f<L: Operand>(lhs: L, rhs: impl Operand) -> Result<(Val<L::Static>, extended_ops!(@second_ty $second)), TypeError>
        where
            Val<L::Static>: AsVal,
        {
            let name = concat!("arith.", stringify!($f));
            let (lhs, rhs) = operands(name, lhs, rhs)?;
            check_binary(name, &lhs, &rhs, Domain::Int)?;
            let [first, second] = Val::all_results($op { lhs, rhs })?.try_into().unwrap();
            Ok((retype(name, first)?, extended_ops!(@second_val $second, name, second)))
        }

        $(#[$doc])*
        /// A literal operand takes the type of the other.
        #[track_caller]
        pub fn $f<L: Operand>(lhs: L, rhs: impl Operand) -> (Val<L::Static>, extended_ops!(@second_ty $second))
        where
            Val<L::Static>: AsVal,
        {
            $try_f(lhs, rhs).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("arith.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new([self.lhs.clone(), self.rhs.clone()].into_iter())
            }

            fn result_types(&self) -> Vec<Ty> {
                let ty = self.lhs.ty();
                extended_ops!(@second $second, ty)
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let (lhs, rhs) = (p.operand(0), p.operand(1));
                let types = extended_ops!(@print_types $second, self.result_types());
                write!(p, "{} {lhs}, {rhs} : {types}", self.name())
            }
        }
    )*};
    (@second_ty bool) => { Val };
    (@second_ty same) => { Val<L::Static> };
    (@second_val bool, $name:ident, $v:ident) => { $v };
    (@second_val same, $name:ident, $v:ident) => { retype($name, $v)? };
    (@second bool, $ty:ident) => { vec![$ty.clone(), $ty.with_element_ty(I1.into())] };
    (@second same, $ty:ident) => { vec![$ty.clone(), $ty] };
    (@print_types bool, $types:expr) => { TypeList(&$types) };
    (@print_types same, $types:expr) => { $types[0].clone() };
}

extended_ops! {
    /// `arith.addui_extended`: the sum and an `i1`-like overflow bit.
    AddUIExtended = addui_extended / try_addui_extended, bool;
    /// `arith.subui_extended`: the difference and an `i1`-like borrow bit.
    SubUIExtended = subui_extended / try_subui_extended, bool;
    /// `arith.mulsi_extended`: the low and high halves of the signed product.
    MulSIExtended = mulsi_extended / try_mulsi_extended, same;
    /// `arith.mului_extended`: the low and high halves of the unsigned product.
    MulUIExtended = mului_extended / try_mului_extended, same;
}

macro_rules! unary_float_ops {
    ($($op:ident = $f:ident / $try_f:ident;)*) => {$(
        #[doc = concat!("`arith.", stringify!($f), "`")]
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub input: Val,
            pub flags: FastMath,
        }

        impl $op {
            #[track_caller]
            pub(crate) fn try_new(input: Val, flags: FastMath) -> Result<Val, TypeError> {
                let name = concat!("arith.", stringify!($f));
                check_domain(name, &value_ty(name, &input)?, Domain::Float)?;
                Val::single($op { input, flags })
            }
        }

        #[doc = concat!("`arith.", stringify!($f), "`")]
        #[track_caller]
        pub fn $try_f<T: ?Sized>(input: &Val<T>, flags: impl Into<FastMath>) -> Result<Val<T>, TypeError>
        where
            Val<T>: AsVal,
        {
            retype(concat!("arith.", stringify!($f)), $op::try_new(input.as_val(), flags.into())?)
        }

        #[doc = concat!("`arith.", stringify!($f), "`")]
        #[track_caller]
        pub fn $f<T: ?Sized>(input: &Val<T>, flags: impl Into<FastMath>) -> Val<T>
        where
            Val<T>: AsVal,
        {
            $try_f(input, flags).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("arith.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(std::iter::once(self.input.clone()))
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![self.input.ty()]
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                self.flags.attributes()
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let (input, ty) = (p.operand(0), p.operand_type(0));
                write!(p, "{} {input}{} : {ty}", self.name(), self.flags.suffix())
            }
        }
    )*};
}

unary_float_ops! {
    NegF = negf / try_negf;
    FlushDenormals = flush_denormals / try_flush_denormals;
}

macro_rules! predicates {
    ($(#[$doc:meta])* $name:ident { $($variant:ident = $kw:literal,)* }) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub enum $name {
            $(#[doc = concat!("`", $kw, "`")] $variant,)*
        }

        impl $name {
            /// The MLIR keyword.
            pub fn keyword(self) -> &'static str {
                match self {
                    $($name::$variant => $kw,)*
                }
            }
        }
    };
}

pub(crate) use predicates;

predicates! {
    /// `arith.cmpi` predicate, in MLIR's numbering.
    CmpIPredicate {
        Eq = "eq", Ne = "ne", Slt = "slt", Sle = "sle", Sgt = "sgt", Sge = "sge",
        Ult = "ult", Ule = "ule", Ugt = "ugt", Uge = "uge",
    }
}

predicates! {
    /// `arith.cmpf` predicate, in MLIR's numbering.
    CmpFPredicate {
        False = "false", Oeq = "oeq", Ogt = "ogt", Oge = "oge", Olt = "olt", Ole = "ole",
        One = "one", Ord = "ord", Ueq = "ueq", Ugt = "ugt", Uge = "uge", Ult = "ult",
        Ule = "ule", Une = "une", Uno = "uno", True = "true",
    }
}

macro_rules! compare_ops {
    ($($op:ident = $f:ident / $try_f:ident, $pred:ident, $domain:ident $(, $flags:ty)?;)*) => {$(
        #[doc = concat!("`arith.", stringify!($f), "`")]
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub predicate: $pred,
            pub lhs: Val,
            pub rhs: Val,
            pub flags: flags_ty!($($flags)?),
        }

        impl $op {
            #[track_caller]
            pub(crate) fn try_new(predicate: $pred, lhs: Val, rhs: Val, flags: flags_ty!($($flags)?)) -> Result<Val, TypeError> {
                check_binary(concat!("arith.", stringify!($f)), &lhs, &rhs, Domain::$domain)?;
                Val::single($op { predicate, lhs, rhs, flags })
            }
        }

        #[doc = concat!("`arith.", stringify!($f), "`; a literal operand takes the type of the other.")]
        #[track_caller]
        pub fn $try_f(
            predicate: $pred,
            lhs: impl Operand,
            rhs: impl Operand,
            $(flags: impl Into<$flags>,)?
        ) -> Result<Val, TypeError> {
            let (lhs, rhs) = operands(concat!("arith.", stringify!($f)), lhs, rhs)?;
            $op::try_new(predicate, lhs, rhs, ($(Into::<$flags>::into(flags))?))
        }

        #[doc = concat!("`arith.", stringify!($f), "`; a literal operand takes the type of the other.")]
        #[track_caller]
        pub fn $f(predicate: $pred, lhs: impl Operand, rhs: impl Operand, $(flags: impl Into<$flags>,)?) -> Val {
            $try_f(predicate, lhs, rhs, $(Into::<$flags>::into(flags))?).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("arith.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new([self.lhs.clone(), self.rhs.clone()].into_iter())
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![self.lhs.ty().with_element_ty(I1.into())]
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                let mut attrs = vec![("predicate".into(), Attr::Int(self.predicate as i64, I64.into()))];
                attrs.extend(self.flags.attributes());
                attrs
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let (lhs, rhs, ty) = (p.operand(0), p.operand(1), p.operand_type(0));
                let (pred, flags) = (self.predicate.keyword(), self.flags.suffix());
                write!(p, "{} {pred}, {lhs}, {rhs}{flags} : {ty}", self.name())
            }
        }
    )*};
}

compare_ops! {
    CmpI = cmpi / try_cmpi, CmpIPredicate, Int;
    CmpF = cmpf / try_cmpf, CmpFPredicate, Float, FastMath;
}

/// `arith.select`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Select {
    pub condition: Val,
    pub true_value: Val,
    pub false_value: Val,
}

impl Select {
    #[track_caller]
    pub(crate) fn try_new(condition: Val, true_value: Val, false_value: Val) -> Result<Val, TypeError> {
        let name = "arith.select";
        let c = value_ty(name, &condition)?;
        let (t, f) = (value_ty(name, &true_value)?, value_ty(name, &false_value)?);
        if t != f {
            return Err(TypeError::new(name, format!("branch types differ: `{t}` and `{f}`")));
        }
        if c != I1 && c != t.with_element_ty(I1.into()) {
            return Err(TypeError::new(name, format!("condition `{c}` does not match `{t}`")));
        }
        Val::single(Select { condition, true_value, false_value })
    }
}

impl HighMlir for Select {
    fn name(&self) -> &str {
        "arith.select"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.condition.clone(), self.true_value.clone(), self.false_value.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.true_value.ty()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (c, ty) = (p.operand_type(0), p.operand_type(1));
        write!(p, "arith.select {} : ", p.operands(0..3))?;
        if c.is_scalar() { write!(p, "{ty}") } else { write!(p, "{c}, {ty}") }
    }
}

fn int_width(t: &Ty) -> Option<u32> {
    t.int_width()
}

fn float_width(t: &Ty) -> Option<u32> {
    t.float_kind().and_then(|_| t.bit_width())
}

fn int_ext(s: &Ty, d: &Ty) -> bool {
    matches!((int_width(s), int_width(d)), (Some(s), Some(d)) if d > s)
}

fn int_trunc(s: &Ty, d: &Ty) -> bool {
    matches!((int_width(s), int_width(d)), (Some(s), Some(d)) if d < s)
}

fn int_to_float(s: &Ty, d: &Ty) -> bool {
    int_width(s).is_some() && float_width(d).is_some()
}

fn float_to_int(s: &Ty, d: &Ty) -> bool {
    float_width(s).is_some() && int_width(d).is_some()
}

fn float_ext(s: &Ty, d: &Ty) -> bool {
    matches!((float_width(s), float_width(d)), (Some(s), Some(d)) if d > s)
}

fn float_trunc(s: &Ty, d: &Ty) -> bool {
    matches!((float_width(s), float_width(d)), (Some(s), Some(d)) if d < s)
}

fn float_convert(s: &Ty, d: &Ty) -> bool {
    s != d && float_width(s).is_some() && float_width(s) == float_width(d)
}

fn is_index_cast(s: &Ty, d: &Ty) -> bool {
    (s.is_index() && int_width(d).is_some()) || (int_width(s).is_some() && d.is_index())
}

fn is_bitcast(s: &Ty, d: &Ty) -> bool {
    !s.is_index() && !d.is_index() && s.bit_width().is_some() && s.bit_width() == d.bit_width()
}

macro_rules! cast_ops {
    ($($op:ident = $f:ident / $try_f:ident, $check:ident $(, $flags:ty)?;)*) => {$(
        #[doc = concat!("`arith.", stringify!($f), "`")]
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub input: Val,
            pub ty: Ty,
            pub flags: flags_ty!($($flags)?),
        }

        impl $op {
            #[track_caller]
            pub(crate) fn try_new(input: Val, ty: Ty, flags: flags_ty!($($flags)?)) -> Result<Val, TypeError> {
                let name = concat!("arith.", stringify!($f));
                let src = value_ty(name, &input)?;
                let same_shape = src.with_element_ty(ty.element().clone()) == ty;
                if !(src.is_int_like() || src.is_float_like()) || !same_shape || !$check(src.element(), ty.element()) {
                    return Err(TypeError::new(name, format!("cannot cast `{src}` to `{ty}`")));
                }
                Val::single($op { input, ty, flags })
            }
        }

        #[doc = concat!("`arith.", stringify!($f), "` of `input` to `ty`")]
        #[track_caller]
        pub fn $try_f<U: IntoTy>(input: impl Into<Val>, ty: U, $(flags: impl Into<$flags>,)?) -> Result<Val<U::Mlir>, TypeError>
        where
            Val<U::Mlir>: AsVal,
        {
            let built = $op::try_new(input.into(), ty.into_ty(), ($(Into::<$flags>::into(flags))?))?;
            retype(concat!("arith.", stringify!($f)), built)
        }

        #[doc = concat!("`arith.", stringify!($f), "` of `input` to `ty`")]
        #[track_caller]
        pub fn $f<U: IntoTy>(input: impl Into<Val>, ty: U, $(flags: impl Into<$flags>,)?) -> Val<U::Mlir>
        where
            Val<U::Mlir>: AsVal,
        {
            $try_f(input, ty, $(Into::<$flags>::into(flags))?).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("arith.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(std::iter::once(self.input.clone()))
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![self.ty.clone()]
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                self.flags.attributes()
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let (input, src) = (p.operand(0), p.operand_type(0));
                write!(p, "{} {input}{} : {src} to {}", self.name(), self.flags.suffix(), self.ty)
            }
        }
    )*};
}

cast_ops! {
    ExtSI = extsi / try_extsi, int_ext;
    ExtUI = extui / try_extui, int_ext, NonNeg;
    TruncI = trunci / try_trunci, int_trunc, Overflow;
    SIToFP = sitofp / try_sitofp, int_to_float;
    UIToFP = uitofp / try_uitofp, int_to_float, NonNeg;
    FPToSI = fptosi / try_fptosi, float_to_int;
    FPToUI = fptoui / try_fptoui, float_to_int;
    ExtF = extf / try_extf, float_ext, FastMath;
    TruncF = truncf / try_truncf, float_trunc, FloatFlags;
    ConvertF = convertf / try_convertf, float_convert, FloatFlags;
    IndexCast = index_cast / try_index_cast, is_index_cast;
    IndexCastUI = index_castui / try_index_castui, is_index_cast, NonNeg;
    Bitcast = bitcast / try_bitcast, is_bitcast;
}

macro_rules! scaling_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident, $check:ident, $flags:ty;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub input: Val,
            pub scale: Val,
            pub ty: Ty,
            pub flags: $flags,
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $try_f<U: IntoTy>(
            input: impl Into<Val>,
            scale: impl Into<Val>,
            ty: U,
            flags: impl Into<$flags>,
        ) -> Result<Val<U::Mlir>, TypeError>
        where
            Val<U::Mlir>: AsVal,
        {
            let name = concat!("arith.", stringify!($f));
            let (input, scale, ty, flags) = (input.into(), scale.into(), ty.into_ty(), flags.into());
            let (src, scale_ty) = (value_ty(name, &input)?, value_ty(name, &scale)?);
            let same_shape = |t: &Ty| src.with_element_ty(t.element().clone()) == *t;
            if !src.is_float_like() || !scale_ty.is_float_like() || !same_shape(&scale_ty) || !same_shape(&ty)
                || !$check(src.element(), ty.element())
            {
                return Err(TypeError::new(name, format!("cannot cast `{src}` scaled by `{scale_ty}` to `{ty}`")));
            }
            retype(name, Val::single($op { input, scale, ty, flags })?)
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f<U: IntoTy>(input: impl Into<Val>, scale: impl Into<Val>, ty: U, flags: impl Into<$flags>) -> Val<U::Mlir>
        where
            Val<U::Mlir>: AsVal,
        {
            $try_f(input, scale, ty, flags).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("arith.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new([self.input.clone(), self.scale.clone()].into_iter())
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![self.ty.clone()]
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                self.flags.attributes()
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let (input, scale) = (p.operand(0), p.operand(1));
                let (src, scale_ty) = (p.operand_type(0), p.operand_type(1));
                write!(p, "{} {input}, {scale}{} : {src}, {scale_ty} to {}", self.name(), self.flags.suffix(), self.ty)
            }
        }
    )*};
}

scaling_ops! {
    /// `arith.scaling_extf`: extends `input` multiplied by `scale` (OCP MXFP).
    ScalingExtF = scaling_extf / try_scaling_extf, float_ext, FastMath;
    /// `arith.scaling_truncf`: truncates `input` divided by `scale` (OCP MXFP).
    ScalingTruncF = scaling_truncf / try_scaling_truncf, float_trunc, FloatFlags;
}

macro_rules! dispatch {
    ($($(#[$doc:meta])* $f:ident / $try_f:ident = $int:ident $(, $float:ident)?;)*) => {$(
        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(lhs: impl Operand, rhs: impl Operand) -> Result<Val, TypeError> {
            let (lhs, rhs) = operands(stringify!($f), lhs, rhs)?;
            $(
                if value_ty(stringify!($f), &lhs)?.is_float_like() {
                    return $float::try_new(lhs, rhs, Default::default());
                }
            )?
            $int::try_new(lhs, rhs, Default::default())
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f(lhs: impl Operand, rhs: impl Operand) -> Val {
            $try_f(lhs, rhs).or_panic()
        }
    )*};
}

dispatch! {
    /// `addi` or `addf`.
    add / try_add = AddI, AddF;
    /// `subi` or `subf`.
    sub / try_sub = SubI, SubF;
    /// `muli` or `mulf`.
    mul / try_mul = MulI, MulF;
    /// Signed `divsi` or `divf`.
    div / try_div = DivSI, DivF;
    /// Signed `remsi` or `remf`.
    rem / try_rem = RemSI, RemF;
    /// Signed `maxsi` or `maximumf`.
    max / try_max = MaxSI, MaximumF;
    /// Signed `minsi` or `minimumf`.
    min / try_min = MinSI, MinimumF;
    /// `andi`.
    and / try_and = AndI;
    /// `ori`.
    or / try_or = OrI;
    /// `xori`.
    xor / try_xor = XOrI;
    /// `shli`.
    shl / try_shl = ShLI;
    /// Arithmetic `shrsi`.
    shr / try_shr = ShRSI;
}

/// `negf`, or `0 - x` for integers.
#[track_caller]
pub fn try_neg(input: impl Operand) -> Result<Val, TypeError> {
    let ty = input.ty_hint().ok_or_else(|| TypeError::new("neg", "cannot infer the type of a literal"))?;
    let input = input.into_val(&ty)?;
    if ty.is_float_like() { NegF::try_new(input, FastMath::NONE) } else { try_sub(0, input) }
}

/// `negf`, or `0 - x` for integers.
#[track_caller]
pub fn neg(input: impl Operand) -> Val {
    try_neg(input).or_panic()
}

/// Bitwise not: `x ^ -1` (logical not for `i1`).
#[track_caller]
pub fn try_not(input: impl Operand) -> Result<Val, TypeError> {
    try_xor(input, -1)
}

/// Bitwise not: `x ^ -1` (logical not for `i1`).
#[track_caller]
pub fn not(input: impl Operand) -> Val {
    try_not(input).or_panic()
}

/// Comparison, mapped to signed `cmpi` or ordered `cmpf` (unordered for `Ne`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Compare {
    /// `==`
    Eq,
    /// `!=` (unordered for floats)
    Ne,
    /// Signed / ordered `<`
    Lt,
    /// Signed / ordered `<=`
    Le,
    /// Signed / ordered `>`
    Gt,
    /// Signed / ordered `>=`
    Ge,
    /// Unsigned for integers, unordered for floats.
    ULt,
    /// Unsigned / unordered `<=`
    ULe,
    /// Unsigned / unordered `>`
    UGt,
    /// Unsigned / unordered `>=`
    UGe,
}

/// `cmpi` or `cmpf` for `kind`; a literal operand takes the type of the other.
#[track_caller]
pub fn try_compare(kind: Compare, lhs: impl Operand, rhs: impl Operand) -> Result<Val, TypeError> {
    use {CmpFPredicate as F, CmpIPredicate as I, Compare as C};
    let (lhs, rhs) = operands("compare", lhs, rhs)?;
    if value_ty("compare", &lhs)?.is_float_like() {
        let p = match kind {
            C::Eq => F::Oeq,
            C::Ne => F::Une,
            C::Lt => F::Olt,
            C::Le => F::Ole,
            C::Gt => F::Ogt,
            C::Ge => F::Oge,
            C::ULt => F::Ult,
            C::ULe => F::Ule,
            C::UGt => F::Ugt,
            C::UGe => F::Uge,
        };
        CmpF::try_new(p, lhs, rhs, FastMath::NONE)
    } else {
        let p = match kind {
            C::Eq => I::Eq,
            C::Ne => I::Ne,
            C::Lt => I::Slt,
            C::Le => I::Sle,
            C::Gt => I::Sgt,
            C::Ge => I::Sge,
            C::ULt => I::Ult,
            C::ULe => I::Ule,
            C::UGt => I::Ugt,
            C::UGe => I::Uge,
        };
        CmpI::try_new(p, lhs, rhs, ())
    }
}

/// `cmpi` or `cmpf` for `kind`; a literal operand takes the type of the other.
#[track_caller]
pub fn compare(kind: Compare, lhs: impl Operand, rhs: impl Operand) -> Val {
    try_compare(kind, lhs, rhs).or_panic()
}

/// `arith.select`; the branches may be literals if one of them is a value.
#[track_caller]
pub fn try_select<L: Operand>(condition: impl Into<Val>, t: L, f: impl Operand) -> Result<Val<L::Static>, TypeError>
where
    Val<L::Static>: AsVal,
{
    let (t, f) = operands("arith.select", t, f)?;
    retype("arith.select", Select::try_new(condition.into(), t, f)?)
}

/// `arith.select`; the branches may be literals if one of them is a value.
#[track_caller]
pub fn select<L: Operand>(condition: impl Into<Val>, t: L, f: impl Operand) -> Val<L::Static>
where
    Val<L::Static>: AsVal,
{
    try_select(condition, t, f).or_panic()
}

/// Converts between integer, index and float types (signed semantics),
/// choosing the right cast operation; a no-op if the types already match.
#[track_caller]
pub fn try_cast<U: IntoTy>(input: impl Into<Val>, ty: U) -> Result<Val<U::Mlir>, TypeError>
where
    Val<U::Mlir>: AsVal,
{
    retype("cast", try_cast_dyn(input.into(), ty.into_ty())?)
}

fn try_cast_dyn(input: Val, ty: Ty) -> Result<Val, TypeError> {
    let src = value_ty("cast", &input)?;
    let (s, d) = (src.element(), ty.element());
    Ok(match () {
        _ if src == ty => input,
        _ if int_ext(s, d) => ExtSI::try_new(input, ty, ())?,
        _ if int_trunc(s, d) => TruncI::try_new(input, ty, Overflow::NONE)?,
        _ if is_index_cast(s, d) => IndexCast::try_new(input, ty, ())?,
        _ if int_to_float(s, d) => SIToFP::try_new(input, ty, ())?,
        _ if float_to_int(s, d) => FPToSI::try_new(input, ty, ())?,
        _ if float_ext(s, d) => ExtF::try_new(input, ty, FastMath::NONE)?,
        _ if float_trunc(s, d) => TruncF::try_new(input, ty, FloatFlags::default())?,
        _ if s.is_index() && float_width(d).is_some() => {
            let i64_ty = src.with_element_ty(I64.into());
            SIToFP::try_new(IndexCast::try_new(input, i64_ty, ())?, ty, ())?
        }
        _ if float_width(s).is_some() && d.is_index() => {
            let i64_ty = src.with_element_ty(I64.into());
            IndexCast::try_new(FPToSI::try_new(input, i64_ty, ())?, ty, ())?
        }
        _ => return Err(TypeError::new("cast", format!("cannot cast `{src}` to `{ty}`"))),
    })
}

/// Converts between integer, index and float types (signed semantics),
/// choosing the right cast operation; a no-op if the types already match.
#[track_caller]
pub fn cast<U: IntoTy>(input: impl Into<Val>, ty: U) -> Val<U::Mlir>
where
    Val<U::Mlir>: AsVal,
{
    try_cast(input, ty).or_panic()
}

/// Rust operators and methods on values of numeric type `$t` (or tensors and
/// vectors of numbers): `arith` operations returning a value of the same
/// static type; comparisons return a `Val<$bool>`. A literal operand takes
/// the type of the other operand. Integer division, remainder and right
/// shift are signed.
macro_rules! numeric_operators {
    ($t:ty, $bool:ty) => {
        numeric_operators!(@binary $t;
            Add::add => try_add, "+";
            Sub::sub => try_sub, "-";
            Mul::mul => try_mul, "*";
            Div::div => try_div, "/";
            Rem::rem => try_rem, "%";
            BitAnd::bitand => try_and, "&";
            BitOr::bitor => try_or, "|";
            BitXor::bitxor => try_xor, "^";
            Shl::shl => try_shl, "<<";
            Shr::shr => try_shr, ">>";
        );
        numeric_operators!(@unary $t; Neg::neg => try_neg, "-"; Not::not => try_not, "!";);

        impl Val<$t> {
            numeric_operators!(@compare $bool;
                /// `==` (`cmpi eq` / `cmpf oeq`).
                equal => Eq;
                /// `!=` (`cmpi ne` / `cmpf une`).
                not_equal => Ne;
                /// Signed / ordered `<`.
                lt => Lt;
                /// Signed / ordered `<=`.
                le => Le;
                /// Signed / ordered `>`.
                gt => Gt;
                /// Signed / ordered `>=`.
                ge => Ge;
                /// Unsigned / unordered `<`.
                ult => ULt;
                /// Unsigned / unordered `<=`.
                ule => ULe;
                /// Unsigned / unordered `>`.
                ugt => UGt;
                /// Unsigned / unordered `>=`.
                uge => UGe;
            );

            /// Signed `max` (`maxsi` / `maximumf`).
            #[track_caller]
            pub fn max(&self, rhs: impl Operand) -> Val<$t> {
                retype("max", try_max(self, rhs).or_panic()).or_panic()
            }

            /// Signed `min` (`minsi` / `minimumf`).
            #[track_caller]
            pub fn min(&self, rhs: impl Operand) -> Val<$t> {
                retype("min", try_min(self, rhs).or_panic()).or_panic()
            }

            /// Converts to `ty` with the matching `arith` cast (signed
            /// semantics); the result has the static type of `ty`.
            #[track_caller]
            pub fn cast<U: IntoTy>(&self, ty: U) -> Val<U::Mlir>
            where
                Val<U::Mlir>: AsVal,
            {
                cast(self, ty)
            }
        }
    };
    (@binary $t:ty; $($trait:ident :: $method:ident => $f:ident, $sym:literal;)*) => {$(
        impl<R: Operand> std::ops::$trait<R> for Val<$t> {
            type Output = Val<$t>;

            #[track_caller]
            fn $method(self, rhs: R) -> Val<$t> {
                retype($sym, $f(&self, rhs).or_panic()).or_panic()
            }
        }

        impl<R: Operand> std::ops::$trait<R> for &Val<$t> {
            type Output = Val<$t>;

            #[track_caller]
            fn $method(self, rhs: R) -> Val<$t> {
                retype($sym, $f(self, rhs).or_panic()).or_panic()
            }
        }

        numeric_operators!(@literals $t; $trait :: $method => $f, $sym; i32, i64, u32, u64, usize, f32, f64, bool);
    )*};
    (@literals $t:ty; $trait:ident :: $method:ident => $f:ident, $sym:literal; $($lit:ty),*) => {$(
        impl std::ops::$trait<Val<$t>> for $lit {
            type Output = Val<$t>;

            #[track_caller]
            fn $method(self, rhs: Val<$t>) -> Val<$t> {
                retype($sym, $f(self, &rhs).or_panic()).or_panic()
            }
        }

        impl std::ops::$trait<&Val<$t>> for $lit {
            type Output = Val<$t>;

            #[track_caller]
            fn $method(self, rhs: &Val<$t>) -> Val<$t> {
                retype($sym, $f(self, rhs).or_panic()).or_panic()
            }
        }
    )*};
    (@unary $t:ty; $($trait:ident :: $method:ident => $f:ident, $sym:literal;)*) => {$(
        impl std::ops::$trait for Val<$t> {
            type Output = Val<$t>;

            #[track_caller]
            fn $method(self) -> Val<$t> {
                retype($sym, $f(&self).or_panic()).or_panic()
            }
        }

        impl std::ops::$trait for &Val<$t> {
            type Output = Val<$t>;

            #[track_caller]
            fn $method(self) -> Val<$t> {
                retype($sym, $f(self).or_panic()).or_panic()
            }
        }
    )*};
    (@compare $bool:ty; $($(#[$doc:meta])* $method:ident => $kind:ident;)*) => {$(
        $(#[$doc])*
        #[track_caller]
        pub fn $method(&self, rhs: impl Operand) -> Val<$bool> {
            retype(stringify!($method), try_compare(Compare::$kind, self, rhs).or_panic()).or_panic()
        }
    )*};
}

numeric_operators!(IntType, IntType);
numeric_operators!(IndexType, IntType);
numeric_operators!(FloatType, IntType);
numeric_operators!(TensorType, TensorType);
numeric_operators!(VectorType, VectorType);

/// `arith.select` with a condition of static type `$t` (an `i1`, or a
/// tensor or vector of `i1`).
macro_rules! select_method {
    ($($t:ty),*) => {$(
        impl Val<$t> {
            /// `arith.select` with this value as the condition; the branches
            /// may be literals if one of them is a value.
            #[track_caller]
            pub fn select<L: Operand>(&self, if_true: L, if_false: impl Operand) -> Val<L::Static>
            where
                Val<L::Static>: AsVal,
            {
                select(self, if_true, if_false)
            }
        }
    )*};
}

select_method!(IntType, TensorType, VectorType);

/// Operations on values of any type.
impl Val {
    /// Converts to `ty` with the matching `arith` cast (signed semantics),
    /// a no-op if the types already match; the result has the static type
    /// of `ty`.
    #[track_caller]
    pub fn cast<U: IntoTy>(&self, ty: U) -> Val<U::Mlir>
    where
        Val<U::Mlir>: AsVal,
    {
        cast(self, ty)
    }

    /// `arith.select` with this value (an `i1`, or a tensor or vector of
    /// `i1`) as the condition; the branches may be literals if one of them
    /// is a value.
    #[track_caller]
    pub fn select<L: Operand>(&self, if_true: L, if_false: impl Operand) -> Val<L::Static>
    where
        Val<L::Static>: AsVal,
    {
        select(self, if_true, if_false)
    }
}
