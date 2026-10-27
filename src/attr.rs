use std::{
    fmt,
    hash::{Hash, Hasher},
};

use crate::{FloatType, I1, I32, I64, Ty};

/// A builtin MLIR attribute.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Attr {
    /// Integer attribute of the given integer/index type (`i1` prints as `true`/`false`).
    Int(i64, Ty),
    /// Float attribute of the given float type.
    Float(Float, Ty),
    /// `true` or `false`.
    Bool(bool),
    /// String attribute.
    Str(String),
    /// Type attribute.
    Type(Ty),
    /// Flat symbol reference `@name`.
    Symbol(String),
    /// Unit attribute, present or absent.
    Unit,
    /// Dense array attribute `array<i32: 1, 2>` (`array<i1: true, false>` for `i1`).
    Array(Ty, Vec<i64>),
    /// Array attribute `[a, b]`.
    List(Vec<Attr>),
    /// Splat elements attribute `dense<1> : tensor<4xi32>`; the scalar is printed without its type.
    Splat(Box<Attr>, Ty),
    /// Elements attribute `dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>`, in row-major order.
    Dense(Vec<Attr>, Ty),
    /// `affine_map<...>`
    AffineMap(AffineMap),
    /// Escape hatch: printed verbatim.
    Raw(String),
}

/// `f64` compared and hashed by bits so that attributes can be `Eq + Hash`.
#[derive(Clone, Copy, Debug)]
pub struct Float(pub f64);

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for Float {}

impl Hash for Float {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state)
    }
}

impl Attr {
    /// Integer attribute normalized (sign-extended) to the width of `ty`.
    pub fn int(value: i64, ty: Ty) -> Attr {
        let value = match ty.int_width() {
            Some(w) if w < 64 => (value << (64 - w)) >> (64 - w),
            _ => value,
        };
        Attr::Int(value, ty)
    }

    /// Float attribute of the float type `ty`.
    pub fn float(value: f64, ty: Ty) -> Attr {
        Attr::Float(Float(value), ty)
    }

    /// String attribute.
    pub fn str(s: impl Into<String>) -> Attr {
        Attr::Str(s.into())
    }

    /// `array<i64: ...>`
    pub fn i64_array(values: impl IntoIterator<Item = i64>) -> Attr {
        Attr::Array(I64.into(), values.into_iter().collect())
    }

    /// `array<i1: ...>`
    pub fn bool_array(values: impl IntoIterator<Item = bool>) -> Attr {
        Attr::Array(I1.into(), values.into_iter().map(i64::from).collect())
    }

    /// `[1, 2]` of `i64` integers, i.e. an `I64ArrayAttr`.
    pub fn i64_list(values: impl IntoIterator<Item = i64>) -> Attr {
        Attr::List(values.into_iter().map(|v| Attr::Int(v, I64.into())).collect())
    }

    /// `operandSegmentSizes` of an operation with variadic operand groups.
    pub fn segments(sizes: impl IntoIterator<Item = usize>) -> (String, Attr) {
        ("operandSegmentSizes".into(), Attr::Array(I32.into(), sizes.into_iter().map(|s| s as i64).collect()))
    }

    /// The attribute without its `: type` suffix (e.g. `dense<1.0>`), as
    /// where the type is implied.
    pub fn untyped(&self) -> String {
        struct Untyped<'a>(&'a Attr);

        impl fmt::Display for Untyped<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.0 {
                    Attr::Splat(v, _) => {
                        f.write_str("dense<")?;
                        v.fmt_bare(f)?;
                        f.write_str(">")
                    }
                    Attr::Dense(vs, ty) => {
                        let shape: Vec<usize> =
                            ty.shape().unwrap_or(&[]).iter().map(|d| d.unwrap_or(0) as usize).collect();
                        f.write_str("dense<")?;
                        fmt_nested(f, vs, &shape)?;
                        f.write_str(">")
                    }
                    a => a.fmt_bare(f),
                }
            }
        }

        Untyped(self).to_string()
    }

    /// The bare value without the `: type` suffix.
    fn fmt_bare(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attr::Int(v, ty) if ty.int_width() == Some(1) => write!(f, "{}", *v != 0),
            Attr::Int(v, _) => write!(f, "{v}"),
            Attr::Float(v, ty) => fmt_float(f, v.0, ty),
            a => write!(f, "{a}"),
        }
    }
}

fn fmt_float(f: &mut fmt::Formatter<'_>, v: f64, ty: &Ty) -> fmt::Result {
    if v.is_finite() {
        // MLIR float literals need a `.` before any exponent.
        let s = format!("{v:?}");
        match s.find('e') {
            Some(i) if !s[..i].contains('.') => write!(f, "{}.0{}", &s[..i], &s[i..]),
            _ => f.write_str(&s),
        }
    } else {
        let f16_bits = || match () {
            _ if v.is_nan() => 0x7E00,
            _ if v > 0.0 => 0x7C00,
            _ => 0xFC00,
        };
        match ty.float_kind() {
            Some(FloatType::F32) => write!(f, "0x{:08X}", (v as f32).to_bits()),
            Some(FloatType::BF16) => write!(f, "0x{:04X}", (v as f32).to_bits() >> 16),
            Some(FloatType::F16) => write!(f, "0x{:04X}", f16_bits()),
            _ => write!(f, "0x{:016X}", v.to_bits()),
        }
    }
}

fn fmt_string(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
    f.write_str("\"")?;
    for b in s.bytes() {
        match b {
            b'"' | b'\\' => write!(f, "\\{}", b as char)?,
            0x20..=0x7e => write!(f, "{}", b as char)?,
            _ => write!(f, "\\{b:02X}")?,
        }
    }
    f.write_str("\"")
}

pub(crate) fn is_bare_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || "_$.".contains(c))
}

pub(crate) struct Symbol<'a>(pub &'a str);

impl fmt::Display for Symbol<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("@")?;
        if is_bare_identifier(self.0) { f.write_str(self.0) } else { fmt_string(f, self.0) }
    }
}

impl fmt::Display for Attr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attr::Int(_, ty) if ty.int_width() == Some(1) => self.fmt_bare(f),
            Attr::Int(_, ty) | Attr::Float(_, ty) => {
                self.fmt_bare(f)?;
                write!(f, " : {ty}")
            }
            Attr::Bool(b) => write!(f, "{b}"),
            Attr::Str(s) => fmt_string(f, s),
            Attr::Type(t) => write!(f, "{t}"),
            Attr::Symbol(s) => write!(f, "{}", Symbol(s)),
            Attr::Unit => f.write_str("unit"),
            Attr::Array(ty, vs) => {
                write!(f, "array<{ty}")?;
                for (i, v) in vs.iter().enumerate() {
                    f.write_str(if i == 0 { ": " } else { ", " })?;
                    if ty.int_width() == Some(1) { write!(f, "{}", *v != 0)? } else { write!(f, "{v}")? }
                }
                f.write_str(">")
            }
            Attr::List(vs) => {
                f.write_str("[")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{v}")?;
                }
                f.write_str("]")
            }
            Attr::Splat(_, ty) | Attr::Dense(_, ty) => write!(f, "{} : {ty}", self.untyped()),
            Attr::AffineMap(m) => write!(f, "affine_map<{m}>"),
            Attr::Raw(s) => f.write_str(s),
        }
    }
}

/// Elements as nested lists following `shape` (a bare scalar for rank 0).
fn fmt_nested(f: &mut fmt::Formatter<'_>, vs: &[Attr], shape: &[usize]) -> fmt::Result {
    let Some((&n, rest)) = shape.split_first() else {
        return vs.first().map_or(Ok(()), |v| v.fmt_bare(f));
    };
    let chunk = rest.iter().product::<usize>().max(1);
    f.write_str("[")?;
    for i in 0..n {
        if i > 0 {
            f.write_str(", ")?;
        }
        fmt_nested(f, vs.get(i * chunk..(i + 1) * chunk).unwrap_or(&[]), rest)?;
    }
    f.write_str("]")
}

/// An affine expression over dimensions `d<i>` and symbols `s<i>`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum AffineExpr {
    /// Dimension `d<i>`.
    Dim(u32),
    /// Symbol `s<i>`.
    Symbol(u32),
    /// Integer constant.
    Const(i64),
    /// `lhs + rhs`
    Add(Box<AffineExpr>, Box<AffineExpr>),
    /// `lhs * rhs`
    Mul(Box<AffineExpr>, Box<AffineExpr>),
    /// `lhs mod rhs`
    Mod(Box<AffineExpr>, Box<AffineExpr>),
    /// `lhs floordiv rhs`
    FloorDiv(Box<AffineExpr>, Box<AffineExpr>),
    /// `lhs ceildiv rhs`
    CeilDiv(Box<AffineExpr>, Box<AffineExpr>),
}

impl AffineExpr {
    fn is_atom(&self) -> bool {
        matches!(self, AffineExpr::Dim(_) | AffineExpr::Symbol(_) | AffineExpr::Const(_))
    }
}

impl std::ops::Add for AffineExpr {
    type Output = AffineExpr;

    fn add(self, rhs: AffineExpr) -> AffineExpr {
        AffineExpr::Add(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Mul for AffineExpr {
    type Output = AffineExpr;

    fn mul(self, rhs: AffineExpr) -> AffineExpr {
        AffineExpr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Sub for AffineExpr {
    type Output = AffineExpr;

    /// `lhs + rhs * -1`
    fn sub(self, rhs: AffineExpr) -> AffineExpr {
        self + -rhs
    }
}

impl std::ops::Neg for AffineExpr {
    type Output = AffineExpr;

    /// `self * -1`
    fn neg(self) -> AffineExpr {
        self * AffineExpr::Const(-1)
    }
}

impl From<i64> for AffineExpr {
    fn from(c: i64) -> AffineExpr {
        AffineExpr::Const(c)
    }
}

impl AffineExpr {
    /// The expression with dimensions and symbols named by `dim` and `symbol`,
    /// e.g. `%arg0 + symbol(%1)` in the operand lists of `affine` operations.
    pub fn render(&self, dim: &dyn Fn(u32) -> String, symbol: &dyn Fn(u32) -> String) -> String {
        let (l, op, r) = match self {
            AffineExpr::Dim(i) => return dim(*i),
            AffineExpr::Symbol(i) => return symbol(*i),
            AffineExpr::Const(c) => return c.to_string(),
            AffineExpr::Add(l, r) => (l, "+", r),
            AffineExpr::Mul(l, r) => (l, "*", r),
            AffineExpr::Mod(l, r) => (l, "mod", r),
            AffineExpr::FloorDiv(l, r) => (l, "floordiv", r),
            AffineExpr::CeilDiv(l, r) => (l, "ceildiv", r),
        };
        let operand = |e: &AffineExpr| {
            let s = e.render(dim, symbol);
            if e.is_atom() { s } else { format!("({s})") }
        };
        format!("{} {op} {}", operand(l), operand(r))
    }
}

impl fmt::Display for AffineExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (l, op, r) = match self {
            AffineExpr::Dim(i) => return write!(f, "d{i}"),
            AffineExpr::Symbol(i) => return write!(f, "s{i}"),
            AffineExpr::Const(c) => return write!(f, "{c}"),
            AffineExpr::Add(l, r) => (l, "+", r),
            AffineExpr::Mul(l, r) => (l, "*", r),
            AffineExpr::Mod(l, r) => (l, "mod", r),
            AffineExpr::FloorDiv(l, r) => (l, "floordiv", r),
            AffineExpr::CeilDiv(l, r) => (l, "ceildiv", r),
        };
        let operand = |f: &mut fmt::Formatter<'_>, e: &AffineExpr| {
            if e.is_atom() { write!(f, "{e}") } else { write!(f, "({e})") }
        };
        operand(f, l)?;
        write!(f, " {op} ")?;
        operand(f, r)
    }
}

/// `(d0, ..)[s0, ..] -> (results)`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct AffineMap {
    /// Number of dimensions.
    pub dims: u32,
    /// Number of symbols.
    pub symbols: u32,
    /// One expression per result.
    pub results: Vec<AffineExpr>,
}

impl AffineMap {
    /// A map from `dims` dimensions and `symbols` symbols to `results`.
    pub fn new(dims: u32, symbols: u32, results: impl IntoIterator<Item = AffineExpr>) -> AffineMap {
        AffineMap { dims, symbols, results: results.into_iter().collect() }
    }

    /// `(d0, ..) -> (d0, ..)`
    pub fn identity(dims: u32) -> AffineMap {
        AffineMap::new(dims, 0, (0..dims).map(AffineExpr::Dim))
    }

    /// `(d0, ..) -> (d<p0>, d<p1>, ..)`
    pub fn permutation(perm: &[u32]) -> AffineMap {
        AffineMap::new(perm.len() as u32, 0, perm.iter().map(|&p| AffineExpr::Dim(p)))
    }

    /// The trailing `results` dimensions of `dims`: `(d0, d1, d2) -> (d1, d2)`.
    pub fn minor_identity(dims: u32, results: u32) -> AffineMap {
        AffineMap::new(dims, 0, (dims - results..dims).map(AffineExpr::Dim))
    }

    /// The dimension of each result, if every result is a dimension or the constant 0 (`None`).
    pub fn projected_dims(&self) -> Option<Vec<Option<u32>>> {
        self.results
            .iter()
            .map(|e| match e {
                AffineExpr::Dim(d) => Some(Some(*d)),
                AffineExpr::Const(0) => Some(None),
                _ => None,
            })
            .collect()
    }
}

impl fmt::Display for AffineMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |prefix: &str, n: u32| (0..n).map(|i| format!("{prefix}{i}")).collect::<Vec<_>>().join(", ");
        write!(f, "({})", list("d", self.dims))?;
        if self.symbols > 0 {
            write!(f, "[{}]", list("s", self.symbols))?;
        }
        let results: Vec<String> = self.results.iter().map(|e| e.to_string()).collect();
        write!(f, " -> ({})", results.join(", "))
    }
}

/// A conjunction of affine constraints over dimensions and symbols:
/// `(d0)[s0] : (d0 - 10 >= 0, s0 - d0 == 0)`, as `affine.if` tests.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct IntegerSet {
    /// Number of dimensions.
    pub dims: u32,
    /// Number of symbols.
    pub symbols: u32,
    /// Each expression with whether it must be `== 0` (else `>= 0`).
    pub constraints: Vec<(AffineExpr, bool)>,
}

impl IntegerSet {
    /// The set where every constraint holds.
    pub fn new(dims: u32, symbols: u32, constraints: impl IntoIterator<Item = (AffineExpr, bool)>) -> IntegerSet {
        IntegerSet { dims, symbols, constraints: constraints.into_iter().collect() }
    }

    /// The constraint `e >= 0`.
    pub fn ge0(e: AffineExpr) -> (AffineExpr, bool) {
        (e, false)
    }

    /// The constraint `e == 0`.
    pub fn eq0(e: AffineExpr) -> (AffineExpr, bool) {
        (e, true)
    }

    /// `affine_set<...>`
    pub fn attr(&self) -> Attr {
        Attr::Raw(format!("affine_set<{self}>"))
    }
}

impl fmt::Display for IntegerSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |prefix: &str, n: u32| (0..n).map(|i| format!("{prefix}{i}")).collect::<Vec<_>>().join(", ");
        write!(f, "({})", list("d", self.dims))?;
        if self.symbols > 0 {
            write!(f, "[{}]", list("s", self.symbols))?;
        }
        let cs: Vec<String> =
            self.constraints.iter().map(|(e, eq)| format!("{e} {} 0", if *eq { "==" } else { ">=" })).collect();
        write!(f, " : ({})", cs.join(", "))
    }
}
