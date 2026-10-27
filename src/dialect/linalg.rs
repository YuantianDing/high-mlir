//! `linalg` dialect: structured operations on tensors and memrefs.
//!
//! Each operation reads its `ins` and writes its `outs` (its "inits"). On
//! tensors, an operation returns one new tensor per tensor out; on memrefs
//! it writes them in place and returns nothing, so every function returns a
//! [`Tuple`] of the tensor results.
//!
//! - [`generic`], [`map`] and [`reduce`] take a Rust closure for their
//!   payload, which receives one element per operand and ends by yielding
//!   one element per out.
//! - Named operations (`matmul`, `add`, `exp`, `fill`, `conv_2d_nhwc_hwcf`,
//!   `pooling_nhwc_max`, ...) build their payload as MLIR defines it, with
//!   inputs cast (signed) to the element type of the outs.

use std::fmt::{self, Write};

use crate::{
    AffineMap, Attr, BuiltRegion, Closure, ClosureBuilder, Edges, HighMlir, IndexType, OpPrinter, TensorType, Ty,
    TypeError, Val,
    arith::{self, FastMath, Operand, predicates},
    dialect::util::{expect, types, value_ty},
    math,
    types::{I64, INDEX, TypeList},
    val::{OrPanic, Tuple},
};

/// How a structured operation prints.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Form {
    /// `name {attrs} ins(..) outs(..) -> results`, the payload implicit.
    Named,
    /// `linalg.generic {maps, iterators} ins(..) outs(..) { payload } -> results`.
    Generic,
    /// `name ins(..) outs(..) attr = [..]`, the payload implicit.
    Trailing(&'static str),
    /// MLIR's generic form.
    Raw,
}

/// A structured operation: its operands, attributes and payload region.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Structured {
    pub name: String,
    pub ins: Vec<Val>,
    pub outs: Vec<Val>,
    /// Attributes printed in the attribute dictionary (or trailing).
    pub attrs: Vec<(String, Attr)>,
    pub body: BuiltRegion,
    form: Form,
    /// Has `operandSegmentSizes`.
    segments: bool,
}

impl Structured {
    #[track_caller]
    fn build(self) -> Result<Tuple, TypeError> {
        Val::all_results(self)
    }
}

impl HighMlir for Structured {
    fn name(&self) -> &str {
        &self.name
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.ins.iter().chain(&self.outs).cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        self.outs.iter().map(|v| v.ty()).filter(|t| t.is::<TensorType>()).collect()
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = self.attrs.clone();
        if self.segments {
            attrs.push(Attr::segments([self.ins.len(), self.outs.len()]));
        }
        attrs
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        if self.form == Form::Raw {
            return p.generic();
        }
        p.results_lhs()?;
        p.write_str(&self.name)?;
        let trailing = match self.form {
            Form::Trailing(attr) => self.attrs.iter().find(|(n, _)| n == attr),
            _ => None,
        };
        if self.form == Form::Generic {
            let attrs: Vec<(String, Attr)> = self.attrs.iter().map(|(n, a)| (n.clone(), printed_iterators(n, a))).collect();
            p.attr_dict(&attrs)?;
        } else if trailing.is_none() {
            p.attr_dict(&self.attrs)?;
        }
        let (ni, no) = (self.ins.len(), self.outs.len());
        if ni > 0 {
            write!(p, " ins({} : {})", p.operands(0..ni), TypeList(&p.operand_types(0..ni)))?;
        }
        write!(p, " outs({} : {})", p.operands(ni..ni + no), TypeList(&p.operand_types(ni..ni + no)))?;
        if let Some((name, Attr::Array(_, vs))) = trailing {
            let vs: Vec<String> = vs.iter().map(|v| v.to_string()).collect();
            write!(p, " {name} = [{}]", vs.join(", "))?;
            return Ok(());
        }
        if self.form == Form::Generic {
            p.write_str(" ")?;
            p.region(0, true)?;
        }
        let results = self.result_types();
        if !results.is_empty() {
            write!(p, " -> {}", TypeList(&results))?;
        }
        Ok(())
    }
}

/// `iterator_types` as the custom form of `linalg.generic` prints them: strings.
fn printed_iterators(name: &str, attr: &Attr) -> Attr {
    match attr {
        Attr::List(its) if name == "iterator_types" => Attr::List(
            its.iter()
                .map(|a| match a {
                    Attr::Raw(s) => Attr::str(s.trim_start_matches("#linalg.iterator_type<").trim_end_matches('>')),
                    a => a.clone(),
                })
                .collect(),
        ),
        a => a.clone(),
    }
}

impl Closure for Structured {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("linalg.yield")
    }
}

/// The operands as values, each a tensor or memref.
fn shaped(op: &str, vals: impl IntoIterator<Item = impl Into<Val>>) -> Result<Vec<Val>, TypeError> {
    let vals: Vec<Val> = vals.into_iter().map(Into::into).collect();
    for t in types(op, &vals)? {
        expect(op, t.is::<TensorType>() || t.as_memref().is_some(), || format!("expected a ranked tensor or memref, not `{t}`"))?;
    }
    Ok(vals)
}

/// One scalar operand of a structured operation (e.g. `fill`'s value), or a shaped one.
fn operand(op: &str, v: Val) -> Result<Val, TypeError> {
    value_ty(op, &v)?;
    Ok(v)
}

fn rank(v: &Val) -> usize {
    v.ty().shape().map_or(0, |s| s.len())
}

/// The payload region: one parameter per element of `ins` then `outs`; it
/// must yield one element per out.
fn payload(op: &str, ins: &[Val], outs: &[Val], body: impl FnOnce(&[Val]) -> Tuple) -> Result<BuiltRegion, TypeError> {
    let element = |v: &Val| v.ty().element().clone();
    let params: Vec<Ty> = ins.iter().chain(outs).map(element).collect();
    let expected: Vec<Ty> = outs.iter().map(element).collect();
    let mut builder = ClosureBuilder::new(false);
    let region = builder.region(&params, body);
    builder.try_finish()?;
    let yielded = types(op, &region.exprs)?;
    expect(op, yielded == expected, || format!("the payload yields ({}), not ({})", TypeList(&yielded), TypeList(&expected)))?;
    Ok(region)
}

predicates! {
    /// The iterator type of a loop of `linalg.generic`.
    IteratorType {
        Parallel = "parallel", Reduction = "reduction",
    }
}

/// `linalg.generic` with `indexing_maps` (one per operand, from the loops
/// to its indices) and one iterator type per loop; `body(elements)` takes
/// one element per in and out, and yields one per out.
#[track_caller]
pub fn try_generic(
    ins: impl IntoIterator<Item = impl Into<Val>>,
    outs: impl IntoIterator<Item = impl Into<Val>>,
    indexing_maps: impl IntoIterator<Item = AffineMap>,
    iterator_types: impl IntoIterator<Item = IteratorType>,
    body: impl FnOnce(&[Val]) -> Tuple,
) -> Result<Tuple, TypeError> {
    let name = "linalg.generic";
    let (ins, outs) = (shaped(name, ins)?, shaped(name, outs)?);
    let maps: Vec<AffineMap> = indexing_maps.into_iter().collect();
    let iterators: Vec<IteratorType> = iterator_types.into_iter().collect();
    expect(name, !outs.is_empty(), || "needs at least one out".into())?;
    expect(name, maps.len() == ins.len() + outs.len(), || format!("{} maps for {} operands", maps.len(), ins.len() + outs.len()))?;
    for (m, v) in maps.iter().zip(ins.iter().chain(&outs)) {
        expect(name, m.dims as usize == iterators.len() && m.symbols == 0 && m.results.len() == rank(v), || {
            format!("`{m}` does not index `{}` with {} loops", v.ty(), iterators.len())
        })?;
    }
    let body = payload(name, &ins, &outs, body)?;
    let attrs = vec![
        ("indexing_maps".into(), Attr::List(maps.into_iter().map(Attr::AffineMap).collect())),
        (
            "iterator_types".into(),
            Attr::List(iterators.iter().map(|i| Attr::Raw(format!("#linalg.iterator_type<{}>", i.keyword()))).collect()),
        ),
    ];
    Structured { name: name.into(), ins, outs, attrs, body, form: Form::Generic, segments: true }.build()
}

/// `linalg.generic`; see [`try_generic`].
#[track_caller]
pub fn generic(
    ins: impl IntoIterator<Item = impl Into<Val>>,
    outs: impl IntoIterator<Item = impl Into<Val>>,
    indexing_maps: impl IntoIterator<Item = AffineMap>,
    iterator_types: impl IntoIterator<Item = IteratorType>,
    body: impl FnOnce(&[Val]) -> Tuple,
) -> Tuple {
    try_generic(ins, outs, indexing_maps, iterator_types, body).or_panic()
}

/// `linalg.map`: `body(elements)` takes one element per in (of one shape)
/// then the element of `init`, and yields the new element of `init`.
#[track_caller]
pub fn try_map(ins: impl IntoIterator<Item = impl Into<Val>>, init: impl Into<Val>, body: impl FnOnce(&[Val]) -> Tuple) -> Result<Tuple, TypeError> {
    let name = "linalg.map";
    let (ins, outs) = (shaped(name, ins)?, shaped(name, [init.into()])?);
    let shape = outs[0].ty().shape().map(<[_]>::to_vec);
    for v in &ins {
        expect(name, v.ty().shape().map(<[_]>::to_vec) == shape, || format!("`{}` is not of the shape of the init", v.ty()))?;
    }
    let body = payload(name, &ins, &outs, body)?;
    Structured { name: name.into(), ins, outs, attrs: vec![], body, form: Form::Raw, segments: false }.build()
}

/// `linalg.map`; see [`try_map`].
#[track_caller]
pub fn map(ins: impl IntoIterator<Item = impl Into<Val>>, init: impl Into<Val>, body: impl FnOnce(&[Val]) -> Tuple) -> Tuple {
    try_map(ins, init, body).or_panic()
}

/// `linalg.reduce` of `ins` along `dimensions` into `inits` (as many, of
/// the shape of the ins without those dimensions): `body(elements)` takes
/// one element per in then per init, and yields the new init elements.
#[track_caller]
pub fn try_reduce(
    ins: impl IntoIterator<Item = impl Into<Val>>,
    inits: impl IntoIterator<Item = impl Into<Val>>,
    dimensions: impl IntoIterator<Item = i64>,
    body: impl FnOnce(&[Val]) -> Tuple,
) -> Result<Tuple, TypeError> {
    let name = "linalg.reduce";
    let (ins, outs) = (shaped(name, ins)?, shaped(name, inits)?);
    let dims: Vec<i64> = dimensions.into_iter().collect();
    expect(name, !ins.is_empty() && ins.len() == outs.len(), || "needs as many ins as inits".into())?;
    for (i, o) in ins.iter().zip(&outs) {
        let kept: Option<Vec<_>> = i.ty().shape().map(|s| {
            s.iter().enumerate().filter(|(d, _)| !dims.contains(&(*d as i64))).map(|(_, d)| *d).collect()
        });
        expect(name, kept.as_deref() == o.ty().shape(), || format!("cannot reduce `{}` along {dims:?} into `{}`", i.ty(), o.ty()))?;
    }
    let body = payload(name, &ins, &outs, body)?;
    let attrs = vec![("dimensions".into(), Attr::i64_array(dims))];
    Structured { name: name.into(), ins, outs, attrs, body, form: Form::Raw, segments: false }.build()
}

/// `linalg.reduce`; see [`try_reduce`].
#[track_caller]
pub fn reduce(
    ins: impl IntoIterator<Item = impl Into<Val>>,
    inits: impl IntoIterator<Item = impl Into<Val>>,
    dimensions: impl IntoIterator<Item = i64>,
    body: impl FnOnce(&[Val]) -> Tuple,
) -> Tuple {
    try_reduce(ins, inits, dimensions, body).or_panic()
}

/// `linalg.index`: in a payload, the iteration index of loop `dim`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Index {
    pub dim: u64,
}

/// `linalg.index`: in a payload, the iteration index of loop `dim`.
#[track_caller]
pub fn try_index(dim: u64) -> Result<Val<IndexType>, TypeError> {
    Val::new_typed(Index { dim })
}

/// `linalg.index`
#[track_caller]
pub fn index(dim: u64) -> Val<IndexType> {
    try_index(dim).or_panic()
}

impl HighMlir for Index {
    fn name(&self) -> &str {
        "linalg.index"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![INDEX.into()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("dim".into(), Attr::Int(self.dim as i64, I64.into()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "linalg.index {} : index", self.dim)
    }
}

/// `x` cast (signed) to `ty`.
fn cast(x: &Val, ty: &Ty) -> Val {
    arith::cast(x, ty.clone())
}

/// A named operation of `ins` and `outs` whose payload `body(ins, out)`
/// computes the out element from the in elements, cast to its type.
#[track_caller]
fn named(
    name: &str,
    ins: Vec<Val>,
    outs: Vec<Val>,
    attrs: Vec<(String, Attr)>,
    form: Form,
    body: impl FnOnce(&[Val], &Val) -> Val,
) -> Result<Tuple, TypeError> {
    let n = ins.len();
    let out_ty = outs[0].ty().element().clone();
    let region = payload(name, &ins, &outs, |p| {
        let args: Vec<Val> = p[..n].iter().map(|x| cast(x, &out_ty)).collect();
        body(&args, &p[n]).into()
    })?;
    // Single in and out operands (`transpose`, `broadcast`) have no segments.
    let segments = !matches!(form, Form::Trailing(_));
    Structured { name: name.into(), ins, outs, attrs, body: region, form, segments }.build()
}

/// Checks the element types of a named elementwise operation.
fn check_elementwise(op: &str, ins: &[Val], out: &Val, float_only: bool) -> Result<(), TypeError> {
    let o = out.ty();
    for v in ins {
        let t = v.ty();
        expect(op, t.shape() == o.shape(), || format!("`{t}` is not of the shape of `{o}`"))?;
    }
    let e = o.element();
    let ok = e.is::<crate::FloatType>() || (!float_only && (e.int_width().is_some() || e.is_index()));
    expect(op, ok, || format!("unsupported element type `{e}`"))
}

macro_rules! unary_ops {
    ($($(#[$doc:meta])* $f:ident / $try_f:ident => |$x:ident| $body:expr;)*) => {$(
        #[doc = concat!("`linalg.", stringify!($f), "` of `input` into `out`.")]
        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(input: impl Into<Val>, out: impl Into<Val>) -> Result<Tuple, TypeError> {
            let name = concat!("linalg.", stringify!($f));
            let (ins, outs) = (shaped(name, [input])?, shaped(name, [out])?);
            check_elementwise(name, &ins, &outs[0], true)?;
            named(name, ins, outs, vec![], Form::Named, |a, _| {
                let $x = a[0].downcast::<crate::FloatType>().expect("checked to be floats");
                $body.into()
            })
        }

        #[doc = concat!("`linalg.", stringify!($f), "` of `input` into `out`.")]
        $(#[$doc])*
        #[track_caller]
        pub fn $f(input: impl Into<Val>, out: impl Into<Val>) -> Tuple {
            $try_f(input, out).or_panic()
        }
    )*};
}

unary_ops! {
    exp / try_exp => |x| x.exp();
    log / try_log => |x| x.log();
    abs / try_abs => |x| x.absf();
    ceil / try_ceil => |x| x.ceil();
    floor / try_floor => |x| x.floor();
    negf / try_negf => |x| arith::negf(&x, FastMath::NONE);
    /// `1 / x`
    reciprocal / try_reciprocal => |x| arith::divf(1.0, &x, ());
    round / try_round => |x| x.round();
    sqrt / try_sqrt => |x| x.sqrt();
    rsqrt / try_rsqrt => |x| x.rsqrt();
    /// `x * x`
    square / try_square => |x| &x * &x;
    tanh / try_tanh => |x| x.tanh();
    erf / try_erf => |x| x.erf();
}

macro_rules! binary_ops {
    ($($(#[$doc:meta])* $f:ident / $try_f:ident, $float_only:literal => |$a:ident, $b:ident| $body:expr;)*) => {$(
        #[doc = concat!("`linalg.", stringify!($f), "` of `lhs` and `rhs` into `out`.")]
        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(lhs: impl Into<Val>, rhs: impl Into<Val>, out: impl Into<Val>) -> Result<Tuple, TypeError> {
            let name = concat!("linalg.", stringify!($f));
            let (ins, outs) = (shaped(name, [lhs.into(), rhs.into()])?, shaped(name, [out])?);
            check_elementwise(name, &ins, &outs[0], $float_only)?;
            named(name, ins, outs, vec![], Form::Named, |args, _| {
                let ($a, $b) = (&args[0], &args[1]);
                $body
            })
        }

        #[doc = concat!("`linalg.", stringify!($f), "` of `lhs` and `rhs` into `out`.")]
        $(#[$doc])*
        #[track_caller]
        pub fn $f(lhs: impl Into<Val>, rhs: impl Into<Val>, out: impl Into<Val>) -> Tuple {
            $try_f(lhs, rhs, out).or_panic()
        }
    )*};
}

binary_ops! {
    add / try_add, false => |a, b| arith::add(a, b);
    sub / try_sub, false => |a, b| arith::sub(a, b);
    mul / try_mul, false => |a, b| arith::mul(a, b);
    /// Signed for integers.
    div / try_div, false => |a, b| arith::div(a, b);
    /// Integers only.
    div_unsigned / try_div_unsigned, false => |a, b| arith::divui(a, b, ());
    /// Signed for integers.
    max / try_max, false => |a, b| arith::max(a, b);
    /// Signed for integers.
    min / try_min, false => |a, b| arith::min(a, b);
    powf / try_powf, true => |a, b| math::powf(a, b, ());
}

/// `linalg.select`: elementwise `cond ? lhs : rhs` into `out`.
#[track_caller]
pub fn try_select(cond: impl Into<Val>, lhs: impl Into<Val>, rhs: impl Into<Val>, out: impl Into<Val>) -> Result<Tuple, TypeError> {
    let name = "linalg.select";
    let (ins, outs) = (shaped(name, [cond.into(), lhs.into(), rhs.into()])?, shaped(name, [out])?);
    check_elementwise(name, &ins[1..], &outs[0], false)?;
    let out_ty = outs[0].ty().element().clone();
    let region = payload(name, &ins, &outs, |p| {
        arith::select(&p[0], cast(&p[1], &out_ty), cast(&p[2], &out_ty)).into()
    })?;
    Structured { name: name.into(), ins, outs, attrs: vec![], body: region, form: Form::Named, segments: true }.build()
}

/// `linalg.select`
#[track_caller]
pub fn select(cond: impl Into<Val>, lhs: impl Into<Val>, rhs: impl Into<Val>, out: impl Into<Val>) -> Tuple {
    try_select(cond, lhs, rhs, out).or_panic()
}

/// `linalg.fill` of `out` with the scalar `value` (a literal takes its element type).
#[track_caller]
pub fn try_fill(value: impl Operand, out: impl Into<Val>) -> Result<Tuple, TypeError> {
    let name = "linalg.fill";
    let outs = shaped(name, [out])?;
    let value = operand(name, value.into_val(outs[0].ty().element())?)?;
    named(name, vec![value], outs, vec![], Form::Named, |a, _| a[0].clone())
}

/// `linalg.fill`
#[track_caller]
pub fn fill(value: impl Operand, out: impl Into<Val>) -> Tuple {
    try_fill(value, out).or_panic()
}

/// `linalg.copy` of `input` into `out`.
#[track_caller]
pub fn try_copy(input: impl Into<Val>, out: impl Into<Val>) -> Result<Tuple, TypeError> {
    let name = "linalg.copy";
    let (ins, outs) = (shaped(name, [input])?, shaped(name, [out])?);
    check_elementwise(name, &ins, &outs[0], false)?;
    named(name, ins, outs, vec![], Form::Named, |a, _| a[0].clone())
}

/// `linalg.copy`
#[track_caller]
pub fn copy(input: impl Into<Val>, out: impl Into<Val>) -> Tuple {
    try_copy(input, out).or_panic()
}

/// `linalg.transpose` of `input` into `out`: dimension `i` of `out` is
/// dimension `permutation[i]` of `input`.
#[track_caller]
pub fn try_transpose(input: impl Into<Val>, out: impl Into<Val>, permutation: impl IntoIterator<Item = i64>) -> Result<Tuple, TypeError> {
    let name = "linalg.transpose";
    let (ins, outs) = (shaped(name, [input])?, shaped(name, [out])?);
    let perm: Vec<i64> = permutation.into_iter().collect();
    let (i, o) = (ins[0].ty(), outs[0].ty());
    let (is, os) = (i.shape().unwrap(), o.shape().unwrap());
    let mut sorted = perm.clone();
    sorted.sort();
    let ok = sorted == (0..is.len() as i64).collect::<Vec<_>>()
        && os.len() == is.len()
        && perm.iter().enumerate().all(|(d, &p)| os[d] == is[p as usize])
        && i.element() == o.element();
    expect(name, ok, || format!("cannot transpose `{i}` by {perm:?} into `{o}`"))?;
    let attrs = vec![("permutation".into(), Attr::i64_array(perm))];
    named(name, ins, outs, attrs, Form::Trailing("permutation"), |a, _| a[0].clone())
}

/// `linalg.transpose`
#[track_caller]
pub fn transpose(input: impl Into<Val>, out: impl Into<Val>, permutation: impl IntoIterator<Item = i64>) -> Tuple {
    try_transpose(input, out, permutation).or_panic()
}

/// `linalg.broadcast` of `input` into `out`, which has the added `dimensions`.
#[track_caller]
pub fn try_broadcast(input: impl Into<Val>, out: impl Into<Val>, dimensions: impl IntoIterator<Item = i64>) -> Result<Tuple, TypeError> {
    let name = "linalg.broadcast";
    let (ins, outs) = (shaped(name, [input])?, shaped(name, [out])?);
    let dims: Vec<i64> = dimensions.into_iter().collect();
    let (i, o) = (ins[0].ty(), outs[0].ty());
    let kept: Vec<_> =
        o.shape().unwrap().iter().enumerate().filter(|(d, _)| !dims.contains(&(*d as i64))).map(|(_, d)| *d).collect();
    expect(name, kept == i.shape().unwrap() && i.element() == o.element(), || {
        format!("cannot broadcast `{i}` into `{o}` along {dims:?}")
    })?;
    let attrs = vec![("dimensions".into(), Attr::i64_array(dims))];
    named(name, ins, outs, attrs, Form::Trailing("dimensions"), |a, _| a[0].clone())
}

/// `linalg.broadcast`
#[track_caller]
pub fn broadcast(input: impl Into<Val>, out: impl Into<Val>, dimensions: impl IntoIterator<Item = i64>) -> Tuple {
    try_broadcast(input, out, dimensions).or_panic()
}

/// Checks operand ranks and that element types are numbers.
fn check_ranks(op: &str, vals: &[Val], ranks: &[usize]) -> Result<(), TypeError> {
    for (v, &r) in vals.iter().zip(ranks) {
        let t = v.ty();
        expect(op, rank(v) == r, || format!("expected an operand of rank {r}, not `{t}`"))?;
        let e = t.element();
        expect(op, e.is_scalar(), || format!("unsupported element type `{e}`"))?;
    }
    Ok(())
}

/// `out + a * b`, in the type of `out`.
fn mul_add(args: &[Val], out: &Val) -> Val {
    arith::add(out, arith::mul(&args[0], &args[1]))
}

macro_rules! contraction_ops {
    ($($(#[$doc:meta])* $f:ident / $try_f:ident, [$($r:literal),*];)*) => {$(
        #[doc = concat!("`linalg.", stringify!($f), "`: accumulates the products of `lhs` and `rhs` into `out`.")]
        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(lhs: impl Into<Val>, rhs: impl Into<Val>, out: impl Into<Val>) -> Result<Tuple, TypeError> {
            let name = concat!("linalg.", stringify!($f));
            let (ins, outs) = (shaped(name, [lhs.into(), rhs.into()])?, shaped(name, [out])?);
            let all: Vec<Val> = ins.iter().chain(&outs).cloned().collect();
            check_ranks(name, &all, &[$($r),*])?;
            named(name, ins, outs, vec![], Form::Named, mul_add)
        }

        #[doc = concat!("`linalg.", stringify!($f), "`: accumulates the products of `lhs` and `rhs` into `out`.")]
        $(#[$doc])*
        #[track_caller]
        pub fn $f(lhs: impl Into<Val>, rhs: impl Into<Val>, out: impl Into<Val>) -> Tuple {
            $try_f(lhs, rhs, out).or_panic()
        }
    )*};
}

contraction_ops! {
    /// `MxK * KxN += MxN`
    matmul / try_matmul, [2, 2, 2];
    /// `BxMxK * BxKxN += BxMxN`
    batch_matmul / try_batch_matmul, [3, 3, 3];
    /// `BxMxK * BxKxN += MxN`, also summing over the batch.
    batch_reduce_matmul / try_batch_reduce_matmul, [3, 3, 2];
    /// `MxN * N += M`
    matvec / try_matvec, [2, 1, 1];
    /// `M * MxN += N`
    vecmat / try_vecmat, [1, 2, 1];
    /// `BxMxN * BxN += BxM`
    batch_matvec / try_batch_matvec, [3, 2, 2];
    /// `BxM * BxMxN += BxN`
    batch_vecmat / try_batch_vecmat, [2, 3, 2];
    /// `N * N += ()`, into a 0-d out.
    dot / try_dot, [1, 1, 0];
    /// Tiled `MxKxmxk * NxKxnxk += MxNxmxn`, the rhs transposed.
    mmt4d / try_mmt4d, [4, 4, 4];
    /// Batched [`mmt4d`].
    batch_mmt4d / try_batch_mmt4d, [5, 5, 5];
}

/// Optional attributes of convolutions and poolings: one stride and
/// dilation per spatial dimension (1 by default).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct ConvOpts {
    pub strides: Option<Vec<i64>>,
    pub dilations: Option<Vec<i64>>,
}

impl From<()> for ConvOpts {
    fn from(_: ()) -> ConvOpts {
        ConvOpts::default()
    }
}

impl ConvOpts {
    fn attributes(self, op: &str, spatial: usize) -> Result<Vec<(String, Attr)>, TypeError> {
        let mut attrs = vec![];
        for (name, v) in [("dilations", self.dilations), ("strides", self.strides)] {
            if let Some(v) = v {
                expect(op, spatial > 0 && v.len() == spatial, || format!("needs {spatial} {name}, got {}", v.len()))?;
                let ty = Ty::tensor([Some(spatial as u64)], I64);
                attrs.push((name.into(), Attr::Dense(v.into_iter().map(|x| Attr::Int(x, I64.into())).collect(), ty)));
            }
        }
        Ok(attrs)
    }
}

#[derive(Clone, Copy)]
enum Pool {
    Sum,
    Max,
    Min,
}

fn pool(kind: Pool) -> impl FnOnce(&[Val], &Val) -> Val {
    move |args, out| match kind {
        Pool::Sum => arith::add(out, &args[0]),
        Pool::Max => arith::max(out, &args[0]),
        Pool::Min => arith::min(out, &args[0]),
    }
}

macro_rules! window_ops {
    ($($(#[$doc:meta])* $f:ident / $try_f:ident, [$($r:literal),*], $spatial:literal, $body:expr;)*) => {$(
        #[doc = concat!("`linalg.", stringify!($f), "` of `input` by `filter` (a window for poolings) into `out`.")]
        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(input: impl Into<Val>, filter: impl Into<Val>, out: impl Into<Val>, opts: impl Into<ConvOpts>) -> Result<Tuple, TypeError> {
            let name = concat!("linalg.", stringify!($f));
            let (ins, outs) = (shaped(name, [input.into(), filter.into()])?, shaped(name, [out])?);
            let all: Vec<Val> = ins.iter().chain(&outs).cloned().collect();
            check_ranks(name, &all, &[$($r),*])?;
            let attrs = opts.into().attributes(name, $spatial)?;
            named(name, ins, outs, attrs, Form::Named, $body)
        }

        #[doc = concat!("`linalg.", stringify!($f), "` of `input` by `filter` (a window for poolings) into `out`.")]
        $(#[$doc])*
        #[track_caller]
        pub fn $f(input: impl Into<Val>, filter: impl Into<Val>, out: impl Into<Val>, opts: impl Into<ConvOpts>) -> Tuple {
            $try_f(input, filter, out, opts).or_panic()
        }
    )*};
}

window_ops! {
    /// Without batch nor channels; no strides or dilations.
    conv_1d / try_conv_1d, [1, 1, 1], 0, mul_add;
    conv_2d / try_conv_2d, [2, 2, 2], 0, mul_add;
    conv_3d / try_conv_3d, [3, 3, 3], 0, mul_add;
    conv_1d_nwc_wcf / try_conv_1d_nwc_wcf, [3, 3, 3], 1, mul_add;
    conv_1d_ncw_fcw / try_conv_1d_ncw_fcw, [3, 3, 3], 1, mul_add;
    conv_2d_nhwc_hwcf / try_conv_2d_nhwc_hwcf, [4, 4, 4], 2, mul_add;
    conv_2d_nhwc_fhwc / try_conv_2d_nhwc_fhwc, [4, 4, 4], 2, mul_add;
    conv_2d_nchw_fchw / try_conv_2d_nchw_fchw, [4, 4, 4], 2, mul_add;
    conv_2d_ngchw_fgchw / try_conv_2d_ngchw_fgchw, [5, 5, 5], 2, mul_add;
    conv_2d_ngchw_gfchw / try_conv_2d_ngchw_gfchw, [5, 5, 5], 2, mul_add;
    conv_2d_nhwgc_gfhwc / try_conv_2d_nhwgc_gfhwc, [5, 5, 5], 2, mul_add;
    conv_3d_ndhwc_dhwcf / try_conv_3d_ndhwc_dhwcf, [5, 5, 5], 3, mul_add;
    conv_3d_ncdhw_fcdhw / try_conv_3d_ncdhw_fcdhw, [5, 5, 5], 3, mul_add;
    depthwise_conv_1d_nwc_wc / try_depthwise_conv_1d_nwc_wc, [3, 2, 3], 1, mul_add;
    depthwise_conv_1d_ncw_cw / try_depthwise_conv_1d_ncw_cw, [3, 2, 3], 1, mul_add;
    depthwise_conv_1d_nwc_wcm / try_depthwise_conv_1d_nwc_wcm, [3, 3, 4], 1, mul_add;
    depthwise_conv_2d_nhwc_hwc / try_depthwise_conv_2d_nhwc_hwc, [4, 3, 4], 2, mul_add;
    depthwise_conv_2d_nchw_chw / try_depthwise_conv_2d_nchw_chw, [4, 3, 4], 2, mul_add;
    depthwise_conv_2d_nhwc_hwcm / try_depthwise_conv_2d_nhwc_hwcm, [4, 4, 5], 2, mul_add;
    depthwise_conv_3d_ndhwc_dhwc / try_depthwise_conv_3d_ndhwc_dhwc, [5, 4, 5], 3, mul_add;
    depthwise_conv_3d_ncdhw_cdhw / try_depthwise_conv_3d_ncdhw_cdhw, [5, 4, 5], 3, mul_add;
    depthwise_conv_3d_ndhwc_dhwcm / try_depthwise_conv_3d_ndhwc_dhwcm, [5, 5, 6], 3, mul_add;
    pooling_nwc_sum / try_pooling_nwc_sum, [3, 1, 3], 1, pool(Pool::Sum);
    pooling_nwc_max / try_pooling_nwc_max, [3, 1, 3], 1, pool(Pool::Max);
    pooling_nwc_min / try_pooling_nwc_min, [3, 1, 3], 1, pool(Pool::Min);
    pooling_ncw_sum / try_pooling_ncw_sum, [3, 1, 3], 1, pool(Pool::Sum);
    pooling_ncw_max / try_pooling_ncw_max, [3, 1, 3], 1, pool(Pool::Max);
    pooling_nhwc_sum / try_pooling_nhwc_sum, [4, 2, 4], 2, pool(Pool::Sum);
    pooling_nhwc_max / try_pooling_nhwc_max, [4, 2, 4], 2, pool(Pool::Max);
    pooling_nhwc_min / try_pooling_nhwc_min, [4, 2, 4], 2, pool(Pool::Min);
    pooling_nchw_sum / try_pooling_nchw_sum, [4, 2, 4], 2, pool(Pool::Sum);
    pooling_nchw_max / try_pooling_nchw_max, [4, 2, 4], 2, pool(Pool::Max);
    pooling_ndhwc_sum / try_pooling_ndhwc_sum, [5, 3, 5], 3, pool(Pool::Sum);
    pooling_ndhwc_max / try_pooling_ndhwc_max, [5, 3, 5], 3, pool(Pool::Max);
    pooling_ndhwc_min / try_pooling_ndhwc_min, [5, 3, 5], 3, pool(Pool::Min);
}

/// `linalg.softmax` of `input` along `dimension`, into `output`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Softmax {
    pub input: Val,
    pub output: Val,
    pub dimension: u64,
}

/// `linalg.softmax` of `input` along `dimension` into `output`: the new
/// tensor, or nothing for memrefs.
#[track_caller]
pub fn try_softmax(input: impl Into<Val>, output: impl Into<Val>, dimension: u64) -> Result<Tuple, TypeError> {
    let name = "linalg.softmax";
    let ops = shaped(name, [input.into(), output.into()])?;
    let (i, o) = (ops[0].ty(), ops[1].ty());
    expect(name, i.shape() == o.shape() && (dimension as usize) < rank(&ops[0]), || {
        format!("cannot compute the softmax of `{i}` along {dimension} into `{o}`")
    })?;
    let [input, output] = <[Val; 2]>::try_from(ops).unwrap();
    Val::all_results(Softmax { input, output, dimension })
}

/// `linalg.softmax`
#[track_caller]
pub fn softmax(input: impl Into<Val>, output: impl Into<Val>, dimension: u64) -> Tuple {
    try_softmax(input, output, dimension).or_panic()
}

impl HighMlir for Softmax {
    fn name(&self) -> &str {
        "linalg.softmax"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.input.clone(), self.output.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        let o = self.output.ty();
        if o.is::<TensorType>() { vec![o] } else { vec![] }
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("dimension".into(), Attr::Int(self.dimension as i64, I64.into()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let (i, o) = (p.operand(0), p.operand(1));
        let (it, ot) = (p.operand_type(0), p.operand_type(1));
        write!(p, "linalg.softmax dimension({}) ins({i} : {it}) outs({o} : {ot})", self.dimension)?;
        if let [t] = self.result_types().as_slice() {
            write!(p, " -> {t}")?;
        }
        Ok(())
    }
}
