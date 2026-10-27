//! `affine` dialect: loops, conditionals and memory accesses whose indices
//! are affine functions of loop variables and symbols.
//!
//! An affine map is applied to `index` operands: its dimensions (e.g. loop
//! induction variables) then its symbols (values fixed in the enclosing
//! function, such as its arguments or constants), as MLIR requires.
//!
//! ```ignore
//! affine::for_(0, &n, 1, (0f32,)).then(|i, acc| (acc + affine::load(&m, &i).downcast().unwrap()).into());
//! affine::if_(set, [&i], [&n]).then(|| ..).else_(|| ..);
//! ```

use std::{
    fmt::{self, Write},
    marker::PhantomData,
};

use crate::{
    AffineExpr, AffineMap, Attr, BuiltRegion, Closure, ClosureBuilder, Edges, HighMlir, IndexType, IntegerSet, OpPrinter,
    Ty, TypeError, Val, VectorType,
    arith::{Operand, Operands},
    dialect::util::{expect, indices, memref_ty, types, value_ty},
    memref::AtomicRMWKind,
    mixed::{Mixed, MixedList},
    scf::{arity_methods, dyn_methods, typed_results},
    types::{I32, I64, INDEX, TypeList},
    val::{AsVal, Dyn, FromTuple, IntoVals, MlirClosure, OrPanic, Tuple, check_params, for_each_arity},
};

/// An affine map applied to `index` operands: one per dimension, then one
/// per symbol.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MapOperands {
    pub map: AffineMap,
    pub dims: Vec<Val>,
    pub symbols: Vec<Val>,
}

impl MapOperands {
    /// Literal operands become `index` constants.
    #[track_caller]
    pub fn new(op: &str, map: AffineMap, dims: impl Operands, symbols: impl Operands) -> Result<MapOperands, TypeError> {
        let (dims, symbols) = (indices(op, dims)?, indices(op, symbols)?);
        expect(op, dims.len() == map.dims as usize && symbols.len() == map.symbols as usize, || {
            format!("`{map}` takes {} dimensions and {} symbols, got {} and {}", map.dims, map.symbols, dims.len(), symbols.len())
        })?;
        Ok(MapOperands { map, dims, symbols })
    }

    /// The identity map of `indices`, as dimensions.
    #[track_caller]
    fn identity(op: &str, indices: impl Operands) -> Result<MapOperands, TypeError> {
        let dims = self::indices(op, indices)?;
        Ok(MapOperands { map: AffineMap::identity(dims.len() as u32), dims, symbols: vec![] })
    }

    fn len(&self) -> usize {
        self.dims.len() + self.symbols.len()
    }

    fn operands(&self) -> impl Iterator<Item = Val> + '_ {
        self.dims.iter().chain(&self.symbols).cloned()
    }

    /// `(%a, %b)[%c]`, the operands from operand `first` on.
    fn format_operands(&self, p: &OpPrinter<'_>, first: usize) -> String {
        let nd = self.dims.len();
        let mut s = format!("({})", p.operands(first..first + nd));
        if !self.symbols.is_empty() {
            write!(s, "[{}]", p.operands(first + nd..first + self.len())).unwrap();
        }
        s
    }

    /// The results with operand names substituted, e.g. `%arg0 + symbol(%1)`.
    fn render(&self, p: &OpPrinter<'_>, first: usize) -> Vec<String> {
        let nd = self.dims.len();
        let dim = |d: u32| p.operand(first + d as usize);
        let symbol = |s: u32| format!("symbol({})", p.operand(first + nd + s as usize));
        self.map.results.iter().map(|e| e.render(&dim, &symbol)).collect()
    }
}

macro_rules! map_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub map: MapOperands,
        }

        $(#[$doc])*
        /// Literal operands become `index` constants.
        #[track_caller]
        pub fn $try_f(map: AffineMap, dims: impl Operands, symbols: impl Operands) -> Result<Val<IndexType>, TypeError> {
            let name = concat!("affine.", stringify!($f));
            let map = MapOperands::new(name, map, dims, symbols)?;
            let one = name == "affine.apply";
            expect(name, !map.map.results.is_empty() && (!one || map.map.results.len() == 1), || {
                format!("`{}` has {} results", map.map, map.map.results.len())
            })?;
            Val::new_typed($op { map })
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f(map: AffineMap, dims: impl Operands, symbols: impl Operands) -> Val<IndexType> {
            $try_f(map, dims, symbols).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("affine.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(self.map.operands())
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![INDEX.into()]
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                vec![("map".into(), Attr::AffineMap(self.map.map.clone()))]
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                let operands = self.map.format_operands(p, 0);
                write!(p, "{} affine_map<{}>{operands}", self.name(), self.map.map)
            }
        }
    )*};
}

map_ops! {
    /// `affine.apply`: the value of a single-result map.
    Apply = apply / try_apply;
    /// `affine.min`: the minimum of the results of a map.
    Min = min / try_min;
    /// `affine.max`: the maximum of the results of a map.
    Max = max / try_max;
}

/// Checks the subscript of an access to `memref` and returns its element type.
fn check_subscript(op: &str, memref: &Val, map: &MapOperands) -> Result<Ty, TypeError> {
    let (ty, m) = memref_ty(op, memref)?;
    expect(op, map.map.results.len() == m.rank(), || {
        format!("{} subscripts for `{ty}` of rank {}", map.map.results.len(), m.rank())
    })?;
    Ok(m.element)
}

fn check_vector(op: &str, element: &Ty, vector: &Ty) -> Result<(), TypeError> {
    let ok = vector.as_vector().is_some_and(|v| v.rank() > 0 && v.element == *element);
    expect(op, ok, || format!("`{vector}` is not a vector of `{element}`"))
}

/// `affine.load`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Load {
    pub memref: Val,
    pub map: MapOperands,
}

/// `affine.load` at `indices` (dimensions; literals become constants).
#[track_caller]
pub fn try_load(memref: impl Into<Val>, indices: impl Operands) -> Result<Val, TypeError> {
    try_load_map(memref, MapOperands::identity("affine.load", indices)?)
}

/// `affine.load` at the results of a map.
#[track_caller]
pub fn try_load_map(memref: impl Into<Val>, map: MapOperands) -> Result<Val, TypeError> {
    let memref = memref.into();
    check_subscript("affine.load", &memref, &map)?;
    Val::single(Load { memref, map })
}

/// `affine.load` at `indices`.
#[track_caller]
pub fn load(memref: impl Into<Val>, indices: impl Operands) -> Val {
    try_load(memref, indices).or_panic()
}

/// `affine.load` at the results of a map.
#[track_caller]
pub fn load_map(memref: impl Into<Val>, map: MapOperands) -> Val {
    try_load_map(memref, map).or_panic()
}

impl HighMlir for Load {
    fn name(&self) -> &str {
        "affine.load"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.memref.clone()).chain(self.map.operands()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.memref.ty().element().clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("map".into(), Attr::AffineMap(self.map.map.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let subscript = self.map.render(p, 1).join(", ");
        write!(p, "affine.load {}[{subscript}] : {}", p.operand(0), p.operand_type(0))
    }
}

/// `affine.store`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Store {
    pub value: Val,
    pub memref: Val,
    pub map: MapOperands,
}

/// `affine.store` of `value` (a literal takes the element type) at `indices`.
#[track_caller]
pub fn try_store(value: impl Operand, memref: impl Into<Val>, indices: impl Operands) -> Result<(), TypeError> {
    try_store_map(value, memref, MapOperands::identity("affine.store", indices)?)
}

/// `affine.store` of `value` at the results of a map.
#[track_caller]
pub fn try_store_map(value: impl Operand, memref: impl Into<Val>, map: MapOperands) -> Result<(), TypeError> {
    let name = "affine.store";
    let memref = memref.into();
    let element = check_subscript(name, &memref, &map)?;
    let value = value.into_val(&element)?;
    let v = value_ty(name, &value)?;
    expect(name, v == element, || format!("cannot store a `{v}` into `{}`", memref.ty()))?;
    Val::unit(Store { value, memref, map })
}

/// `affine.store` of `value` at `indices`.
#[track_caller]
pub fn store(value: impl Operand, memref: impl Into<Val>, indices: impl Operands) {
    try_store(value, memref, indices).or_panic()
}

/// `affine.store` of `value` at the results of a map.
#[track_caller]
pub fn store_map(value: impl Operand, memref: impl Into<Val>, map: MapOperands) {
    try_store_map(value, memref, map).or_panic()
}

impl HighMlir for Store {
    fn name(&self) -> &str {
        "affine.store"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.value.clone(), self.memref.clone()].into_iter().chain(self.map.operands()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("map".into(), Attr::AffineMap(self.map.map.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let subscript = self.map.render(p, 2).join(", ");
        write!(p, "affine.store {}, {}[{subscript}] : {}", p.operand(0), p.operand(1), p.operand_type(1))
    }
}

/// `affine.vector_load`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VectorLoad {
    pub memref: Val,
    pub map: MapOperands,
    pub ty: Ty,
}

/// `affine.vector_load` of a `ty` vector at the results of a map.
#[track_caller]
pub fn try_vector_load(memref: impl Into<Val>, map: MapOperands, ty: impl Into<Ty>) -> Result<Val<VectorType>, TypeError> {
    let name = "affine.vector_load";
    let (memref, ty) = (memref.into(), ty.into());
    let element = check_subscript(name, &memref, &map)?;
    check_vector(name, &element, &ty)?;
    Val::new_typed(VectorLoad { memref, map, ty })
}

/// `affine.vector_load` of a `ty` vector at the results of a map.
#[track_caller]
pub fn vector_load(memref: impl Into<Val>, map: MapOperands, ty: impl Into<Ty>) -> Val<VectorType> {
    try_vector_load(memref, map, ty).or_panic()
}

impl HighMlir for VectorLoad {
    fn name(&self) -> &str {
        "affine.vector_load"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.memref.clone()).chain(self.map.operands()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("map".into(), Attr::AffineMap(self.map.map.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let subscript = self.map.render(p, 1).join(", ");
        write!(p, "affine.vector_load {}[{subscript}] : {}, {}", p.operand(0), p.operand_type(0), self.ty)
    }
}

/// `affine.vector_store`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VectorStore {
    pub value: Val,
    pub memref: Val,
    pub map: MapOperands,
}

/// `affine.vector_store` of a vector at the results of a map.
#[track_caller]
pub fn try_vector_store(value: impl Into<Val>, memref: impl Into<Val>, map: MapOperands) -> Result<(), TypeError> {
    let name = "affine.vector_store";
    let (value, memref) = (value.into(), memref.into());
    let element = check_subscript(name, &memref, &map)?;
    check_vector(name, &element, &value_ty(name, &value)?)?;
    Val::unit(VectorStore { value, memref, map })
}

/// `affine.vector_store` of a vector at the results of a map.
#[track_caller]
pub fn vector_store(value: impl Into<Val>, memref: impl Into<Val>, map: MapOperands) {
    try_vector_store(value, memref, map).or_panic()
}

impl HighMlir for VectorStore {
    fn name(&self) -> &str {
        "affine.vector_store"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.value.clone(), self.memref.clone()].into_iter().chain(self.map.operands()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("map".into(), Attr::AffineMap(self.map.map.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let subscript = self.map.render(p, 2).join(", ");
        let (mt, vt) = (p.operand_type(1), p.operand_type(0));
        write!(p, "affine.vector_store {}, {}[{subscript}] : {mt}, {vt}", p.operand(0), p.operand(1))
    }
}

/// `affine.prefetch`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Prefetch {
    pub memref: Val,
    pub map: MapOperands,
    pub is_write: bool,
    /// From 0 (no locality) to 3 (keep in cache).
    pub locality: u32,
    pub is_data_cache: bool,
}

/// `affine.prefetch` of the element at the results of a map.
#[track_caller]
pub fn try_prefetch(memref: impl Into<Val>, map: MapOperands, is_write: bool, locality: u32, is_data_cache: bool) -> Result<(), TypeError> {
    let name = "affine.prefetch";
    let memref = memref.into();
    check_subscript(name, &memref, &map)?;
    expect(name, locality <= 3, || format!("locality must be at most 3, not {locality}"))?;
    Val::unit(Prefetch { memref, map, is_write, locality, is_data_cache })
}

/// `affine.prefetch` of the element at the results of a map.
#[track_caller]
pub fn prefetch(memref: impl Into<Val>, map: MapOperands, is_write: bool, locality: u32, is_data_cache: bool) {
    try_prefetch(memref, map, is_write, locality, is_data_cache).or_panic()
}

impl HighMlir for Prefetch {
    fn name(&self) -> &str {
        "affine.prefetch"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.memref.clone()).chain(self.map.operands()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![
            ("isWrite".into(), Attr::Bool(self.is_write)),
            ("localityHint".into(), Attr::Int(self.locality as i64, I32.into())),
            ("isDataCache".into(), Attr::Bool(self.is_data_cache)),
            ("map".into(), Attr::AffineMap(self.map.map.clone())),
        ]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let subscript = self.map.render(p, 1).join(", ");
        let rw = if self.is_write { "write" } else { "read" };
        let cache = if self.is_data_cache { "data" } else { "instr" };
        write!(p, "affine.prefetch {}[{subscript}], {rw}, locality<{}>, {cache} : {}", p.operand(0), self.locality, p.operand_type(0))
    }
}

/// A bound of `affine.for`: a constant, an `index` value (a dimension), or
/// the maximum (lower bound) / minimum (upper bound) of a map's results.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum AffineBound {
    /// A constant.
    Const(i64),
    /// An `index` value.
    Val(Val),
    /// The results of a map.
    Map(MapOperands),
}

macro_rules! const_bounds {
    ($($t:ty),*) => {$(
        impl From<$t> for AffineBound {
            fn from(c: $t) -> AffineBound {
                AffineBound::Const(c as i64)
            }
        }
    )*};
}

const_bounds!(i32, i64, isize, u32, u64, usize);

impl<T: ?Sized> From<Val<T>> for AffineBound
where
    Val<T>: AsVal,
{
    fn from(v: Val<T>) -> AffineBound {
        AffineBound::Val(v.as_val())
    }
}

impl<T: ?Sized> From<&Val<T>> for AffineBound
where
    Val<T>: AsVal,
{
    fn from(v: &Val<T>) -> AffineBound {
        AffineBound::Val(v.as_val())
    }
}

impl From<MapOperands> for AffineBound {
    fn from(m: MapOperands) -> AffineBound {
        AffineBound::Map(m)
    }
}

impl AffineBound {
    fn into_map(self, op: &str) -> Result<MapOperands, TypeError> {
        Ok(match self {
            AffineBound::Const(c) => MapOperands { map: AffineMap::new(0, 0, [AffineExpr::Const(c)]), dims: vec![], symbols: vec![] },
            AffineBound::Val(v) => {
                let ty = value_ty(op, &v)?;
                expect(op, ty.is_index(), || format!("bounds must be `index`, not `{ty}`"))?;
                MapOperands { map: AffineMap::identity(1), dims: vec![v], symbols: vec![] }
            }
            AffineBound::Map(m) => {
                expect(op, !m.map.results.is_empty(), || format!("`{}` has no results", m.map))?;
                m
            }
        })
    }
}

/// A bound as `affine.for` prints it: `0`, or `max affine_map<..>(%a)[%b]`.
fn format_bound(p: &OpPrinter<'_>, bound: &MapOperands, first: usize, keyword: &str) -> String {
    match (bound.len(), bound.map.results.as_slice()) {
        (0, [AffineExpr::Const(c)]) => c.to_string(),
        (_, results) => {
            let keyword = if results.len() > 1 { keyword } else { "" };
            format!("{keyword}affine_map<{}>{}", bound.map, bound.format_operands(p, first))
        }
    }
}

/// `affine.for`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct For {
    pub lower: MapOperands,
    pub upper: MapOperands,
    pub step: u64,
    pub inits: Vec<Val>,
    pub body: BuiltRegion,
}

/// `affine::for_(lb, ub, step, inits)`, waiting for its body.
///
/// `A` is the marker of the body: `(Val<IndexType>, Val<T>, ..)` for the
/// induction variable and each loop-carried value (of the static type of
/// its initial value), or [`Dyn`]. `R` is the shape of the results.
#[must_use = "a loop is built by `.then(body)`"]
pub struct ForBuilder<A = Dyn, R = Tuple> {
    bounds: Result<(MapOperands, MapOperands), TypeError>,
    step: u64,
    inits: Vec<Val>,
    marker: PhantomData<fn() -> (A, R)>,
}

/// `affine.for` from `lower` to `upper` (exclusive) by a positive `step`,
/// with loop-carried values of initial values `inits` (a Rust tuple); see
/// [`ForBuilder`].
#[track_caller]
pub fn for_<I: IntoVals>(
    lower: impl Into<AffineBound>,
    upper: impl Into<AffineBound>,
    step: u64,
    inits: I,
) -> ForBuilder<I::WithFirst<IndexType>, I::Results> {
    let bounds = lower.into().into_map("affine.for").and_then(|l| Ok((l, upper.into().into_map("affine.for")?)));
    ForBuilder { bounds, step, inits: inits.into_vals(), marker: PhantomData }
}

impl<A, R: FromTuple> ForBuilder<A, R> {
    /// `body(iv, iter_args..)` returns the next values of the loop-carried
    /// values, which are also the loop's results.
    #[track_caller]
    pub fn try_then_with<C: MlirClosure<A>>(self, body: C) -> Result<R, TypeError> {
        let name = "affine.for";
        let (lower, upper) = self.bounds?;
        expect(name, self.step > 0, || "the step must be positive".into())?;
        let init_types = types(name, &self.inits)?;
        let param_types: Vec<Ty> = std::iter::once(INDEX.into()).chain(init_types.iter().cloned()).collect();
        check_params::<A, C>(name, &param_types)?;
        let mut builder = ClosureBuilder::new(false);
        let region = builder.region(&param_types, |p| body.call(p));
        builder.try_finish()?;
        let yielded = types(name, &region.exprs)?;
        expect(name, yielded == init_types, || {
            format!("body yields ({}) but iter_args are ({})", TypeList(&yielded), TypeList(&init_types))
        })?;
        let op = For { lower, upper, step: self.step, inits: self.inits, body: region };
        typed_results(name, Val::all_results(op)?)
    }
}

dyn_methods!(ForBuilder, "Builds the loop; `body(iv, iter_args..)` yields the next loop-carried values.");
for_each_arity!(arity_methods ForBuilder, "Builds the loop; `body(iv, iter_args..)` yields the next loop-carried values.");

impl HighMlir for For {
    fn name(&self) -> &str {
        "affine.for"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.lower.operands().chain(self.upper.operands()).chain(self.inits.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        self.inits.iter().map(|v| v.ty()).collect()
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![
            Attr::segments([self.lower.len(), self.upper.len(), self.inits.len()]),
            ("lowerBoundMap".into(), Attr::AffineMap(self.lower.map.clone())),
            ("upperBoundMap".into(), Attr::AffineMap(self.upper.map.clone())),
            ("step".into(), Attr::Int(self.step as i64, INDEX.into())),
        ]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let iv = p.region_arg(0, 0);
        let (nl, nu) = (self.lower.len(), self.upper.len());
        let lower = format_bound(p, &self.lower, 0, "max ");
        let upper = format_bound(p, &self.upper, nl, "min ");
        write!(p, "affine.for {iv} = {lower} to {upper}")?;
        if self.step != 1 {
            write!(p, " step {}", self.step)?;
        }
        if !self.inits.is_empty() {
            let first = nl + nu;
            let iter_args: Vec<String> =
                (0..self.inits.len()).map(|i| format!("{} = {}", p.region_arg(0, i + 1), p.operand(first + i))).collect();
            let types = p.operand_types(first..first + self.inits.len());
            write!(p, " iter_args({}) -> ({})", iter_args.join(", "), TypeList(&types))?;
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
        Some("affine.yield")
    }
}

/// `affine.if`: branches on whether the operands are in an integer set.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct If {
    pub set: IntegerSet,
    pub dims: Vec<Val>,
    pub symbols: Vec<Val>,
    pub then_branch: BuiltRegion,
    pub else_branch: BuiltRegion,
}

/// `affine::if_(set, dims, symbols)`, waiting for its branches.
#[must_use = "a branch is built by `.then(..).else_(..)` or `.then(..).end()`"]
pub struct IfBuilder {
    condition: Result<(IntegerSet, Vec<Val>, Vec<Val>), TypeError>,
}

/// `affine::if_(..).then(..)`, waiting for its else branch.
#[must_use = "a branch is built by `.else_(..)` or `.end()`"]
pub struct IfThen<'a> {
    condition: Result<(IntegerSet, Vec<Val>, Vec<Val>), TypeError>,
    then_branch: Box<dyn FnOnce() -> Tuple + 'a>,
}

/// `affine.if` on whether `dims` and `symbols` (literals become constants)
/// satisfy every constraint of `set`.
#[track_caller]
pub fn if_(set: IntegerSet, dims: impl Operands, symbols: impl Operands) -> IfBuilder {
    let name = "affine.if";
    let condition = (|| {
        let (dims, symbols) = (indices(name, dims)?, indices(name, symbols)?);
        expect(name, dims.len() == set.dims as usize && symbols.len() == set.symbols as usize, || {
            format!("`{set}` takes {} dimensions and {} symbols", set.dims, set.symbols)
        })?;
        Ok((set, dims, symbols))
    })();
    IfBuilder { condition }
}

impl IfBuilder {
    /// The branch run when the operands are in the set.
    pub fn then<'a>(self, then_branch: impl FnOnce() -> Tuple + 'a) -> IfThen<'a> {
        IfThen { condition: self.condition, then_branch: Box::new(then_branch) }
    }
}

impl IfThen<'_> {
    /// The else branch, yielding values of the types the then branch yields.
    #[track_caller]
    pub fn try_else_(self, else_branch: impl FnOnce() -> Tuple) -> Result<Tuple, TypeError> {
        let name = "affine.if";
        let (set, dims, symbols) = self.condition?;
        let mut builder = ClosureBuilder::new(false);
        let then_branch = builder.region(&[], |_| (self.then_branch)());
        let else_branch = builder.region(&[], |_| else_branch());
        builder.try_finish()?;
        let (t, e) = (types(name, &then_branch.exprs)?, types(name, &else_branch.exprs)?);
        expect(name, t == e, || format!("then yields ({}) but else yields ({})", TypeList(&t), TypeList(&e)))?;
        Val::all_results(If { set, dims, symbols, then_branch, else_branch })
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

impl HighMlir for If {
    fn name(&self) -> &str {
        "affine.if"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.dims.iter().chain(&self.symbols).cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        self.then_branch.exprs.iter().map(|v| v.ty()).collect()
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("condition".into(), self.set.attr())]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let nd = self.dims.len();
        write!(p, "affine.if affine_set<{}>({})", self.set, p.operands(0..nd))?;
        if !self.symbols.is_empty() {
            write!(p, "[{}]", p.operands(nd..nd + self.symbols.len()))?;
        }
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
    fn num_regions(&self) -> usize {
        2
    }

    fn region(&self, r: usize) -> &BuiltRegion {
        if r == 0 { &self.then_branch } else { &self.else_branch }
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("affine.yield")
    }
}

/// `affine.parallel`: a parallel loop nest whose iterations yield values
/// combined by reductions.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Parallel {
    /// One result per dimension, over the dimensions `operands[..lower_operands]`.
    pub lower: AffineMap,
    /// One result per dimension, over the dimensions `operands[lower_operands..]`.
    pub upper: AffineMap,
    pub lower_operands: usize,
    pub operands: Vec<Val>,
    pub steps: Vec<i64>,
    pub reductions: Vec<AtomicRMWKind>,
    pub body: BuiltRegion,
}

/// One map result per entry: constants stay, values become dimensions.
fn mixed_map(list: Vec<Mixed>, operands: &mut Vec<Val>) -> AffineMap {
    let mut dims = 0;
    let results: Vec<AffineExpr> = list
        .into_iter()
        .map(|m| match m {
            Mixed::Static(c) => AffineExpr::Const(c),
            Mixed::Dynamic(v) => {
                operands.push(v);
                dims += 1;
                AffineExpr::Dim(dims - 1)
            }
        })
        .collect();
    AffineMap::new(dims, 0, results)
}

/// `affine.parallel` over `lower..upper` by `steps` (one entry per
/// dimension; literals are constants, values `index` dimensions).
/// `body(ivs)` yields one value per reduction, and the results are the
/// reduced values.
#[track_caller]
pub fn try_parallel(
    lower: impl MixedList,
    upper: impl MixedList,
    steps: impl IntoIterator<Item = i64>,
    reductions: impl IntoIterator<Item = AtomicRMWKind>,
    body: impl FnOnce(&[Val<IndexType>]) -> Tuple,
) -> Result<Tuple, TypeError> {
    let name = "affine.parallel";
    let (lower, upper) = (lower.into_mixed_list(), upper.into_mixed_list());
    let steps: Vec<i64> = steps.into_iter().collect();
    let reductions: Vec<AtomicRMWKind> = reductions.into_iter().collect();
    let n = lower.len();
    expect(name, n > 0 && upper.len() == n && steps.len() == n, || "needs as many lower and upper bounds and steps".into())?;
    expect(name, steps.iter().all(|&s| s > 0), || "steps must be positive".into())?;
    let mut operands = vec![];
    let lower = mixed_map(lower, &mut operands);
    let lower_operands = operands.len();
    let upper = mixed_map(upper, &mut operands);
    for t in types(name, &operands)? {
        expect(name, t.is_index(), || format!("bounds must be `index`, not `{t}`"))?;
    }
    let mut builder = ClosureBuilder::new(false);
    let region = builder.region(&vec![INDEX.into(); n], |ivs| {
        let ivs: Vec<Val<IndexType>> = ivs.iter().map(|v| v.downcast().unwrap()).collect();
        body(&ivs)
    });
    builder.try_finish()?;
    let yielded = types(name, &region.exprs)?;
    expect(name, yielded.len() == reductions.len(), || {
        format!("the body yields {} values for {} reductions", yielded.len(), reductions.len())
    })?;
    Val::all_results(Parallel { lower, upper, lower_operands, operands, steps, reductions, body: region })
}

/// `affine.parallel`; see [`try_parallel`].
#[track_caller]
pub fn parallel(
    lower: impl MixedList,
    upper: impl MixedList,
    steps: impl IntoIterator<Item = i64>,
    reductions: impl IntoIterator<Item = AtomicRMWKind>,
    body: impl FnOnce(&[Val<IndexType>]) -> Tuple,
) -> Tuple {
    try_parallel(lower, upper, steps, reductions, body).or_panic()
}

impl HighMlir for Parallel {
    fn name(&self) -> &str {
        "affine.parallel"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.operands.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        self.body.exprs.iter().map(|v| v.ty()).collect()
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let n = self.steps.len();
        let groups = || Attr::Splat(Box::new(Attr::Int(1, I32.into())), Ty::tensor([Some(n as u64)], I32));
        vec![
            ("reductions".into(), Attr::List(self.reductions.iter().map(|&k| Attr::Int(k as i64, I64.into())).collect())),
            ("lowerBoundsMap".into(), Attr::AffineMap(self.lower.clone())),
            ("lowerBoundsGroups".into(), groups()),
            ("upperBoundsMap".into(), Attr::AffineMap(self.upper.clone())),
            ("upperBoundsGroups".into(), groups()),
            ("steps".into(), Attr::i64_list(self.steps.clone())),
        ]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = self.steps.len();
        let ivs: Vec<String> = (0..n).map(|i| p.region_arg(0, i)).collect();
        let bounds = |map: &AffineMap, first: usize| {
            let dim = |d: u32| p.operand(first + d as usize);
            map.results.iter().map(|e| e.render(&dim, &|_| unreachable!())).collect::<Vec<_>>().join(", ")
        };
        let (lower, upper) = (bounds(&self.lower, 0), bounds(&self.upper, self.lower_operands));
        write!(p, "affine.parallel ({}) = ({lower}) to ({upper})", ivs.join(", "))?;
        if self.steps.iter().any(|&s| s != 1) {
            let steps: Vec<String> = self.steps.iter().map(|s| s.to_string()).collect();
            write!(p, " step ({})", steps.join(", "))?;
        }
        if !self.reductions.is_empty() {
            let kinds: Vec<String> = self.reductions.iter().map(|k| format!("\"{}\"", k.keyword())).collect();
            write!(p, " reduce ({}) -> ({})", kinds.join(", "), TypeList(&self.result_types()))?;
        }
        p.write_str(" ")?;
        p.region(0, false)
    }
}

impl Closure for Parallel {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("affine.yield")
    }
}

/// `affine.delinearize_index`: the multi-dimensional index of a linear one.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct DelinearizeIndex {
    pub linear_index: Val,
    pub basis: crate::mixed::MixedValues,
}

/// `affine.delinearize_index` of `linear_index` into the dimensions of
/// sizes `basis` (literals are static): one `index` per basis entry.
#[track_caller]
pub fn try_delinearize_index(linear_index: impl Into<Val>, basis: impl MixedList) -> Result<Vec<Val<IndexType>>, TypeError> {
    let name = "affine.delinearize_index";
    let linear_index = linear_index.into();
    let ty = value_ty(name, &linear_index)?;
    expect(name, ty.is_index(), || format!("expected an `index`, not `{ty}`"))?;
    let basis = crate::mixed::MixedValues::new(name, basis)?;
    expect(name, !basis.is_empty(), || "the basis is empty".into())?;
    let Tuple(results) = Val::all_results(DelinearizeIndex { linear_index, basis })?;
    Ok(results.into_iter().map(|v| v.downcast().unwrap()).collect())
}

/// `affine.delinearize_index`
#[track_caller]
pub fn delinearize_index(linear_index: impl Into<Val>, basis: impl MixedList) -> Vec<Val<IndexType>> {
    try_delinearize_index(linear_index, basis).or_panic()
}

impl HighMlir for DelinearizeIndex {
    fn name(&self) -> &str {
        "affine.delinearize_index"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.linear_index.clone()).chain(self.basis.dynamic.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![INDEX.into(); self.basis.len()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("static_basis".into(), Attr::i64_array(self.basis.statics.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let basis = self.basis.format(|k| p.operand(1 + k));
        let basis = &basis[1..basis.len() - 1];
        write!(p, "affine.delinearize_index {} into ({basis}) : {}", p.operand(0), TypeList(&self.result_types()))
    }
}

/// `affine.linearize_index`: the linear index of a multi-dimensional one.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct LinearizeIndex {
    pub multi_index: Vec<Val>,
    pub basis: crate::mixed::MixedValues,
}

/// `affine.linearize_index` of `multi_index` in the dimensions of sizes
/// `basis` (literals are static; one entry per index).
#[track_caller]
pub fn try_linearize_index(multi_index: impl Operands, basis: impl MixedList) -> Result<Val<IndexType>, TypeError> {
    let name = "affine.linearize_index";
    let multi_index = indices(name, multi_index)?;
    let basis = crate::mixed::MixedValues::new(name, basis)?;
    expect(name, basis.len() == multi_index.len(), || {
        format!("{} indices for a basis of {} sizes", multi_index.len(), basis.len())
    })?;
    Val::new_typed(LinearizeIndex { multi_index, basis })
}

/// `affine.linearize_index`
#[track_caller]
pub fn linearize_index(multi_index: impl Operands, basis: impl MixedList) -> Val<IndexType> {
    try_linearize_index(multi_index, basis).or_panic()
}

impl HighMlir for LinearizeIndex {
    fn name(&self) -> &str {
        "affine.linearize_index"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.multi_index.iter().chain(&self.basis.dynamic).cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![INDEX.into()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![
            Attr::segments([self.multi_index.len(), self.basis.dynamic.len()]),
            ("static_basis".into(), Attr::i64_array(self.basis.statics.clone())),
        ]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = self.multi_index.len();
        let basis = self.basis.format(|k| p.operand(n + k));
        let basis = &basis[1..basis.len() - 1];
        write!(p, "affine.linearize_index [{}] by ({basis}) : index", p.operands(0..n))
    }
}
