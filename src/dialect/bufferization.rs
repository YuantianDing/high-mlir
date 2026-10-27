//! `bufferization` dialect: the boundary between tensors and memrefs.
//!
//! Each operation is built by the function of its name
//! (`bufferization::to_tensor`, ...), with a `try_` variant returning a
//! [`TypeError`].

use std::fmt::{self, Write};

use crate::{
    Attr, Edges, HighMlir, IntType, MemRefType, OpPrinter, TensorType, Ty, TypeError, Val,
    arith::Operands,
    dialect::util::{expect, indices, memref_ty, tensor_ty, types, value_ty},
    types::{I1, TypeList},
    val::{OrPanic, Tuple},
};

/// Optional operands and attributes of `bufferization.alloc_tensor`.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct AllocTensorOpts {
    /// A tensor whose contents the new one starts with (then there are no sizes).
    pub copy: Option<Val>,
    /// Hint of the number of elements, for sparse tensors.
    pub size_hint: Option<Val>,
    /// The memory space of the buffer.
    pub memory_space: Option<Attr>,
}

impl From<()> for AllocTensorOpts {
    fn from(_: ()) -> AllocTensorOpts {
        AllocTensorOpts::default()
    }
}

/// `bufferization.alloc_tensor`: a tensor that bufferizes to a new allocation.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct AllocTensor {
    pub ty: Ty,
    pub sizes: Vec<Val>,
    pub opts: AllocTensorOpts,
}

/// `bufferization.alloc_tensor` of type `ty`, with one size per dynamic
/// dimension (none when copying).
#[track_caller]
pub fn try_alloc_tensor(ty: impl Into<Ty>, sizes: impl Operands, opts: impl Into<AllocTensorOpts>) -> Result<Val<TensorType>, TypeError> {
    let name = "bufferization.alloc_tensor";
    let (ty, sizes, opts) = (ty.into(), indices(name, sizes)?, opts.into());
    let Some(t) = ty.downcast_ref::<TensorType>() else {
        return Err(TypeError::new(name, format!("expected a ranked tensor type, not `{ty}`")));
    };
    let dynamic = if opts.copy.is_some() { 0 } else { t.shape.iter().filter(|d| d.is_none()).count() };
    expect(name, sizes.len() == dynamic, || format!("`{ty}` needs {dynamic} sizes, got {}", sizes.len()))?;
    if let Some(copy) = &opts.copy {
        let c = value_ty(name, copy)?;
        expect(name, c == ty, || format!("copies a `{c}` into a `{ty}`"))?;
    }
    if let Some(hint) = &opts.size_hint {
        expect(name, value_ty(name, hint)?.is_index(), || "the size hint must be an `index`".into())?;
    }
    Val::new_typed(AllocTensor { ty, sizes, opts })
}

/// `bufferization.alloc_tensor`
#[track_caller]
pub fn alloc_tensor(ty: impl Into<Ty>, sizes: impl Operands, opts: impl Into<AllocTensorOpts>) -> Val<TensorType> {
    try_alloc_tensor(ty, sizes, opts).or_panic()
}

impl HighMlir for AllocTensor {
    fn name(&self) -> &str {
        "bufferization.alloc_tensor"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.sizes.iter().chain(&self.opts.copy).chain(&self.opts.size_hint).cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = vec![Attr::segments([
            self.sizes.len(),
            self.opts.copy.is_some() as usize,
            self.opts.size_hint.is_some() as usize,
        ])];
        attrs.extend(self.opts.memory_space.clone().map(|m| ("memory_space".into(), m)));
        attrs
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let n = self.sizes.len();
        write!(p, "bufferization.alloc_tensor({})", p.operands(0..n))?;
        let mut next = n;
        if self.opts.copy.is_some() {
            write!(p, " copy({})", p.operand(next))?;
            next += 1;
        }
        if self.opts.size_hint.is_some() {
            write!(p, " size_hint={}", p.operand(next))?;
        }
        let space: Vec<_> = self.opts.memory_space.clone().map(|m| ("memory_space".to_string(), m)).into_iter().collect();
        p.attr_dict(&space)?;
        write!(p, " : {}", self.ty)
    }
}

/// `bufferization.clone`: a new buffer with a copy of the contents.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CloneOp {
    pub input: Val,
}

/// `bufferization.clone` of a memref.
#[track_caller]
pub fn try_clone(input: impl Into<Val>) -> Result<Val, TypeError> {
    let input = input.into();
    let ty = value_ty("bufferization.clone", &input)?;
    expect("bufferization.clone", ty.is_memref_like(), || format!("expected a memref, not `{ty}`"))?;
    Val::single(CloneOp { input })
}

/// `bufferization.clone` of a memref.
#[track_caller]
pub fn clone(input: impl Into<Val>) -> Val {
    try_clone(input).or_panic()
}

impl HighMlir for CloneOp {
    fn name(&self) -> &str {
        "bufferization.clone"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.input.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.input.ty()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let ty = p.operand_type(0);
        write!(p, "bufferization.clone {} : {ty} to {ty}", p.operand(0))
    }
}

/// `bufferization.to_tensor`: the contents of a memref as a tensor.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ToTensor {
    pub buffer: Val,
    /// The memref is the only way to access its buffer.
    pub restrict: bool,
    /// The tensor may be written in place.
    pub writable: bool,
}

/// `bufferization.to_tensor` of a ranked memref; set `restrict` when the
/// memref is not otherwise accessed (as one-shot bufferization requires).
#[track_caller]
pub fn try_to_tensor(buffer: impl Into<Val>, restrict: bool, writable: bool) -> Result<Val<TensorType>, TypeError> {
    let buffer = buffer.into();
    memref_ty("bufferization.to_tensor", &buffer)?;
    Val::new_typed(ToTensor { buffer, restrict, writable })
}

/// `bufferization.to_tensor` of a ranked memref.
#[track_caller]
pub fn to_tensor(buffer: impl Into<Val>, restrict: bool, writable: bool) -> Val<TensorType> {
    try_to_tensor(buffer, restrict, writable).or_panic()
}

fn unit_flags(flags: &[(&str, bool)]) -> Vec<(String, Attr)> {
    flags.iter().filter(|(_, on)| *on).map(|(name, _)| (name.to_string(), Attr::Unit)).collect()
}

fn keywords(flags: &[(&str, bool)]) -> String {
    flags.iter().filter(|(_, on)| *on).map(|(name, _)| format!(" {name}")).collect()
}

impl HighMlir for ToTensor {
    fn name(&self) -> &str {
        "bufferization.to_tensor"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.buffer.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        let m = self.buffer.ty().as_memref().cloned().expect("checked when built");
        vec![TensorType { shape: m.shape, element: m.element }.into()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        unit_flags(&[("restrict", self.restrict), ("writable", self.writable)])
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let flags = keywords(&[("restrict", self.restrict), ("writable", self.writable)]);
        write!(p, "bufferization.to_tensor {}{flags} : {} to {}", p.operand(0), p.operand_type(0), self.result_types()[0])
    }
}

/// `bufferization.to_buffer`: a memref holding the contents of a tensor.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ToBuffer {
    pub tensor: Val,
    pub read_only: bool,
    pub ty: Ty,
}

/// `bufferization.to_buffer` of a tensor, as a memref of identity layout.
#[track_caller]
pub fn try_to_buffer(tensor: impl Into<Val>, read_only: bool) -> Result<Val<MemRefType>, TypeError> {
    let tensor = tensor.into();
    let (_, t) = tensor_ty("bufferization.to_buffer", &tensor)?;
    try_to_buffer_with_type(tensor, read_only, MemRefType::new(t.shape, t.element).into())
}

/// `bufferization.to_buffer` of a tensor as a memref of type `ty` (e.g. strided).
#[track_caller]
pub fn try_to_buffer_with_type(tensor: impl Into<Val>, read_only: bool, ty: Ty) -> Result<Val<MemRefType>, TypeError> {
    let name = "bufferization.to_buffer";
    let tensor = tensor.into();
    let (src, t) = tensor_ty(name, &tensor)?;
    let ok = ty.as_memref().is_some_and(|m| m.shape == t.shape && m.element == t.element);
    expect(name, ok, || format!("cannot take `{src}` as a `{ty}`"))?;
    Val::new_typed(ToBuffer { tensor, read_only, ty })
}

/// `bufferization.to_buffer` of a tensor, as a memref of identity layout.
#[track_caller]
pub fn to_buffer(tensor: impl Into<Val>, read_only: bool) -> Val<MemRefType> {
    try_to_buffer(tensor, read_only).or_panic()
}

impl HighMlir for ToBuffer {
    fn name(&self) -> &str {
        "bufferization.to_buffer"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.tensor.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        unit_flags(&[("read_only", self.read_only)])
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let flags = keywords(&[("read_only", self.read_only)]);
        write!(p, "bufferization.to_buffer {}{flags} : {} to {}", p.operand(0), p.operand_type(0), self.ty)
    }
}

/// `bufferization.materialize_in_destination`: stores a tensor into a destination.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MaterializeInDestination {
    pub source: Val,
    pub dest: Val,
    pub restrict: bool,
    pub writable: bool,
}

/// `bufferization.materialize_in_destination` of `source` into a tensor
/// `dest` (giving the new tensor) or a memref `dest` (giving nothing, and
/// then `writable` must be set).
#[track_caller]
pub fn try_materialize_in_destination(source: impl Into<Val>, dest: impl Into<Val>, restrict: bool, writable: bool) -> Result<Option<Val<TensorType>>, TypeError> {
    let name = "bufferization.materialize_in_destination";
    let (source, dest) = (source.into(), dest.into());
    let (src, s) = tensor_ty(name, &source)?;
    let d = value_ty(name, &dest)?;
    let ok = match (d.downcast_ref::<TensorType>(), d.as_memref()) {
        (Some(t), _) => t == &s && !writable,
        (_, Some(m)) => m.shape == s.shape && m.element == s.element && writable,
        _ => false,
    };
    expect(name, ok, || format!("cannot materialize `{src}` in `{d}` (writable: {writable})"))?;
    Ok(Val::optional(MaterializeInDestination { source, dest, restrict, writable })?.map(|v| v.downcast().unwrap()))
}

/// `bufferization.materialize_in_destination`
#[track_caller]
pub fn materialize_in_destination(source: impl Into<Val>, dest: impl Into<Val>, restrict: bool, writable: bool) -> Option<Val<TensorType>> {
    try_materialize_in_destination(source, dest, restrict, writable).or_panic()
}

impl HighMlir for MaterializeInDestination {
    fn name(&self) -> &str {
        "bufferization.materialize_in_destination"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.source.clone(), self.dest.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        let d = self.dest.ty();
        if d.is::<TensorType>() { vec![d] } else { vec![] }
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        unit_flags(&[("restrict", self.restrict), ("writable", self.writable)])
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let flags = keywords(&[("restrict", self.restrict), ("writable", self.writable)]);
        let types = TypeList(&p.operand_types(0..2)).to_string();
        let results = self.result_types();
        let results = if results.is_empty() { "()".to_string() } else { results[0].to_string() };
        write!(p, "bufferization.materialize_in_destination {} in{flags} {} : ({types}) -> {results}", p.operand(0), p.operand(1))
    }
}

/// `bufferization.dealloc_tensor`: frees the buffer of a tensor.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct DeallocTensor {
    pub tensor: Val,
}

/// `bufferization.dealloc_tensor`
#[track_caller]
pub fn try_dealloc_tensor(tensor: impl Into<Val>) -> Result<(), TypeError> {
    let tensor = tensor.into();
    tensor_ty("bufferization.dealloc_tensor", &tensor)?;
    Val::unit(DeallocTensor { tensor })
}

/// `bufferization.dealloc_tensor`
#[track_caller]
pub fn dealloc_tensor(tensor: impl Into<Val>) {
    try_dealloc_tensor(tensor).or_panic()
}

impl HighMlir for DeallocTensor {
    fn name(&self) -> &str {
        "bufferization.dealloc_tensor"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.tensor.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        write!(p, "bufferization.dealloc_tensor {} : {}", p.operand(0), p.operand_type(0))
    }
}

/// `bufferization.dealloc`: frees memrefs under conditions, unless retained.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Dealloc {
    pub memrefs: Vec<Val>,
    pub conditions: Vec<Val>,
    pub retained: Vec<Val>,
}

/// `bufferization.dealloc` of each memref whose `i1` condition holds,
/// unless it aliases a retained one; one updated `i1` per retained memref.
#[track_caller]
pub fn try_dealloc(memrefs: impl IntoIterator<Item = impl Into<Val>>, conditions: impl IntoIterator<Item = impl Into<Val>>, retained: impl IntoIterator<Item = impl Into<Val>>) -> Result<Vec<Val<IntType>>, TypeError> {
    let name = "bufferization.dealloc";
    let memrefs: Vec<Val> = memrefs.into_iter().map(Into::into).collect();
    let conditions: Vec<Val> = conditions.into_iter().map(Into::into).collect();
    let retained: Vec<Val> = retained.into_iter().map(Into::into).collect();
    expect(name, memrefs.len() == conditions.len(), || "needs one condition per memref".into())?;
    for t in types(name, &memrefs)?.iter().chain(&types(name, &retained)?) {
        expect(name, t.is_memref_like(), || format!("expected memrefs, not `{t}`"))?;
    }
    for c in types(name, &conditions)? {
        expect(name, c == I1, || format!("conditions must be `i1`, not `{c}`"))?;
    }
    let Tuple(results) = Val::all_results(Dealloc { memrefs, conditions, retained })?;
    Ok(results.into_iter().map(|v| v.downcast().unwrap()).collect())
}

/// `bufferization.dealloc`
#[track_caller]
pub fn dealloc(memrefs: impl IntoIterator<Item = impl Into<Val>>, conditions: impl IntoIterator<Item = impl Into<Val>>, retained: impl IntoIterator<Item = impl Into<Val>>) -> Vec<Val<IntType>> {
    try_dealloc(memrefs, conditions, retained).or_panic()
}

impl HighMlir for Dealloc {
    fn name(&self) -> &str {
        "bufferization.dealloc"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.memrefs.iter().chain(&self.conditions).chain(&self.retained).cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![I1.into(); self.retained.len()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![Attr::segments([self.memrefs.len(), self.conditions.len(), self.retained.len()])]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        p.write_str("bufferization.dealloc")?;
        let (m, r) = (self.memrefs.len(), self.retained.len());
        if m > 0 {
            let types = TypeList(&p.operand_types(0..m)).to_string();
            write!(p, " ({} : {types}) if ({})", p.operands(0..m), p.operands(m..2 * m))?;
        }
        if r > 0 {
            let types = TypeList(&p.operand_types(2 * m..2 * m + r)).to_string();
            write!(p, " retain ({} : {types})", p.operands(2 * m..2 * m + r))?;
        }
        Ok(())
    }
}
