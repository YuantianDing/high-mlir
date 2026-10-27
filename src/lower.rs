//! Numbers the operations of an IR for printing and conversion: the regions,
//! in build order, with their SSA values.

use std::{
    collections::HashMap,
    fmt::{self, Write},
    sync::Arc,
};

use crate::{
    Closure, Edges, HighMlir, OpPrinter, Ty, Val,
    types::TypeList,
    val::{Node, detached},
};

/// Failure to print or convert an IR.
#[derive(Debug)]
pub enum Error {
    /// An operand is not defined before its use, in a visible region.
    NotVisible(String),
    /// MLIR rejected a type or attribute.
    Parse(String),
    /// Melior failed to build the IR.
    #[cfg(feature = "melior")]
    Melior(melior::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotVisible(op) => write!(f, "a value of `{op}` is used where it is not defined"),
            Error::Parse(s) => write!(f, "MLIR could not parse `{s}`"),
            #[cfg(feature = "melior")]
            Error::Melior(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(feature = "melior")]
impl From<melior::Error> for Error {
    fn from(e: melior::Error) -> Error {
        Error::Melior(e)
    }
}

/// Reference to a value from an operand.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Ref {
    Result { id: usize, port: u32, multi: bool },
    /// Region parameter, numbered uniquely across the IR.
    Arg(usize),
    /// Not defined in what is lowered (when printing a single operation).
    Undefined,
}

pub(crate) struct SOp {
    pub node: Arc<Node>,
    pub id: usize,
    pub num_results: usize,
    pub operands: Vec<Ref>,
    pub regions: Vec<SRegion>,
}

pub(crate) struct SRegion {
    pub args: Vec<(usize, Ty)>,
    pub ops: Vec<SOp>,
    pub empty: bool,
}

/// The terminator of a region, made from the values it yields.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Terminator {
    name: String,
    values: Vec<Val>,
}

impl HighMlir for Terminator {
    fn name(&self) -> &str {
        &self.name
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.values.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.write_str(&self.name)?;
        if !self.values.is_empty() {
            let types: Vec<Ty> = self.values.iter().map(|v| v.ty()).collect();
            let operands = p.operands(0..self.values.len());
            write!(p, " {operands} : {}", TypeList(&types))?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct Lowering {
    results: HashMap<*const (), (usize, usize)>,
    args: HashMap<*const (), usize>,
    next_id: usize,
    next_arg: usize,
    /// Undefined operands are errors (rather than `Ref::Undefined`).
    strict: bool,
}

/// Lowers `node` and everything in its regions; with `strict`, operands
/// must be defined in them.
pub(crate) fn lower(node: &Arc<Node>, strict: bool) -> Result<SOp, Error> {
    Lowering { strict, ..Lowering::default() }.op(node)
}

fn key(node: &Arc<Node>) -> *const () {
    Arc::as_ptr(node) as *const ()
}

impl Lowering {
    /// SSA number for an operation; only operations with results use one up.
    fn fresh_id(&mut self, num_results: usize) -> usize {
        let id = self.next_id;
        if num_results > 0 {
            self.next_id += 1;
        }
        id
    }

    fn operand(&self, v: &Val) -> Result<Ref, Error> {
        if let Some(&k) = self.args.get(&v.key()) {
            return Ok(Ref::Arg(k));
        }
        match self.results.get(&v.key()) {
            Some(&(id, n)) => Ok(Ref::Result { id, port: v.port(), multi: n > 1 }),
            None if self.strict => Err(Error::NotVisible(v.name())),
            None => Ok(Ref::Undefined),
        }
    }

    fn op(&mut self, node: &Arc<Node>) -> Result<SOp, Error> {
        let operands = node.inputs().map(|v| self.operand(&v)).collect::<Result<Vec<_>, _>>()?;
        let mut regions = vec![];
        if let Some(c) = node.as_closure() {
            for r in 0..c.num_regions() {
                regions.push(self.region(c, r)?);
            }
        }
        let num_results = node.result_types().len();
        let id = self.fresh_id(num_results);
        self.results.insert(key(node), (id, num_results));
        Ok(SOp { node: node.clone(), id, num_results, operands, regions })
    }

    fn region(&mut self, c: &dyn Closure, r: usize) -> Result<SRegion, Error> {
        let region = c.region(r);
        let args = region
            .params
            .iter()
            .map(|p| {
                let k = self.next_arg;
                self.next_arg += 1;
                self.args.insert(key(p), k);
                (k, p.result_types().remove(0))
            })
            .collect();
        let mut ops = region.ops.iter().map(|op| self.op(op)).collect::<Result<Vec<_>, _>>()?;
        let empty = c.is_empty(r);
        if let Some(name) = c.terminator(r).filter(|_| !empty) {
            let operands = region.exprs.iter().map(|v| self.operand(v)).collect::<Result<Vec<_>, _>>()?;
            let node = detached(Terminator { name: name.to_string(), values: region.exprs.clone() });
            let id = self.fresh_id(0);
            ops.push(SOp { node, id, num_results: 0, operands, regions: vec![] });
        }
        Ok(SRegion { args, ops, empty })
    }
}
