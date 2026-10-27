//! Textual MLIR via `Display`.

use std::fmt::{self, Write};

use indenter::indented;

use crate::{
    HighMlir, Root, Ty, Val, annotations,
    attr::is_bare_identifier,
    lower::{Ref, SOp, SRegion, lower},
    types::{ResultTypes, TypeList},
};

/// Prints one lowered operation; handed to [`HighMlir::print`].
pub struct OpPrinter<'a> {
    w: &'a mut dyn Write,
    op: &'a SOp,
    inputs: Vec<Val>,
}

fn ref_name(r: Ref) -> String {
    match r {
        Ref::Result { id, port, multi: true } => format!("%{id}#{port}"),
        Ref::Result { id, .. } => format!("%{id}"),
        Ref::Arg(k) => format!("%arg{k}"),
        Ref::Undefined => "%?".to_string(),
    }
}

fn print_ops(w: &mut dyn Write, ops: &[SOp]) -> fmt::Result {
    for op in ops {
        let mut p = OpPrinter { w: &mut *w, op, inputs: op.node.inputs().collect() };
        // An annotated operation has the attribute dictionary of the generic form.
        if annotations::op_attributes(&op.node).is_empty() { op.node.print(&mut p)? } else { p.generic()? }
        if let Some(location) = annotations::op_location(&op.node) {
            write!(w, " {location}")?;
        }
        w.write_char('\n')?;
    }
    Ok(())
}

impl<'a> OpPrinter<'a> {
    /// The operation being printed.
    pub fn op(&self) -> &dyn HighMlir {
        &**self.op.node
    }

    /// Number of operands.
    pub fn num_operands(&self) -> usize {
        self.op.operands.len()
    }

    /// SSA name of operand `i`, e.g. `%3`.
    pub fn operand(&self, i: usize) -> String {
        ref_name(self.op.operands[i])
    }

    /// Comma separated names of operands `range`.
    pub fn operands(&self, range: impl IntoIterator<Item = usize>) -> String {
        range.into_iter().map(|i| self.operand(i)).collect::<Vec<_>>().join(", ")
    }

    /// Type of operand `i`.
    pub fn operand_type(&self, i: usize) -> Ty {
        self.inputs[i].ty()
    }

    /// Types of operands `range`.
    pub fn operand_types(&self, range: impl IntoIterator<Item = usize>) -> Vec<Ty> {
        range.into_iter().map(|i| self.operand_type(i)).collect()
    }

    /// Number of results.
    pub fn num_results(&self) -> usize {
        self.op.num_results
    }

    /// Writes `%id = `, `%id:n = ` or nothing, depending on the result count.
    pub fn results_lhs(&mut self) -> fmt::Result {
        match self.op.num_results {
            0 => Ok(()),
            1 => write!(self.w, "%{} = ", self.op.id),
            n => write!(self.w, "%{}:{n} = ", self.op.id),
        }
    }

    /// SSA name of parameter `i` of region `r`.
    pub fn region_arg(&self, r: usize, i: usize) -> String {
        ref_name(Ref::Arg(self.op.regions[r].args[i].0))
    }

    /// Whether region `r` has no block.
    pub fn region_is_empty(&self, r: usize) -> bool {
        self.op.regions[r].empty
    }

    /// Writes region `r` as `{ ... }`; the entry block header is written only
    /// with `header` (and when there are arguments).
    pub fn region(&mut self, r: usize, header: bool) -> fmt::Result {
        let op = self.op;
        let region: &SRegion = &op.regions[r];
        self.w.write_str("{\n")?;
        if header && !region.args.is_empty() {
            let args: Vec<String> =
                region.args.iter().map(|(k, ty)| format!("{}: {ty}", ref_name(Ref::Arg(*k)))).collect();
            writeln!(self.w, "^bb0({}):", args.join(", "))?;
        }
        let mut w: &mut dyn Write = &mut *self.w;
        print_ops(&mut indented(&mut w).with_str("  "), &region.ops)?;
        self.w.write_str("}")
    }

    /// Writes `{name = value, ...}` for the given attributes (nothing if empty).
    pub fn attr_dict(&mut self, attrs: &[(String, crate::Attr)]) -> fmt::Result {
        if attrs.is_empty() {
            return Ok(());
        }
        self.w.write_str(" {")?;
        for (i, (name, value)) in attrs.iter().enumerate() {
            if i > 0 {
                self.w.write_str(", ")?;
            }
            if is_bare_identifier(name) {
                write!(self.w, "{name} = {value}")?;
            } else {
                write!(self.w, "{name:?} = {value}")?;
            }
        }
        self.w.write_str("}")
    }

    /// MLIR's generic operation form.
    pub fn generic(&mut self) -> fmt::Result {
        self.results_lhs()?;
        let operands = self.operands(0..self.num_operands());
        write!(self.w, "\"{}\"({operands})", self.op.node.name())?;
        if !self.op.regions.is_empty() {
            self.w.write_str(" (")?;
            for r in 0..self.op.regions.len() {
                if r > 0 {
                    self.w.write_str(", ")?;
                }
                self.region(r, true)?;
            }
            self.w.write_str(")")?;
        }
        let mut attrs = self.op.node.attributes();
        attrs.extend(annotations::op_attributes(&self.op.node));
        self.attr_dict(&attrs)?;
        let inputs = self.operand_types(0..self.num_operands());
        let results = self.op.node.result_types();
        write!(self.w, " : ({}) -> {}", TypeList(&inputs), ResultTypes(&results))
    }

    /// `%r = name %a, %b : type`, the common form of elementwise operations.
    pub fn same_type_form(&mut self) -> fmt::Result {
        self.results_lhs()?;
        let ty = self.op.node.result_types()[0].clone();
        write!(self.w, "{} {} : {ty}", self.op.node.name(), self.operands(0..self.num_operands()))
    }
}

impl Write for OpPrinter<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.w.write_str(s)
    }
}

/// Prints the IR.
impl fmt::Display for Root {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match lower(self.op(), true) {
            Ok(op) => print_ops(f, std::slice::from_ref(&op)),
            Err(e) => write!(f, "<error: {e}>"),
        }
    }
}

/// Prints the operation of this value, with its regions; operands defined
/// outside of it are printed as `%?`.
impl<T: ?Sized> fmt::Display for Val<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(node) = self.try_op() else {
            return f.write_str("<dropped>");
        };
        match lower(&node, false) {
            Ok(op) => print_ops(f, std::slice::from_ref(&op)),
            Err(e) => write!(f, "<error: {e}>"),
        }
    }
}
