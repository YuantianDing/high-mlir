//! `builtin` dialect.

use std::fmt::{self, Write};

use crate::{
    BuiltRegion, Closure, ClosureBuilder, Edges, HighMlir, OpPrinter, Root, Ty, TypeError, Tuple,
    val::OrPanic,
};

/// `builtin.module` holding top-level operations such as functions.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Module {
    pub body: BuiltRegion,
}

/// `builtin.module` of the operations `body` builds (functions, globals,
/// ...), the outermost closure of an IR; fails if an operation uses a value
/// from outside of the module.
#[track_caller]
pub fn try_module(body: impl FnOnce()) -> Result<Root, TypeError> {
    let mut builder = ClosureBuilder::new(true);
    let body = builder.region(&[], |_| {
        body();
        Tuple::default()
    });
    builder.try_finish()?;
    Ok(Root::new(Module { body }))
}

/// `builtin.module` of the operations `body` builds (functions, globals,
/// ...), the outermost closure of an IR.
#[track_caller]
pub fn module(body: impl FnOnce()) -> Root {
    try_module(body).or_panic()
}

impl HighMlir for Module {
    fn name(&self) -> &str {
        "builtin.module"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.write_str("module ")?;
        p.region(0, false)
    }
}

impl Closure for Module {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        None
    }
}
