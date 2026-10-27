//! Conversion to melior IR.

use std::collections::HashMap;

use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{
        Attribute, Block, BlockLike, Identifier, Location, Module, Operation, Region, RegionLike,
        Type, Value, ValueLike, operation::OperationBuilder,
    },
    utility::register_all_dialects,
};
use mlir_sys::MlirValue;

use crate::{
    Root, annotations,
    builtin::Module as ModuleOp,
    lower::{Error, Ref, SOp, SRegion, lower},
};

/// A melior context with every upstream dialect loaded.
pub fn context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context
}

/// SSA values built so far. Values are kept raw: they live as long as the
/// operations being built, which outlive this conversion.
#[derive(Default)]
struct Env {
    results: HashMap<usize, Vec<MlirValue>>,
    args: HashMap<usize, MlirValue>,
}

impl Env {
    fn get<'c, 'a>(&self, r: Ref) -> Value<'c, 'a> {
        let raw = match r {
            Ref::Result { id, port, .. } => self.results[&id][port as usize],
            Ref::Arg(k) => self.args[&k],
            Ref::Undefined => unreachable!("conversion lowers strictly"),
        };
        // SAFETY: `raw` was produced by an operation or block of the IR under
        // construction, which is never freed while the conversion runs.
        unsafe { Value::from_raw(raw) }
    }
}

fn parse_type<'c>(context: &'c Context, ty: &crate::Ty) -> Result<Type<'c>, Error> {
    let text = ty.to_string();
    Type::parse(context, &text).ok_or(Error::Parse(text))
}

fn build<'c>(context: &'c Context, op: &SOp, env: &mut Env) -> Result<Operation<'c>, Error> {
    let node = &op.node;
    let location = match annotations::op_location(node) {
        Some(annotations::Location::FileLineCol { file, line, column }) => {
            Location::new(context, &file, line as usize, column as usize)
        }
        Some(annotations::Location::Name(name)) => Location::name(context, &name, Location::unknown(context)),
        Some(annotations::Location::Unknown) | None => Location::unknown(context),
    };
    let operands: Vec<Value> = op.operands.iter().map(|&r| env.get(r)).collect();
    let results = node
        .result_types()
        .iter()
        .map(|ty| parse_type(context, ty))
        .collect::<Result<Vec<_>, _>>()?;
    let attributes = node
        .attributes()
        .into_iter()
        .chain(annotations::op_attributes(node))
        .map(|(name, value)| {
            let text = value.to_string();
            let attr = Attribute::parse(context, &text).ok_or(Error::Parse(text))?;
            Ok((Identifier::new(context, &name), attr))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let regions = op
        .regions
        .iter()
        .map(|r| build_region(context, r, env))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OperationBuilder::new(node.name(), location)
        .add_operands(&operands)
        .add_results(&results)
        .add_attributes(&attributes)
        .add_regions_vec(regions)
        .build()?)
}

fn build_region<'c>(context: &'c Context, region: &SRegion, env: &mut Env) -> Result<Region<'c>, Error> {
    let result = Region::new();
    if region.empty {
        return Ok(result);
    }
    let location = Location::unknown(context);
    let args = region
        .args
        .iter()
        .map(|(_, ty)| Ok((parse_type(context, ty)?, location)))
        .collect::<Result<Vec<_>, Error>>()?;
    let block = Block::new(&args);
    for (i, (k, _)) in region.args.iter().enumerate() {
        env.args.insert(*k, block.argument(i)?.to_raw());
    }
    for op in &region.ops {
        let built = build(context, op, env)?;
        let op_ref = block.append_operation(built);
        let values = (0..op.num_results)
            .map(|i| Ok(op_ref.result(i)?.to_raw()))
            .collect::<Result<Vec<_>, Error>>()?;
        if !values.is_empty() {
            env.results.insert(op.id, values);
        }
    }
    result.append_block(block);
    Ok(result)
}

impl Root {
    /// Builds the melior operation of the IR.
    pub fn to_melior_operation<'c>(&self, context: &'c Context) -> Result<Operation<'c>, Error> {
        build(context, &lower(self.op(), true)?, &mut Env::default())
    }

    /// Builds the melior module of a `builtin.module`.
    pub fn to_melior<'c>(&self, context: &'c Context) -> Result<Module<'c>, Error> {
        let op = self.to_melior_operation(context)?;
        let is_module = (&***self.op() as &dyn std::any::Any).is::<ModuleOp>();
        Ok(Module::from_operation(op).filter(|_| is_module).expect("the root of an IR is a `builtin.module`"))
    }
}
