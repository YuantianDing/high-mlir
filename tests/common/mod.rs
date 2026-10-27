#![allow(dead_code)]

use high_mlir::{Root, builtin, context};
use melior::ir::{Module as MeliorModule, operation::OperationLike};

/// Prints `module`, checks that MLIR parses and verifies the text, that the
/// melior conversion verifies, and that both give the same IR.
pub fn check(module: &Root) -> String {
    let text = module.to_string();
    let ctx = context();
    let parsed = MeliorModule::parse(&ctx, &text).unwrap_or_else(|| panic!("MLIR rejected:\n{text}"));
    assert!(parsed.as_operation().verify(), "parsed module does not verify:\n{text}");
    let built = module.to_melior(&ctx).unwrap_or_else(|e| panic!("conversion failed: {e}\n{text}"));
    assert!(built.as_operation().verify(), "built module does not verify:\n{text}");
    assert_eq!(parsed.as_operation().to_string(), built.as_operation().to_string());
    text
}

/// Checks the module of what `body` builds and returns the printed text.
pub fn check_fn(body: impl FnOnce()) -> String {
    check(&builtin::module(body))
}

/// The operation names of the printed lines, in order.
pub fn op_names(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|l| {
            let l = l.trim_start();
            let l = l.split_once(" = ").map_or(l, |(_, r)| r);
            let name = l.split([' ', '(']).next()?;
            name.contains('.').then_some(name)
        })
        .collect()
}
