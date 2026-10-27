//! Its own test binary: `ADD_RUST_LOC` is global, and the other tests
//! compare printed text without locations.

mod common;

use std::sync::{Arc, atomic::Ordering};

use common::check;
use high_mlir::{
    Attr, F32, I32, INDEX, IntType, Val, arith, builtin,
    annotations::{ADD_RUST_LOC, Location},
    memref, scf, sig,
};

fn file() -> &'static str {
    file!()
}

#[test]
fn rust_locations() {
    ADD_RUST_LOC.store(true, Ordering::Relaxed);
    let (mut add_line, mut alloc_line, mut for_line) = (0, 0, 0);
    let module = builtin::module(|| {
        sig!(fn located(I32, INDEX) -> F32).define(|x, n| {
            let y = &x + 1;
            add_line = line!() - 1;
            let m = memref::alloc(memref![F32; ?], &n, ());
            alloc_line = line!() - 1;
            let sum = scf! {for 0 to n step 1 iter_args(0f32)}.then(|i, acc| (acc + m.load(i)).into());
            for_line = line!() - 1;
            (sum + arith::sitofp(y, F32)).into()
        });
    });
    let text = check(&module);
    for line in [add_line, alloc_line, for_line] {
        let loc = format!("loc(\"{}\":{line}:", file());
        assert!(text.contains(&loc), "missing `{loc}`:\n{text}");
    }
    // literal operands are located at the operator too
    let constant = text.lines().find(|l| l.contains("arith.constant 1 : i32")).unwrap();
    assert!(constant.contains(&format!(":{add_line}:")), "{text}");

    // the melior operations carry the locations
    #[cfg(feature = "melior")]
    {
        use melior::ir::operation::{OperationLike, OperationPrintingFlags};
        let ctx = high_mlir::context();
        let built = module.to_melior(&ctx).unwrap();
        let flags = OperationPrintingFlags::new().enable_debug_info(true, false);
        let debug = built.as_operation().to_string_with_flags(flags).unwrap();
        assert!(debug.contains(&format!("\"{}\":{add_line}:", file())), "{debug}");
    }
    ADD_RUST_LOC.store(false, Ordering::Relaxed);
}

#[test]
fn explicit_annotations() {
    let root = builtin::module(|| {
        sig!(fn tagged(I32) -> I32).define(|x| {
            let y = &x * 3;
            y.annotate("my.tag", Attr::str("triple")).set_location(Location::Name("triple".into()));
            y.annotate("my.count", Attr::Int(3, I32.into())).annotate("my.tag", Attr::str("thrice"));
            (y + 1).into()
        });
    });
    let text = check(&root);
    assert!(
        text.contains("\"arith.muli\"(%arg0, %0) {my.tag = \"thrice\", my.count = 3 : i32} : (i32, i32) -> i32 loc(\"triple\")"),
        "{text}"
    );
    assert!(text.contains("arith.addi %"), "unannotated operations keep their custom form:\n{text}");
}

#[test]
fn entries_are_removed_with_their_operation() {
    let mut address = 0;
    let root = builtin::module(|| {
        let x: Val<IntType> = 1.into();
        let tagged = &x + 2;
        tagged.annotate("my.tag", Attr::Unit).set_location(Location::Name("old".into()));
        assert_eq!(tagged.annotations().len(), 1);
        address = Arc::as_ptr(&tagged.op()) as *const () as usize;
    });
    // the module owns its operations
    drop(root);
    // an operation reusing the address does not inherit them (it may get a
    // Rust location: `rust_locations` sets `ADD_RUST_LOC` concurrently)
    for _ in 0..100 {
        let mut reused = false;
        builtin::module(|| {
            let x: Val<IntType> = 1.into();
            for _ in 0..100 {
                let fresh = &x + 2;
                if Arc::as_ptr(&fresh.op()) as *const () as usize == address {
                    assert!(fresh.annotations().is_empty());
                    assert_ne!(fresh.location(), Some(Location::Name("old".into())));
                    reused = true;
                    return;
                }
            }
        });
        if reused {
            return;
        }
    }
}
