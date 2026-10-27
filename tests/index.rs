mod common;

use common::check_fn;
use high_mlir::{I32, I64, INDEX, index, index::IndexCmpPredicate, sig};

#[test]
fn index_ops() {
    let text = check_fn(|| {
        sig!(fn f(INDEX, I32) -> (INDEX, I64)).define(|a, b| {
            let c = index::constant(42);
            let x = index::add(&a, &c);
            let x = index::mul(index::sub(&x, 1), 3);
            let ops = [
                index::divs(&x, 2), index::divu(&x, 2), index::ceildivs(&x, 2), index::ceildivu(&x, 2),
                index::floordivs(&x, 2), index::rems(&x, 2), index::remu(&x, 2), index::maxs(&x, &a),
                index::maxu(&x, &a), index::mins(&x, &a), index::minu(&x, &a), index::shl(&x, 1),
                index::shrs(&x, 1), index::shru(&x, 1), index::and(&x, 7), index::or(&x, 7), index::xor(&x, 7),
            ];
            let s = ops.into_iter().fold(index::sizeof(), index::add);
            let lt = index::cmp(IndexCmpPredicate::Slt, &s, &a);
            let t = index::bool_constant(true);
            let pick = (lt & t).select(&s, &a);
            let w = index::casts(&b, INDEX);
            let u = index::castu(&pick, I64);
            (index::add(&pick, &w), u).into()
        });
    });
    assert!(text.contains("index.constant 42"), "{text}");
    assert!(text.contains("index.cmp slt(%"), "{text}");
    assert!(text.contains("index.casts %arg1 : i32 to index"), "{text}");
    assert!(text.contains("index.bool.constant true"), "{text}");
}

#[test]
fn type_errors() {
    high_mlir::builtin::module(|| {
        sig!(fn f(I32)).define(|b| {
            assert!(index::try_add(&b, 1).is_err());
            assert!(index::try_casts(&b, I64).is_err());
            ().into()
        });
    });
}
