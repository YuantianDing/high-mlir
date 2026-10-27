mod common;

use common::{check_fn, op_names};
use high_mlir::{
    AffineExpr::{self, Const, Dim, Symbol},
    AffineMap, F32, FloatType, INDEX, IntegerSet, Val, affine,
    affine::MapOperands,
    memref, memref::AtomicRMWKind,
    sig, vector,
};

fn float(v: Val) -> Val<FloatType> {
    v.downcast().unwrap()
}

#[test]
fn maps_and_accesses() {
    let text = check_fn(|| {
        sig!(fn f(memref![F32; ?, 8], INDEX) -> INDEX).define(|m, n| {
            let shifted = AffineMap::new(1, 1, [Dim(0) + Symbol(0)]);
            let a = affine::apply(shifted.clone(), &n, &n);
            let lo = affine::min(AffineMap::new(1, 0, [Dim(0), Const(4)]), &a, ());
            let hi = affine::max(AffineMap::new(0, 1, [Symbol(0) * Const(2), Const(1)]), (), &n);
            affine::for_(0, &n, 1, ()).then(|i| {
                let x = affine::load(&m, (&i, 3));
                let sub = MapOperands::new("test", AffineMap::new(2, 1, [Dim(0), Dim(1) - Symbol(0) + Const(7)]), (&i, 0), &n).unwrap();
                let y = affine::load_map(&m, sub.clone());
                affine::store(float(x) + float(y), &m, (&i, 0));
                affine::store_map(1.0, &m, sub.clone());
                let v = affine::vector_load(&m, sub.clone(), vector![F32; 4]);
                affine::vector_store(&v, &m, sub.clone());
                affine::prefetch(&m, sub, false, 3, true);
                ().into()
            });
            let parts = affine::delinearize_index(&lo, (&hi, 4));
            let lin = affine::linearize_index((&parts[0], &parts[1]), (&hi, 4));
            lin.into()
        });
    });
    assert!(text.contains("affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg1)[%arg1]"), "{text}");
    assert!(text.contains("affine.load %arg0[%arg2, %"), "{text}");
    assert!(text.contains("symbol(%arg1)"), "{text}");
    assert!(text.contains("affine.prefetch"), "{text}");
    assert!(text.contains("affine.delinearize_index"), "{text}");
    assert!(op_names(&text).contains(&"affine.linearize_index"), "{text}");
}

#[test]
fn loops_and_conditionals() {
    let text = check_fn(|| {
        sig!(fn f(memref![F32; ?], INDEX) -> F32).define(|m, n| {
            let total = affine::for_(0, &n, 2, (0f32,)).then(|i, acc| {
                let inner = affine::for_(&i, MapOperands::new("t", AffineMap::new(1, 0, [Dim(0) + Const(2), Dim(0) * Const(3)]), &i, ()).unwrap(), 1, (acc,))
                    .then(|j, acc| (acc + float(affine::load(&m, &j))).into());
                let set = IntegerSet::new(1, 1, [IntegerSet::ge0(Dim(0) - Const(1)), IntegerSet::ge0(Symbol(0) - Dim(0) - AffineExpr::from(2))]);
                let picked = affine::if_(set.clone(), &i, &n).then(|| inner.clone().into()).else_(|| 0f32.into());
                affine::if_(set, &i, &n).then(|| { affine::store(1.0, &m, &i); ().into() }).end();
                float(picked[0].clone()).into()
            });
            let sums = affine::parallel((0,), (&n,), [1], [AtomicRMWKind::AddF], |ivs| {
                affine::load(&m, &ivs[0]).into()
            });
            let _ = affine::parallel((0, 0), (&n, 8), [2, 4], [], |_| ().into());
            let _ = memref::alloc(memref![F32; 2], (), ());
            (total + float(sums[0].clone())).into()
        });
    });
    assert!(text.contains("affine.for %arg2 = 0 to affine_map<(d0) -> (d0)>(%arg1) step 2 iter_args("), "{text}");
    assert!(text.contains("to min affine_map<(d0) -> (d0 + 2, d0 * 3)>(%arg2)"), "{text}");
    assert!(text.contains("affine.if affine_set<"), "{text}");
    assert!(text.contains("reduce (\"addf\") -> (f32)"), "{text}");
    assert!(text.contains("step (2, 4)"), "{text}");
}

#[test]
fn type_errors() {
    high_mlir::builtin::module(|| {
        sig!(fn f(memref![F32; 4], INDEX)).define(|m, n| {
            assert!(affine::try_apply(AffineMap::new(1, 0, [Dim(0), Dim(0)]), &n, ()).is_err());
            assert!(affine::try_apply(AffineMap::identity(2), &n, ()).is_err());
            assert!(affine::try_load(&m, (0, 0)).is_err());
            assert!(affine::try_store(1i32, &m, &n).is_ok());
            assert!(affine::for_(0, 4, 1, ()).try_then(|_| 1f32.into()).is_err());
            ().into()
        });
    });
}
