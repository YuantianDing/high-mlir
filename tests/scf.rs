mod common;

use common::{check, op_names};
use high_mlir::{
    Attr, F32, FloatType, I1, I32, I64, INDEX, IntType, MemRefType, Ty, Val, arith, builtin, func::Signature, memref,
    scf,
    sig,
};

/// A loaded `i32`: loads are dynamically typed, by the element type.
fn int(v: Val) -> Val<IntType> {
    v.downcast().unwrap()
}

#[test]
fn unsigned_for() {
    let root = builtin::module(|| {
        sig!(fn count(I32) -> I32).define(|n| {
            let r = scf! {for 0 to n step 1 iter_args(0i32)}.unsigned().then(|i, acc| (acc + i).into());
            r.into()
        });
    });
    let text = check(&root);
    assert!(text.contains("scf.for unsigned %arg1 = %"), "{text}");
    assert!(text.contains(" : i32 {"), "{text}");
}

#[test]
fn parallel_reductions() {
    let root = builtin::module(|| {
        sig!(fn sums(memref![F32; ?], INDEX) -> (F32, I32)).define(|m, n| {
            let r = scf! {parallel (0) to (n) step (1) init(0f32, 0i32)}.then_reduce(
                |i| (m.load(&i), i.cast(I32)).into(),
                // the reductions have different types: combine dynamically typed values
                |k, a, b| if k == 0 { (a.downcast::<FloatType>().unwrap() + b).into() } else { int(a).max(b).into() },
            );
            r.into()
        });
    });
    let text = check(&root);
    for s in [
        ":2 = scf.parallel (%arg2) = (%",
        "init (%",
        ") -> (f32, i32) {",
        "scf.reduce(%",
        ": f32, i32) {\n      ^bb0(%arg3: f32, %arg4: f32):",
        "scf.reduce.return %",
        "arith.maxsi %",
    ] {
        assert!(text.contains(s), "missing `{s}`:\n{text}");
    }
}

#[test]
fn while_loops() {
    let root = builtin::module(|| {
        // collatz steps of n
        sig!(fn collatz(I64) -> I64).define(|n| {
            let r = scf! {while (n, 0i64)}.cond(|x, steps| (x.gt(1), x, steps).into()).body(|x, steps| {
                let odd = (&x & 1).equal(1);
                let [next] = scf! {if odd}.then(|| (&x * 3 + 1).into()).else_(|| (&x / 2).into()).try_into().unwrap();
                (next, steps + 1).into()
            });
            r.1.into()
        });
    });
    let text = check(&root);
    let root = builtin::module(|| {
        assert!(text.contains(":2 = scf.while (%arg1 = %arg0, %arg2 = %"), "{text}");
        assert!(text.contains(": (i64, i64) -> (i64, i64) {"), "{text}");
        assert!(text.contains("scf.condition(%"), "{text}");
        assert!(text.contains("} do {\n    ^bb0(%arg3: i64, %arg4: i64):"), "{text}");

        // forwarding fewer values than the loop carries, and no values at all
        sig!(fn first(I32, memref![I32; 1]) -> I32).define(|n, m| {
            let r = scf! {while (n, 1i32)}
                .cond(|x, y| (x.lt(&y), x + y).into())
                .body_dyn(|args| (args[0].clone(), args[0].clone()).into());
            // a loop without values is kept for its effects
            scf! {while ()}.cond(|| int(m.load(0)).lt(10).into()).body(|| {
                m.store(int(m.load(0)) + 1, 0);
                ().into()
            });
            r
        });
    });
    let text = check(&root);
    assert!(text.contains("scf.while : () -> () {"), "{text}");
}

#[test]
fn execute_region_and_switch() {
    let root = builtin::module(|| {
        sig!(fn pick(INDEX, F32) -> F32).define(|i, x| {
            let doubled = scf::execute_region(false, || (&x * 2.0).into());
            let kept = scf::try_execute_region(true, || x.clone().into()).unwrap();
            let s = scf::index_switch(&i)
                .case(0, || doubled.clone())
                .case(3, || kept.clone())
                .default(|| (-&x).into());
            scf::index_switch(2).case(1, || ().into()).default(|| ().into());
            s
        });
    });
    let text = check(&root);
    for s in [
        "scf.execute_region -> (f32) {",
        "scf.execute_region -> (f32) no_inline {",
        "scf.index_switch %arg0 -> f32\n    case 0 {",
        "\n    case 3 {",
        "\n    default {",
    ] {
        assert!(text.contains(s), "missing `{s}`:\n{text}");
    }
}

#[test]
fn forall() {
    let root = builtin::module(|| {
        sig!(fn fill(memref![F32; ?, 8], INDEX)).define(|m, n| {
            scf! {forall (n, 8)}.then(|i, j| {
                m.store((&i + &j).cast(F32), (&i, &j));
                ().into()
            });
            scf! {forall (0, 0) to (n, 8) step (2, 4)}
                .mapping(Attr::Raw("[#gpu.thread<y>, #gpu.thread<x>]".into()))
                .then(|i, j| {
                    m.store(0.0, (&i, &j));
                    ().into()
                });
            ().into()
        });
    });
    let text = check(&root);
    assert!(text.contains("scf.forall (%arg2, %arg3) in (%arg1, 8) {"), "{text}");
    assert!(text.contains("scf.forall (%arg4, %arg5) = (0, 0) to (%arg1, 8) step (2, 4) {"), "{text}");
    assert!(text.contains("} {mapping = [#gpu.thread<y>, #gpu.thread<x>]}"), "{text}");
    assert_eq!(text.matches("scf.forall.in_parallel").count(), 2, "{text}");
}

#[test]
fn build_order_across_loops() {
    let root = builtin::module(|| {
        sig!(fn ordered(memref![F32; 4], I1) -> F32).define(|m, c| {
            m.store(1.0, 0);
            let r = scf! {while (0.0f32)}.cond(|x| (c.clone(), x).into()).body(|x| {
                m.store(&x, 1);
                (x + 1.0).into()
            });
            scf::execute_region(false, || {
                m.store(&r, 2);
                ().into()
            });
            scf::index_switch(1)
                .case(1, || {
                    m.store(3.0, 3);
                    ().into()
                })
                .default(|| ().into());
            m.load(0).into()
        });
    });
    let text = check(&root);
    let order: Vec<&str> = op_names(&text)
        .into_iter()
        .filter(|op| ["memref.store", "memref.load", "scf.while", "scf.execute_region", "scf.index_switch"].contains(op))
        .collect();
    assert_eq!(
        order,
        [
            "memref.store",
            "scf.while",
            "memref.store",
            "scf.execute_region",
            "memref.store",
            "scf.index_switch",
            "memref.store",
            "memref.load"
        ],
        "{text}"
    );
}

#[test]
fn calls_in_ifs_keep_build_order() {
    let root = builtin::module(|| {
        // Pools produced by calls in `scf.if`s, as in a size-staged enumerator:
        // `s` and `a` alternate, each call reads the previous pool of its kind,
        // and each else branch yields an empty view at a cursor advanced by the
        // sizes of the pools before it. Every operation is emitted where it is
        // built, so each load, `if` and call stays in step.
        let pool = || Ty::new(MemRefType::new([None], I64).with_strides([Some(1)], None)).erase();
        let next = Signature::new("next", [memref![I64; ?].erase(), pool()], [pool()]);
        next.declare();
        Signature::new("f", [memref![I64; ?].erase()], [INDEX.into()]).define(|args| {
            let arena = &args[0];
            let zero: Val = arith::Constant::index(0).into();
            let view = |at: &Val| -> Val { memref::reinterpret_cast(arena, (at,), (&zero,), (1,)).into() };
            let pick = |input: &Val, at: &Val| -> Val {
                let empty = view(at);
                let searching = int(memref::load(arena, 0, ())).equal(0);
                let [r] = scf! {if searching}.then(|| next.call((arena, input))).else_(|| empty.into()).try_into().unwrap();
                r
            };
            let mut cursor = zero.clone();
            let empty = view(&cursor);
            let (mut s, mut a) = (empty.clone(), empty);
            for _ in 0..3 {
                for pool in [&mut s, &mut a] {
                    *pool = pick(pool, &cursor);
                    cursor = arith::addi(&cursor, memref::dim(&*pool, 0), ());
                }
            }
            cursor.into()
        });
    });
    let text = check(&root);
    let order: Vec<&str> =
        op_names(&text).into_iter().filter(|op| ["memref.load", "scf.if", "func.call"].contains(op)).collect();
    let step = ["memref.load", "scf.if", "func.call"];
    assert_eq!(order, step.repeat(6), "{text}");
}

#[test]
fn type_errors() {
    builtin::module(|| {
        let i: Val<IntType> = 1.into();
        let f: Val = 1.0f32.into();
        assert!(scf! {while (&i)}.cond(|x| (x.clone(), x).into()).try_body(|x| x.into()).is_err(), "condition is not i1");
        let err = scf! {while (&i)}.cond(|x| (x.lt(0), x).into()).try_body(|_| f.clone().into()).unwrap_err();
        assert_eq!(err.to_string(), "type error in `scf.while`: the body yields (f32) but the loop carries (i32)");
        let err = scf::index_switch(0).case(1, || i.clone().into()).try_default(|| f.clone().into()).unwrap_err();
        assert!(err.to_string().contains("case 1 yields (i32) but default yields (f32)"), "{err}");
        assert!(scf::index_switch(0).case(1, || ().into()).case(1, || ().into()).try_default(|| ().into()).is_err());
        let err = scf! {parallel (0) to (8) step (1) init(0f32)}.try_then(|_| ().into()).unwrap_err();
        assert!(err.to_string().contains("then_reduce"), "{err}");
        let err =
            scf! {parallel (0) to (8) step (1) init(0f32)}.try_then_reduce(|_| i.clone().into(), |_, a, _| a).unwrap_err();
        assert!(err.to_string().contains("yields (i32) but the reductions are (f32)"), "{err}");
        assert!(scf! {forall (8)}.try_then(|_| i.clone().into()).is_err());
    });
}
