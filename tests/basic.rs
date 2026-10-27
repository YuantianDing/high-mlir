use std::panic::AssertUnwindSafe;

mod common;

use common::check;
use high_mlir::{F32, I1, I32, I64, INDEX, IntType, Val, arith, builtin, func::Signature, memref, scf, sig};

/// The exact-text tests compare text without the default Rust locations.
fn without_locations() {
    high_mlir::annotations::ADD_RUST_LOC.store(false, std::sync::atomic::Ordering::Relaxed);
}

#[test]
fn arith_expression() {
    without_locations();
    let root = builtin::module(|| {
        let a: Val<IntType> = 3.into();
        let b = (&a + 2) * &a - 1;
        let c = b.lt(10).select(&b, 0);
        assert_eq!(c.to_string(), "%0 = arith.select %?, %?, %? : i32\n", "a value prints its own operation");
    });
    assert_eq!(
        root.to_string(),
        "module {
  %0 = arith.constant 3 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.addi %0, %1 : i32
  %3 = arith.muli %2, %0 : i32
  %4 = arith.constant 1 : i32
  %5 = arith.subi %3, %4 : i32
  %6 = arith.constant 10 : i32
  %7 = arith.cmpi slt, %5, %6 : i32
  %8 = arith.constant 0 : i32
  %9 = arith.select %7, %5, %8 : i32
}
"
    );
}

#[test]
fn types() {
    builtin::module(|| {
        assert_eq!(memref![F32; ?, 4].to_string(), "memref<?x4xf32>");
        assert_eq!(memref![I1].to_string(), "memref<i1>");
        assert_eq!(high_mlir::tensor![I32; 2, 3].to_string(), "tensor<2x3xi32>");
        assert_eq!(sig!(fn f(I32, F32) -> (F32, I1)).ty().to_string(), "(i32, f32) -> (f32, i1)");
        assert_eq!(sig!(fn f()).ty().to_string(), "() -> ()");
        let ty: high_mlir::Ty = I64.into();
        assert!(ty == I64 && ty != I32);
    });
}

#[test]
fn functions_and_calls() {
    let root = builtin::module(|| {
        without_locations();
        let add = sig!(fn add(I32, F32) -> F32);
        add.define(|x, y| (x.cast(F32) + y).into());
        sig!(fn main() -> F32).define(|| (-add.call((1, 2.5)) * 2.0).into());
    });
    let text = check(&root);
    assert_eq!(
        text,
        "module {
  func.func @add(%arg0: i32, %arg1: f32) -> f32 {
    %0 = arith.sitofp %arg0 : i32 to f32
    %1 = arith.addf %0, %arg1 : f32
    func.return %1 : f32
  }
  func.func @main() -> f32 {
    %2 = arith.constant 1 : i32
    %3 = arith.constant 2.5 : f32
    %4 = func.call @add(%2, %3) : (i32, f32) -> f32
    %5 = arith.negf %4 : f32
    %6 = arith.constant 2.0 : f32
    %7 = arith.mulf %5, %6 : f32
    func.return %7 : f32
  }
}
"
    );
}

#[test]
fn operations_are_emitted_in_build_order() {
    let root = builtin::module(|| {
        let tick = sig!(fn tick(I64) -> I64);
        let hash = sig!(fn hash(I64) -> I64);
        tick.declare();
        hash.declare();
        sig!(fn f(I64) -> I64).define(|x| {
            // unused results are kept
            tick.call(&x);
            hash.call(&x);
            let first = tick.call(1i64);
            let second = tick.call(2i64);
            // used only in a branch: the call stays where it is built
            let late = tick.call(3i64);
            let [r] = scf! {if first.lt(&second)}.then(|| late.clone().into()).else_(|| hash.call(&x).into()).try_into().unwrap();
            r.into()
        });
    });
    let text = check(&root);
    let calls: Vec<&str> = text.lines().filter(|l| l.contains("func.call")).map(str::trim).collect();
    assert_eq!(calls.len(), 6, "{text}");
    assert!(calls[0].contains("@tick(%arg0)") && calls[1].contains("@hash(%arg0)"), "{text}");
    for (call, n) in calls[2..5].iter().zip(1..) {
        assert!(call.contains("@tick(") && text.contains(&format!("arith.constant {n} : i64")), "{text}");
    }
    let branch = text.find("scf.if").unwrap();
    assert!(text.rfind("func.call @hash(").unwrap() > branch, "{text}");
    assert!(text.rfind("func.call @tick(").unwrap() < branch, "{text}");
}

#[test]
fn multiple_results() {
    let root = builtin::module(|| {
        sig!(fn swap(I32, F32) -> (F32, I32)).define(|a, b| (b, a).into());
    });
    let text = check(&root);
    assert!(text.contains("func.return %arg1, %arg0 : f32, i32"), "{text}");
}

#[test]
fn loops_and_branches() {
    let root = builtin::module(|| {
        // sum of i for i in 0..n, with odd i doubled
        sig!(fn sum(INDEX) -> I64).define(|n| {
            let r = scf! {for 0 to n step 1 iter_args(0i64)}.then(|iv, acc| {
                let i = iv.cast(I64);
                let [term] = scf! {if (&i & 1).equal(1)}.then(|| (&i * 2).into()).else_(|| i.clone().into()).try_into().unwrap();
                (acc + term).into()
            });
            r.into()
        });
    });
    let text = check(&root);
    assert!(text.contains("scf.for %arg1 = %"), "{text}");
    assert!(text.contains("iter_args(%arg2 = %"), "{text}");
    assert!(text.contains("scf.if %"), "{text}");
}

#[test]
fn loop_with_several_iter_args() {
    let root = builtin::module(|| {
        // fibonacci
        sig!(fn fib(INDEX) -> I64).define(|n| {
            let (a, _) = scf! {for 0 to n step 1 iter_args(0i64, 1i64)}.then(|_, a, b| (b.clone(), a + b).into());
            a.into()
        });
    });
    let text = check(&root);
    assert!(text.contains(":2 = scf.for") && text.contains("scf.yield %arg3, %4 : i64, i64"), "{text}");
}

#[test]
fn stores_and_loads_keep_build_order() {
    let root = builtin::module(|| {
        sig!(fn swap() -> (F32, F32)).define(|| {
            let m = memref::alloc(memref![F32; 2], (), ());
            m.store(1.0, 0);
            m.store(2.0, 1);
            let a = m.load(0);
            let b = m.load(1);
            // swap: the stores must stay after both loads
            m.store(&b, 0);
            m.store(&a, 1);
            (m.load(0), a).into()
        });
    });
    let text = check(&root);
    let order: Vec<&str> = text
        .lines()
        .filter_map(|l| ["memref.store", "memref.load"].into_iter().find(|op| l.contains(op)))
        .collect();
    assert_eq!(
        order,
        ["memref.store", "memref.store", "memref.load", "memref.load", "memref.store", "memref.store", "memref.load"],
        "{text}"
    );
}

#[test]
fn closures_keep_build_order() {
    let root = builtin::module(|| {
        // fill a memref argument in parallel, then conditionally overwrite one element
        sig!(fn fill(memref![F32; ?, ?], INDEX, I1)).define(|m, n, c| {
            scf! {parallel (0, 0) to (n, n) step (1, 1)}.then(|i, j| {
                m.store((&i + &j).cast(F32), (&i, &j));
                ().into()
            });
            scf! {if c}
                .then(|| {
                    m.store(-1.0, (0, 0));
                    ().into()
                })
                .end();
            ().into()
        });
    });
    let text = check(&root);
    let parallel = text.find("scf.parallel").expect(&text);
    let branch = text.find("scf.if").expect(&text);
    assert!(parallel < branch, "{text}");
    assert_eq!(text.matches("memref.store").count(), 2, "{text}");
}

#[test]
fn external_declarations_and_recursion() {
    let root = builtin::module(|| {
        let ext = sig!(fn ext(I64) -> I64);
        let fact = sig!(fn fact(I64) -> I64);
        ext.declare();
        fact.define(|n| {
            let [r] = scf! {if n.le(1)}.then(|| 1i64.into()).else_(|| (&n * fact.call(&n - 1)).into()).try_into().unwrap();
            ext.call(r).into()
        });
    });
    let text = check(&root);
    assert!(text.contains("func.func private @ext(i64) -> i64"), "{text}");
}

#[test]
fn dynamic_arity() {
    let root = builtin::module(|| {
        Signature::new("first", [I32.into(), I32.into()], [I32.into()]).define(|args| args[0].clone().into());
    });
    check(&root);
}

#[test]
fn type_errors() {
    builtin::module(|| {
        let i: Val<IntType> = 1.into();
        let f: Val = 1.0f32.into();
        let err = arith::try_add(&i, &f).unwrap_err();
        assert_eq!(err.to_string(), "type error in `arith.addi`: operand types differ: `i32` and `f32`");
        assert!(arith::try_and(&f, &f).is_err());
        assert!(arith::try_add(1, 2).is_err());
        assert!(scf! {if &i}.then(|| ().into()).try_end().is_err());
        let err = scf! {if i.lt(0)}.then(|| i.clone().into()).try_else_(|| f.clone().into()).unwrap_err();
        assert_eq!(err.to_string(), "type error in `scf.if`: then yields (i32) but else yields (f32)");
        let err = scf! {for 0 to 10 step 1 iter_args(0i64)}.try_then(|_, _| 0.into()).unwrap_err();
        assert_eq!(err.to_string(), "type error in `scf.for`: body yields (i32) but iter_args are (i64)");
        let g = sig!(fn g(I32));
        assert!(g.try_call(1.5).is_err());
        assert!(g.try_define(|x| x.into()).is_err());
        assert!(memref![F32; 4].clone() != memref![F32; ?]);

        let panic = std::panic::catch_unwind(AssertUnwindSafe(|| &i + &f)).unwrap_err();
        assert!(panic.downcast_ref::<String>().unwrap().contains("operand types differ"));
    });
}

#[test]
fn isolated_functions_reject_captures() {
    builtin::module(|| {
        let outer: Val = 1.into();
        let define = || sig!(fn f() -> I32).define(|| outer.clone().into());
        assert!(std::panic::catch_unwind(AssertUnwindSafe(define)).is_err());
    });
}

#[test]
fn operations_need_an_open_region() {
    let message = |r: std::thread::Result<()>| r.unwrap_err().downcast_ref::<String>().cloned().unwrap();
    let outside = std::panic::catch_unwind(|| {
        let _: Val<IntType> = 1.into();
    });
    assert!(message(outside).contains("built outside of a module"));

    // a value of a finished module cannot be used in another one
    let mut stale = None;
    let root = builtin::module(|| stale = Some(Val::<IntType>::from(1)));
    let stale = stale.unwrap();
    let finished = std::panic::catch_unwind(AssertUnwindSafe(|| {
        builtin::module(|| {
            let _ = &stale + 1;
        });
    }));
    assert!(message(finished).contains("not visible here"));

    // the module owns its operations
    drop(root);
    assert!(stale.try_op().is_none());
    assert!(stale.to_string().contains("<dropped>"));
}

/// A dialect type, whose operations are added to its values by an extension trait.
#[derive(Debug, PartialEq, Eq, Hash)]
struct Money;

impl std::fmt::Display for Money {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("!bank.money")
    }
}

impl high_mlir::MlirType for Money {}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Deposit {
    lhs: Val,
    rhs: Val,
}

impl high_mlir::HighMlir for Deposit {
    fn name(&self) -> &str {
        "bank.deposit"
    }

    fn inputs(&self) -> high_mlir::Edges<'_> {
        Box::new([self.lhs.clone(), self.rhs.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<high_mlir::Ty> {
        vec![Money.into()]
    }
}

trait Bank {
    fn deposit(&self, other: &Val<Money>) -> Val<Money>;
}

impl Bank for Val<Money> {
    fn deposit(&self, other: &Val<Money>) -> Val<Money> {
        let node = high_mlir::Node::build(Deposit { lhs: self.into(), rhs: other.into() });
        Val::new(&node.unwrap(), 0).downcast().unwrap()
    }
}

#[test]
fn operations_by_value_type() {
    let root = builtin::module(|| {
        sig!(fn pay(Money, Money) -> Money).define(|a, b| a.deposit(&b).into());
    });
    let text = root.to_string();
    let root = builtin::module(|| {
        assert!(text.contains("\"bank.deposit\"(%arg0, %arg1) : (!bank.money, !bank.money) -> !bank.money"), "{text}");

        // numbers give numbers, tensors give tensors
        sig!(fn tensors(high_mlir::tensor![F32; 4], high_mlir::tensor![F32; 4]) -> high_mlir::tensor![F32; 4])
            .define(|x, y| ((&x + &y) * 2.0).into());
    });
    let text = check(&root);
    assert!(text.contains("arith.addf %arg0, %arg1 : tensor<4xf32>"), "{text}");
    assert!(text.contains("arith.constant dense<2.0> : tensor<4xf32>"), "{text}");
}

#[test]
fn values_are_typed_by_their_mlir_type() {
    use high_mlir::{IndexType, IntType, MemRefType, TensorType, Ty, arith::Constant};

    let root = builtin::module(|| {
        let a: Val<IntType> = 3.into();
        let b: Val<IntType> = (&a + 2) * &a; // number op number is a number
        let c: Val<IntType> = -b.max(1);
        assert_eq!(c.static_ty().0, 32);
        let erased: Val = c.clone().into();
        assert_eq!(erased.downcast::<IntType>(), Some(c));
        assert!(erased.downcast::<TensorType>().is_none());

        sig!(fn t(high_mlir::tensor![F32; 4]) -> high_mlir::tensor![F32; 4]).define(|x| {
            let y: Val<TensorType> = &x + &x; // tensor op tensor is a tensor
            assert_eq!(y.static_ty().shape, [Some(4)]);
            y.into()
        });
    });
    check(&root);
    let root = builtin::module(|| {

        sig!(fn m(INDEX) -> F32).define(|n| {
            let m: Val<MemRefType> = memref::alloc(memref![F32; ?], &n, ());
            let d: Val<IndexType> = m.dim(0);
            let i: Val<IndexType> = d - Constant::index(1);
            m.store(1.0, &i);
            m.load(i).into()
        });
    });
    let text = check(&root);
    assert!(text.contains("memref.dim %"), "{text}");
    let ty: Ty = memref![F32; ?].into();
    assert!(ty.as_memref().is_some());
}

#[test]
fn indirect_calls() {
    let text = common::check_fn(|| {
        let twice = high_mlir::sig!(fn twice(high_mlir::I32) -> high_mlir::I32);
        twice.define(|x| (&x + &x).into());
        high_mlir::sig!(fn main(high_mlir::I32) -> high_mlir::I32).define(|x| {
            let f = twice.constant();
            let r = high_mlir::func::call_indirect(&f, (&x,));
            r[0].clone().into()
        });
    });
    assert!(text.contains("func.constant @twice : (i32) -> i32"), "{text}");
    assert!(text.contains("func.call_indirect %1(%arg1) : (i32) -> i32"), "{text}");
}
