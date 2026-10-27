mod common;

use common::{check, op_names};
use high_mlir::{
    AffineMap, Attr, F32, FloatType, I8, I32, I64, INDEX, IntType, Layout, MemRefType, Ty, Val,
    builtin,
    memref,
    memref::{Access, AllocOpts, AtomicRMWKind},
    sig,
};

/// A loaded `f32`: loads are dynamically typed, by the element type.
fn float(v: Val) -> Val<FloatType> {
    v.downcast().unwrap()
}

fn int(v: Val) -> Val<IntType> {
    v.downcast().unwrap()
}

#[test]
fn allocation_and_access() {
    let root = builtin::module(|| {
        sig!(fn alloc(INDEX) -> F32).define(|n| {
            let a = memref::try_alloc(memref![F32; ?, 4], &n, AllocOpts { alignment: Some(64), ..Default::default() }).unwrap();
            let b = memref::alloca(memref![F32; 4], (), ());
            let access = Access { nontemporal: true, alignment: Some(16), invariant: false };
            memref::store(1.0, &a, (0, 1), access);
            b.store(2.0, 3);
            let x = memref::load(&a, (0, 1), Access { invariant: true, ..access });
            let grown = memref::realloc(memref::alloc(memref![I8; 16], (), ()), memref![I8; ?], Some(n.clone().into()), None);
            grown.store(0i8, 0);
            let c = memref::realloc(memref::alloc(memref![F32; 2], (), ()), memref![F32; 4], None, None);
            memref::copy(&b, &c);
            let y = c.load(0);
            a.dealloc();
            grown.dealloc();
            (float(x) + y + b.load(3)).into()
        });
    });
    let text = check(&root);
    assert!(text.contains("memref.alloc(%arg0) {alignment = 64 : i64} : memref<?x4xf32>"), "{text}");
    assert!(text.contains("{alignment = 16 : i64, nontemporal = true} : memref<?x4xf32>"), "{text}");
    assert!(text.contains("invariant = true"), "{text}");
    assert!(text.contains("memref.realloc %"), "{text}");
    assert!(text.contains("memref.copy %"), "{text}");
}

#[test]
fn views() {
    let root = builtin::module(|| {
        sig!(fn views(memref![F32; ?, ?], INDEX) -> INDEX).define(|m, n| {
            let sub = m.subview((&n, 0), (4, 4), (1, 2));
            sub.store(1.0, (0, 0));
            let cast = memref::cast(&sub, MemRefType::new([Some(4), None], F32).with_strides([None, None], None));
            let unranked = memref::cast(&m, memref![F32; *]);
            let back = memref::cast(&unranked, memref![F32; ?, ?]);
            let t = memref::transpose(&back, [1, 0]);
            let aligned = memref::assume_alignment(&t, 16);
            let space = memref::memory_space_cast(&m, Some(Attr::Int(1, I64.into())));
            let re = memref::reinterpret_cast(&m, 0, (&n, 4), (4, 1));
            let flat = memref::alloc(memref![F32; 2, 3, 4], (), ());
            let collapsed = memref::collapse_shape(&flat, vec![vec![0, 1], vec![2]]);
            let expanded = memref::expand_shape(&collapsed, vec![vec![0, 1], vec![2]], (2, 3, 4));
            let strided_collapse = memref::collapse_shape(&sub, vec![vec![0], vec![1]]);
            let bytes = memref::alloc(memref![I8; 64], (), ());
            let view = memref::view(&bytes, 0, &n, memref![F32; ?]);
            let shape = memref::alloca(memref![INDEX; 1], (), ());
            shape.store(&n, 0);
            let reshaped = memref::reshape(&view, &shape, memref![F32; ?]);
            let [d0, d1] = memref::try_distinct_objects([&flat, &bytes]).unwrap().try_into().unwrap();
            let meta = memref::extract_strided_metadata(&re);
            let rank = memref::rank(&unranked);
            let ptr = memref::extract_aligned_pointer_as_index(&m);
            let total = [
                memref::dim(&cast, 1),
                aligned.dim(0),
                memref::dim(&space, 0),
                expanded.dim(2),
                strided_collapse.dim(1),
                memref::dim(&reshaped, 0),
                memref::try_dim(&d0, 0).unwrap(),
                memref::try_dim(&d1, 0).unwrap(),
            ]
            .into_iter()
            .fold(rank, |acc, d| acc + d);
            let (_, offset, sizes, strides) = meta;
            (total + offset + &sizes[0] + &strides[1] + ptr).into()
        });
    });
    let text = check(&root);
    for s in [
        "memref.subview %arg0[%arg1, 0] [4, 4] [1, 2] : memref<?x?xf32> to memref<4x4xf32, strided<[?, 2], offset: ?>>",
        "memref.transpose %",
        "(d0, d1) -> (d1, d0) : memref<?x?xf32> to memref<?x?xf32, strided<[1, ?]>>",
        "memref.reinterpret_cast %arg0 to offset: [0], sizes: [%arg1, 4], strides: [4, 1]",
        "memref.collapse_shape %",
        "[[0, 1], [2]] : memref<2x3x4xf32> into memref<6x4xf32>",
        "output_shape [2, 3, 4] : memref<6x4xf32> into memref<2x3x4xf32>",
        "memref<4x4xf32, strided<[?, 2], offset: ?>> into memref<4x4xf32, strided<[?, 2], offset: ?>>",
        "memref.memory_space_cast %arg0 : memref<?x?xf32> to memref<?x?xf32, 1>",
        "memref.cast %arg0 : memref<?x?xf32> to memref<*xf32>",
        "memref.extract_strided_metadata %",
        "memref.distinct_objects %",
        "memref.reshape %",
    ] {
        assert!(text.contains(s), "missing `{s}`:\n{text}");
    }
}

#[test]
fn accesses_through_views_are_ordered() {
    let root = builtin::module(|| {
        // a store through a view stays between the accesses through the original
        sig!(fn alias(memref![F32; 8]) -> F32).define(|m| {
            m.store(1.0, 2);
            let sub = m.subview(2, 4, 1);
            let before = m.load(2);
            sub.store(5.0, 0);
            let after = m.load(2);
            (float(before) + after).into()
        });
    });
    let text = check(&root);
    let order: Vec<&str> =
        op_names(&text).into_iter().filter(|op| ["memref.store", "memref.load"].contains(op)).collect();
    assert_eq!(order, ["memref.store", "memref.load", "memref.store", "memref.load"], "{text}");
}

#[test]
fn atomics_prefetch_and_dma() {
    let root = builtin::module(|| {
        sig!(fn atomics(memref![F32; ?], memref![I32; 1], INDEX) -> F32).define(|m, tag, n| {
            let old = memref::try_atomic_rmw(AtomicRMWKind::AddF, 1.0, &m, 0).unwrap();
            let generic = memref::generic_atomic_rmw(&m, 1, |current| (float(current) * 2.0).into());
            memref::try_prefetch(&m, &n, false, 3, true).unwrap();
            let buf = memref::alloc(memref![F32; 16], (), ());
            memref::try_dma_start(&m, 0, &buf, 0, &n, &tag, 0, None).unwrap();
            memref::try_dma_wait(&tag, 0, &n).unwrap();
            let s: Val = high_mlir::arith::Constant::index(4).into();
            memref::try_dma_start(&buf, 0, &m, 0, 8, &tag, 0, Some((s.clone(), s))).unwrap();
            let after = m.load(0);
            (float(old) + generic + after).into()
        });
    });
    let text = check(&root);
    for s in [
        "memref.atomic_rmw addf %",
        ": (f32, memref<?xf32>) -> f32",
        "memref.generic_atomic_rmw %arg0[%",
        "memref.atomic_yield %",
        "memref.prefetch %arg0[%arg2], read, locality<3>, data : memref<?xf32>",
        "memref.dma_start %arg0[%",
        "memref.dma_wait %arg1[%",
    ] {
        assert!(text.contains(s), "missing `{s}`:\n{text}");
    }
    let names = op_names(&text);
    let pos = |name| names.iter().position(|n| *n == name).unwrap();
    assert!(pos("memref.dma_wait") < pos("memref.load"), "{text}");
}

#[test]
fn alloca_scope() {
    let root = builtin::module(|| {
        sig!(fn scoped(INDEX) -> F32).define(|n| {
            let r = memref::alloca_scope(|| {
                let tmp = memref::alloca(memref![F32; ?], &n, ());
                tmp.store(3.0, 0);
                tmp.load(0).into()
            });
            memref::alloca_scope(|| ().into());
            r
        });
    });
    let text = check(&root);
    assert!(text.contains("memref.alloca_scope -> (f32) {"), "{text}");
    assert!(text.contains("memref.alloca_scope.return %"), "{text}");
}

#[test]
fn globals() {
    let root = builtin::module(|| {
        let table = memref::global("table", memref![F32; 2, 2]).constant().floats([1.0, 2.0, 3.0, 4.0]).alignment(64).build();
        let zeros = memref::global("zeros", memref![I32; 4]).public().ints([0]).build();
        let scratch = memref::global("scratch", memref![I32; 4]).build();
        let external = memref::global("external", memref![I32; 4]).declaration().build();
        sig!(fn read() -> I32).define(|| {
            let t = memref::get_global(&table);
            let s = memref::get_global(&scratch);
            s.store(memref::get_global(&zeros).load(1), 0);
            let x = int(s.load(0)) + memref::get_global(&external).load(0);
            (high_mlir::arith::cast(t.load((0, 1)), I32) + x).into()
        });
    });
    let text = check(&root);
    let declaration = text.lines().find(|l| l.contains("@external :")).unwrap();
    assert!(!declaration.contains(" = "), "a declaration has no initial value:\n{text}");
    for s in [
        "memref.global \"private\" constant @table : memref<2x2xf32> = dense<[[1.0, 2.0], [3.0, 4.0]]> {alignment = 64 : i64}",
        "memref.global @zeros : memref<4xi32> = dense<0>",
        "memref.global \"private\" @scratch : memref<4xi32> = uninitialized",
        "memref.get_global @table : memref<2x2xf32>",
    ] {
        assert!(text.contains(s), "missing `{s}`:\n{text}");
    }
}

#[test]
fn layouts_and_symbols() {
    let root = builtin::module(|| {
        let map = AffineMap::new(1, 1, [high_mlir::AffineExpr::Dim(0) + high_mlir::AffineExpr::Symbol(0)]);
        let ty: Ty = MemRefType::new([Some(4)], F32).with_layout(Layout::Map(map)).into();
        sig!(fn shifted(INDEX) -> F32).define(|s| {
            let m = memref::try_alloc(ty.clone(), (), AllocOpts { symbols: vec![s.clone().into()], alignment: None }).unwrap();
            m.store(0.0, 0);
            m.load(1).into()
        });
    });
    let text = check(&root);
    assert!(text.contains("memref.alloc()[%arg0] : memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>>"), "{text}");
}

#[test]
fn type_errors() {
    builtin::module(|| {
        let m: Val = memref::alloc(memref![F32; 4], (), ()).into();
        assert!(memref::try_load(&m, (0, 0), ()).is_err());
        assert!(memref::try_store(1, &m, 0, ()).is_ok());
        assert!(memref::try_store(Val::<high_mlir::IntType>::from(1i32), &m, 0, ()).is_err());
        assert!(memref::try_alloc(memref![F32; ?], (), ()).is_err());
        assert!(memref::try_alloc(memref![F32; 4], (), AllocOpts { alignment: Some(3), ..Default::default() }).is_err());
        assert!(memref::try_cast(&m, memref![F32; 5]).is_err());
        assert!(memref::try_cast(&m, memref![F32; ?]).is_ok());
        assert!(memref::try_copy(&m, memref::alloc(memref![F32; 8], (), ())).is_err());
        assert!(memref::try_transpose(&m, [1]).is_err());
        assert!(memref::try_collapse_shape(&m, vec![vec![0], vec![1]]).is_err());
        assert!(memref::try_atomic_rmw(AtomicRMWKind::AddI, 1.0, &m, 0).is_err());
        assert!(memref::try_view(&m, 0, (), memref![F32; 1]).is_err());
        assert!(memref::global("g", memref![F32; ?]).try_build().is_err());
    });
}

#[test]
fn value_methods() {
    let root = builtin::module(|| {
        sig!(fn methods(memref![F32; 4, 8], memref![I8; 64], INDEX) -> INDEX).define(|m, bytes, n| {
            let copy = memref::alloc(memref![F32; 4, 8], (), ());
            m.copy_to(&copy);
            let t = copy.transpose([1, 0]).assume_alignment(16);
            let flat = copy.collapse_shape(vec![vec![0, 1]]).expand_shape(vec![vec![0, 1]], (8, 4));
            let r = m.reinterpret_cast(0, (2, 16), (16, 1));
            let v = bytes.view(0, &n, memref![F32; ?]);
            let old = m.atomic_rmw(AtomicRMWKind::AddF, 1.0, (0, 0));
            m.generic_atomic_rmw((0, 1), |current| (float(current) + 1.0).into());
            m.prefetch((1, 1), true, 1, false);
            v.store(old, 0);
            let (_, offset, _, strides) = r.extract_strided_metadata();
            let ptr = flat.extract_aligned_pointer_as_index();
            let grown = memref::alloc(memref![F32; 4], (), ()).realloc(memref![F32; 8], None);
            grown.store(0.0, 7);
            memref::dim(m.cast(memref![F32; ?, 8]), 0);
            (t.dim(0) + offset + &strides[0] + ptr + m.rank() + grown.dim(0) + v.dim(0)).into()
        });
    });
    let text = check(&root);
    for op in [
        "memref.copy %arg0",
        "memref.transpose %",
        "memref.assume_alignment %",
        "memref.collapse_shape %",
        "memref.expand_shape %",
        "memref.reinterpret_cast %arg0",
        "memref.view %arg1",
        "memref.atomic_rmw addf",
        "memref.generic_atomic_rmw %arg0",
        "memref.prefetch %arg0",
        "memref.extract_strided_metadata %",
        "memref.extract_aligned_pointer_as_index %",
        "memref.realloc %",
        "memref.rank %arg0",
    ] {
        assert!(text.contains(op), "missing `{op}`:\n{text}");
    }
}
