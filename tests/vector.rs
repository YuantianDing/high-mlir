mod common;

use common::{check, op_names};
use high_mlir::{
    builtin,
    AffineExpr, AffineMap, F16, F32, FloatType, I1, I32, INDEX, Val, VectorType,
    arith::{Constant, FastMath},
    memref,
    memref::Access,
    sig, tensor,
    vector,
    vector::{CombiningKind as K, IteratorType, Punctuation, Transfer},
};

#[test]
fn construction_and_shape() {
    let root = builtin::module(|| {
        let v4 = vector![F32; 4];
        sig!(fn shapes(F32, vector![F32; 4], vector![F32; [4]], INDEX) -> F32).define(|x, v, s, n| {
            let b = vector::broadcast(&x, vector![F32; 2, 4]);
            let b2 = vector::broadcast(&v, vector![F32; 3, 4]);
            let bits = vector::bitcast(&v, vector![F16; 8]);
            let flat = vector::shape_cast(&b, vector![F32; 8]);
            let el = vector::from_elements((&x, 1.0, &x, 2.0), v4.clone());
            let first = vector::to_elements(&el)[0].clone();
            let row: Val<VectorType> = vector::extract(&b, 1).downcast().unwrap();
            let dyn_el = vector::extract(&b, (0, &n));
            let ins = vector::insert(&x, &v, 2);
            let ins_row = vector::insert(&row, &b, (&n,));
            let slice = vector::extract_strided_slice(&b2, [1, 0], [2, 4], [1, 1]);
            let ins_slice = vector::insert_strided_slice(&slice, &b2, [0, 0], [1, 1]);
            let fma = vector::fma(&v, &el, &ins);
            let sh = vector::shuffle(&v, &fma, [0, 5, 2, 7, 1]);
            let il = vector::interleave(&v, &v);
            let de = vector::deinterleave(&il);
            let t = vector::transpose(&b, [1, 0]);
            let ext = vector::scalable_extract(&s, 0, v4.clone());
            let sins = vector::scalable_insert(&v, &s, 4);
            let step = vector::step(vector![INDEX; 4]);
            let vs = vector::vscale();
            let total = [first, row.extract(3), dyn_el, ins_row.extract((1, 1)), ins_slice.extract((0, 0)), sh.extract(4)]
                .into_iter()
                .fold(x.clone(), |a, b| a + b);
            let total = total + de.1.extract(0) + t.extract((0, 0)) + ext.extract(0) + bits.extract(0).cast(F32);
            let total = total + flat.extract(7) + sins.extract(0) + vector::extract(step, 3).cast(F32) + vs.cast(F32);
            total.into()
        });
    });
    let text = check(&root);
    for s in [
        "vector.broadcast %arg0 : f32 to vector<2x4xf32>",
        "vector.bitcast %arg1 : vector<4xf32> to vector<8xf16>",
        "vector.shape_cast %",
        "vector.from_elements %arg0, %",
        ":4 = vector.to_elements %",
        "vector.extract %",
        "[0, %arg3] : f32 from vector<2x4xf32>",
        "vector.insert %arg0, %arg1 [2] : f32 into vector<4xf32>",
        "vector.extract_strided_slice %",
        "{offsets = [1, 0], sizes = [2, 4], strides = [1, 1]} : vector<3x4xf32> to vector<2x4xf32>",
        "vector.insert_strided_slice %",
        "vector.fma %",
        "[0, 5, 2, 7, 1] : vector<4xf32>, vector<4xf32>",
        "vector.interleave %arg1, %arg1 : vector<4xf32> -> vector<8xf32>",
        ":2 = vector.deinterleave %",
        "vector.transpose %",
        "vector.scalable.extract %arg2[0] : vector<4xf32> from vector<[4]xf32>",
        "vector.scalable.insert %arg1, %arg2[4] : vector<4xf32> into vector<[4]xf32>",
        "vector.step : vector<4xindex>",
        "vector.vscale",
    ] {
        assert!(text.contains(s), "missing `{s}`:\n{text}");
    }
}

#[test]
fn reductions_and_contractions() {
    let root = builtin::module(|| {
        sig!(fn reduce(vector![F32; 4, 8], vector![F32; 8, 4], vector![F32; 4, 4], vector![I32; 8]) -> F32).define(
            |a, b, c, i| {
                let sum = vector::reduction(K::Add, vector::extract(&a, 0), None, FastMath::FAST);
                let acc: Val = 1.0f32.into();
                let max = vector::reduction(K::MaximumF, vector::extract(&a, 1), Some(acc), ());
                let imax = vector::reduction(K::MaxSI, &i, None, ());
                let rows = vector::multi_reduction(K::Add, &a, vector::broadcast(0.0f32, vector![F32; 4]), [1]);
                let zero: Val = 0.0f32.into();
                let all = vector::multi_reduction(K::Mul, &c, &zero, [0, 1]);
                let init = vector::broadcast(0.0f32, vector![F32; 8]);
                let scanned = vector::scan(K::Add, &a, &init, 0, true);
                let outer = vector::outerproduct(vector::extract(&a, 0), vector::extract(&b, 0), None, K::Add);
                let axpy = vector::outerproduct(vector::extract(&a, 0), &sum, Some(vector::extract(&a, 1)), K::Mul);
                let (d0, d1, d2) = (AffineExpr::Dim(0), AffineExpr::Dim(1), AffineExpr::Dim(2));
                let maps = [
                    AffineMap::new(3, 0, [d0.clone(), d2.clone()]),
                    AffineMap::new(3, 0, [d2, d1.clone()]),
                    AffineMap::new(3, 0, [d0, d1]),
                ];
                let iterators = [IteratorType::Parallel, IteratorType::Parallel, IteratorType::Reduction];
                let mm = vector::contract(&a, &b, &c, maps, iterators, K::Add, FastMath::NNAN);
                let sum: Val<FloatType> = sum.downcast().unwrap();
                let total = &sum + max + imax.cast(F32) + vector::extract(rows, 0) + all + vector::extract(&scanned.1, 0);
                let total = total + scanned.0.extract((0, 0)) + outer.extract((0, 0)) + axpy.extract(0) + vector::extract(&mm, (0, 0));
                total.into()
            },
        );
    });
    let text = check(&root);
    for s in [
        "vector.reduction <add>, %",
        "fastmath<fast> : vector<8xf32> into f32",
        "vector.reduction <maximumf>, %",
        "vector.multi_reduction <add>, %arg0, %",
        "[1] : vector<4x8xf32> to vector<4xf32>",
        "[0, 1] : vector<4x4xf32> to f32",
        "{inclusive = true, reduction_dim = 0 : i64} : vector<4x8xf32>, vector<8xf32>",
        "{kind = #vector.kind<add>} : vector<8xf32>, vector<4xf32>",
        "{kind = #vector.kind<mul>} : vector<8xf32>, f32",
        "iterator_types = [\"parallel\", \"parallel\", \"reduction\"], kind = #vector.kind<add>, fastmath = #arith.fastmath<nnan>}",
    ] {
        assert!(text.contains(s), "missing `{s}`:\n{text}");
    }
}

#[test]
fn memory_accesses() {
    let root = builtin::module(|| {
        sig!(fn mem(memref![F32; ?, 8], memref![F32; 16], INDEX) -> vector![F32; 4]).define(|m, flat, n| {
            let mask = vector::create_mask(&n, vector![I1; 4]);
            let all = vector::constant_mask([2], vector![I1; 4]);
            let zeros = vector::broadcast(0.0f32, vector![F32; 4]);
            let a = vector::load(&m, (0, 0), vector![F32; 4], Access { alignment: Some(16), ..Default::default() });
            vector::store(&a, &m, (1, 0), ());
            let b = vector::maskedload(&flat, 0, &mask, &zeros, None);
            vector::maskedstore(&flat, 4, &all, &b, Some(4));
            let c = vector::expandload(&flat, 0, &mask, &zeros, None);
            vector::compressstore(&flat, 8, &mask, &c, None);
            let idx = vector::step(vector![INDEX; 4]);
            let d = vector::gather(&flat, 0, &idx, &all, &zeros, None);
            vector::scatter(&flat, 4, &idx, &mask, &d, None);
            let e = vector::transfer_read(&m, (&n, 0), 0.0, vector![F32; 4], ());
            let map = AffineMap::new(2, 0, [AffineExpr::Dim(0)]);
            let col = Transfer { permutation_map: Some(map), mask: Some(mask.clone().into()), in_bounds: Some(vec![false]) };
            let g = vector::transfer_read(&m, (0, 1), 0.0, vector![F32; 4], col);
            vector::transfer_write(&e, &m, (2, 0), Transfer { in_bounds: Some(vec![true]), ..().into() });
            let after = vector::load(&flat, 0, vector![F32; 4], ());
            (after + g).into()
        });
    });
    let text = check(&root);
    for s in [
        "vector.create_mask %arg2 : vector<4xi1>",
        "vector.constant_mask [2] : vector<4xi1>",
        "{alignment = 16 : i64} : memref<?x8xf32>, vector<4xf32>",
        "vector.store %",
        "vector.maskedload %arg1[%",
        "into vector<4xf32>",
        "vector.maskedstore %arg1[%",
        "{alignment = 4 : i64}",
        "vector.expandload %",
        "vector.compressstore %",
        "vector.gather %arg1[%",
        "vector.scatter %arg1[%",
        "{in_bounds = [false], permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<?x8xf32>, vector<4xf32>",
        "permutation_map = affine_map<(d0, d1) -> (d0)>",
        "vector.transfer_write %",
        "{in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d1)>}",
    ] {
        assert!(text.contains(s), "missing `{s}`:\n{text}");
    }
    // effects on `flat` keep their order
    let order: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("%arg1["))
        .filter_map(|l| l.split_whitespace().find(|w| w.starts_with("vector.")))
        .collect();
    assert_eq!(
        order,
        [
            "vector.maskedload",
            "vector.maskedstore",
            "vector.expandload",
            "vector.compressstore",
            "vector.gather",
            "vector.scatter",
            "vector.load"
        ],
        "{text}"
    );
}

#[test]
fn tensors_and_type_cast() {
    let root = builtin::module(|| {
        sig!(fn t(tensor![F32; 8], vector![F32; 4], vector![INDEX; 4], vector![I1; 4]) -> tensor![F32; 8]).define(
            |t, v, idx, m| {
                let r = vector::transfer_read(&t, 0, 0.0, vector![F32; 4], ());
                let w = vector::transfer_write(&v + r, &t, 4, ()).unwrap();
                vector::scatter(&w, 0, &idx, &m, &v, None).unwrap().into()
            },
        );
    });
    let text = check(&root);
    let root = builtin::module(|| {
        assert!(text.contains(": vector<4xf32>, tensor<8xf32>"), "{text}");
        assert!(text.contains("-> tensor<8xf32>"), "{text}");

        sig!(fn tc() -> F32).define(|| {
            let m = memref::alloc(memref![F32; 4, 8], (), ());
            memref::store(1.0, &m, (0, 0), ());
            let as_vector = vector::type_cast(&m);
            let v = memref::load(&as_vector, (), ());
            vector::extract(v, (0, 0)).into()
        });
    });
    let text = check(&root);
    assert!(text.contains("vector.type_cast %0 : memref<4x8xf32> to memref<vector<4x8xf32>>"), "{text}");
    // the store through the original memref comes before the load through the cast
    let names = op_names(&text);
    assert!(names.iter().position(|n| *n == "memref.store") < names.iter().position(|n| *n == "memref.load"), "{text}");
}

#[test]
fn masks_and_printing() {
    let root = builtin::module(|| {
        sig!(fn masked(memref![F32; 8], vector![I1; 4], vector![F32; 4]) -> F32).define(|m, mask, v| {
            let zero: Val = 0.0f32.into();
            let i: Val = Constant::index(0).into();
            let [r] = vector::mask(&mask, None, || vector::reduction(K::Add, &v, None, ()).into()).try_into().unwrap();
            let [read] = vector::mask(&mask, Some(v.clone().into()), || vector::transfer_read(&m, &i, &zero, vector![F32; 4], ()).into())
                .try_into()
                .unwrap();
            vector::mask(&mask, None, || {
                vector::transfer_write(&read, &m, &i, ());
                ().into()
            });
            vector::print(&r, Punctuation::Comma);
            vector::print(&read, ());
            vector::print_str("done");
            vector::print_punctuation(Punctuation::Open);
            r.into()
        });
    });
    let text = check(&root);
    for s in [
        "vector.mask %arg1 {",
        "vector.yield %",
        "} : vector<4xi1> -> f32",
        "vector.mask %arg1, %arg2 {",
        "} : vector<4xi1> -> vector<4xf32>",
        "vector.print %",
        " : f32 punctuation <comma>",
        "vector.print str \"done\"",
        "vector.print punctuation <open>",
    ] {
        assert!(text.contains(s), "missing `{s}`:\n{text}");
    }
    let no_results = |l: &&str| l.trim_start().starts_with("} : vector<4xi1>") && !l.contains("->");
    assert!(text.lines().any(|l| no_results(&l)), "a mask without results:\n{text}");
    let prints: Vec<&str> = text.lines().filter(|l| l.contains("vector.print")).collect();
    assert_eq!(prints.len(), 4, "prints are kept, in order:\n{text}");
    assert!(prints[2].contains("str") && prints[3].contains("open"), "{text}");
}

#[test]
fn prints_in_loops_are_kept() {
    let root = builtin::module(|| {
        sig!(fn count(INDEX)).define(|n| {
            high_mlir::scf! {for 0 to n step 1}.then(|i| {
                vector::print(&i, ());
                ().into()
            });
            ().into()
        });
    });
    let text = check(&root);
    assert!(text.contains("scf.for") && text.contains("vector.print %arg1 : index"), "{text}");
}

#[test]
fn type_errors() {
    builtin::module(|| {
        let v: Val = vector::broadcast(0.0f32, vector![F32; 4]).into();
        let x: Val = 1.0f32.into();
        assert!(vector::try_broadcast(&v, vector![F32; 8]).is_err());
        assert!(vector::try_bitcast(&v, vector![F16; 4]).is_err());
        assert!(vector::try_extract(&v, (0, 0)).is_err());
        assert!(vector::try_insert(&v, &v, 0).is_err());
        assert!(vector::try_shuffle(&v, &v, [9]).is_err());
        assert!(vector::try_transpose(&v, [1]).is_err());
        assert!(vector::try_reduction(K::MaxSI, &v, None, ()).is_err());
        assert!(vector::try_from_elements([&x], vector![F32; 4]).is_err());
        assert!(vector::try_constant_mask([5], vector![I1; 4]).is_err());
        assert!(vector::try_step(vector![F32; 4]).is_err());
        assert!(vector::try_fma(&v, &x, &v).is_err());
        let ty: VectorType = (*vector![F32; 4]).clone();
        assert_eq!(ty.dims(), [4]);
    });
}

#[test]
fn value_methods() {
    let root = builtin::module(|| {
        sig!(fn methods(memref![F32; 4, 8], vector![F32; 8], vector![F32; [4]], F32) -> F32).define(|m, v, s, x| {
            let read = m.transfer_read((0, 0), 0.0, vector![F32; 8], ());
            let loaded = m.load_vector((1, 0), vector![F32; 8]);
            let sum = &read + &loaded;
            m.transfer_write(&sum, (2, 0), ());
            sum.store(&m, (3, 0));
            let scaled = sum.fma(&v, &read);
            let wide = scaled.shape_cast(vector![F32; 2, 4]).transpose([1, 0]).shape_cast(vector![F32; 8]);
            let (even, odd) = wide.deinterleave();
            let both = even.interleave(&odd).shuffle(&v, [0, 9]).insert(&x, 1);
            let (scan, total) = v.scan(K::Add, vector::broadcast(&x, vector![F32]), 0, true);
            let piece = v.extract_strided_slice([2], [4], [1]).insert_strided_slice(s.scalable_extract(0, vector![F32; 4]), [0], [1]);
            let outer = piece.outerproduct(&piece, None, K::Add);
            let reduced = outer.multi_reduction(K::Add, vector::broadcast(&x, vector![F32; 4]), [1]);
            let bits = both.bitcast(vector![I32; 2]).cast(vector![F32; 2]);
            let elems = bits.to_elements();
            let as_vector = m.type_cast();
            let whole = memref::load(&as_vector, (), ());
            (x + scan.reduction(K::Add, None, ()) + vector::extract(&total, ()) + vector::extract(&reduced, 0) + &elems[1]
                + vector::extract(&whole, (0, 0)) + piece.scalable_insert(&s, 0).extract(0))
            .into()
        });
    });
    let text = check(&root);
    for op in [
        "vector.transfer_read %arg0",
        "vector.load %arg0",
        "vector.transfer_write %",
        "vector.store %",
        "vector.fma %",
        "vector.deinterleave %",
        "vector.interleave %",
        "vector.shuffle %",
        "vector.scan <add>",
        "vector.outerproduct %",
        "vector.multi_reduction <add>",
        "vector.bitcast %",
        "arith.sitofp %",
        "vector.to_elements %",
        "vector.type_cast %arg0",
        "vector.scalable.insert %",
    ] {
        assert!(text.contains(op), "missing `{op}`:\n{text}");
    }
}
