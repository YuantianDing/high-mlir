mod common;

use common::{check_fn, op_names};
use high_mlir::{F32, FloatType, I32, I64, INDEX, IndexType, TensorType, Val, arith, sig, tensor};

fn float(v: Val) -> Val<FloatType> {
    v.downcast().unwrap()
}

#[test]
fn create_and_access() {
    let text = check_fn(|| {
        sig!(fn f(tensor![F32; ?, 4], INDEX) -> F32).define(|t, n| {
            let e = tensor::empty(tensor![F32; ?, 8], &n);
            let x = t.extract((0, 1));
            let t2 = t.insert(2.0, (&n, 3));
            let d = t2.dim(0) + tensor::rank(&e);
            let lit = tensor::from_elements((1.0, 2.0, 3.0, 4.0), tensor![F32; 2, 2]);
            let s = tensor::splat(float(x.clone()), tensor![F32; ?], &d);
            let s2 = tensor::splat(1.5, tensor![F32; 3], ());
            let c = tensor::cast(&t2, tensor![F32; 2, 4]);
            let back = tensor::cast(&c, tensor![F32; ?, ?]);
            let bits = tensor::bitcast(&lit, tensor![I32; 2, 2]);
            let _ = (s, s2, back, bits);
            (float(x) + float(lit.extract((1, 1)))).into()
        });
    });
    assert!(text.contains("tensor.empty(%arg1) : tensor<?x8xf32>"), "{text}");
    assert!(text.contains("tensor.insert %"), "{text}");
    assert!(text.contains("tensor.from_elements"), "{text}");
    assert!(text.contains("tensor.splat %"), "{text}");
    assert!(op_names(&text).contains(&"tensor.bitcast"), "{text}");
}

#[test]
fn slices_and_reshapes() {
    let text = check_fn(|| {
        sig!(fn f(tensor![F32; 8, 16], INDEX) -> tensor![F32; 8, 16]).define(|t, n| {
            let s = t.extract_slice((0, &n), (4, 8), (1, 1));
            let s2 = tensor::extract_slice(&t, (&n, 0), (&n, 16), (2, 1));
            let _ = s2;
            let r = s.insert_slice_into(&t, (4, 0), (4, 8), (1, 2));
            let flat = r.collapse_shape(vec![vec![0, 1]]);
            let e = flat.expand_shape(vec![vec![0, 1, 2]], (2, 4, 16));
            assert!(tensor::try_expand_shape(&flat, vec![vec![0, 1]], (&n, 32)).is_err());
            let dynamic = tensor::cast(&flat, tensor![F32; ?]);
            let _ = tensor::expand_shape(&dynamic, vec![vec![0, 1]], (&n, 32));
            let shape = tensor::from_elements((128i64,), tensor![I64; 1]);
            let re = tensor::reshape(&e, &shape, tensor![F32; 128]);
            let cat = tensor::concat(0, [&re, &re]);
            let _: Val<TensorType> = cat;
            let p = s.pad((1, 0), (0, &n), 0.0);
            let _ = p;
            r.into()
        });
    });
    assert!(text.contains("tensor.extract_slice %arg0[0, %arg1] [4, 8] [1, 1] : tensor<8x16xf32> to tensor<4x8xf32>"), "{text}");
    assert!(text.contains("tensor.insert_slice %"), "{text}");
    assert!(text.contains("tensor.collapse_shape %"), "{text}");
    assert!(text.contains("output_shape [2, 4, 16]"), "{text}");
    assert!(text.contains("tensor.concat dim(0)"), "{text}");
    assert!(text.contains("tensor.pad %"), "{text}");
}

#[test]
fn regions_gather_scatter() {
    let text = check_fn(|| {
        sig!(fn f(tensor![F32; 4, 5], tensor![INDEX; 3, 1], INDEX) -> tensor![F32; 4, 5]).define(|t, idx, n| {
            let g = tensor::generate(tensor![F32; ?, 3], &n, |ij: &[Val<IndexType>]| {
                arith::cast(&ij[0] + &ij[1], F32).into()
            });
            let _ = g;
            let p = tensor::try_pad_with(&t, (1, 1), (1, 1), true, |ij| arith::cast(&ij[0], F32).into()).unwrap();
            let _ = p;
            let gathered = tensor::gather(&t, &idx, [0], true);
            assert_eq!(gathered.ty().to_string(), "tensor<3x1x5xf32>");
            tensor::scatter(&gathered, &t, &idx, [0], true).into()
        });
    });
    assert!(text.contains("tensor.generate %arg2 {"), "{text}");
    assert!(text.contains("tensor.yield %"), "{text}");
    assert!(text.contains("gather_dims([0]) unique"), "{text}");
    assert!(text.contains("scatter_dims([0]) unique"), "{text}");
    assert!(text.contains("nofold"), "{text}");
}

#[test]
fn type_errors() {
    high_mlir::builtin::module(|| {
        sig!(fn f(tensor![F32; 4], INDEX)).define(|t, n| {
            assert!(tensor::try_empty(tensor![F32; ?], ()).is_err());
            assert!(tensor::try_extract(&t, (0, 0)).is_err());
            assert!(tensor::try_insert(&n, &t, 0).is_err());
            assert!(tensor::try_cast(&t, tensor![F32; 5]).is_err());
            assert!(tensor::try_from_elements((1.0,), tensor![F32; 2]).is_err());
            assert!(tensor::try_dim(&n, 0).is_err());
            ().into()
        });
    });
}
