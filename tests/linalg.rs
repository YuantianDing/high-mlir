mod common;

use common::{check_fn, op_names};
use high_mlir::{
    AffineExpr::Dim, AffineMap, F32, F64, FloatType, I32, INDEX, Val, arith, linalg,
    linalg::{ConvOpts, IteratorType::{Parallel, Reduction}},
    memref, sig, tensor,
};

fn float(v: &Val) -> Val<FloatType> {
    v.downcast().unwrap()
}

#[test]
fn named_ops_on_tensors() {
    let text = check_fn(|| {
        sig!(fn f(tensor![F32; 4, 8], tensor![F32; 8, 16], tensor![I32; 4, 8]) -> tensor![F32; 4, 16]).define(|a, b, i| {
            let init = tensor::empty(tensor![F32; 4, 16], ());
            let zero = linalg::fill(0.0, &init);
            let mm = linalg::matmul(&a, &b, &zero[0]);
            let e = tensor::empty(tensor![F32; 4, 8], ());
            let s = linalg::add(&a, &a, &e);
            let s = linalg::exp(&s[0], &e);
            let s = linalg::max(&s[0], &a, &e);
            let s = linalg::powf(&s[0], &a, &e);
            let cast = linalg::copy(&i, &e);
            let _ = linalg::select(tensor::empty(tensor![high_mlir::I1; 4, 8], ()), &s[0], &cast[0], &e);
            let t = linalg::transpose(&a, tensor::empty(tensor![F32; 8, 4], ()), [1, 0]);
            let row = tensor::empty(tensor![F32; 8], ());
            let bc = linalg::broadcast(&row, &e, [0]);
            let _ = (t, bc);
            let v = linalg::matvec(&a, tensor::empty(tensor![F32; 8], ()), tensor::empty(tensor![F32; 4], ()));
            let d = linalg::dot(&row, &row, tensor::empty(tensor![F32], ()));
            let sm = linalg::softmax(&a, &e, 1);
            let _ = (v, d, sm);
            mm[0].clone().into()
        });
    });
    assert!(text.contains("linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>) outs(%"), "{text}");
    assert!(text.contains("linalg.fill ins(%"), "{text}");
    assert!(text.contains("permutation = [1, 0]"), "{text}");
    assert!(text.contains("dimensions = [0]"), "{text}");
    assert!(text.contains("linalg.softmax dimension(1)"), "{text}");
    for op in ["linalg.add", "linalg.exp", "linalg.max", "linalg.powf", "linalg.copy", "linalg.select", "linalg.matvec", "linalg.dot"] {
        assert!(op_names(&text).contains(&op), "{op}: {text}");
    }
}

#[test]
fn named_ops_on_memrefs() {
    let text = check_fn(|| {
        sig!(fn f(memref![F32; 1, 8, 8, 3], memref![F32; 3, 3, 3, 4], memref![F32; 1, 6, 6, 4])).define(|x, w, y| {
            assert!(linalg::conv_2d_nhwc_hwcf(&x, &w, &y, ()).is_empty());
            let opts = ConvOpts { strides: Some(vec![1, 1]), dilations: Some(vec![1, 1]) };
            linalg::conv_2d_nhwc_hwcf(&x, &w, &y, opts);
            let window = memref::alloc(memref![F32; 3, 3], (), ());
            linalg::pooling_nhwc_max(&x, &window, memref::alloc(memref![F32; 1, 6, 6, 3], (), ()), ());
            let m = memref::alloc(memref![F64; 2, 2], (), ());
            linalg::matmul(&m, &m, &m);
            linalg::batch_matmul(memref::alloc(memref![F32; 2, 3, 4], (), ()), memref::alloc(memref![F32; 2, 4, 5], (), ()), memref::alloc(memref![F32; 2, 3, 5], (), ()));
            ().into()
        });
    });
    assert!(text.contains("linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins("), "{text}");
    assert!(text.contains("linalg.pooling_nhwc_max ins("), "{text}");
}

#[test]
fn generic_map_reduce() {
    let text = check_fn(|| {
        sig!(fn f(tensor![F32; 4, 8], tensor![F32; 4]) -> tensor![F32; 4]).define(|a, acc| {
            let maps = [AffineMap::identity(2), AffineMap::new(2, 0, [Dim(0)])];
            let r = linalg::generic([&a], [&acc], maps, [Parallel, Reduction], |p| {
                let i = linalg::index(1);
                let x = float(&p[0]) * arith::cast(&i, F32);
                (x + float(&p[1])).into()
            });
            let m = linalg::map([&a, &a], tensor::empty(tensor![F32; 4, 8], ()), |p| (float(&p[0]) * float(&p[1]) + float(&p[2])).into());
            let red = linalg::reduce([&m[0]], [&r[0]], [1], |p| (float(&p[0]) + float(&p[1])).into());
            let _ = INDEX;
            red[0].clone().into()
        });
    });
    assert!(text.contains("linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = [\"parallel\", \"reduction\"]} ins(%arg0 : tensor<4x8xf32>) outs(%arg1 : tensor<4xf32>) {"), "{text}");
    assert!(text.contains("linalg.index 1 : index"), "{text}");
    assert!(text.contains("linalg.yield %"), "{text}");
    assert!(text.contains("\"linalg.map\""), "{text}");
    assert!(text.contains("\"linalg.reduce\""), "{text}");
}

#[test]
fn type_errors() {
    high_mlir::builtin::module(|| {
        sig!(fn f(tensor![F32; 4, 8], tensor![F32; 4])).define(|a, v| {
            assert!(linalg::try_matmul(&a, &v, &a).is_err());
            assert!(linalg::try_add(&a, &v, &a).is_err());
            assert!(linalg::try_transpose(&a, &a, [1, 0]).is_err());
            assert!(linalg::try_generic([&a], [&v], [AffineMap::identity(2)], [Parallel, Parallel], |_| ().into()).is_err());
            assert!(linalg::try_reduce([&a], [&v], [0], |p| p[0].clone().into()).is_err());
            assert!(linalg::try_conv_2d_nhwc_hwcf(&a, &a, &a, ()).is_err());
            ().into()
        });
    });
}
