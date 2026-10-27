mod common;

use common::check_fn;
use high_mlir::{F32, I1, INDEX, Val, bufferization, bufferization::AllocTensorOpts, memref, sig, tensor};

#[test]
fn bufferization_ops() {
    let text = check_fn(|| {
        sig!(fn f(memref![F32; 4, ?], tensor![F32; 4, ?], INDEX, I1) -> tensor![F32; 4, ?]).define(|m, t, n, c| {
            let a = bufferization::alloc_tensor(tensor![F32; ?, 8], &n, ());
            let copy = bufferization::alloc_tensor(tensor![F32; 4, ?], (), AllocTensorOpts { copy: Some(t.clone().into()), ..Default::default() });
            let _ = (a, copy.clone());
            let from = bufferization::to_tensor(&m, true, false);
            let buf = bufferization::to_buffer(&from, true);
            let cl = bufferization::clone(&buf);
            let out = bufferization::materialize_in_destination(&t, &copy, false, false).unwrap();
            assert!(bufferization::materialize_in_destination(&out, &m, true, true).is_none());
            let tmp = memref::alloc(memref![F32; 4], (), ());
            let kept = bufferization::dealloc([&tmp], [&c], [&cl]);
            assert_eq!(kept.len(), 1);
            bufferization::dealloc([&cl], [Val::from(&kept[0])], Vec::<Val>::new());
            bufferization::dealloc_tensor(tensor::empty(tensor![F32; 2], ()));
            out.into()
        });
    });
    assert!(text.contains("bufferization.alloc_tensor(%arg2) : tensor<?x8xf32>"), "{text}");
    assert!(text.contains("copy(%arg1)"), "{text}");
    assert!(text.contains("bufferization.to_tensor %arg0 restrict : memref<4x?xf32> to tensor<4x?xf32>"), "{text}");
    assert!(text.contains("read_only"), "{text}");
    assert!(text.contains("retain ("), "{text}");
}

#[test]
fn type_errors() {
    high_mlir::builtin::module(|| {
        sig!(fn f(memref![F32; 4], tensor![F32; 4])).define(|m, t| {
            assert!(bufferization::try_alloc_tensor(tensor![F32; ?], (), ()).is_err());
            assert!(bufferization::try_to_tensor(&t, false, false).is_err());
            assert!(bufferization::try_to_buffer(&m, false).is_err());
            assert!(bufferization::try_materialize_in_destination(&t, &m, false, false).is_err());
            ().into()
        });
    });
}
