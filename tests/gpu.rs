mod common;

use common::{check, op_names};
use high_mlir::{
    F32, FloatType, I32, INDEX, Val, builtin, gpu,
    gpu::{Dimension::{X, Y}, ReduceOperation, ShuffleMode},
    memref, sig,
};

#[test]
fn kernels_and_launch_func() {
    let kernel = sig!(fn scale(memref![F32; ?], F32));
    let root = builtin::module(|| {
        gpu::module("kernels", || {
            kernel.define_kernel(|m, k| {
                let i = gpu::global_id(X);
                let _ = (gpu::thread_id(Y), gpu::block_id(X), gpu::block_dim(X), gpu::grid_dim(X), gpu::lane_id());
                let _ = (gpu::subgroup_id(), gpu::num_subgroups(), gpu::subgroup_size());
                let x: Val<FloatType> = m.load(&i).downcast().unwrap();
                let (y, _valid) = gpu::shuffle(ShuffleMode::Xor, &x, 1i32, 32i32);
                let s = gpu::subgroup_reduce(ReduceOperation::Add, &y, false);
                let t = gpu::all_reduce(ReduceOperation::MaximumF, &s, true);
                gpu::barrier();
                gpu::printf("x: %f\n", [&t]);
                let shared = gpu::dynamic_shared_memory();
                let _ = shared;
                m.store(t * &k, &i);
                ().into()
            });
            sig!(fn helper(F32) -> F32).define_gpu(false, |x| (&x + 1.0).into());
        });
        sig!(fn host(memref![F32; ?], INDEX)).define(|h, n| {
            let d = gpu::alloc(memref![F32; ?], &n, false);
            gpu::memcpy(&d, &h);
            gpu::memset(&d, Val::<FloatType>::from(0f32));
            gpu::launch_func("kernels", &kernel, (&n, 1, 1), (32, 1, 1), (&d, 2.0f32));
            gpu::wait();
            gpu::memcpy(&h, &d);
            gpu::dealloc(&d);
            ().into()
        });
    });
    gpu::container_module(&root);
    let text = check(&root);
    assert!(text.contains("gpu.module @kernels {"), "{text}");
    assert!(text.contains("gpu.func @scale(%arg0: memref<?xf32>, %arg1: f32) kernel {"), "{text}");
    assert!(text.contains("gpu.shuffle xor %"), "{text}");
    assert!(text.contains("gpu.subgroup_reduce add %"), "{text}");
    assert!(text.contains("gpu.launch_func @kernels::@scale blocks in ("), "{text}");
    for op in ["gpu.barrier", "gpu.printf", "gpu.all_reduce", "gpu.alloc", "gpu.memset", "gpu.wait", "gpu.dealloc", "gpu.return"] {
        assert!(op_names(&text).contains(&op), "{op}: {text}");
    }
}

#[test]
fn inline_launch() {
    let text = check(&builtin::module(|| {
        sig!(fn f(memref![I32; ?], INDEX)).define(|m, n| {
            gpu::launch((&n, 1, 1), (64, 1, 1), |ids| {
                let i = &ids.block_id[0] * &ids.block_size[0] + &ids.thread_id[0];
                m.store(7i32, &i);
            });
            ().into()
        });
    }));
    assert!(text.contains("gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %arg1,"), "{text}");
    assert!(text.contains("gpu.terminator"), "{text}");
}

#[test]
fn type_errors() {
    builtin::module(|| {
        sig!(fn f(memref![F32; 4], memref![I32; 4])).define(|a, b| {
            assert!(gpu::try_memcpy(&a, &b).is_err());
            assert!(gpu::try_alloc(memref![F32; ?], (), false).is_err());
            assert!(sig!(fn k() -> F32).try_define_gpu_with(true, || 1f32.into()).is_err());
            let _ = memref::alloc(memref![F32; 1], (), ());
            ().into()
        });
    });
}
