use high_mlir::{F32, I1, INDEX, builtin, memref, scf, sig};
use melior::ir::operation::OperationLike;

fn main() {
    // Fill `m[i] = i` in parallel, overwrite m[0] if `c`, then sum m[0..n].
    let module = builtin::module(|| {
        sig!(fn fill_and_sum(memref![F32; ?], INDEX, I1) -> F32).define(|m, n, c| {
            scf! {parallel (0) to (n) step (1)}.then(|i| {
                m.store(i.cast(F32), &i);
                ().into()
            });
            scf! {if c}
                .then(|| {
                    m.store(-1.0, 0);
                    ().into()
                })
                .end();
            let sum =
                scf! {for 0 to n step 1 iter_args(0f32)}.then(|i, acc| (acc + m.load(i)).into());
            sum.into()
        });
    });
    print!("{module}");
    let ctx = high_mlir::context();
    let built = module.to_melior(&ctx).unwrap();
    println!("verifies: {}", built.as_operation().verify());
}
