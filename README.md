# high-mlir

[![crates.io](https://img.shields.io/crates/v/high-mlir.svg)](https://crates.io/crates/high-mlir)
[![docs.rs](https://img.shields.io/docsrs/high-mlir)](https://docs.rs/high-mlir)
[![license](https://img.shields.io/crates/l/high-mlir.svg)](#license)

Build [MLIR](https://mlir.llvm.org) as typed Rust expressions.

Every operation is an expression whose value has a static MLIR type
(`Val<IntType>`, `Val<MemRefType>`, …), so `x + 1` on an `i32` builds
`arith.addi` and gives back an `i32` value. Operands are type-checked as
the IR is built, operations are emitted in the order they are built, and
regions (modules, functions, loop bodies) are Rust closures. The result prints as textual MLIR, or converts to
[melior](https://crates.io/crates/melior) IR, ready for MLIR's passes.

```rust
use high_mlir::{F32, INDEX, builtin, memref, scf, sig};
use high_mlir::melior::ir::operation::OperationLike;

// sum(m, n) = m[0] + .. + m[n - 1]
let module = builtin::module(|| {
    sig!(fn sum(memref![F32; ?], INDEX) -> F32).define(|m, n| {
        let total = scf! {for 0 to n step 1 iter_args(0f32)}.then(|i, acc| (acc + m.load(i)).into());
        total.into()
    });
});
println!("{module}");

let context = high_mlir::context();
let ir = module.to_melior(&context).unwrap();
assert!(ir.as_operation().verify());
```

prints

```text
module {
  func.func @sum(%arg0: memref<?xf32>, %arg1: index) -> f32 {
    %0 = arith.constant 0 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 0.0 : f32
    %5 = scf.for %arg2 = %0 to %arg1 step %1 iter_args(%arg3 = %2) -> (f32) {
      %3 = memref.load %arg0[%arg2] : memref<?xf32>
      %4 = arith.addf %arg3, %3 : f32
      scf.yield %4 : f32
    }
    func.return %5 : f32
  }
}
```

(with a `loc(...)` per operation pointing at the Rust code that built it;
see [Source locations](#annotations-and-source-locations)).

## Requirements

`high-mlir` builds on [`mlir-sys`](https://crates.io/crates/mlir-sys), which
links against an **LLVM/MLIR 23** installation. Point it to the installation
prefix (the directory with `bin/llvm-config`):

```sh
export MLIR_SYS_230_PREFIX=/path/to/llvm
export TABLEGEN_230_PREFIX=/path/to/llvm
```

or set them for a project in `.cargo/config.toml`:

```toml
[env]
MLIR_SYS_230_PREFIX = "/path/to/llvm"
TABLEGEN_230_PREFIX = "/path/to/llvm"
```

The minimum supported Rust version is 1.88.

## Dialects

| Dialect   | Coverage |
|-----------|----------|
| `arith`   | every operation, including the overflow, `exact`, `nneg`, fast-math and rounding-mode flags |
| `memref`  | every operation: allocation, loads and stores, views (`subview`, `reinterpret_cast`, `expand_shape`, …), globals, atomics, DMA |
| `scf`     | `for`, `if`, `while`, `parallel` with reductions, `forall`, `index_switch`, `execute_region` |
| `vector`  | every operation: construction and shape, reductions and contractions, masked/gather/scatter/transfer accesses, `mask`, `print` |
| `func`    | `func.func` (with external declarations), `func.call`, `func.return` |
| `builtin` | `builtin.module` |

Operations are built by functions named after them, with operands first
and optional attributes last (`()` for none):

```rust
use high_mlir::{F32, I64, Val, FloatType, arith, arith::{FastMath, Overflow}, builtin};

builtin::module(|| {
    let a: Val<FloatType> = 1.5f32.into();
    let fast = arith::addf(&a, 2.0, FastMath::FAST);   // arith.addf .. fastmath<fast>
    let wide = arith::extf(&fast, high_mlir::F64, ());  // a Val<FloatType>
    let n = arith::fptosi(&wide, I64);                  // a Val<IntType>
    let doubled = arith::muli(&n, 2, Overflow::NSW);    // arith.muli .. overflow<nsw>
    assert!(doubled.to_string().contains("arith.muli"));
});
```

`scf` regions are closures, written with the `scf!` macro; `vector` and
`memref` operations are also methods of vector and memref values.

## Concepts

- **Typed values.** `Val<T>` is an output of an operation, of MLIR type `T`.
  The default `Val = Val<dyn MlirType>` is dynamically typed. Operators and
  type-specific methods exist for the concrete types: arithmetic on
  integers, `index`, floats, tensors and vectors, `load`/`subview`/… on
  memrefs, `extract`/`shuffle`/… on vectors. `sig!` gives function bodies
  typed parameters, and loops type their induction variables and results.
- **Results by count.** Functions return one `Val<T>`, a Rust tuple for a
  fixed number of results, a `Tuple` for a variable number, or `()`.
- **Regions.** A thread-local stack tracks the closures being built:
  `builtin::module(|| ..)` is the outermost one, and function and loop
  bodies push their own. Every operation built while a region is open is
  appended to it, in build order, used or not, so side effects keep the
  order of the Rust code. Regions own their operations; a `Val` is a weak
  reference to its operation plus a result number. Building an operation
  outside of any module, or using a value of a region that is no longer
  open (or across a function boundary), is an error.
- **Extending.** A dialect defines operations by implementing `HighMlir`
  (and `Closure` for regions), and types by implementing `MlirType`.

## Annotations and source locations

Any operation can carry discardable attributes and a location:

```rust
use high_mlir::{Attr, Val, IntType, annotations::Location, builtin};

builtin::module(|| {
    let x: Val<IntType> = 1.into();
    let y = &x + 2;
    y.annotate("my.tag", Attr::str("sum")).set_location(Location::Name("sum".into()));
    assert!(y.to_string().contains("my.tag = \"sum\""));
});
```

With `annotations::ADD_RUST_LOC` set (the default), every operation records
the Rust source location that built it, and prints it as `loc(...)`.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
