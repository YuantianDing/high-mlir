# Changelog

All notable changes to this crate are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the crate follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First release.

- Expression-based IR: operations are values typed by their MLIR type
  (`Val<T>`), and regions are Rust closures. A thread-local stack tracks the
  regions being built; each operation is appended to the innermost open
  region, in build order, and owned by it, while a `Val` holds a weak
  reference to its operation and its result number.
- `builtin::module(|| ..)` is the outermost region; building an operation
  outside of a module is an error, as is using a value of a region that is
  not open.
- The `arith`, `memref`, `scf` and `vector` dialects (every operation of
  MLIR 23), plus `func` and `builtin`.
- Printing as textual MLIR and conversion to melior IR.
- Annotations: discardable attributes and source locations, including the
  Rust location of the code that built each operation (`ADD_RUST_LOC`).
