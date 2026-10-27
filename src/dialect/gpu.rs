//! `gpu` dialect: kernels, launches and the operations of a GPU thread.
//!
//! Kernels live in a [`module`] (a `gpu.module`) and are defined from a
//! signature with [`Signature::define_kernel`](crate::func::Signature),
//! then launched with [`launch_func`]; or a kernel body is written inline
//! with [`launch`]. The enclosing `builtin.module` of a `launch_func` needs
//! the `gpu.container_module` attribute: see [`container_module`].
//!
//! Memory operations (`alloc`, `memcpy`, ...) are the synchronous forms
//! (without `!gpu.async.token`s).

use std::fmt::{self, Write};

use crate::{
    Attr, BuiltRegion, Closure, ClosureBuilder, Edges, HighMlir, IndexType, IntType, MemRefType, OpPrinter, Root, Ty,
    TypeError, Val,
    arith::{Operand, Operands, predicates},
    attr::Symbol,
    dialect::util::{expect, indices, memref_ty, types, value_ty},
    func::Signature,
    types::{I1, I32, INDEX, ResultTypes, TypeList},
    val::{AsVal, Dyn, FromTuple, MlirClosure, OrPanic, Tuple, check_params, for_each_arity, retype},
};

/// Marks the root module as holding `gpu.module`s, as `gpu.launch_func` requires.
pub fn container_module(root: &Root) {
    root.val().annotate("gpu.container_module", Attr::Unit);
}

predicates! {
    /// A dimension of a grid, block or cluster.
    Dimension {
        X = "x", Y = "y", Z = "z",
    }
}

macro_rules! dimension_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            pub dimension: Dimension,
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $try_f(dimension: Dimension) -> Result<Val<IndexType>, TypeError> {
            Val::new_typed($op { dimension })
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f(dimension: Dimension) -> Val<IndexType> {
            $try_f(dimension).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("gpu.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(std::iter::empty())
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![INDEX.into()]
            }

            fn attributes(&self) -> Vec<(String, Attr)> {
                vec![("dimension".into(), Attr::Raw(format!("#gpu<dim {}>", self.dimension.keyword())))]
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                write!(p, "{} {}", self.name(), self.dimension.keyword())
            }
        }
    )*};
}

dimension_ops! {
    /// `gpu.thread_id`: the thread index within its block.
    ThreadId = thread_id / try_thread_id;
    /// `gpu.block_id`: the block index within the grid.
    BlockId = block_id / try_block_id;
    /// `gpu.global_id`: the thread index within the grid.
    GlobalId = global_id / try_global_id;
    /// `gpu.block_dim`: the number of threads per block.
    BlockDim = block_dim / try_block_dim;
    /// `gpu.grid_dim`: the number of blocks.
    GridDim = grid_dim / try_grid_dim;
    /// `gpu.cluster_id`: the cluster index within the grid.
    ClusterId = cluster_id / try_cluster_id;
    /// `gpu.cluster_dim`: the number of clusters.
    ClusterDim = cluster_dim / try_cluster_dim;
    /// `gpu.cluster_dim_blocks`: the number of blocks per cluster.
    ClusterDimBlocks = cluster_dim_blocks / try_cluster_dim_blocks;
    /// `gpu.cluster_block_id`: the block index within its cluster.
    ClusterBlockId = cluster_block_id / try_cluster_block_id;
}

macro_rules! subgroup_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident, $suffix:literal;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op;

        $(#[$doc])*
        #[track_caller]
        pub fn $try_f() -> Result<Val<IndexType>, TypeError> {
            Val::new_typed($op)
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f() -> Val<IndexType> {
            $try_f().or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("gpu.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new(std::iter::empty())
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![INDEX.into()]
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                p.results_lhs()?;
                write!(p, "{}{}", self.name(), $suffix)
            }
        }
    )*};
}

subgroup_ops! {
    /// `gpu.lane_id`: the thread index within its subgroup.
    LaneId = lane_id / try_lane_id, "";
    /// `gpu.subgroup_id`: the subgroup index within its block.
    SubgroupId = subgroup_id / try_subgroup_id, " : index";
    /// `gpu.num_subgroups`: the number of subgroups per block.
    NumSubgroups = num_subgroups / try_num_subgroups, " : index";
    /// `gpu.subgroup_size`: the number of threads per subgroup.
    SubgroupSize = subgroup_size / try_subgroup_size, " : index";
}

/// `gpu.barrier`: waits for every thread of the block.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Barrier;

/// `gpu.barrier`
#[track_caller]
pub fn try_barrier() -> Result<(), TypeError> {
    Val::unit(Barrier)
}

/// `gpu.barrier`
#[track_caller]
pub fn barrier() {
    try_barrier().or_panic()
}

impl HighMlir for Barrier {
    fn name(&self) -> &str {
        "gpu.barrier"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.write_str("gpu.barrier")
    }
}

predicates! {
    /// How `gpu.shuffle` picks the lane to read, in MLIR's numbering.
    ShuffleMode {
        Xor = "xor", Down = "down", Up = "up", Idx = "idx",
    }
}

/// `gpu.shuffle`: a value of another lane of the subgroup.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Shuffle {
    pub mode: ShuffleMode,
    pub value: Val,
    pub offset: Val,
    pub width: Val,
}

/// `gpu.shuffle` of `value` with the lane given by `mode` and `offset`
/// (an `i32`), among the first `width` lanes (an `i32`); literals are
/// `i32`. Returns the value and whether the lane was valid.
#[track_caller]
pub fn try_shuffle<T: ?Sized>(mode: ShuffleMode, value: &Val<T>, offset: impl Operand, width: impl Operand) -> Result<(Val<T>, Val<IntType>), TypeError>
where
    Val<T>: AsVal,
{
    let name = "gpu.shuffle";
    let value = value.as_val();
    let (offset, width) = (offset.into_val(&I32.into())?, width.into_val(&I32.into())?);
    let v = value_ty(name, &value)?;
    let scalar = |t: &Ty| t.int_width().is_some() || t.float_kind().is_some();
    let ok = scalar(&v) || v.as_vector().is_some_and(|vt| vt.rank() == 1 && scalar(&vt.element));
    expect(name, ok, || format!("cannot shuffle a `{v}`"))?;
    for t in types(name, &[offset.clone(), width.clone()])? {
        expect(name, t == I32, || format!("offset and width must be `i32`, not `{t}`"))?;
    }
    let [value, valid] = Val::all_results(Shuffle { mode, value, offset, width })?.try_into().unwrap();
    Ok((retype(name, value)?, valid.downcast().unwrap()))
}

/// `gpu.shuffle`; see [`try_shuffle`].
#[track_caller]
pub fn shuffle<T: ?Sized>(mode: ShuffleMode, value: &Val<T>, offset: impl Operand, width: impl Operand) -> (Val<T>, Val<IntType>)
where
    Val<T>: AsVal,
{
    try_shuffle(mode, value, offset, width).or_panic()
}

impl HighMlir for Shuffle {
    fn name(&self) -> &str {
        "gpu.shuffle"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new([self.value.clone(), self.offset.clone(), self.width.clone()].into_iter())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.value.ty(), I1.into()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("mode".into(), Attr::Raw(format!("#gpu<shuffle_mode {}>", self.mode.keyword())))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "gpu.shuffle {} {} : {}", self.mode.keyword(), p.operands(0..3), p.operand_type(0))
    }
}

predicates! {
    /// How `gpu.all_reduce` and `gpu.subgroup_reduce` combine values, in MLIR's numbering.
    ReduceOperation {
        Add = "add", Mul = "mul", MinUI = "minui", MinSI = "minsi", MinNumF = "minnumf", MaxUI = "maxui",
        MaxSI = "maxsi", MaxNumF = "maxnumf", And = "and", Or = "or", Xor = "xor", MinimumF = "minimumf",
        MaximumF = "maximumf",
    }
}

impl ReduceOperation {
    fn attr(self) -> Attr {
        Attr::Raw(format!("#gpu<all_reduce_op {}>", self.keyword()))
    }
}

/// `gpu.all_reduce` (over the block) or `gpu.subgroup_reduce` (over the subgroup).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Reduce {
    subgroup: bool,
    pub op: ReduceOperation,
    pub value: Val,
    /// Every thread takes part.
    pub uniform: bool,
    /// Empty: the reduction is `op`.
    body: BuiltRegion,
}

macro_rules! reduce_ops {
    ($($(#[$doc:meta])* $f:ident / $try_f:ident, $subgroup:literal;)*) => {$(
        $(#[$doc])*
        #[track_caller]
        pub fn $try_f<T: ?Sized>(op: ReduceOperation, value: &Val<T>, uniform: bool) -> Result<Val<T>, TypeError>
        where
            Val<T>: AsVal,
        {
            let name = if $subgroup { "gpu.subgroup_reduce" } else { "gpu.all_reduce" };
            let value = value.as_val();
            let v = value_ty(name, &value)?;
            let scalar = |t: &Ty| t.int_width().is_some() || t.float_kind().is_some();
            let ok = scalar(&v) || ($subgroup && v.as_vector().is_some_and(|vt| vt.rank() == 1 && scalar(&vt.element)));
            expect(name, ok, || format!("cannot reduce a `{v}`"))?;
            retype(name, Val::single(Reduce { subgroup: $subgroup, op, value, uniform, body: BuiltRegion::empty() })?)
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f<T: ?Sized>(op: ReduceOperation, value: &Val<T>, uniform: bool) -> Val<T>
        where
            Val<T>: AsVal,
        {
            $try_f(op, value, uniform).or_panic()
        }
    )*};
}

reduce_ops! {
    /// `gpu.all_reduce`: `value` reduced over the threads of the block.
    all_reduce / try_all_reduce, false;
    /// `gpu.subgroup_reduce`: `value` reduced over the threads of the subgroup.
    subgroup_reduce / try_subgroup_reduce, true;
}

impl HighMlir for Reduce {
    fn name(&self) -> &str {
        if self.subgroup { "gpu.subgroup_reduce" } else { "gpu.all_reduce" }
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.value.clone()))
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.value.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = vec![("op".into(), self.op.attr())];
        if self.uniform {
            attrs.push(("uniform".into(), Attr::Unit));
        }
        attrs
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        (!self.subgroup).then_some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let uniform = if self.uniform { " uniform" } else { "" };
        write!(p, "{} {} {}{uniform}", self.name(), self.op.keyword(), p.operand(0))?;
        if !self.subgroup {
            p.write_str(" ")?;
            p.region(0, false)?;
        }
        let ty = p.operand_type(0);
        write!(p, " : ({ty}) -> {ty}")
    }
}

impl Closure for Reduce {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        None
    }

    fn is_empty(&self, _: usize) -> bool {
        true
    }
}

/// `gpu.printf`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Printf {
    pub format: String,
    pub args: Vec<Val>,
}

/// `gpu.printf` of `args` (scalars) with a C `format` string.
#[track_caller]
pub fn try_printf(format: impl Into<String>, args: impl IntoIterator<Item = impl Into<Val>>) -> Result<(), TypeError> {
    let args: Vec<Val> = args.into_iter().map(Into::into).collect();
    for t in types("gpu.printf", &args)? {
        expect("gpu.printf", t.is_scalar(), || format!("cannot print a `{t}`"))?;
    }
    Val::unit(Printf { format: format.into(), args })
}

/// `gpu.printf`
#[track_caller]
pub fn printf(format: impl Into<String>, args: impl IntoIterator<Item = impl Into<Val>>) {
    try_printf(format, args).or_panic()
}

impl HighMlir for Printf {
    fn name(&self) -> &str {
        "gpu.printf"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.args.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("format".into(), Attr::str(&self.format))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        write!(p, "gpu.printf {}", Attr::str(&self.format))?;
        let n = self.args.len();
        if n > 0 {
            write!(p, ", {} : {}", p.operands(0..n), TypeList(&p.operand_types(0..n)))?;
        }
        Ok(())
    }
}

/// `gpu.dynamic_shared_memory`: the dynamic shared memory of the block.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct DynamicSharedMemory {
    pub ty: Ty,
}

/// The type of [`dynamic_shared_memory`]: `memref<?xi8, #gpu.address_space<workgroup>>`.
pub fn workgroup_bytes() -> Ty {
    MemRefType::new([None], crate::I8).with_memory_space(Some(Attr::Raw("#gpu.address_space<workgroup>".into()))).into()
}

/// `gpu.dynamic_shared_memory`, of type [`workgroup_bytes`].
#[track_caller]
pub fn try_dynamic_shared_memory() -> Result<Val<MemRefType>, TypeError> {
    Val::new_typed(DynamicSharedMemory { ty: workgroup_bytes() })
}

/// `gpu.dynamic_shared_memory`
#[track_caller]
pub fn dynamic_shared_memory() -> Val<MemRefType> {
    try_dynamic_shared_memory().or_panic()
}

impl HighMlir for DynamicSharedMemory {
    fn name(&self) -> &str {
        "gpu.dynamic_shared_memory"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "gpu.dynamic_shared_memory : {}", self.ty)
    }
}

/// `gpu.alloc`: device memory.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Alloc {
    pub ty: Ty,
    pub sizes: Vec<Val>,
    /// Shared with the host.
    pub host_shared: bool,
}

/// `gpu.alloc` of type `ty`, with one size per dynamic dimension.
#[track_caller]
pub fn try_alloc(ty: impl Into<Ty>, sizes: impl Operands, host_shared: bool) -> Result<Val<MemRefType>, TypeError> {
    let name = "gpu.alloc";
    let (ty, sizes) = (ty.into(), indices(name, sizes)?);
    let Some(m) = ty.as_memref() else {
        return Err(TypeError::new(name, format!("expected a memref type, not `{ty}`")));
    };
    let dynamic = m.shape.iter().filter(|d| d.is_none()).count();
    expect(name, sizes.len() == dynamic, || format!("`{ty}` needs {dynamic} sizes, got {}", sizes.len()))?;
    Val::new_typed(Alloc { ty, sizes, host_shared })
}

/// `gpu.alloc`
#[track_caller]
pub fn alloc(ty: impl Into<Ty>, sizes: impl Operands, host_shared: bool) -> Val<MemRefType> {
    try_alloc(ty, sizes, host_shared).or_panic()
}

impl HighMlir for Alloc {
    fn name(&self) -> &str {
        "gpu.alloc"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.sizes.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.ty.clone()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = vec![Attr::segments([0, self.sizes.len(), 0])];
        if self.host_shared {
            attrs.push(("hostShared".into(), Attr::Unit));
        }
        attrs
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let shared = if self.host_shared { " host_shared" } else { "" };
        write!(p, "gpu.alloc{shared} ({}) : {}", p.operands(0..self.sizes.len()), self.ty)
    }
}

/// Synchronous memory operations: `name operands : types`.
macro_rules! memory_ops {
    ($($(#[$doc:meta])* $op:ident = $f:ident / $try_f:ident ($($arg:ident),*) $check:expr;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub struct $op {
            $(pub $arg: Val,)*
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $try_f($($arg: impl Into<Val>),*) -> Result<(), TypeError> {
            let name = concat!("gpu.", stringify!($f));
            $(let $arg = $arg.into();)*
            let check: fn(&str, &[Val]) -> Result<(), TypeError> = $check;
            let args: [Val; [$(stringify!($arg)),*].len()] = [$($arg.clone()),*];
            check(name, &args)?;
            Val::unit($op { $($arg),* })
        }

        $(#[$doc])*
        #[track_caller]
        pub fn $f($($arg: impl Into<Val>),*) {
            $try_f($($arg),*).or_panic()
        }

        impl HighMlir for $op {
            fn name(&self) -> &str {
                concat!("gpu.", stringify!($f))
            }

            fn inputs(&self) -> Edges<'_> {
                Box::new([$(self.$arg.clone()),*].into_iter())
            }

            fn result_types(&self) -> Vec<Ty> {
                vec![]
            }

            fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
                let n = self.inputs().count();
                write!(p, "{} {} : {}", self.name(), p.operands(0..n), TypeList(&p.operand_types(0..n)))
            }
        }
    )*};
}

memory_ops! {
    /// `gpu.dealloc` of memory from [`alloc`].
    Dealloc = dealloc / try_dealloc (memref) |op, v| { memref_ty(op, &v[0]).map(drop) };
    /// `gpu.memcpy` from `src` into `dst`, of the same shape and element type.
    Memcpy = memcpy / try_memcpy (dst, src) |op, v| {
        let ((d, dm), (s, sm)) = (memref_ty(op, &v[0])?, memref_ty(op, &v[1])?);
        expect(op, dm.shape == sm.shape && dm.element == sm.element, || format!("cannot copy `{s}` into `{d}`"))
    };
    /// `gpu.memset` of every element of `dst` to `value`.
    Memset = memset / try_memset (dst, value) |op, v| {
        let ((d, dm), t) = (memref_ty(op, &v[0])?, value_ty(op, &v[1])?);
        expect(op, dm.element == t, || format!("cannot set `{d}` to a `{t}`"))
    };
}

/// `gpu.wait`: waits for the device (without async tokens).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Wait;

/// `gpu.wait`
#[track_caller]
pub fn try_wait() -> Result<(), TypeError> {
    Val::unit(Wait)
}

/// `gpu.wait`
#[track_caller]
pub fn wait() {
    try_wait().or_panic()
}

impl HighMlir for Wait {
    fn name(&self) -> &str {
        "gpu.wait"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.write_str("gpu.wait")
    }
}

/// `gpu.module`: a symbol table of kernels, compiled for the device.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Module {
    pub name: String,
    pub body: BuiltRegion,
}

/// `gpu.module @name` of the operations (kernels) `body` builds.
#[track_caller]
pub fn try_module(name: impl Into<String>, body: impl FnOnce()) -> Result<Val, TypeError> {
    let mut builder = ClosureBuilder::new(true);
    let region = builder.region(&[], |_| {
        body();
        Tuple::default()
    });
    builder.try_finish()?;
    Val::single(Module { name: name.into(), body: region })
}

/// `gpu.module @name`
#[track_caller]
pub fn module(name: impl Into<String>, body: impl FnOnce()) -> Val {
    try_module(name, body).or_panic()
}

impl HighMlir for Module {
    fn name(&self) -> &str {
        "gpu.module"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("sym_name".into(), Attr::str(&self.name))]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        write!(p, "gpu.module {} ", Symbol(&self.name))?;
        p.region(0, false)
    }
}

impl Closure for Module {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        None
    }
}

/// `gpu.func`: a device function, or a kernel.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Func {
    pub signature: Signature,
    pub kernel: bool,
    pub body: BuiltRegion,
}

impl<A, R: FromTuple> Signature<A, R> {
    /// `gpu.func` (a `kernel`, without results, or a device function) whose
    /// `body(args..)` returns the results; build it in a [`module`].
    #[track_caller]
    pub fn try_define_gpu_with<C: MlirClosure<A>>(&self, kernel: bool, body: C) -> Result<Val, TypeError> {
        let name = "gpu.func";
        expect(name, !kernel || self.results.is_empty(), || format!("kernel `@{}` has results", self.name))?;
        check_params::<A, C>(name, &self.inputs)?;
        let mut builder = ClosureBuilder::new(true);
        let region = builder.region(&self.inputs, |args| body.call(args));
        builder.try_finish()?;
        let returned = types("gpu.return", &region.exprs)?;
        expect("gpu.return", returned == self.results, || {
            format!("`@{}` returns ({}) but is declared to return ({})", self.name, TypeList(&returned), TypeList(&self.results))
        })?;
        let signature = Signature::new(self.name.clone(), self.inputs.clone(), self.results.clone());
        Val::single(Func { signature, kernel, body: region })
    }
}

impl<R: FromTuple> Signature<Dyn, R> {
    /// `gpu.func` whose `body(&args)` returns the results.
    #[track_caller]
    pub fn define_gpu(&self, kernel: bool, body: impl FnOnce(&[Val]) -> Tuple) -> Val {
        self.try_define_gpu_with(kernel, body).or_panic()
    }

    /// A kernel: a `gpu.func` without results.
    #[track_caller]
    pub fn define_kernel(&self, body: impl FnOnce(&[Val]) -> Tuple) -> Val {
        self.try_define_gpu_with(true, body).or_panic()
    }
}

macro_rules! define_methods {
    (; $($a:ident)*) => {
        #[allow(non_camel_case_types)]
        impl<R: FromTuple, $($a: ?Sized),*> Signature<($(Val<$a>,)*), R>
        where
            $(Val<$a>: AsVal,)*
        {
            /// `gpu.func` (a `kernel` or a device function) whose `body(args..)` returns the results.
            #[track_caller]
            pub fn define_gpu(&self, kernel: bool, body: impl FnOnce($(Val<$a>),*) -> Tuple) -> Val {
                self.try_define_gpu_with(kernel, body).or_panic()
            }

            /// A kernel: a `gpu.func` without results.
            #[track_caller]
            pub fn define_kernel(&self, body: impl FnOnce($(Val<$a>),*) -> Tuple) -> Val {
                self.try_define_gpu_with(true, body).or_panic()
            }
        }
    };
}

for_each_arity!(define_methods);

impl HighMlir for Func {
    fn name(&self) -> &str {
        "gpu.func"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        let mut attrs = vec![
            ("sym_name".into(), Attr::str(&self.signature.name)),
            ("function_type".into(), Attr::Type(self.signature.ty())),
        ];
        if self.kernel {
            attrs.push(("kernel".into(), Attr::Unit));
        }
        attrs
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let sig = &self.signature;
        let args: Vec<String> = sig.inputs.iter().enumerate().map(|(i, ty)| format!("{}: {ty}", p.region_arg(0, i))).collect();
        write!(p, "gpu.func {}({})", Symbol(&sig.name), args.join(", "))?;
        if !sig.results.is_empty() {
            write!(p, " -> {}", ResultTypes(&sig.results))?;
        }
        if self.kernel {
            p.write_str(" kernel")?;
        }
        p.write_str(" ")?;
        p.region(0, false)
    }
}

impl Closure for Func {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("gpu.return")
    }
}

/// Three `index` sizes: of a grid (in blocks) or of a block (in threads).
fn sizes3(op: &str, sizes: impl Operands) -> Result<Vec<Val>, TypeError> {
    let sizes = indices(op, sizes)?;
    expect(op, sizes.len() == 3, || format!("needs 3 sizes (x, y, z), got {}", sizes.len()))?;
    Ok(sizes)
}

/// `gpu.launch_func`: launches a kernel of a `gpu.module`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct LaunchFunc {
    pub module: String,
    pub kernel: Signature,
    pub grid: Vec<Val>,
    pub block: Vec<Val>,
    pub args: Vec<Val>,
}

/// `gpu.launch_func` of `kernel` in the `gpu.module` named `module`, on a
/// `grid` of blocks of `block` threads (3 sizes each; literals are
/// `index`); literal arguments take the parameter types.
#[track_caller]
pub fn try_launch_func<A, R>(
    module: impl Into<String>,
    kernel: &Signature<A, R>,
    grid: impl Operands,
    block: impl Operands,
    args: impl Operands,
) -> Result<(), TypeError> {
    let name = "gpu.launch_func";
    let (grid, block) = (sizes3(name, grid)?, sizes3(name, block)?);
    let args = args.into_vals(&|i| kernel.inputs.get(i).cloned())?;
    let found = types(name, &args)?;
    expect(name, found == kernel.inputs, || {
        format!("`@{}` takes ({}) but got ({})", kernel.name, TypeList(&kernel.inputs), TypeList(&found))
    })?;
    let kernel = Signature::new(kernel.name.clone(), kernel.inputs.clone(), kernel.results.clone());
    Val::unit(LaunchFunc { module: module.into(), kernel, grid, block, args })
}

/// `gpu.launch_func`; see [`try_launch_func`].
#[track_caller]
pub fn launch_func<A, R>(module: impl Into<String>, kernel: &Signature<A, R>, grid: impl Operands, block: impl Operands, args: impl Operands) {
    try_launch_func(module, kernel, grid, block, args).or_panic()
}

impl HighMlir for LaunchFunc {
    fn name(&self) -> &str {
        "gpu.launch_func"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.grid.iter().chain(&self.block).chain(&self.args).cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![
            ("kernel".into(), Attr::Raw(format!("{}::{}", Symbol(&self.module), Symbol(&self.kernel.name)))),
            Attr::segments([0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, self.args.len(), 0]),
        ]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let kernel = format!("{}::{}", Symbol(&self.module), Symbol(&self.kernel.name));
        write!(p, "gpu.launch_func {kernel} blocks in ({}) threads in ({})", p.operands(0..3), p.operands(3..6))?;
        let n = self.args.len();
        if n > 0 {
            let args: Vec<String> = (6..6 + n).map(|i| format!("{} : {}", p.operand(i), p.operand_type(i))).collect();
            write!(p, " args({})", args.join(", "))?;
        }
        Ok(())
    }
}

/// The values a [`launch`] body is given: indices and sizes, in `x, y, z` order.
#[derive(Clone, Debug)]
pub struct LaunchIds {
    pub block_id: [Val<IndexType>; 3],
    pub thread_id: [Val<IndexType>; 3],
    pub grid_size: [Val<IndexType>; 3],
    pub block_size: [Val<IndexType>; 3],
}

/// `gpu.launch`: runs its body as a kernel.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Launch {
    pub grid: Vec<Val>,
    pub block: Vec<Val>,
    pub body: BuiltRegion,
}

/// `gpu.launch` on a `grid` of blocks of `block` threads (3 sizes each;
/// literals are `index`), running `body(ids)` in every thread.
#[track_caller]
pub fn try_launch(grid: impl Operands, block: impl Operands, body: impl FnOnce(&LaunchIds)) -> Result<(), TypeError> {
    let name = "gpu.launch";
    let (grid, block) = (sizes3(name, grid)?, sizes3(name, block)?);
    let mut builder = ClosureBuilder::new(false);
    let region = builder.region(&vec![INDEX.into(); 12], |p| {
        let three = |k: usize| -> [Val<IndexType>; 3] { std::array::from_fn(|i| p[3 * k + i].downcast().unwrap()) };
        body(&LaunchIds { block_id: three(0), thread_id: three(1), grid_size: three(2), block_size: three(3) });
        Tuple::default()
    });
    builder.try_finish()?;
    Val::unit(Launch { grid, block, body: region })
}

/// `gpu.launch`; see [`try_launch`].
#[track_caller]
pub fn launch(grid: impl Operands, block: impl Operands, body: impl FnOnce(&LaunchIds)) {
    try_launch(grid, block, body).or_panic()
}

impl HighMlir for Launch {
    fn name(&self) -> &str {
        "gpu.launch"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.grid.iter().chain(&self.block).cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![Attr::segments([0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])]
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let arg = |i: usize| p.region_arg(0, i);
        let ids = |k: usize| (0..3).map(|i| arg(3 * k + i)).collect::<Vec<_>>().join(", ");
        let sizes = |k: usize, first: usize| {
            (0..3).map(|i| format!("{} = {}", arg(3 * k + i), p.operand(first + i))).collect::<Vec<_>>().join(", ")
        };
        let text = format!("gpu.launch blocks({}) in ({}) threads({}) in ({}) ", ids(0), sizes(2, 0), ids(1), sizes(3, 3));
        p.write_str(&text)?;
        p.region(0, false)
    }
}

impl Closure for Launch {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        Some("gpu.terminator")
    }
}
