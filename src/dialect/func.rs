//! `func` dialect.

use std::{
    fmt::{self, Write},
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use crate::{
    Attr, BuiltRegion, Closure, ClosureBuilder, Edges, FunctionType, HighMlir, OpPrinter, Ty, TypeError, Val,
    arith::Operands,
    attr::Symbol,
    types::{ResultTypes, TypeList},
    val::{AsVal, Dyn, FromTuple, IntoTys, MlirClosure, OrPanic, Tuple, check_params, for_each_arity},
};

/// The name and type of a function; calls refer to functions through it, so
/// a function can be called before (or while) it is defined.
///
/// `A` is the marker of the body passed to `define`: `(Val<T>, ..)` with one
/// typed value per parameter (as built by [`sig!`](crate::sig!)), or [`Dyn`].
/// `R` is the shape of the results of a call: `()`, `Val<T>`, a Rust tuple
/// of them, or a dynamic [`Tuple`].
pub struct Signature<A = Dyn, R = Tuple> {
    /// The symbol name.
    pub name: String,
    /// Parameter types.
    pub inputs: Vec<Ty>,
    /// Result types.
    pub results: Vec<Ty>,
    marker: PhantomData<fn() -> (A, R)>,
}

impl Signature {
    /// A signature of dynamic arity: `define` takes `FnOnce(&[Val]) -> Tuple`
    /// and `call` returns a [`Tuple`].
    pub fn new(
        name: impl Into<String>,
        inputs: impl IntoIterator<Item = Ty>,
        results: impl IntoIterator<Item = Ty>,
    ) -> Signature {
        Signature {
            name: name.into(),
            inputs: inputs.into_iter().collect(),
            results: results.into_iter().collect(),
            marker: PhantomData,
        }
    }

    /// A signature whose parameters and results have the static types of
    /// `inputs` and `results`, Rust tuples of types (e.g. `(I32, memref![F32; ?])`).
    pub fn typed<I: IntoTys, O: IntoTys>(name: impl Into<String>, inputs: I, results: O) -> Signature<I::Vals, O::Results> {
        Signature::new(name, inputs.into_tys(), results.into_tys()).with_marker()
    }
}

impl<A, R: FromTuple> Signature<A, R> {
    /// The same signature, with other static types.
    pub fn with_marker<B, S>(self) -> Signature<B, S> {
        Signature { name: self.name, inputs: self.inputs, results: self.results, marker: PhantomData }
    }

    /// The function type.
    pub fn ty(&self) -> Ty {
        Ty::function(self.inputs.clone(), self.results.clone())
    }

    /// `func.func` whose `body(args..)` returns the results.
    #[track_caller]
    pub fn try_define_with<C: MlirClosure<A>>(&self, body: C) -> Result<Val, TypeError> {
        check_params::<A, C>("func.func", &self.inputs)?;
        let mut builder = ClosureBuilder::new(true);
        let region = builder.region(&self.inputs, |args| body.call(args));
        builder.try_finish()?;
        let returned: Vec<Ty> = region
            .exprs
            .iter()
            .map(|v| v.try_ty())
            .collect::<Option<_>>()
            .ok_or_else(|| TypeError::new("func.return", "returns an operation without results"))?;
        if returned != self.results {
            return Err(TypeError::new(
                "func.return",
                format!(
                    "`@{}` returns ({}) but is declared to return ({})",
                    self.name,
                    TypeList(&returned),
                    TypeList(&self.results)
                ),
            ));
        }
        Val::single(Func { signature: self.clone().with_marker(), body: region, external: false })
    }

    /// A private external `func.func` without a body.
    #[track_caller]
    pub fn try_declare(&self) -> Result<Val, TypeError> {
        Val::single(Func { signature: self.clone().with_marker(), body: BuiltRegion::empty(), external: true })
    }

    /// A private external `func.func` without a body.
    #[track_caller]
    pub fn declare(&self) -> Val {
        self.try_declare().or_panic()
    }

    /// `func.call`; literal arguments take the parameter types.
    #[track_caller]
    pub fn try_call(&self, args: impl Operands) -> Result<R, TypeError> {
        let args = args.into_vals(&|i| self.inputs.get(i).cloned())?;
        let types: Vec<Ty> = args
            .iter()
            .map(|v| v.try_ty())
            .collect::<Option<_>>()
            .ok_or_else(|| TypeError::new("func.call", "an argument is an operation without results"))?;
        if types != self.inputs {
            return Err(TypeError::new(
                "func.call",
                format!("`@{}` takes ({}) but got ({})", self.name, TypeList(&self.inputs), TypeList(&types)),
            ));
        }
        let call = Call { callee: self.clone().with_marker(), args };
        R::from_tuple(Val::all_results(call)?).ok_or_else(|| TypeError::new("func.call", "results of unexpected types"))
    }

    /// `func.call`; literal arguments take the parameter types.
    #[track_caller]
    pub fn call(&self, args: impl Operands) -> R {
        self.try_call(args).or_panic()
    }

}

impl<R: FromTuple> Signature<Dyn, R> {
    /// `func.func` whose `body(&args)` returns the results.
    #[track_caller]
    pub fn try_define(&self, body: impl FnOnce(&[Val]) -> Tuple) -> Result<Val, TypeError> {
        self.try_define_with(body)
    }

    /// `func.func` whose `body(&args)` returns the results.
    #[track_caller]
    pub fn define(&self, body: impl FnOnce(&[Val]) -> Tuple) -> Val {
        self.try_define_with(body).or_panic()
    }
}

/// `define`/`try_define` taking one value per parameter, so closure
/// parameter types are inferred.
macro_rules! define_methods {
    (; $($a:ident)*) => {
        #[allow(non_camel_case_types)]
        impl<R: FromTuple, $($a: ?Sized),*> Signature<($(Val<$a>,)*), R>
        where
            $(Val<$a>: AsVal,)*
        {
            /// `func.func` whose `body(args..)` returns the results.
            #[track_caller]
            pub fn try_define(&self, body: impl FnOnce($(Val<$a>),*) -> Tuple) -> Result<Val, TypeError> {
                self.try_define_with(body)
            }

            /// `func.func` whose `body(args..)` returns the results.
            #[track_caller]
            pub fn define(&self, body: impl FnOnce($(Val<$a>),*) -> Tuple) -> Val {
                self.try_define_with(body).or_panic()
            }
        }
    };
}

for_each_arity!(define_methods);

impl<A, R> Clone for Signature<A, R> {
    fn clone(&self) -> Self {
        Signature {
            name: self.name.clone(),
            inputs: self.inputs.clone(),
            results: self.results.clone(),
            marker: PhantomData,
        }
    }
}

impl<A, R> PartialEq for Signature<A, R> {
    fn eq(&self, other: &Self) -> bool {
        (&self.name, &self.inputs, &self.results) == (&other.name, &other.inputs, &other.results)
    }
}

impl<A, R> Eq for Signature<A, R> {}

impl<A, R> Hash for Signature<A, R> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&self.name, &self.inputs, &self.results).hash(state)
    }
}

impl<A, R> fmt::Debug for Signature<A, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Signature({} : {})", self.name, Ty::function(self.inputs.clone(), self.results.clone()))
    }
}

/// `func.func`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Func {
    pub signature: Signature,
    pub body: BuiltRegion,
    /// An external declaration, without a body.
    pub external: bool,
}

impl HighMlir for Func {
    fn name(&self) -> &str {
        "func.func"
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
        if self.external {
            attrs.push(("sym_visibility".into(), Attr::str("private")));
        }
        attrs
    }

    fn as_closure(&self) -> Option<&dyn Closure> {
        Some(self)
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        let sig = &self.signature;
        let results = if sig.results.is_empty() { String::new() } else { format!(" -> {}", ResultTypes(&sig.results)) };
        match self.external {
            true => write!(p, "func.func private {}({}){results}", Symbol(&sig.name), TypeList(&sig.inputs)),
            false => {
                let args: Vec<String> =
                    sig.inputs.iter().enumerate().map(|(i, ty)| format!("{}: {ty}", p.region_arg(0, i))).collect();
                write!(p, "func.func {}({}){results} ", Symbol(&sig.name), args.join(", "))?;
                p.region(0, false)
            }
        }
    }
}

impl Closure for Func {
    fn region(&self, _: usize) -> &BuiltRegion {
        &self.body
    }

    fn terminator(&self, _: usize) -> Option<&str> {
        (!self.external).then_some("func.return")
    }

    fn is_empty(&self, _: usize) -> bool {
        self.external
    }
}

/// `func.call`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Call {
    pub callee: Signature,
    pub args: Vec<Val>,
}

impl HighMlir for Call {
    fn name(&self) -> &str {
        "func.call"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(self.args.iter().cloned())
    }

    fn result_types(&self) -> Vec<Ty> {
        self.callee.results.clone()
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("callee".into(), Attr::Symbol(self.callee.name.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let callee = &self.callee;
        let args = p.operands(0..self.args.len());
        write!(p, "func.call {}({args}) : {}", Symbol(&callee.name), callee.ty())
    }
}

impl<A, R: FromTuple> Signature<A, R> {
    /// `func.constant`: this function as a value, for [`call_indirect`].
    #[track_caller]
    pub fn try_constant(&self) -> Result<Val<FunctionType>, TypeError> {
        Val::new_typed(Constant { callee: self.clone().with_marker() })
    }

    /// `func.constant`: this function as a value, for [`call_indirect`].
    #[track_caller]
    pub fn constant(&self) -> Val<FunctionType> {
        self.try_constant().or_panic()
    }
}

/// `func.constant`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Constant {
    pub callee: Signature,
}

impl HighMlir for Constant {
    fn name(&self) -> &str {
        "func.constant"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::empty())
    }

    fn result_types(&self) -> Vec<Ty> {
        vec![self.callee.ty()]
    }

    fn attributes(&self) -> Vec<(String, Attr)> {
        vec![("value".into(), Attr::Symbol(self.callee.name.clone()))]
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        write!(p, "func.constant {} : {}", Symbol(&self.callee.name), self.callee.ty())
    }
}

/// `func.call_indirect`
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CallIndirect {
    pub callee: Val,
    pub args: Vec<Val>,
}

/// `func.call_indirect` of a function value; literal arguments take the parameter types.
#[track_caller]
pub fn try_call_indirect(callee: impl Into<Val>, args: impl Operands) -> Result<Tuple, TypeError> {
    let name = "func.call_indirect";
    let callee = callee.into();
    let ty = callee.try_ty().ok_or_else(|| TypeError::new(name, "the callee is an operation without results"))?;
    let Some(f) = ty.downcast_ref::<FunctionType>().cloned() else {
        return Err(TypeError::new(name, format!("expected a function, not `{ty}`")));
    };
    let args = args.into_vals(&|i| f.inputs.get(i).cloned())?;
    let types: Vec<Ty> = args
        .iter()
        .map(|v| v.try_ty())
        .collect::<Option<_>>()
        .ok_or_else(|| TypeError::new(name, "an argument is an operation without results"))?;
    if types != f.inputs {
        return Err(TypeError::new(name, format!("`{ty}` takes ({}) but got ({})", TypeList(&f.inputs), TypeList(&types))));
    }
    Val::all_results(CallIndirect { callee, args })
}

/// `func.call_indirect` of a function value.
#[track_caller]
pub fn call_indirect(callee: impl Into<Val>, args: impl Operands) -> Tuple {
    try_call_indirect(callee, args).or_panic()
}

impl HighMlir for CallIndirect {
    fn name(&self) -> &str {
        "func.call_indirect"
    }

    fn inputs(&self) -> Edges<'_> {
        Box::new(std::iter::once(self.callee.clone()).chain(self.args.iter().cloned()))
    }

    fn result_types(&self) -> Vec<Ty> {
        self.callee.ty().downcast_ref::<FunctionType>().expect("checked when built").results.clone()
    }

    fn print(&self, p: &mut OpPrinter<'_>) -> fmt::Result {
        p.results_lhs()?;
        let ty = p.operand_type(0);
        write!(p, "func.call_indirect {}({}) : {ty}", p.operand(0), p.operands(1..=self.args.len()))
    }
}
