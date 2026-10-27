mod common;

use common::{check_fn, op_names};
use high_mlir::{
    F32, F64, FloatType, I32, INDEX, IntType, Val, arith::{FastMath, RoundingMode}, math, sig, tensor, ub, vector,
};

#[test]
fn float_functions() {
    let text = check_fn(|| {
        sig!(fn f(F32, F32) -> F32).define(|x, y| {
            let a = x.sqrt() + x.exp() + x.log() + math::tanh(&x, FastMath::FAST);
            let b = x.powf(&y) + x.powf(2.0) + x.atan2(&y) + x.copysign(-1.0);
            let c = x.fma(&y, 1.0) + math::fma(&x, &y, &a, RoundingMode::TowardZero);
            let d = x.clampf(0.0, 1.0) + math::clampf(&x, &y, &b, FastMath::NNAN);
            let (s, co) = x.sincos();
            let e = x.fpowi(Val::<IntType>::from(3)) + s + co;
            let all = [x.absf(), x.ceil(), x.floor(), x.round(), x.roundeven(), x.trunc(), x.erf(), x.erfc(),
                x.cbrt(), x.rsqrt(), x.log2(), x.log10(), x.log1p(), x.exp2(), x.expm1(), x.sin(), x.cos(),
                x.tan(), x.asin(), x.acos(), x.atan(), x.sinh(), x.cosh(), x.asinh(), x.acosh(), x.atanh()];
            let sum = all.into_iter().fold(a + b + c + d + e, |acc, v| acc + v);
            let nan = x.isnan() | x.isinf() | math::isnormal(&x, ()).downcast::<IntType>().unwrap() | x.isfinite();
            nan.select(&sum, 0.0).into()
        });
    });
    assert!(text.contains("math.tanh %arg0 fastmath<fast> : f32"), "{text}");
    assert!(text.contains("math.fma %arg0, %arg1, %0 toward_zero : f32") || text.contains("toward_zero : f32"), "{text}");
    assert!(text.contains("to [%arg1,"), "{text}");
    assert!(text.contains("math.fpowi"), "{text}");
    assert!(text.contains("math.isnan %arg0 : f32"), "{text}");
    assert!(op_names(&text).contains(&"math.sincos"), "{text}");
}

#[test]
fn integer_and_shaped() {
    let text = check_fn(|| {
        sig!(fn f(I32, vector![F64; 4], tensor![F32; 2, 2], INDEX) -> I32).define(|i, v, t, n| {
            let _ = math::sqrt(&v, ());
            let _ = math::isnan(&v, ());
            let _: Val<high_mlir::TensorType> = math::exp(&t, FastMath::AFN);
            let _ = math::absi(&n, ());
            let _ = math::ipowi(&n, 2usize, ());
            let _ = math::fpowi(&v, ub::poison(vector![I32; 4]), ());
            (i.absi() + i.ctlz() + i.cttz() + i.ctpop() + i.ipowi(3)).into()
        });
    });
    assert!(text.contains("math.isnan %arg1 : vector<4xf64>"), "{text}");
    assert!(text.contains("ub.poison : vector<4xi32>"), "{text}");
    assert!(text.contains("math.fpowi %arg1, %"), "{text}");
}

#[test]
fn type_errors() {
    high_mlir::builtin::module(|| {
        sig!(fn f(I32, F32)).define(|i, x| {
            assert!(math::try_sqrt(&i, ()).is_err());
            assert!(math::try_absi(&x, ()).is_err());
            assert!(math::try_powf(&x, &i, ()).is_err());
            assert!(math::try_fpowi(&x, &x, ()).is_err());
            let _: Val<FloatType> = x;
            ().into()
        });
    });
}
