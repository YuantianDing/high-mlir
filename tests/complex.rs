mod common;

use common::check_fn;
use high_mlir::{ComplexType, F32, F64, I32, I64, arith::FastMath, complex, sig};

#[test]
fn complex_ops() {
    let text = check_fn(|| {
        sig!(fn f(ComplexType::new(F32), F32, I32) -> (F32, I64)).define(|z, x, n| {
            let w = complex::create(&x, 2.0);
            let k = complex::constant(1.0, -0.5, F32);
            let s = (&z + &w) * &k - &z / &w;
            let s = -s;
            let s = complex::pow(&s, &z, FastMath::FAST);
            let s = complex::powi(&s, &n, ());
            let s = complex::atan2(&s, &z, ());
            let fs = [complex::cos, complex::exp, complex::expm1, complex::log, complex::log1p, complex::rsqrt,
                complex::sign, complex::sin, complex::sqrt, complex::tan, complex::tanh];
            let s = fs.into_iter().fold(s, |acc, f| f(acc, FastMath::NONE));
            let e = s.equal(&z) | complex::neq(&s, &w);
            let r = s.re() + s.im() + s.abs() + s.conj().angle();
            let bits = complex::bitcast(&s, I64);
            let r = e.select(&r, 0.0);
            (r, bits).into()
        });
    });
    assert!(text.contains("complex.constant [1.0 : f32, -5.0e-01 : f32] : complex<f32>")
        || text.contains("complex.constant [1.0 : f32, -0.5 : f32] : complex<f32>"), "{text}");
    assert!(text.contains("complex.pow %"), "{text}");
    assert!(text.contains("fastmath<fast> : complex<f32>"), "{text}");
    assert!(text.contains("complex.bitcast %"), "{text}");
    let _ = F64;
}

#[test]
fn type_errors() {
    high_mlir::builtin::module(|| {
        sig!(fn f(F32, F64)).define(|x, y| {
            assert!(complex::try_create(&x, &y).is_err());
            assert!(complex::try_re(&x, ()).is_err());
            ().into()
        });
    });
}
