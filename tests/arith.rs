mod common;

use common::{check, op_names};
use high_mlir::{
    builtin,
    BF16, F16, F32, F64, FloatType, I1, I8, I16, I32, I64, INDEX, IntType, Val,
    arith::{self, CmpFPredicate, CmpIPredicate, Constant, Exact, FastMath, FloatFlags, NonNeg, Overflow, RoundingMode},
    sig, tensor, vector,
};

#[test]
fn integer_ops() {
    let root = builtin::module(|| {
        sig!(fn ints(I32, I32, INDEX) -> I32).define(|a, b, n| {
            let ops: Vec<Val<IntType>> = vec![
                arith::addi(&a, &b, Overflow::NSW | Overflow::NUW),
                arith::subi(&a, &b, Overflow::NSW),
                arith::muli(&a, &b, ()),
                arith::divsi(&a, &b, Exact(true)),
                arith::divui(&a, &b, ()),
                arith::ceildivsi(&a, &b),
                arith::ceildivui(&a, &b),
                arith::floordivsi(&a, &b),
                arith::remsi(&a, &b),
                arith::remui(&a, &b),
                arith::andi(&a, &b),
                arith::ori(&a, &b),
                arith::xori(&a, &b),
                arith::shli(&a, &b, Overflow::NUW),
                arith::shrsi(&a, &b, ()),
                arith::shrui(&a, &b, Exact(true)),
                arith::maxsi(&a, &b),
                arith::minsi(&a, &b),
                arith::maxui(&a, &b),
                arith::minui(&a, 7),
            ];
            let (sum, overflow) = arith::addui_extended(&a, &b);
            let (diff, under) = arith::subui_extended(&a, &b);
            let (_, high) = arith::mulsi_extended(&a, &b);
            let (_, uhigh) = arith::mului_extended(&a, &b);
            let carry = arith::select(overflow, sum, &diff);
            let borrow = arith::select(under, high, uhigh);
            let total = ops.into_iter().fold(carry + borrow, |acc, v| acc + v);
            let wide = arith::extui(&total, I64, NonNeg(true));
            let back = arith::trunci(wide, I32, Overflow::NSW);
            let idx = arith::index_castui(&back, INDEX, NonNeg(true));
            let i = arith::index_cast(&n + idx, I32);
            arith::select(arith::cmpi(CmpIPredicate::Uge, &i, 3), i, back).into()
        });
    });
    let text = check(&root);
    for op in [
        "arith.addi %arg0, %arg1 overflow<nsw, nuw> : i32",
        "arith.divsi %arg0, %arg1 exact : i32",
        "arith.shrui %arg0, %arg1 exact : i32",
        "arith.addui_extended %arg0, %arg1 : i32, i1",
        "arith.mulsi_extended %arg0, %arg1 : i32",
        "arith.extui %",
        "nneg : i32 to i64",
        "overflow<nsw> : i64 to i32",
        "arith.cmpi uge",
    ] {
        assert!(text.contains(op), "missing `{op}`:\n{text}");
    }
}

#[test]
fn float_ops() {
    let root = builtin::module(|| {
        sig!(fn floats(F32, F32) -> F32).define(|a, b| {
            let fast = FloatFlags { rounding: None, fastmath: FastMath::FAST };
            let rounded = FloatFlags { rounding: Some(RoundingMode::TowardZero), fastmath: FastMath::NNAN | FastMath::NINF };
            let ops: Vec<Val<FloatType>> = vec![
                arith::addf(&a, &b, fast),
                arith::subf(&a, &b, rounded),
                arith::mulf(&a, &b, ()),
                arith::divf(&a, &b, RoundingMode::Upward),
                arith::remf(&a, &b, ()),
                arith::maximumf(&a, &b, FastMath::NSZ),
                arith::minimumf(&a, &b, ()),
                arith::maxnumf(&a, &b, ()),
                arith::minnumf(&a, &b, ()),
                arith::negf(&a, FastMath::CONTRACT),
                arith::flush_denormals(&a, ()),
            ];
            let total = ops.into_iter().fold(a.clone(), |acc, v| acc + v);
            let wide = arith::extf(&total, F64, ());
            let half = arith::truncf(&wide, F16, RoundingMode::ToNearestEven);
            let brain = arith::convertf(&half, BF16, ());
            let back = arith::extf(brain, F32, ());
            let c = arith::cmpf(CmpFPredicate::Ord, &back, &b, FastMath::NNAN);
            arith::select(c, back, b).into()
        });
    });
    let text = check(&root);
    for op in [
        "arith.addf %arg0, %arg1 fastmath<fast> : f32",
        "arith.subf %arg0, %arg1 toward_zero fastmath<nnan,ninf> : f32",
        "arith.divf %arg0, %arg1 upward : f32",
        "arith.negf %arg0 fastmath<contract> : f32",
        "arith.flush_denormals %arg0 : f32",
        "to_nearest_even : f64 to f16",
        "arith.convertf",
        "arith.cmpf ord",
    ] {
        assert!(text.contains(op), "missing `{op}`:\n{text}");
    }
}

#[test]
fn typed_results() {
    builtin::module(|| {
        // operations of one type keep the static type of their first operand
        let a: Val<IntType> = 3.into();
        let b: Val<IntType> = arith::addi(&a, 4, ());
        let c: Val<IntType> = arith::maxsi(&b, &a);
        let d: Val = arith::addi(4, &c, ());
        assert_eq!(d.ty(), I32);
    });
}

#[test]
fn conversions() {
    let root = builtin::module(|| {
        sig!(fn casts(I8, F32, INDEX) -> F32).define(|i, x, n| {
            let a = arith::extsi(&i, I32);
            let b = arith::sitofp(&a, F32);
            let c = arith::uitofp(&i, F32, NonNeg(true));
            let d = arith::fptosi(&x, I16);
            let e = arith::fptoui(&x, I16);
            let g = arith::bitcast(d + e, F16);
            let h = arith::bitcast(&x, I32);
            let k = arith::index_cast(&n, I32) + h;
            let total = b + c + arith::extf(g, F32, ()) + arith::sitofp(k, F32);
            // generic casts pick the operation
            (arith::cast(&n, F32) + arith::cast(&i, F32) + total).into()
        });
    });
    let text = check(&root);
    assert!(text.contains("arith.uitofp %arg0 nneg : i8 to f32"), "{text}");
}

#[test]
fn scaling_casts() {
    let root = builtin::module(|| {
        let e8m0 = FloatType::F8E8M0FNU;
        sig!(fn mx(vector![FloatType::F8E4M3FN; 4], vector![e8m0; 4]) -> vector![F32; 4]).define(|x, s| {
            let up = arith::scaling_extf(&x, &s, vector![F32; 4], ());
            let down = arith::scaling_truncf(&up, &s, vector![FloatType::F4E2M1FN; 4], RoundingMode::ToNearestEven);
            let again = arith::scaling_extf(down, &s, vector![F32; 4], ());
            (up + again).into()
        });
    });
    let text = check(&root);
    assert!(text.contains("arith.scaling_extf %arg0, %arg1 : vector<4xf8E4M3FN>, vector<4xf8E8M0FNU> to vector<4xf32>"));
    assert!(text.contains("to_nearest_even : vector<4xf32>, vector<4xf8E8M0FNU> to vector<4xf4E2M1FN>"), "{text}");
}

#[test]
fn vectors_and_tensors() {
    let root = builtin::module(|| {
        let v4 = vector![I32; 4];
        let ty = (v4.clone(), vector![F32; [4]], tensor![F32; 2], vector![I32; 2, 2]);
        sig!(fn vec(vector![I32; 4], vector![F32; [4]], tensor![F32; 2]) -> (ty.0, ty.1, ty.2, ty.3)).define(|v, s, t| {
            let dense = Constant::ints([1, 2, 3, 4], v4.clone());
            let mask = Constant::ints([1, 0, 1, 0], vector![I1; 4]);
            let doubled = &v * 2 + &dense;
            let picked = arith::select(mask, &doubled, &v);
            let cmp = picked.lt(&v);
            let scaled = &s * 2.0 + 1;
            let clamped = scaled.lt(0.0).select(&scaled, 0.0);
            let shifted = &t + Constant::floats([1.0, 2.0], tensor![F32; 2]);
            let matrix = Constant::ints([1, 2, 3, 4], vector![I8; 2, 2]);
            (arith::select(cmp, picked, v), clamped, shifted, arith::cast(matrix, vector![I32; 2, 2])).into()
        });
    });
    let text = check(&root);
    assert!(text.contains("arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>"), "{text}");
    assert!(text.contains("arith.constant dense<[true, false, true, false]> : vector<4xi1>"), "{text}");
    assert!(text.contains("arith.constant dense<2.0> : vector<[4]xf32>"), "{text}");
    assert!(text.contains("dense<[[1, 2], [3, 4]]> : vector<2x2xi8>"), "{text}");
    assert!(text.contains("arith.select %"), "{text}");
    assert!(op_names(&text).contains(&"arith.extsi"), "{text}");
}

#[test]
fn type_errors() {
    builtin::module(|| {
        let i: Val = 1.into();
        let f: Val = 1.0f32.into();
        assert!(arith::try_convertf(&f, F32, ()).is_err(), "convertf needs a different type of the same width");
        assert!(arith::try_convertf(&f, F16, ()).is_err());
        assert!(arith::try_extf(&f, F16, ()).is_err());
        assert!(arith::try_trunci(&i, I64, ()).is_err());
        assert!(arith::try_addui_extended(&f, &f).is_err());
        assert!(arith::try_addf(&i, &i, ()).is_err());
        assert!(arith::try_addi(1, 2, ()).is_err());
        assert!(Constant::try_ints([1, 2], vector![I32; 3]).is_err());
        assert!(Constant::try_floats([1.0], tensor![F32; ?]).is_err());
        assert!(arith::try_add(Constant::int(0, vector![I32; 4]), &i).is_err());
    });
}
