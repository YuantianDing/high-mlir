//! Type-checking helpers shared by the dialects.

use crate::{MemRefType, Ty, TypeError, Val, VectorType, arith::Operands, types::INDEX};

/// The type of a result (not of an operation without results).
pub(crate) fn value_ty(op: &str, v: &Val) -> Result<Ty, TypeError> {
    v.try_ty()
        .filter(|_| v.is_result())
        .ok_or_else(|| TypeError::new(op, format!("port #{} of `{}` is not a value", v.port(), v.name())))
}

pub(crate) fn types(op: &str, vs: &[Val]) -> Result<Vec<Ty>, TypeError> {
    vs.iter().map(|v| value_ty(op, v)).collect()
}

/// `index` values; literals become `index` constants.
#[track_caller]
pub(crate) fn indices(op: &str, indices: impl Operands) -> Result<Vec<Val>, TypeError> {
    let indices = indices.into_vals(&|_| Some(INDEX.into())).map_err(|e| TypeError::new(op, e.message))?;
    for i in &indices {
        let ty = value_ty(op, i)?;
        if !ty.is_index() {
            return Err(TypeError::new(op, format!("indices must be `index`, not `{ty}`")));
        }
    }
    Ok(indices)
}

pub(crate) fn memref_ty(op: &str, v: &Val) -> Result<(Ty, MemRefType), TypeError> {
    let ty = value_ty(op, v)?;
    let m = ty.as_memref().cloned();
    m.map(|m| (ty.clone(), m)).ok_or_else(|| TypeError::new(op, format!("expected a memref, not `{ty}`")))
}

pub(crate) fn vector_ty(op: &str, v: &Val) -> Result<(Ty, VectorType), TypeError> {
    let ty = value_ty(op, v)?;
    let vt = ty.as_vector().cloned();
    vt.map(|vt| (ty.clone(), vt)).ok_or_else(|| TypeError::new(op, format!("expected a vector, not `{ty}`")))
}

/// Checks that `n` indices address a value of rank `rank`.
pub(crate) fn check_rank(op: &str, ty: &Ty, rank: usize, n: usize) -> Result<(), TypeError> {
    if n != rank {
        return Err(TypeError::new(op, format!("{n} indices for `{ty}` of rank {rank}")));
    }
    Ok(())
}

pub(crate) fn expect(op: &str, ok: bool, message: impl FnOnce() -> String) -> Result<(), TypeError> {
    if ok { Ok(()) } else { Err(TypeError::new(op, message())) }
}
