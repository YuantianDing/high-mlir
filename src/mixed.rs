//! Lists mixing static integers and dynamic `index` values, such as the
//! offsets of `memref.subview` or the position of `vector.extract`.
//!
//! Integer literals stay static (they are printed inline and stored in a
//! `static_*` attribute); values become operands.

use crate::{TypeError, Val, val::AsVal};

/// `ShapedType::kDynamic`: marks a dynamic entry in a `static_*` attribute.
pub const DYNAMIC: i64 = i64::MIN;

/// A static integer or a dynamic `index` value (MLIR's `OpFoldResult`).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Mixed {
    /// A static integer.
    Static(i64),
    /// An `index` value.
    Dynamic(Val),
}

/// Conversion to [`Mixed`]: integer literals are static, values dynamic.
pub trait IntoMixed {
    /// This entry as a [`Mixed`].
    fn into_mixed(self) -> Mixed;
}

impl IntoMixed for Mixed {
    fn into_mixed(self) -> Mixed {
        self
    }
}

impl<T: ?Sized> IntoMixed for Val<T>
where
    Val<T>: AsVal,
{
    fn into_mixed(self) -> Mixed {
        Mixed::Dynamic(self.as_val())
    }
}

impl<T: IntoMixed + Clone> IntoMixed for &T {
    fn into_mixed(self) -> Mixed {
        self.clone().into_mixed()
    }
}

macro_rules! static_mixed {
    ($($t:ty),*) => {$(
        impl IntoMixed for $t {
            fn into_mixed(self) -> Mixed {
                Mixed::Static(self as i64)
            }
        }
    )*};
}

static_mixed!(i32, i64, isize, u32, u64, usize);

/// One [`IntoMixed`], an array, a `Vec` or a Rust tuple of them.
pub trait MixedList {
    /// The entries, in order.
    fn into_mixed_list(self) -> Vec<Mixed>;
}

impl<T: IntoMixed> MixedList for T {
    fn into_mixed_list(self) -> Vec<Mixed> {
        vec![self.into_mixed()]
    }
}

impl<T: IntoMixed, const N: usize> MixedList for [T; N] {
    fn into_mixed_list(self) -> Vec<Mixed> {
        self.into_iter().map(IntoMixed::into_mixed).collect()
    }
}

impl<T: IntoMixed> MixedList for Vec<T> {
    fn into_mixed_list(self) -> Vec<Mixed> {
        self.into_iter().map(IntoMixed::into_mixed).collect()
    }
}

macro_rules! tuple_mixed {
    ($(($($t:ident $i:tt),*))*) => {$(
        impl<$($t: IntoMixed),*> MixedList for ($($t,)*) {
            fn into_mixed_list(self) -> Vec<Mixed> {
                vec![$(self.$i.into_mixed()),*]
            }
        }
    )*};
}

tuple_mixed! {
    ()
    (A 0)
    (A 0, B 1)
    (A 0, B 1, C 2)
    (A 0, B 1, C 2, D 3)
    (A 0, B 1, C 2, D 3, E 4)
    (A 0, B 1, C 2, D 3, E 4, F 5)
}

/// A mixed list split into its dynamic values and its static attribute
/// (with [`DYNAMIC`] at the positions of the values).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct MixedValues {
    pub dynamic: Vec<Val>,
    pub statics: Vec<i64>,
}

impl MixedValues {
    /// Checks that the dynamic entries are `index` values.
    pub fn new(op: &str, list: impl MixedList) -> Result<MixedValues, TypeError> {
        let mut values = MixedValues::default();
        for m in list.into_mixed_list() {
            match m {
                Mixed::Static(DYNAMIC) => return Err(TypeError::new(op, "static value is `kDynamic`")),
                Mixed::Static(s) => values.statics.push(s),
                Mixed::Dynamic(v) => {
                    let ty = v.try_ty().filter(|_| v.is_result());
                    if !ty.as_ref().is_some_and(|t| t.is_index()) {
                        let ty = ty.map_or("an operation without results".to_string(), |t| format!("`{t}`"));
                        return Err(TypeError::new(op, format!("dynamic entries must be `index`, not {ty}")));
                    }
                    values.dynamic.push(v);
                    values.statics.push(DYNAMIC);
                }
            }
        }
        Ok(values)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.statics.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.statics.is_empty()
    }

    /// Each entry as a static value, or `None` if dynamic.
    pub fn static_values(&self) -> Vec<Option<i64>> {
        self.statics.iter().map(|&s| (s != DYNAMIC).then_some(s)).collect()
    }

    /// `[%a, 4, %b]`, naming the dynamic entries with `operand(k)` for the k-th of them.
    pub fn format(&self, mut operand: impl FnMut(usize) -> String) -> String {
        let mut k = 0;
        let items: Vec<String> = self
            .statics
            .iter()
            .map(|&s| {
                if s == DYNAMIC {
                    k += 1;
                    operand(k - 1)
                } else {
                    s.to_string()
                }
            })
            .collect();
        format!("[{}]", items.join(", "))
    }
}
