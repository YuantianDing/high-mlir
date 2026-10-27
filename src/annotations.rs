//! Annotations of operations: extra (discardable) attributes and source
//! locations, kept in static tables keyed by the address of the operation.
//!
//! Annotations are not part of an operation's identity. They are printed
//! (an annotated operation is printed in generic form, and its location as
//! a trailing `loc(..)`) and converted to melior. An operation's entries are
//! removed when it is dropped, with its module.
//!
//! With [`ADD_RUST_LOC`] set, every operation built records the Rust source
//! location of the call that built it, through `#[track_caller]`:
//!
//! ```
//! use std::sync::atomic::Ordering;
//!
//! use high_mlir::{Attr, IntType, Val, annotations::{ADD_RUST_LOC, Location}, builtin};
//!
//! ADD_RUST_LOC.store(true, Ordering::Relaxed);
//! builtin::module(|| {
//!     let x: Val<IntType> = 1.into();
//!     let y = &x + 2;
//!     let line = line!() - 1;
//!     assert!(matches!(y.location(), Some(Location::FileLineCol { line: l, .. }) if l == line));
//!     y.annotate("tag", Attr::str("sum"));
//!     assert!(y.to_string().contains("tag = \"sum\""));
//! });
//! ADD_RUST_LOC.store(false, Ordering::Relaxed);
//! ```

use std::{
    collections::HashMap,
    fmt,
    sync::{
        LazyLock,
        atomic::{AtomicBool, Ordering},
    },
};

use spin::Mutex;

use crate::{Attr, Val, val::Node};

/// The source location of an operation.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Location {
    /// An unknown location, `loc(unknown)`.
    Unknown,
    /// `loc("file":line:column)`
    FileLineCol { file: String, line: u32, column: u32 },
    /// `loc("name")`
    Name(String),
}

impl Location {
    /// The location of a Rust call, e.g. `std::panic::Location::caller()`.
    pub fn rust(location: &std::panic::Location<'_>) -> Location {
        Location::FileLineCol { file: location.file().to_string(), line: location.line(), column: location.column() }
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Location::Unknown => f.write_str("loc(unknown)"),
            Location::FileLineCol { file, line, column } => write!(f, "loc({}:{line}:{column})", Attr::str(file)),
            Location::Name(name) => write!(f, "loc({})", Attr::str(name)),
        }
    }
}

/// Whether building an operation records the Rust source location of the
/// call that built it (through `#[track_caller]`). On by default.
pub static ADD_RUST_LOC: AtomicBool = AtomicBool::new(true);

type Table<V> = LazyLock<Mutex<HashMap<u64, V>>>;

/// Discardable attributes of operations, by operation address.
static DICTIONARY_ATTR: Table<Vec<(String, Attr)>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Source locations of operations, by operation address.
static LOCATIONS: Table<Location> = LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn op_attributes(op: &Node) -> Vec<(String, Attr)> {
    DICTIONARY_ATTR.lock().get(&op.key()).cloned().unwrap_or_default()
}

pub(crate) fn op_location(op: &Node) -> Option<Location> {
    LOCATIONS.lock().get(&op.key()).cloned()
}

/// Records the location of the caller of a newly built operation, if
/// [`ADD_RUST_LOC`] is set.
#[track_caller]
pub(crate) fn record_rust_location(op: &Node) {
    record_rust_location_at(op, std::panic::Location::caller());
}

/// Records `caller` as the location of an operation built on its behalf
/// (e.g. inside a closure), if [`ADD_RUST_LOC`] is set.
pub(crate) fn record_rust_location_at(op: &Node, caller: &std::panic::Location<'_>) {
    if ADD_RUST_LOC.load(Ordering::Relaxed) {
        LOCATIONS.lock().insert(op.key(), Location::rust(caller));
    }
}

/// Removes the entries of the operation at `key`, which is being dropped.
pub(crate) fn forget(key: u64) {
    // Taken out of the lock: dropping attributes must not reenter it.
    let attrs = DICTIONARY_ATTR.lock().remove(&key);
    let location = LOCATIONS.lock().remove(&key);
    drop((attrs, location));
}

impl<T: ?Sized> Val<T> {
    /// Adds (or replaces) the discardable attribute `name` of the operation.
    pub fn annotate(&self, name: impl Into<String>, value: Attr) -> &Self {
        let name = name.into();
        let mut table = DICTIONARY_ATTR.lock();
        let attrs = table.entry(self.op().key()).or_default();
        match attrs.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = value,
            None => attrs.push((name, value)),
        }
        self
    }

    /// The annotated attributes of the operation, in the order they were added.
    pub fn annotations(&self) -> Vec<(String, Attr)> {
        op_attributes(&self.op())
    }

    /// Sets the location of the operation.
    pub fn set_location(&self, location: Location) -> &Self {
        LOCATIONS.lock().insert(self.op().key(), location);
        self
    }

    /// The location of the operation, if one was set or recorded.
    pub fn location(&self) -> Option<Location> {
        op_location(&self.op())
    }
}
