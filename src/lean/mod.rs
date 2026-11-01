//! The lean lane: a library cell's whole replay -- rules, orders, fills,
//! account, daily curve -- on one symbol's precomputed market, for the shape
//! most of the library has.
//!
//! It is the account-coupled native lane (`python::replay_session` driving
//! `portfolio::kernel` through `python::rule_session`) restated for that one
//! shape, so a cell costs its decisions and fills rather than its bars. It
//! answers exactly what that lane answers or declines: where a cell takes a
//! branch the lane transcribes and this does not, it returns [`Declined`]
//! and the caller runs the cell on the full lane.

pub mod kernel;
pub mod market;
pub mod replay;
pub mod rules;

/// The cell takes a branch the lean lane does not transcribe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Declined(pub &'static str);
