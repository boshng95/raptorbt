//! Multi-position ledger.
//!
//! Generalizes the one-position assumption the kernel was built on. Two
//! policies:
//!
//! - [`PositionPolicy::Net`] — at most one open position, opened in the
//!   kernel's direction. This is the historical behavior; every arithmetic
//!   step matches the original [`PositionManager`] path bit-for-bit (the
//!   golden fixture suite enforces it).
//! - [`PositionPolicy::NetAveraging`] — one net position, but an opening
//!   fill that agrees with it makes it bigger at a size-weighted average
//!   entry instead of being refused. This is how an exchange account
//!   behaves, and what a partially filled entry needs: the strategy
//!   re-sends the unfilled remainder next bar and the two fills have to
//!   become one position, not a rejection.
//! - [`PositionPolicy::Independent`] — hedging: each opening order creates
//!   its own entry with its own direction, protective levels, and running
//!   extremes; longs and shorts coexist. Closes target a position id.
//!
//! [`PositionManager`]: crate::portfolio::position::PositionManager

use crate::core::decimals::quantize_money;
use crate::core::types::{Direction, ExitReason, Position, Price, Timestamp, Trade};
use crate::execution::indian_costs::FeeBreakdown;
use crate::portfolio::position::ExitDetails;

/// How the ledger treats additional opening fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PositionPolicy {
    /// One net position at a time; further opening fills are refused
    /// (historical behavior).
    #[default]
    Net,
    /// One net position at a time, grown by further opening fills at a
    /// size-weighted average entry.
    NetAveraging,
    /// Independent concurrent positions, both directions (hedging).
    Independent,
}

/// One open position plus the bookkeeping the kernel used to hold globally.
#[derive(Debug, Clone)]
pub struct ManagedPosition {
    /// Ledger-assigned id, unique within a session.
    pub id: u64,
    /// Price/size/protective state (shared struct with the legacy path).
    pub position: Position,
    /// Entry timestamp, carried onto the trade record.
    pub entry_timestamp: Timestamp,
    /// Itemized entry costs, combined with exit costs at close.
    pub entry_breakdown: Option<FeeBreakdown>,
    /// Closing fills accumulated so far, empty until the first reduction.
    ///
    /// A position that is reduced rather than closed keeps building this
    /// until it goes flat, so one round trip still produces exactly one
    /// [`Trade`] however many fills it took to get in and out.
    closing: ClosingFills,
    /// Realized PnL settled fill by fill, when the settlement currency has
    /// a smallest unit to settle in; `None` when it declares none.
    ///
    /// A venue does not carry a running total in full precision and round
    /// it at the end -- it books each fill into the account in whole
    /// currency units, and the position's realized PnL is the sum of those
    /// bookings. Over a position unwound in nine fills the two figures
    /// part company, and it is the venue's that is the true one.
    realized: Option<f64>,
}

impl ManagedPosition {
    /// Book one fill into the position's realized PnL.
    ///
    /// `gross_pnl` is what the fill realized before costs -- zero for an
    /// opening fill, which only pays -- and `fees` is what it cost. The two
    /// are combined into the fill's own contribution first and then added
    /// to a total that is already whole money, which is the order the venue
    /// books them in.
    fn settle(&mut self, gross_pnl: f64, fees: f64, precision: Option<u32>) {
        let Some(realized) = self.realized.as_mut() else {
            return;
        };
        let booked = -fees + gross_pnl;
        *realized = quantize_money(*realized + booked, precision);
    }
}

/// Running summary of the fills that are unwinding a position.
///
/// Holds the exit side of the trade record while the position is still
/// partly open. `price` is the size-weighted mean of the fills seen so far,
/// maintained incrementally rather than as `notional / size`, so a position
/// closed by a single fill reports that fill's price bit-for-bit.
#[derive(Debug, Clone, Default)]
struct ClosingFills {
    /// Units closed so far.
    size: f64,
    /// Size-weighted mean exit price.
    price: Price,
    /// Gross PnL realized so far, before fees.
    gross_pnl: f64,
    /// Exit-side fees paid so far.
    fees: f64,
    /// Itemized exit costs accumulated across fills.
    breakdown: Option<FeeBreakdown>,
    /// Bar index, timestamp and reason of the most recent closing fill.
    last: Option<(usize, Timestamp, ExitReason)>,
}

impl ClosingFills {
    /// Fold one closing fill into the running summary.
    fn absorb(
        &mut self,
        size: f64,
        price: Price,
        gross_pnl: f64,
        fees: f64,
        breakdown: Option<FeeBreakdown>,
        at: (usize, Timestamp, ExitReason),
    ) {
        // The first fill adopts its price outright. Averaging it against a
        // zero-size history would compute `price * size / size`, which is
        // not always `price` in binary and would move every existing
        // single-fill trade record by an ULP.
        self.price = if self.size > 0.0 {
            (self.price * self.size + price * size) / (self.size + size)
        } else {
            price
        };
        self.size += size;
        self.gross_pnl += gross_pnl;
        self.fees += fees;
        merge_breakdown(&mut self.breakdown, breakdown);
        self.last = Some(at);
    }
}

/// Fold an itemized cost into a slot that may not have one yet.
fn merge_breakdown(slot: &mut Option<FeeBreakdown>, next: Option<FeeBreakdown>) {
    let Some(next) = next else { return };
    match slot {
        Some(total) => total.add(&next),
        None => *slot = Some(next),
    }
}

/// What a reduction did to a position.
#[derive(Debug, Clone)]
pub enum ReduceOutcome {
    /// The fill was refused: no such position, or a non-positive size.
    None,
    /// Size came off and the position is still open.
    Reduced {
        /// Units actually closed by this fill.
        size: f64,
        /// Units still open.
        remaining: f64,
        /// Gross PnL this fill realized, before any fee.
        gross_pnl: f64,
    },
    /// The last of the position came off; here is the round trip.
    Closed {
        /// Units actually closed by this fill.
        size: f64,
        /// The completed round trip, spanning every fill it took.
        trade: Box<Trade>,
        /// Gross PnL this fill realized, before any fee.
        ///
        /// This fill's share alone, not the round trip's: an account is
        /// settled once per fill, and the earlier fills settled themselves.
        gross_pnl: f64,
    },
}

impl ManagedPosition {
    /// Whether this position's stop level is touched by the bar range.
    pub fn is_stop_hit(&self, low: Price, high: Price) -> bool {
        match self.position.stop_price {
            Some(stop) => match self.position.direction {
                Direction::Long => low <= stop,
                Direction::Short => high >= stop,
            },
            None => false,
        }
    }

    /// Whether this position's target level is touched by the bar range.
    pub fn is_target_hit(&self, low: Price, high: Price) -> bool {
        match self.position.target_price {
            Some(target) => match self.position.direction {
                Direction::Long => high >= target,
                Direction::Short => low <= target,
            },
            None => false,
        }
    }

    /// Ratchet a percent trailing stop off the running extreme.
    pub fn update_trailing_stop(&mut self, trail_percent: f64) {
        let new_stop = match self.position.direction {
            Direction::Long => self.position.highest_since_entry * (1.0 - trail_percent),
            Direction::Short => self.position.lowest_since_entry * (1.0 + trail_percent),
        };
        let improves = match (self.position.stop_price, self.position.direction) {
            (None, _) => true,
            (Some(current), Direction::Long) => new_stop > current,
            (Some(current), Direction::Short) => new_stop < current,
        };
        if improves {
            self.position.stop_price = Some(new_stop);
        }
    }
}

/// Open positions and trade-record bookkeeping for one instrument.
#[derive(Debug)]
pub struct PositionLedger {
    policy: PositionPolicy,
    symbol: String,
    open: Vec<ManagedPosition>,
    trade_counter: u64,
    next_position_id: u64,
    /// Contract point value; see `PositionManager::set_contract_multiplier`.
    contract_multiplier: f64,
    /// Decimal places the settlement currency counts in, when it declares
    /// any; see [`ManagedPosition::realized`].
    currency_precision: Option<u32>,
}

impl PositionLedger {
    pub fn new(symbol: String, policy: PositionPolicy) -> Self {
        Self {
            policy,
            symbol,
            open: Vec::new(),
            trade_counter: 0,
            next_position_id: 0,
            contract_multiplier: 1.0,
            currency_precision: None,
        }
    }

    /// Declare the settlement currency's precision.
    ///
    /// Positions opened after this settle each fill in whole currency
    /// units. Left unset, realized PnL stays the raw floating-point round
    /// trip stock Raptor has always reported.
    pub fn set_currency_precision(&mut self, precision: Option<u32>) {
        self.currency_precision = precision;
    }

    /// Set the contract point value used for PnL and notional calculations.
    pub fn set_contract_multiplier(&mut self, multiplier: f64) {
        self.contract_multiplier = if multiplier > 0.0 { multiplier } else { 1.0 };
    }

    #[inline]
    pub fn policy(&self) -> PositionPolicy {
        self.policy
    }

    /// Symbol this ledger tracks.
    #[inline]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    #[inline]
    pub fn contract_multiplier(&self) -> f64 {
        self.contract_multiplier
    }

    /// Whether any position is open.
    #[inline]
    pub fn is_in_position(&self) -> bool {
        !self.open.is_empty()
    }

    /// Number of open positions.
    #[inline]
    pub fn open_count(&self) -> usize {
        self.open.len()
    }

    /// The earliest-opened position — the legacy single-position view.
    #[inline]
    pub fn first(&self) -> Option<&ManagedPosition> {
        self.open.first()
    }

    /// Mutable view of the earliest-opened position.
    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut ManagedPosition> {
        self.open.first_mut()
    }

    /// All open positions, in opening order.
    #[inline]
    pub fn positions(&self) -> &[ManagedPosition] {
        &self.open
    }

    /// Mutable iteration over open positions, in opening order.
    #[inline]
    pub fn positions_mut(&mut self) -> impl Iterator<Item = &mut ManagedPosition> {
        self.open.iter_mut()
    }

    /// A position by ledger id.
    pub fn get(&self, id: u64) -> Option<&ManagedPosition> {
        self.open.iter().find(|p| p.id == id)
    }

    /// Mutable view of a position by ledger id.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut ManagedPosition> {
        self.open.iter_mut().find(|p| p.id == id)
    }

    /// Open a position; returns its ledger id, or `None` when the Net
    /// policy already holds one.
    #[allow(clippy::too_many_arguments)]
    pub fn open_position(
        &mut self,
        idx: usize,
        timestamp: Timestamp,
        price: Price,
        size: f64,
        direction: Direction,
        stop_price: Option<Price>,
        target_price: Option<Price>,
        entry_fees: f64,
        entry_breakdown: Option<FeeBreakdown>,
    ) -> Option<u64> {
        if self.policy != PositionPolicy::Independent && !self.open.is_empty() {
            return None;
        }
        let mut position = Position::new();
        position.open(idx, price, size, direction, stop_price, target_price, entry_fees);
        let id = self.next_position_id;
        self.next_position_id += 1;
        let mut managed = ManagedPosition {
            id,
            position,
            entry_timestamp: timestamp,
            entry_breakdown,
            closing: ClosingFills::default(),
            realized: self.currency_precision.is_some().then_some(0.0),
        };
        managed.settle(0.0, entry_fees, self.currency_precision);
        self.open.push(managed);
        Some(id)
    }

    /// Add units to an open position at a size-weighted average entry.
    ///
    /// This is the netting behavior of a real exchange account: a second
    /// buy while long does not open a second trade, it makes the one
    /// position bigger and moves its average entry. Only
    /// [`PositionPolicy::NetAveraging`] routes here — [`PositionPolicy::Net`]
    /// refuses the fill outright and [`PositionPolicy::Independent`] opens a
    /// separate position instead.
    ///
    /// Returns `false` when the id is unknown, the size is not positive, or
    /// the fill opposes the position's direction (a reduction, not an add).
    pub fn add_to_position(
        &mut self,
        id: u64,
        price: Price,
        size: f64,
        direction: Direction,
        entry_fees: f64,
        entry_breakdown: Option<FeeBreakdown>,
    ) -> bool {
        if !(size > 0.0) {
            return false;
        }
        let precision = self.currency_precision;
        let Some(managed) = self.open.iter_mut().find(|p| p.id == id) else {
            return false;
        };
        let pos = &mut managed.position;
        if pos.direction != direction || !(pos.size > 0.0) {
            return false;
        }
        // Weighted incrementally, for the same reason the exit side is: a
        // position that never gets added to must keep the exact entry price
        // it opened at.
        pos.entry_price = (pos.entry_price * pos.size + price * size) / (pos.size + size);
        pos.size += size;
        pos.entry_fees += entry_fees;
        merge_breakdown(&mut managed.entry_breakdown, entry_breakdown);
        managed.settle(0.0, entry_fees, precision);
        true
    }

    /// Take `size` units off a position, closing it if that is all of it.
    ///
    /// The size is clamped to what the position actually holds, so a caller
    /// may always ask for more than is open. Fees and the exit price in
    /// `exit` describe *this fill*; they are accumulated onto the position
    /// and reported once, on the fill that takes it flat.
    ///
    /// Trade ids keep the legacy numbering: sequential in close order.
    pub fn reduce_position(&mut self, id: u64, size: f64, exit: ExitDetails) -> ReduceOutcome {
        let Some(index) = self.open.iter().position(|p| p.id == id) else {
            return ReduceOutcome::None;
        };
        let contract_multiplier = self.contract_multiplier;
        let precision = self.currency_precision;
        let managed = &mut self.open[index];
        let filled = size.min(managed.position.size);
        if !(filled > 0.0) {
            return ReduceOutcome::None;
        }

        let multiplier = managed.position.direction.multiplier() * contract_multiplier;
        let gross_pnl = (exit.price - managed.position.entry_price) * filled * multiplier;
        managed.closing.absorb(
            filled,
            exit.price,
            gross_pnl,
            exit.fees,
            exit.fee_breakdown,
            (exit.idx, exit.timestamp, exit.reason),
        );
        managed.settle(gross_pnl, exit.fees, precision);
        managed.position.size -= filled;

        if managed.position.size > 0.0 {
            return ReduceOutcome::Reduced {
                size: filled,
                remaining: managed.position.size,
                gross_pnl,
            };
        }

        let managed = self.open.remove(index);
        let trade = self.create_trade(&managed, exit.entry_timestamp);
        self.trade_counter += 1;
        ReduceOutcome::Closed { size: filled, trade: Box::new(trade), gross_pnl }
    }

    /// Close a position by id and produce its trade record.
    ///
    /// Convenience over [`Self::reduce_position`] for the callers that
    /// always take the whole position: a stop, a target, a liquidation, or
    /// end-of-data finalization.
    pub fn close_position(&mut self, id: u64, exit: ExitDetails) -> Option<Trade> {
        match self.reduce_position(id, f64::INFINITY, exit) {
            ReduceOutcome::Closed { trade, .. } => Some(*trade),
            _ => None,
        }
    }

    /// Trade record for a position that has just gone flat.
    ///
    /// Reads the closing accumulator rather than a single exit, so one
    /// record describes the whole round trip however many fills it took.
    /// For the common case -- opened by one fill, closed by one fill -- every
    /// term is bit-identical to the legacy single-fill arithmetic.
    ///
    /// Note that `size` is the size *closed*, taken from the accumulator:
    /// `position.size` has already been drawn down to zero by then.
    fn create_trade(&self, managed: &ManagedPosition, entry_timestamp: Timestamp) -> Trade {
        let pos = &managed.position;
        let closing = &managed.closing;
        let (exit_idx, exit_timestamp, exit_reason) =
            closing.last.unwrap_or((pos.entry_idx, entry_timestamp, ExitReason::Signal));

        let total_fees = pos.entry_fees + closing.fees;
        // The settled total when the currency has units to settle in, the
        // raw round trip when it has none. The two agree on a position
        // taken off in one fill; they part company on one taken off in
        // nine, and the settled figure is what the venue's books carry.
        let pnl = managed.realized.unwrap_or(closing.gross_pnl - total_fees);

        let cost_basis = pos.entry_price * closing.size * self.contract_multiplier;
        let return_pct = if cost_basis > 0.0 { pnl / cost_basis * 100.0 } else { 0.0 };

        Trade {
            id: self.trade_counter,
            symbol: self.symbol.clone(),
            entry_idx: pos.entry_idx,
            exit_idx,
            entry_price: pos.entry_price,
            exit_price: closing.price,
            size: closing.size,
            direction: pos.direction,
            pnl,
            return_pct,
            entry_time: entry_timestamp,
            exit_time: exit_timestamp,
            fees: total_fees,
            entry_fees: pos.entry_fees,
            exit_fees: closing.fees,
            fee_breakdown: match (managed.entry_breakdown, closing.breakdown) {
                (Some(entry), Some(exit)) => {
                    let mut total = entry;
                    total.add(&exit);
                    Some(total)
                }
                (entry, exit) => entry.or(exit),
            },
            exit_reason,
        }
    }

    /// Track bar extremes on every open position (for trailing stops).
    pub fn update_price(&mut self, high: Price, low: Price) {
        for managed in &mut self.open {
            managed.position.update_extremes(high, low);
        }
    }

    /// Direction-aware unrealized PnL across open positions.
    pub fn unrealized_total(&self, price: Price) -> f64 {
        self.open.iter().map(|p| p.position.unrealized_pnl(price) * self.contract_multiplier).sum()
    }

    /// Notional value of open positions at the given price (unsigned).
    pub fn notional_total(&self, price: Price) -> f64 {
        self.open.iter().map(|p| price * p.position.size * self.contract_multiplier).sum()
    }

    /// Total market value of open positions at the given price.
    ///
    /// `price * size` for every direction — the fully-funded model the
    /// engine has always used (shorts included; the golden suite pins it).
    /// Direction-aware marking arrives with the margin account layer, which
    /// owns short cash-flow properly.
    pub fn position_value(&self, price: Price) -> f64 {
        self.open.iter().map(|p| price * p.position.size * self.contract_multiplier).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::ExitReason;

    fn exit(idx: usize, price: Price) -> ExitDetails {
        ExitDetails {
            idx,
            timestamp: idx as i64,
            price,
            entry_timestamp: 0,
            reason: ExitReason::Signal,
            fees: 0.0,
            fee_breakdown: None,
        }
    }

    fn exit_with(idx: usize, price: Price, fees: f64) -> ExitDetails {
        ExitDetails { fees, ..exit(idx, price) }
    }

    #[test]
    fn a_position_unwound_in_pieces_reports_one_round_trip() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Net);
        let id =
            ledger.open_position(0, 0, 100.0, 10.0, Direction::Long, None, None, 2.0, None).unwrap();

        // Three units off at 110, the rest at 120. No trade until flat --
        // one position is one trade, however many fills it took.
        match ledger.reduce_position(id, 3.0, exit_with(1, 110.0, 0.3)) {
            ReduceOutcome::Reduced { size, remaining, gross_pnl } => {
                assert_eq!(size, 3.0);
                assert_eq!(remaining, 7.0);
                // This fill's own gross: 3 units, 10 a unit.
                assert_eq!(gross_pnl, 30.0);
            }
            other => panic!("expected a reduction, got {other:?}"),
        }
        assert_eq!(ledger.open_count(), 1);

        let trade = match ledger.reduce_position(id, 7.0, exit_with(2, 120.0, 0.7)) {
            ReduceOutcome::Closed { size, trade, gross_pnl } => {
                assert_eq!(size, 7.0);
                // The closing fill's own gross, not the round trip's 170.
                assert_eq!(gross_pnl, 140.0);
                *trade
            }
            other => panic!("expected a close, got {other:?}"),
        };
        assert_eq!(ledger.open_count(), 0);

        assert_eq!(trade.size, 10.0);
        // Size-weighted: (110*3 + 120*7) / 10.
        assert_eq!(trade.exit_price, 117.0);
        assert_eq!(trade.entry_price, 100.0);
        // Gross is accumulated per fill: 3*10 + 7*20.
        assert_eq!(trade.exit_fees, 1.0);
        assert_eq!(trade.entry_fees, 2.0);
        assert_eq!(trade.fees, 3.0);
        assert_eq!(trade.pnl, 170.0 - 3.0);
        assert_eq!(trade.exit_idx, 2);
    }

    #[test]
    fn a_currency_with_units_settles_every_fill_in_them() {
        // A venue books each fill into the account in whole currency units
        // and the position's realized PnL is the sum of those bookings.
        // Three fills each leave four tenths of a cent behind; none of them
        // survives to the total.
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Net);
        ledger.set_currency_precision(Some(2));
        let id =
            ledger.open_position(0, 0, 100.0, 3.0, Direction::Long, None, None, 0.0, None).unwrap();
        ledger.reduce_position(id, 1.0, exit(1, 101.004));
        ledger.reduce_position(id, 1.0, exit(2, 101.004));
        let trade = match ledger.reduce_position(id, 1.0, exit(3, 101.004)) {
            ReduceOutcome::Closed { trade, .. } => *trade,
            other => panic!("expected a close, got {other:?}"),
        };
        assert_eq!(trade.pnl, 3.0);
    }

    #[test]
    fn a_currency_with_no_declared_units_keeps_the_raw_round_trip() {
        // Nothing to settle in, so the trade reports the floating-point
        // difference it always has -- tails and all.
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Net);
        let id =
            ledger.open_position(0, 0, 100.0, 3.0, Direction::Long, None, None, 0.0, None).unwrap();
        ledger.reduce_position(id, 1.0, exit(1, 101.004));
        ledger.reduce_position(id, 1.0, exit(2, 101.004));
        let trade = match ledger.reduce_position(id, 1.0, exit(3, 101.004)) {
            ReduceOutcome::Closed { trade, .. } => *trade,
            other => panic!("expected a close, got {other:?}"),
        };
        let fill = 101.004 - 100.0;
        assert_eq!(trade.pnl, fill + fill + fill);
        assert_ne!(trade.pnl, 3.0);
    }

    #[test]
    fn an_entry_fee_is_booked_when_it_is_paid() {
        // The fee is realized the moment it is charged, so it settles with
        // the entry. Six tenths of a cent charged at the open is a cent off
        // the account, not a fraction waiting to be rounded away at the
        // close.
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Net);
        ledger.set_currency_precision(Some(2));
        let id = ledger
            .open_position(0, 0, 100.0, 1.0, Direction::Long, None, None, 0.006, None)
            .unwrap();
        let trade = ledger.close_position(id, exit(1, 100.0)).unwrap();
        assert_eq!(trade.pnl, -0.01);
    }

    #[test]
    fn a_single_fill_close_keeps_its_exit_price_bit_for_bit() {
        // The weighted mean must not touch a price that had nothing to be
        // averaged against: `price * size / size` is not always `price`.
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Net);
        let id = ledger
            .open_position(0, 0, 0.1, 3.0, Direction::Long, None, None, 0.0, None)
            .unwrap();
        let trade = ledger.close_position(id, exit(1, 92_104.5)).unwrap();
        assert_eq!(trade.exit_price, 92_104.5);
        assert_eq!(trade.size, 3.0);
    }

    #[test]
    fn reducing_more_than_is_open_closes_the_position() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Net);
        let id =
            ledger.open_position(0, 0, 100.0, 4.0, Direction::Long, None, None, 0.0, None).unwrap();
        match ledger.reduce_position(id, 99.0, exit(1, 105.0)) {
            ReduceOutcome::Closed { size, trade, .. } => {
                assert_eq!(size, 4.0, "the fill is clamped to what is open");
                assert_eq!(trade.size, 4.0);
            }
            other => panic!("expected a close, got {other:?}"),
        }
    }

    #[test]
    fn averaging_policy_grows_one_position_instead_of_refusing() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::NetAveraging);
        let id =
            ledger.open_position(0, 0, 100.0, 6.0, Direction::Long, None, None, 1.0, None).unwrap();
        assert!(ledger.add_to_position(id, 110.0, 4.0, Direction::Long, 0.5, None));
        assert_eq!(ledger.open_count(), 1, "averaging must not open a second position");

        let pos = &ledger.get(id).unwrap().position;
        assert_eq!(pos.size, 10.0);
        // (100*6 + 110*4) / 10.
        assert_eq!(pos.entry_price, 104.0);
        assert_eq!(pos.entry_fees, 1.5);
    }

    #[test]
    fn averaging_refuses_a_fill_that_opposes_the_position() {
        // Reducing is not adding; routing an opposing fill here would
        // silently grow the position it was meant to close.
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::NetAveraging);
        let id =
            ledger.open_position(0, 0, 100.0, 6.0, Direction::Long, None, None, 0.0, None).unwrap();
        assert!(!ledger.add_to_position(id, 110.0, 4.0, Direction::Short, 0.0, None));
        assert!(!ledger.add_to_position(id, 110.0, 0.0, Direction::Long, 0.0, None));
        assert_eq!(ledger.get(id).unwrap().position.size, 6.0);
    }

    #[test]
    fn net_policy_refuses_second_position() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Net);
        let a = ledger.open_position(0, 0, 100.0, 10.0, Direction::Long, None, None, 0.0, None);
        assert!(a.is_some());
        let b = ledger.open_position(1, 1, 101.0, 10.0, Direction::Long, None, None, 0.0, None);
        assert!(b.is_none());
        assert_eq!(ledger.open_count(), 1);
    }

    #[test]
    fn independent_policy_holds_both_directions() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Independent);
        let long = ledger
            .open_position(0, 0, 100.0, 10.0, Direction::Long, None, None, 0.0, None)
            .unwrap();
        let short = ledger
            .open_position(1, 1, 102.0, 5.0, Direction::Short, None, None, 0.0, None)
            .unwrap();
        assert_eq!(ledger.open_count(), 2);
        assert_ne!(long, short);

        // Close the short at a profit; the long stays open.
        let trade = ledger.close_position(short, exit(2, 98.0)).unwrap();
        assert!((trade.pnl - (102.0 - 98.0) * 5.0).abs() < 1e-9);
        assert_eq!(ledger.open_count(), 1);
        assert_eq!(ledger.first().unwrap().id, long);
    }

    #[test]
    fn trade_ids_are_sequential_in_close_order() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Independent);
        let a =
            ledger.open_position(0, 0, 100.0, 1.0, Direction::Long, None, None, 0.0, None).unwrap();
        let b =
            ledger.open_position(0, 0, 100.0, 1.0, Direction::Long, None, None, 0.0, None).unwrap();
        // Close b first: it takes trade id 0.
        assert_eq!(ledger.close_position(b, exit(1, 101.0)).unwrap().id, 0);
        assert_eq!(ledger.close_position(a, exit(2, 102.0)).unwrap().id, 1);
    }

    #[test]
    fn per_position_stops_and_trailing() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Independent);
        let long = ledger
            .open_position(0, 0, 100.0, 1.0, Direction::Long, Some(95.0), None, 0.0, None)
            .unwrap();
        let short = ledger
            .open_position(0, 0, 100.0, 1.0, Direction::Short, Some(105.0), None, 0.0, None)
            .unwrap();

        // Bar range 97..103 hits neither stop.
        assert!(!ledger.get(long).unwrap().is_stop_hit(97.0, 103.0));
        assert!(!ledger.get(short).unwrap().is_stop_hit(97.0, 103.0));
        // 94 low hits the long's stop only.
        assert!(ledger.get(long).unwrap().is_stop_hit(94.0, 103.0));
        assert!(!ledger.get(short).unwrap().is_stop_hit(94.0, 103.0));

        // Trailing ratchets each side toward its own extreme.
        ledger.update_price(110.0, 94.0);
        ledger.get_mut(long).unwrap().update_trailing_stop(0.05);
        ledger.get_mut(short).unwrap().update_trailing_stop(0.05);
        assert!((ledger.get(long).unwrap().position.stop_price.unwrap() - 104.5).abs() < 1e-9);
        assert!((ledger.get(short).unwrap().position.stop_price.unwrap() - 98.7).abs() < 1e-9);
    }

    #[test]
    fn multiplier_scales_trade_pnl() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Net);
        ledger.set_contract_multiplier(50.0);
        let id =
            ledger.open_position(0, 0, 100.0, 2.0, Direction::Long, None, None, 0.0, None).unwrap();
        let trade = ledger.close_position(id, exit(1, 101.0)).unwrap();
        assert!((trade.pnl - 1.0 * 2.0 * 50.0).abs() < 1e-9);
    }
}
