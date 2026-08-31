//! Order types and the order state machine.

use serde::{Deserialize, Serialize};

use crate::core::types::{Price, Timestamp};

/// Which way an order trades.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    /// The opposite side.
    #[inline]
    pub fn flip(self) -> Self {
        match self {
            OrderSide::Buy => OrderSide::Sell,
            OrderSide::Sell => OrderSide::Buy,
        }
    }
}

/// How much an order trades.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QtySpec {
    /// Explicit unit/contract count.
    Units(f64),
    /// Fraction of available capital, resolved to units at fill time —
    /// equity moves while an order rests, so resolving at accept time would
    /// size against a stale account.
    CapitalFrac(f64),
    /// Whatever the open position holds when the order fills (close-all).
    FullPosition,
}

/// How a trailing offset is measured.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrailOffset {
    /// Absolute price distance from the watermark.
    Price(f64),
    /// Basis points of the watermark.
    Bps(f64),
}

impl TrailOffset {
    /// Offset in price terms at the given watermark.
    #[inline]
    pub fn at(&self, watermark: Price) -> f64 {
        match self {
            TrailOffset::Price(p) => *p,
            TrailOffset::Bps(b) => watermark * b / 10_000.0,
        }
    }
}

/// Order flavor and its price levels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderKind {
    /// Fill at the engine's configured fill-price model on the current bar.
    Market,
    /// Rest until the market trades at or through `price`.
    Limit { price: Price },
    /// Become marketable once the market trades adversely through `trigger`.
    StopMarket { trigger: Price },
    /// Once triggered, rest as a limit at `price`.
    StopLimit { trigger: Price, price: Price },
    /// Become marketable once the market trades *favorably* to `trigger`
    /// (a buy triggers when price falls to it) — the stop's mirror.
    MarketIfTouched { trigger: Price },
    /// Favorable touch at `trigger`, then rest as a limit at `price`.
    LimitIfTouched { trigger: Price, price: Price },
    /// Fill at the next bar's open. Without partial fills the "remainder
    /// rests as a limit" phase never occurs, so this is exactly an
    /// at-the-open market order; the distinction returns with book depth.
    MarketToLimit,
    /// Stop whose trigger trails the running favorable extreme by `offset`.
    TrailingStopMarket { offset: TrailOffset },
    /// Trailing stop that, once triggered, rests as a limit `limit_offset`
    /// through the trigger (more marketable, bounded slippage).
    TrailingStopLimit { offset: TrailOffset, limit_offset: f64 },
}

/// How long an order stays working.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Rest until filled or canceled.
    Gtc,
    /// Expire when the trading date rolls over past the submission date.
    /// The date is UTC unless `session_tz_offset_ns` shifts it.
    Day,
    /// Expire at an explicit timestamp.
    Gtd { expire_ns: Timestamp },
    /// Match against exactly one bar, then cancel.
    Ioc,
    /// Fill completely against one bar or cancel. Without partial fills this
    /// behaves like [`TimeInForce::Ioc`]; the distinction becomes real when
    /// the position ledger introduces partial fills.
    Fok,
    /// Market order queued to fill at the next bar's open.
    AtOpen,
    /// Market order queued to fill at the next bar's close.
    AtClose,
}

/// Order lifecycle states.
///
/// Transitions are enforced by [`Order::transition`]; anything not listed
/// there is a logic error, caught in debug builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    /// Created, not yet evaluated by the engine.
    Submitted,
    /// Working: resting in the engine's order book.
    Accepted,
    /// Stop trigger touched; a stop-limit now rests as a limit.
    Triggered,
    /// Working: some size has filled and the rest is still resting.
    ///
    /// Reached only when a liquidity model bounds a fill -- see
    /// [`BarLiquidity`]. Without one every fill is for the order's whole
    /// size and this state is unreachable.
    ///
    /// [`BarLiquidity`]: crate::execution::fill::BarLiquidity
    PartiallyFilled,
    /// Terminal: filled.
    Filled,
    /// Terminal: canceled by the strategy or by IOC/FOK exhaustion.
    Canceled,
    /// Terminal: time-in-force expired.
    Expired,
    /// Terminal: refused (e.g. opening order while a position is open).
    Rejected,
}

impl OrderStatus {
    /// Whether this is an end state.
    ///
    /// [`PartiallyFilled`](Self::PartiallyFilled) is deliberately not one:
    /// the unfilled remainder is still working and must keep matching.
    #[inline]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            OrderStatus::Filled
                | OrderStatus::Canceled
                | OrderStatus::Expired
                | OrderStatus::Rejected
        )
    }
}

/// One working or finished order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Order {
    /// Engine-assigned id, unique within a session, monotonically increasing
    /// in submission order — matching iterates ids ascending, so submission
    /// order is the deterministic tiebreak.
    pub id: u64,
    /// Caller-supplied identifier, echoed on every event.
    pub client_id: String,
    pub side: OrderSide,
    pub qty: QtySpec,
    pub kind: OrderKind,
    pub tif: TimeInForce,
    pub status: OrderStatus,
    /// Bar index on which the order was submitted. Resting orders begin
    /// matching on the *next* bar: an order cannot rest into a bar that had
    /// already closed when it was placed.
    pub submitted_idx: usize,
    /// Submission timestamp, for DAY expiry.
    pub submitted_ts: Timestamp,
    /// Protective stop attached to the position this order opens.
    pub stop_price: Option<Price>,
    /// Protective target attached to the position this order opens.
    pub target_price: Option<Price>,
    /// Limit orders only: reject instead of filling if marketable at the
    /// open of the first bar the order rests into.
    pub post_only: bool,
    /// The order reached the venue before the bar it was submitted on did,
    /// so it meets the book the *previous* step left behind.
    ///
    /// A venue processes market data one instrument at a time. A strategy
    /// trading a basket decides on the bar of whichever name printed first
    /// and sends orders for the rest, whose bars for that same instant have
    /// not reached the venue yet: those orders are matched against a book
    /// one bar older than their own timestamp. `false`, the default, keeps
    /// the book the submission bar leaves behind.
    #[serde(default)]
    pub arrives_before_bar: bool,
    /// Reject fills that would open a position (closing fills only).
    pub reduce_only: bool,
    /// One-triggers-other: held (not matched) until the parent order fills;
    /// canceled if the parent dies unfilled.
    pub parent_id: Option<u64>,
    /// Schedule that released this order, when it is an algo slice.
    /// Purely a back-pointer: slices match like any other order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algo_id: Option<u64>,
    /// One-cancels-other group: when any member fills, working siblings are
    /// canceled. One-updates-other reduces to this without partial fills.
    pub oco_group: Option<u64>,
    /// Trailing orders: running favorable extreme since acceptance.
    pub trail_watermark: Option<Price>,
    /// Trailing stop-limit: the limit price fixed at trigger time.
    pub trail_limit: Option<Price>,
    /// Whether this order's stop trigger has been touched.
    ///
    /// Kept apart from [`OrderStatus::Triggered`] because the two answer
    /// different questions. The status is a lifecycle state and a partly
    /// filled order has to report [`OrderStatus::PartiallyFilled`]; whether
    /// its trigger already fired is a separate fact that must survive that,
    /// or a stop-limit would arm itself twice.
    #[serde(default)]
    pub triggered: bool,
    /// Units filled so far. Nonzero only under a bounding liquidity model.
    #[serde(default)]
    pub filled_qty: f64,
    /// The unit count this order's [`QtySpec`] resolved to, fixed at its
    /// first fill.
    ///
    /// A capital-fraction order has no unit count until something prices
    /// it, and the remainder after a partial fill has to be the rest of
    /// *that* size -- re-sizing it against a later bar would let one order
    /// quietly grow or shrink between fills.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_qty: Option<f64>,
}

impl Order {
    /// A plain unlinked GTC-style order in `Submitted` state.
    ///
    /// Construction sites set ids, timestamps, flags, and links on top;
    /// keeping one canonical literal avoids churn as fields grow.
    pub fn plain(side: OrderSide, qty: QtySpec, kind: OrderKind, tif: TimeInForce) -> Self {
        Self {
            id: 0,
            client_id: String::new(),
            side,
            qty,
            kind,
            tif,
            status: OrderStatus::Submitted,
            submitted_idx: 0,
            submitted_ts: 0,
            stop_price: None,
            target_price: None,
            post_only: false,
            arrives_before_bar: false,
            reduce_only: false,
            parent_id: None,
            algo_id: None,
            oco_group: None,
            trail_watermark: None,
            trail_limit: None,
            triggered: false,
            filled_qty: 0.0,
            resolved_qty: None,
        }
    }

    /// The remainder under which an order counts as wholly filled.
    ///
    /// Fill sizes are accumulated in floating point and can land a fraction
    /// of an ULP short of the total they were measured against; a venue
    /// holds quantities as fixed-precision decimals and has no such
    /// remainder. One tolerance governs both the leaves and the status, so
    /// an order can never report itself filled while still showing size to
    /// fill, nor chase a sliver no instrument can express.
    #[inline]
    fn residual_tolerance(total: f64) -> f64 {
        f64::EPSILON * total.abs().max(1.0) * 4.0
    }

    /// Units still to fill, once the order's size has been resolved.
    ///
    /// `None` before the first fill, when a capital-fraction order has no
    /// unit count yet.
    #[inline]
    pub fn leaves_qty(&self) -> Option<f64> {
        self.resolved_qty.map(|total| {
            let leaves = total - self.filled_qty;
            if leaves <= Self::residual_tolerance(total) { 0.0 } else { leaves }
        })
    }

    /// Record `units` as filled, returning the status the order now holds.
    ///
    /// `total` is the size the order resolved to; it is pinned on the first
    /// fill and every later fill is measured against that same number.
    pub fn record_fill(&mut self, units: f64, total: f64) -> OrderStatus {
        let total = *self.resolved_qty.get_or_insert(total);
        self.filled_qty += units;
        // A fill that lands within a rounding error of the whole order is
        // the whole order.
        let done = total - self.filled_qty <= Self::residual_tolerance(total);
        if done {
            OrderStatus::Filled
        } else {
            OrderStatus::PartiallyFilled
        }
    }

    /// Move to a new status, checking the transition is legal.
    ///
    /// Returns `false` (and leaves the order untouched) on an illegal
    /// transition; callers treat that as a bug, not a user error.
    #[must_use]
    pub fn transition(&mut self, to: OrderStatus) -> bool {
        use OrderStatus::*;
        let ok = matches!(
            (self.status, to),
            (Submitted, Accepted)
                | (Submitted, Rejected)
                | (Submitted, Canceled)
                | (Accepted, Triggered)
                | (Accepted, Filled)
                | (Accepted, PartiallyFilled)
                | (Accepted, Canceled)
                | (Accepted, Expired)
                | (Accepted, Rejected)
                | (Triggered, Filled)
                | (Triggered, PartiallyFilled)
                | (Triggered, Canceled)
                | (Triggered, Expired)
                | (Triggered, Rejected)
                // A partly-filled order is still working: it may take more
                // size, be finished off, or die with its remainder unfilled.
                | (PartiallyFilled, PartiallyFilled)
                | (PartiallyFilled, Filled)
                | (PartiallyFilled, Canceled)
                | (PartiallyFilled, Expired)
                | (PartiallyFilled, Rejected)
        );
        if ok {
            self.status = to;
            // Latch the trigger here rather than at the call sites, so no
            // path can arm an order without recording that it armed.
            self.triggered |= to == Triggered;
        }
        debug_assert!(ok, "illegal order transition {:?} -> {to:?}", self.status);
        ok
    }

    /// The price a resting order would currently fill or trigger at.
    #[inline]
    pub fn working_price(&self) -> Option<Price> {
        let triggered = self.triggered;
        match self.kind {
            OrderKind::Market | OrderKind::MarketToLimit => None,
            OrderKind::Limit { price } => Some(price),
            OrderKind::StopMarket { trigger } | OrderKind::MarketIfTouched { trigger } => {
                Some(trigger)
            }
            OrderKind::StopLimit { trigger, price }
            | OrderKind::LimitIfTouched { trigger, price } => {
                Some(if triggered { price } else { trigger })
            }
            OrderKind::TrailingStopMarket { offset } => {
                self.trail_watermark.map(|wm| self.trail_trigger(wm, offset))
            }
            OrderKind::TrailingStopLimit { offset, .. } => {
                if triggered {
                    self.trail_limit
                } else {
                    self.trail_watermark.map(|wm| self.trail_trigger(wm, offset))
                }
            }
        }
    }

    /// Trigger level implied by a watermark for a trailing order.
    ///
    /// A sell trail protects a long: trigger below the running high. A buy
    /// trail mirrors it below-market entry style: trigger above the running
    /// low.
    #[inline]
    pub fn trail_trigger(&self, watermark: Price, offset: TrailOffset) -> Price {
        match self.side {
            OrderSide::Sell => watermark - offset.at(watermark),
            OrderSide::Buy => watermark + offset.at(watermark),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(kind: OrderKind) -> Order {
        let mut o = Order::plain(OrderSide::Buy, QtySpec::Units(1.0), kind, TimeInForce::Gtc);
        o.id = 1;
        o.client_id = "t-1".into();
        o
    }

    #[test]
    fn legal_lifecycle_paths() {
        let mut o = order(OrderKind::Limit { price: 100.0 });
        assert!(o.transition(OrderStatus::Accepted));
        assert!(o.transition(OrderStatus::Filled));
        assert!(o.status.is_terminal());
    }

    #[test]
    fn stop_limit_triggers_then_fills() {
        let mut o = order(OrderKind::StopLimit { trigger: 105.0, price: 104.5 });
        assert!(o.transition(OrderStatus::Accepted));
        assert_eq!(o.working_price(), Some(105.0));
        assert!(o.transition(OrderStatus::Triggered));
        assert_eq!(o.working_price(), Some(104.5));
        assert!(o.transition(OrderStatus::Filled));
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn illegal_transition_is_refused() {
        let mut o = order(OrderKind::Market);
        assert!(o.transition(OrderStatus::Accepted));
        assert!(o.transition(OrderStatus::Filled));
        assert!(!o.transition(OrderStatus::Accepted));
        assert_eq!(o.status, OrderStatus::Filled);
    }
}
