//! Resting-order book and bar matching.
//!
//! [`OrderEngine`] owns every order submitted to a session and evaluates the
//! working ones against each incoming bar. It is deliberately ignorant of
//! positions and cash: it reports *what matched at which price*, and the
//! kernel decides whether the fill opens or closes a position (and may still
//! reject it there). Market orders never fill here — the kernel sweeps them
//! itself, on the submission bar (same-bar timing) or the bar after
//! (next-bar-open), mirroring the signal-entry path.

use crate::core::types::{Direction, OhlcvBar, Price, Timestamp};
use crate::execution::fill::{BarTape, FillDepth, FillModel, StepKind};
use crate::execution::orders::order::{Order, OrderKind, OrderStatus, TimeInForce};
use crate::execution::orders::OrderSide;

const NS_PER_DAY: i64 = 86_400_000_000_000;

/// Whether a match pass is driven by a bar or a single trade print.
#[derive(Debug, Clone, Copy, PartialEq)]
enum MatchMode {
    Bar,
    Trade,
}

/// What happened to one order during a bar.
#[derive(Debug, Clone, PartialEq)]
pub enum MatchOutcome {
    /// The order is marketable at `price` (limit price, or stop fill price
    /// including gap-through adjustment), against `depth` of book.
    ///
    /// `depth` carries raw, un-quantized sizes: the kernel rounds them to
    /// the instrument's size grid before clamping, because a venue cannot
    /// fill a fraction of a lot any more than it can quote one.
    ///
    /// `on_arrival` marks a fill taken from the book standing when the
    /// order reached the venue, ahead of this bar -- it happened at that
    /// instant, not when the bar printed. Every other fill, including one
    /// the same order takes from the bar it beat, happened with its print.
    Fill { order_id: u64, price: Price, depth: FillDepth, on_arrival: bool },
    /// A stop-limit's trigger was touched; its limit now rests and becomes
    /// marketable from the next bar.
    Trigger { order_id: u64 },
    /// Time-in-force lapsed before a fill.
    Expire { order_id: u64 },
    /// An IOC/FOK order found no fill on its evaluation bar, or a held
    /// child's parent died unfilled.
    Cancel { order_id: u64 },
    /// The book refused the order (e.g. a post-only limit that was
    /// immediately marketable).
    Reject { order_id: u64, reason: &'static str },
}

/// Map an order side onto the (direction, is_entry) pairs the fill helpers
/// key on. Buy behaves like a long entry, sell like a long exit; the other
/// two pairs are aliases of these.
#[inline]
fn side_as_fill_args(side: OrderSide) -> (Direction, bool) {
    match side {
        OrderSide::Buy => (Direction::Long, true),
        OrderSide::Sell => (Direction::Long, false),
    }
}

/// Owns all orders for one session and matches resting ones per bar.
#[derive(Debug, Default)]
pub struct OrderEngine {
    orders: Vec<Order>,
    next_id: u64,
    /// Offset applied before deriving the trading date for DAY expiry.
    /// `0` is UTC, which is what `Default` yields.
    tz_offset_ns: i64,
    /// Opt-in compatibility for decisions made by a composite callback just
    /// before the primary close with the same timestamp.
    same_bar_marketable_limit_on_close: bool,
    /// The venue's tape: what last traded and the size showing there. It
    /// carries across bars because a bar that never leaves the last traded
    /// price prints nothing and leaves the book untouched.
    tape: BarTape,
}

impl OrderEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// An engine whose DAY orders expire on a trading date offset from UTC.
    pub fn with_tz_offset(tz_offset_ns: i64, same_bar_marketable_limit_on_close: bool) -> Self {
        Self { tz_offset_ns, same_bar_marketable_limit_on_close, ..Self::default() }
    }

    /// Size the venue's book is showing, unbounded until something trades.
    ///
    /// The kernel fills market orders itself, outside the resting-order
    /// pass, and they meet the same book any other order submitted during
    /// a bar meets.
    pub fn book_size(&self) -> f64 {
        self.tape.book_size()
    }

    /// Tag an order as a slice of an execution schedule.
    pub fn set_algo_id(&mut self, order_id: u64, algo_id: Option<u64>) {
        if let Some(order) = self.get_mut(order_id) {
            order.algo_id = algo_id;
        }
    }

    /// Working orders released by a schedule.
    pub fn algo_order_ids(&self, algo_id: u64) -> Vec<u64> {
        self.orders
            .iter()
            .filter(|o| o.algo_id == Some(algo_id) && !o.status.is_terminal())
            .map(|o| o.id)
            .collect()
    }

    /// Register a new order and return its engine id.
    ///
    /// The order arrives `Submitted`; the caller decides whether it is
    /// accepted (resting kinds) or handled immediately (market kinds).
    #[allow(clippy::too_many_arguments)]
    pub fn submit(&mut self, mut order: Order) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        order.id = id;
        self.orders.push(order);
        id
    }

    /// Shared view of an order by id.
    pub fn get(&self, id: u64) -> Option<&Order> {
        self.orders.iter().find(|o| o.id == id)
    }

    /// Mutable view of an order by id.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Order> {
        self.orders.iter_mut().find(|o| o.id == id)
    }

    /// All non-terminal orders, in submission order.
    pub fn working(&self) -> impl Iterator<Item = &Order> {
        self.orders.iter().filter(|o| !o.status.is_terminal())
    }

    /// Every order ever submitted, in submission order.
    pub fn all(&self) -> &[Order] {
        &self.orders
    }

    /// Cancel a working order. Returns `false` for unknown or finished ids.
    pub fn cancel(&mut self, id: u64) -> bool {
        match self.get_mut(id) {
            Some(order) if !order.status.is_terminal() => order.transition(OrderStatus::Canceled),
            _ => false,
        }
    }

    /// Cancel every working order, returning the ids canceled.
    pub fn cancel_all(&mut self) -> Vec<u64> {
        let ids: Vec<u64> =
            self.orders.iter().filter(|o| !o.status.is_terminal()).map(|o| o.id).collect();
        for id in &ids {
            let _ = self.cancel(*id);
        }
        ids
    }

    /// Replace a working order's price levels and/or quantity.
    ///
    /// A triggered stop-limit only accepts a new limit price. Returns
    /// `false` for unknown, finished, or market orders.
    pub fn modify(
        &mut self,
        id: u64,
        qty: Option<crate::execution::orders::QtySpec>,
        limit_price: Option<Price>,
        trigger_price: Option<Price>,
    ) -> bool {
        let Some(order) = self.get_mut(id) else { return false };
        if order.status.is_terminal() {
            return false;
        }
        let triggered = order.triggered;
        match &mut order.kind {
            OrderKind::Market | OrderKind::MarketToLimit => return false,
            // Trailing offsets are the order's identity; replacing one is a
            // new order. Cancel-and-replace instead.
            OrderKind::TrailingStopMarket { .. } | OrderKind::TrailingStopLimit { .. } => {
                return false
            }
            OrderKind::MarketIfTouched { trigger } => {
                if let Some(t) = trigger_price {
                    *trigger = t;
                }
                if limit_price.is_some() {
                    return false;
                }
            }
            OrderKind::LimitIfTouched { trigger, price } => {
                if let Some(p) = limit_price {
                    *price = p;
                }
                match trigger_price {
                    Some(_) if triggered => return false,
                    Some(t) => *trigger = t,
                    None => {}
                }
            }
            OrderKind::Limit { price } => {
                if let Some(p) = limit_price {
                    *price = p;
                }
                if trigger_price.is_some() {
                    return false;
                }
            }
            OrderKind::StopMarket { trigger } => {
                if let Some(t) = trigger_price {
                    *trigger = t;
                }
                if limit_price.is_some() {
                    return false;
                }
            }
            OrderKind::StopLimit { trigger, price } => {
                if let Some(p) = limit_price {
                    *price = p;
                }
                match trigger_price {
                    // The trigger has already fired; re-arming it would be a
                    // different order.
                    Some(_) if triggered => return false,
                    Some(t) => *trigger = t,
                    None => {}
                }
            }
        }
        if let Some(q) = qty {
            order.qty = q;
        }
        true
    }

    /// Evaluate all working orders against one bar.
    ///
    /// Only orders submitted on an earlier bar participate: an order placed
    /// while bar `i` was being observed cannot rest into bar `i`, which had
    /// already closed. Outcome order is deterministic — expiries first, then
    /// fills/triggers/cancels in submission order.
    ///
    /// The engine reports outcomes but does not apply status transitions for
    /// fills; the kernel confirms or rejects each fill (position state may
    /// refuse it) and transitions the order itself.
    pub fn match_bar(
        &mut self,
        idx: usize,
        bar: &OhlcvBar,
        fill_model: &FillModel,
    ) -> Vec<MatchOutcome> {
        self.match_events(idx, bar, fill_model, MatchMode::Bar)
    }

    /// Evaluate all working orders against one trade print.
    ///
    /// The print is carried as a degenerate bar (`open == high == low ==
    /// close`), which the fill model's predicates reduce over correctly: a
    /// limit fills when the print reaches it, a stop triggers when the print
    /// crosses it, and gap handling collapses to a no-op because a print has
    /// no range.
    ///
    /// Two kinds do *not* reduce, and are refused rather than mispriced:
    /// `AT_OPEN`/`AT_CLOSE` market orders queue for a bar phase that a print
    /// does not have, so they keep resting. Everything else — including
    /// trailing stops, which ratchet off the print — matches as it would on
    /// a bar, at tick resolution.
    pub fn match_trade(
        &mut self,
        idx: usize,
        tick: &OhlcvBar,
        fill_model: &FillModel,
    ) -> Vec<MatchOutcome> {
        self.match_events(idx, tick, fill_model, MatchMode::Trade)
    }

    fn match_events(
        &mut self,
        idx: usize,
        bar: &OhlcvBar,
        fill_model: &FillModel,
        mode: MatchMode,
    ) -> Vec<MatchOutcome> {
        let mut expiries = Vec::new();
        let mut actions = Vec::new();
        let tz = self.tz_offset_ns;

        // Replay the step onto the tape first: resting orders match against
        // the prints it puts up, and orders submitted while it was observed
        // match against the book it leaves behind. Both are taken from the
        // tape rather than from the bar, because a bar that never leaves
        // the last traded price prints nothing at all and an order meeting
        // it is filled against liquidity that traded earlier.
        let step_kind = match mode {
            MatchMode::Bar => StepKind::Bar,
            MatchMode::Trade => StepKind::Print,
        };
        // The book as it stood before this step, for an order that reached
        // the venue ahead of the bar it was submitted on.
        let prior_book = self.tape;
        let tape = self.tape.replay(
            bar,
            fill_model.bar_liquidity,
            fill_model.size_quantum,
            step_kind,
        );
        let book = self.tape;

        // Parent states for one-triggers-other gating, resolved up front so
        // the mutable iteration below stays borrow-clean. Held children of a
        // filled parent become matchable; children of a dead parent cancel.
        let parent_state: std::collections::HashMap<u64, OrderStatus> =
            self.orders.iter().map(|o| (o.id, o.status)).collect();

        for order in &mut self.orders {
            if order.status.is_terminal() || order.submitted_idx > idx {
                continue;
            }
            let submitted_this_bar = order.submitted_idx == idx;
            // An order that arrived ahead of its bar has already met a book
            // -- the one standing when it was sent -- so it matches here
            // whether or not same-bar matching is enabled generally.
            if submitted_this_bar
                && !(self.same_bar_marketable_limit_on_close || order.arrives_before_bar)
            {
                continue;
            }

            if let Some(parent_id) = order.parent_id {
                match parent_state.get(&parent_id) {
                    Some(OrderStatus::Filled) => {} // active: fall through
                    Some(status) if status.is_terminal() => {
                        let _ = order.transition(OrderStatus::Canceled);
                        actions.push(MatchOutcome::Cancel { order_id: order.id });
                        continue;
                    }
                    // Parent still working: the child is held — no matching,
                    // no expiry clock.
                    _ => continue,
                }
            }

            if expired(order, bar.timestamp, tz) {
                let _ = order.transition(OrderStatus::Expired);
                expiries.push(MatchOutcome::Expire { order_id: order.id });
                continue;
            }

            // The first bar this order rests through -- the one whose open
            // it would have crossed had it been there for it. For an order
            // that reached the venue ahead of its bar, that is the very bar
            // it beat; for any other, the bar after the one it was sent
            // from, since the bar it was sent from had already printed.
            let first_bar = match order.arrives_before_bar {
                true => submitted_this_bar,
                false => order.submitted_idx + 1 == idx,
            };
            let (direction, is_entry) = side_as_fill_args(order.side);
            let oid = order.id;
            // Copied out of `order` and `bar` so the fill closure stays
            // borrow-clean while the loop transitions the order itself.
            let buying = order.side == OrderSide::Buy;
            // IOC and FOK are canceled the instant their first fill lands,
            // so they never walk past the print in front of them.
            let immediate = matches!(order.tif, TimeInForce::Ioc | TimeInForce::Fok);
            // A resting order is passive: the prints come to it. One that
            // offers it nothing is not a fill at all -- the step never
            // reached it -- so it yields no outcome and an IOC dies as it
            // would on any other barren bar.
            let fill = move |p: Price| {
                let depth = tape.offered(p, buying, immediate);
                (!depth.is_empty()).then_some(MatchOutcome::Fill {
                    order_id: oid,
                    price: p,
                    depth,
                    on_arrival: false,
                })
            };

            // An order that reached the venue while this bar was being
            // observed meets a standing book, not the bar: only a plain
            // limit which crosses that book participates here. This bar's
            // high and low are not read for that -- they are either behind
            // the order, for one sent from this bar's close, or ahead of
            // it, for one that beat the bar; neither is a book it could
            // have crossed on arrival.
            if submitted_this_bar {
                // Whichever book was standing when the order reached the
                // venue: the one this step leaves behind, or -- for an
                // order sent before this bar arrived -- the one before it.
                let standing = if order.arrives_before_bar { prior_book } else { book };
                let outcome = match order.kind {
                    // The standing book is what is in front of such an
                    // order. It crosses that book, so it fills at the
                    // book's price, not at its own limit, and a limit
                    // strictly through it empties the level beneath.
                    OrderKind::Limit { price } => {
                        let depth = standing.offered(price, buying, immediate);
                        let at = standing.book().map(|(price, _)| price);
                        match (depth.is_empty(), at) {
                            (false, Some(at)) => Some(MatchOutcome::Fill {
                                order_id: oid,
                                price: at,
                                depth,
                                // Only an order that beat the bar met its
                                // book at an instant of its own; one sent
                                // from this bar met it as the bar closed.
                                on_arrival: order.arrives_before_bar,
                            }),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                match outcome {
                    Some(outcome) => {
                        actions.push(outcome);
                        continue;
                    }
                    // An immediate order is executed against the book it
                    // arrived at and killed; it never waits for a print.
                    None if immediate => {
                        let _ = order.transition(OrderStatus::Canceled);
                        actions.push(MatchOutcome::Cancel { order_id: order.id });
                        continue;
                    }
                    // An order that beat this bar to the venue was working
                    // while the bar printed, so the range is not ahead of
                    // it in time: it is the market coming to a resting
                    // order, and it matches below like any other. One
                    // submitted *from* this bar has no such claim on it.
                    None if order.arrives_before_bar => {}
                    None => continue,
                }
            }

            let outcome = match order.kind {
                // Plain market orders are filled by the kernel on their
                // submission bar; the ones that reach the book are either
                // queued to a bar phase (AT_OPEN/AT_CLOSE) or activated
                // one-triggers-other children, which fill at the open.
                OrderKind::Market => match order.tif {
                    // A print has no open or close to queue against, so these
                    // keep resting until a bar event arrives.
                    TimeInForce::AtOpen if mode == MatchMode::Bar => fill(bar.open),
                    TimeInForce::AtClose if mode == MatchMode::Bar => fill(bar.close),
                    TimeInForce::AtOpen | TimeInForce::AtClose => None,
                    // "Next available price": the bar's open, or the print.
                    _ if order.parent_id.is_some() => fill(bar.open),
                    _ => None,
                },
                OrderKind::MarketToLimit => fill(bar.open),
                OrderKind::Limit { price } => {
                    // Post-only: marketable at the first resting open means
                    // the order would have crossed the book — refuse it.
                    let crosses_open = match direction {
                        Direction::Long if is_entry => bar.open <= price,
                        _ => bar.open >= price,
                    };
                    if order.post_only && first_bar && crosses_open {
                        let _ = order.transition(OrderStatus::Rejected);
                        Some(MatchOutcome::Reject { order_id: oid, reason: "post_only" })
                    } else {
                        fill_model.get_limit_fill_price(price, bar, direction, is_entry).and_then(fill)
                    }
                }
                OrderKind::StopMarket { trigger } => {
                    fill_model.get_stop_fill_price(trigger, bar, direction, is_entry).and_then(fill)
                }
                // If-touched orders trigger on the *favorable* side — the
                // limit-style touch — then fill like a market at the touch
                // price (conservative: no better-than-trigger assumption).
                OrderKind::MarketIfTouched { trigger } => {
                    fill_model.get_limit_fill_price(trigger, bar, direction, is_entry).and_then(fill)
                }
                OrderKind::StopLimit { trigger, price } => match order.triggered {
                    true => {
                        fill_model.get_limit_fill_price(price, bar, direction, is_entry).and_then(fill)
                    }
                    false => {
                        if fill_model.would_trigger_stop(trigger, bar, direction, is_entry) {
                            let _ = order.transition(OrderStatus::Triggered);
                            Some(MatchOutcome::Trigger { order_id: oid })
                        } else {
                            None
                        }
                    }
                },
                OrderKind::LimitIfTouched { trigger, price } => match order.triggered {
                    true => {
                        fill_model.get_limit_fill_price(price, bar, direction, is_entry).and_then(fill)
                    }
                    false => {
                        if fill_model.would_fill_limit(trigger, bar, direction, is_entry) {
                            let _ = order.transition(OrderStatus::Triggered);
                            Some(MatchOutcome::Trigger { order_id: oid })
                        } else {
                            None
                        }
                    }
                },
                OrderKind::TrailingStopMarket { offset } => {
                    // Ratchet the watermark from this bar's favorable
                    // extreme, then test the trigger against the adverse one
                    // — both same-bar, mirroring position trailing stops.
                    let favorable = match order.side {
                        OrderSide::Sell => bar.high,
                        OrderSide::Buy => bar.low,
                    };
                    let wm = match (order.trail_watermark, order.side) {
                        (None, _) => favorable,
                        (Some(w), OrderSide::Sell) => w.max(favorable),
                        (Some(w), OrderSide::Buy) => w.min(favorable),
                    };
                    order.trail_watermark = Some(wm);
                    let trigger = order.trail_trigger(wm, offset);
                    fill_model.get_stop_fill_price(trigger, bar, direction, is_entry).and_then(fill)
                }
                OrderKind::TrailingStopLimit { offset, limit_offset } => match order.triggered {
                    true => {
                        let price = order.trail_limit.unwrap_or(bar.close);
                        fill_model.get_limit_fill_price(price, bar, direction, is_entry).and_then(fill)
                    }
                    false => {
                        let favorable = match order.side {
                            OrderSide::Sell => bar.high,
                            OrderSide::Buy => bar.low,
                        };
                        let wm = match (order.trail_watermark, order.side) {
                            (None, _) => favorable,
                            (Some(w), OrderSide::Sell) => w.max(favorable),
                            (Some(w), OrderSide::Buy) => w.min(favorable),
                        };
                        order.trail_watermark = Some(wm);
                        let trigger = order.trail_trigger(wm, offset);
                        if fill_model.would_trigger_stop(trigger, bar, direction, is_entry) {
                            // Fix the limit through the trigger: bounded
                            // slippage, more marketable than the trigger.
                            order.trail_limit = Some(match order.side {
                                OrderSide::Sell => trigger - limit_offset,
                                OrderSide::Buy => trigger + limit_offset,
                            });
                            let _ = order.transition(OrderStatus::Triggered);
                            Some(MatchOutcome::Trigger { order_id: oid })
                        } else {
                            None
                        }
                    }
                },
            };

            match outcome {
                Some(outcome) => actions.push(outcome),
                None => {
                    // IOC/FOK live for exactly one evaluation bar. Plain
                    // (parentless) market orders are exempt: they are the
                    // kernel sweep's responsibility, not this matcher's, and
                    // under next-bar-open timing they legitimately rest here
                    // for the one bar between submission and their fill.
                    let kernel_swept =
                        matches!(order.kind, OrderKind::Market) && order.parent_id.is_none();
                    if !kernel_swept && matches!(order.tif, TimeInForce::Ioc | TimeInForce::Fok) {
                        let _ = order.transition(OrderStatus::Canceled);
                        actions.push(MatchOutcome::Cancel { order_id: order.id });
                    }
                }
            }
        }

        expiries.extend(actions);
        expiries
    }
}

/// Whether an order's time-in-force has lapsed at the bar timestamp.
fn expired(order: &Order, ts: Timestamp, tz_offset_ns: i64) -> bool {
    match order.tif {
        TimeInForce::Gtd { expire_ns } => ts >= expire_ns,
        TimeInForce::Day => {
            // Compare trading dates, not UTC dates: the two diverge for any
            // session whose local hours cross UTC midnight.
            let local = |t: Timestamp| (t + tz_offset_ns).div_euclid(NS_PER_DAY);
            local(ts) > local(order.submitted_ts)
        }
        TimeInForce::Gtc
        | TimeInForce::Ioc
        | TimeInForce::Fok
        | TimeInForce::AtOpen
        | TimeInForce::AtClose => false,
    }
}

#[cfg(test)]
mod tests {

    /// The expected fill for a test running the default, unbounded fill
    /// model — where every match offers more depth than any order needs.
    fn filled(order_id: u64, price: Price) -> MatchOutcome {
        MatchOutcome::Fill { order_id, price, depth: FillDepth::UNLIMITED, on_arrival: false }
    }

    /// The same, for a fill taken from the book standing when the order
    /// reached the venue rather than from the bar it beat.
    fn filled_on_arrival(order_id: u64, price: Price) -> MatchOutcome {
        MatchOutcome::Fill { order_id, price, depth: FillDepth::UNLIMITED, on_arrival: true }
    }

    use super::*;
    use crate::execution::orders::QtySpec;

    fn bar(ts: i64, open: f64, high: f64, low: f64, close: f64) -> OhlcvBar {
        OhlcvBar { timestamp: ts, open, high, low, close, volume: 1_000.0 }
    }

    fn engine_with(kind: OrderKind, side: OrderSide, tif: TimeInForce) -> (OrderEngine, u64) {
        let mut engine = OrderEngine::new();
        let mut order = Order::plain(side, QtySpec::Units(1.0), kind, tif);
        order.client_id = "t-0".into();
        let _ = order.transition(OrderStatus::Accepted);
        let id = engine.submit(order);
        (engine, id)
    }

    #[test]
    fn buy_limit_fills_when_low_touches() {
        let (mut engine, id) =
            engine_with(OrderKind::Limit { price: 99.0 }, OrderSide::Buy, TimeInForce::Gtc);
        let fm = FillModel::default();

        // Bar stays above the limit: no fill.
        assert!(engine.match_bar(1, &bar(1, 100.0, 101.0, 99.5, 100.5), &fm).is_empty());
        // Bar trades down to it: fill at limit.
        let outcomes = engine.match_bar(2, &bar(2, 100.0, 100.5, 98.5, 99.5), &fm);
        assert_eq!(outcomes, vec![filled(id, 99.0)]);
    }

    #[test]
    fn same_bar_submission_never_matches() {
        let (mut engine, _) =
            engine_with(OrderKind::Limit { price: 99.0 }, OrderSide::Buy, TimeInForce::Gtc);
        let fm = FillModel::default();
        // idx == submitted_idx: the bar had closed when the order was placed.
        assert!(engine.match_bar(0, &bar(0, 100.0, 100.5, 98.0, 99.5), &fm).is_empty());
    }

    #[test]
    fn compatibility_market_limit_fills_at_same_close_without_using_range() {
        let (mut engine, id) = engine_with(
            OrderKind::Limit { price: 101.0 },
            OrderSide::Buy,
            TimeInForce::Gtd { expire_ns: 12 },
        );
        engine.same_bar_marketable_limit_on_close = true;
        let fm = FillModel::default();
        let outcomes = engine.match_bar(0, &bar(0, 100.0, 102.0, 98.0, 100.5), &fm);
        assert_eq!(outcomes, vec![filled(id, 100.5)]);

        let (mut engine, _) =
            engine_with(OrderKind::Limit { price: 99.0 }, OrderSide::Buy, TimeInForce::Gtc);
        engine.same_bar_marketable_limit_on_close = true;
        assert!(engine.match_bar(0, &bar(0, 100.0, 102.0, 98.0, 100.5), &fm).is_empty());
    }

    /// An order that reached the venue before its bar meets the book the
    /// previous bar left behind -- not the price this bar closes at, and
    /// not its own limit.
    #[test]
    fn order_arriving_before_its_bar_meets_the_standing_book() {
        let (mut engine, id) = engine_with(
            OrderKind::Limit { price: 101.0 },
            OrderSide::Buy,
            TimeInForce::Gtd { expire_ns: 12 },
        );
        engine.get_mut(id).unwrap().arrives_before_bar = true;
        let fm = FillModel::default();

        // A first bar leaves the book at its close; the order was submitted
        // on the second, before that bar reached the venue.
        engine.orders[0].submitted_idx = 1;
        assert!(engine.match_bar(0, &bar(0, 99.0, 100.0, 98.5, 99.5), &fm).is_empty());
        let outcomes = engine.match_bar(1, &bar(1, 100.0, 102.0, 98.0, 100.5), &fm);
        assert_eq!(outcomes, vec![filled_on_arrival(id, 99.5)]);
    }

    /// The same order without the flag takes this bar's close, which is
    /// what a decision made *from* this bar meets.
    #[test]
    fn order_submitted_on_its_bar_meets_the_book_that_bar_leaves() {
        let (mut engine, id) = engine_with(
            OrderKind::Limit { price: 101.0 },
            OrderSide::Buy,
            TimeInForce::Gtd { expire_ns: 12 },
        );
        engine.same_bar_marketable_limit_on_close = true;
        let fm = FillModel::default();

        engine.orders[0].submitted_idx = 1;
        assert!(engine.match_bar(0, &bar(0, 99.0, 100.0, 98.5, 99.5), &fm).is_empty());
        let outcomes = engine.match_bar(1, &bar(1, 100.0, 102.0, 98.0, 100.5), &fm);
        assert_eq!(outcomes, vec![filled(id, 100.5)]);
    }

    /// Arriving ahead of the bar is a fact about one order, not a session
    /// mode: it matches with same-bar matching off, and orders without it
    /// still wait for the next bar.
    #[test]
    fn arriving_before_the_bar_does_not_need_the_session_flag() {
        let (mut engine, id) = engine_with(
            OrderKind::Limit { price: 101.0 },
            OrderSide::Buy,
            TimeInForce::Gtd { expire_ns: 12 },
        );
        engine.get_mut(id).unwrap().arrives_before_bar = true;
        assert!(!engine.same_bar_marketable_limit_on_close);
        let fm = FillModel::default();

        engine.orders[0].submitted_idx = 1;
        assert!(engine.match_bar(0, &bar(0, 99.0, 100.0, 98.5, 99.5), &fm).is_empty());
        assert_eq!(
            engine.match_bar(1, &bar(1, 100.0, 102.0, 98.0, 100.5), &fm),
            vec![filled_on_arrival(id, 99.5)]
        );
    }

    /// Nothing has traded before the first bar, so an order that arrives
    /// ahead of it meets no book at all -- and then rests through that very
    /// bar, which prints through its limit and fills it there.
    #[test]
    fn arriving_before_the_first_bar_meets_no_book_and_rests_into_it() {
        let (mut engine, id) = engine_with(
            OrderKind::Limit { price: 101.0 },
            OrderSide::Buy,
            TimeInForce::Gtc,
        );
        engine.get_mut(id).unwrap().arrives_before_bar = true;
        let fm = FillModel::default();

        assert_eq!(
            engine.match_bar(0, &bar(0, 100.0, 102.0, 98.0, 100.5), &fm),
            vec![filled(id, 101.0)]
        );
    }

    /// An order that reached the venue before its bar and did not cross the
    /// book standing there was working while that bar printed: the range is
    /// ahead of it in time, so the bar it beat is the one that fills it --
    /// at its own limit, like any resting order the market comes to.
    #[test]
    fn order_arriving_before_its_bar_rests_into_the_bar_it_beat() {
        let (mut engine, id) = engine_with(
            OrderKind::Limit { price: 99.0 },
            OrderSide::Buy,
            TimeInForce::Gtc,
        );
        engine.get_mut(id).unwrap().arrives_before_bar = true;
        let fm = FillModel::default();

        // The first bar leaves the book at 99.5: a buy at 99.0 does not
        // cross it. The second, which the order beat to the venue, trades
        // down to 98.0 -- through the limit -- and fills it there.
        engine.orders[0].submitted_idx = 1;
        assert!(engine.match_bar(0, &bar(0, 99.0, 100.0, 98.5, 99.5), &fm).is_empty());
        assert_eq!(
            engine.match_bar(1, &bar(1, 100.0, 102.0, 98.0, 100.5), &fm),
            vec![filled(id, 99.0)]
        );
    }

    /// The same order arriving with its own bar rather than ahead of it has
    /// no claim on that bar's range: the prints came before the decision
    /// that sent it, and reading them would be look-ahead.
    #[test]
    fn order_submitted_from_its_bar_never_meets_that_bar_range() {
        let (mut engine, id) = engine_with(
            OrderKind::Limit { price: 99.0 },
            OrderSide::Buy,
            TimeInForce::Gtc,
        );
        let fm = FillModel::default();

        engine.orders[0].submitted_idx = 1;
        assert!(engine.match_bar(0, &bar(0, 99.0, 100.0, 98.5, 99.5), &fm).is_empty());
        assert!(engine.match_bar(1, &bar(1, 100.0, 102.0, 98.0, 100.5), &fm).is_empty());
        // It is working from the next bar on, like any resting limit.
        assert_eq!(
            engine.match_bar(2, &bar(2, 100.0, 102.0, 98.0, 100.5), &fm),
            vec![filled(id, 99.0)]
        );
    }

    /// An immediate order that beat its bar is executed against the book it
    /// arrived at and killed -- it does not rest into the bar's prints.
    #[test]
    fn an_immediate_order_arriving_before_its_bar_does_not_rest_into_it() {
        let (mut engine, id) = engine_with(
            OrderKind::Limit { price: 99.0 },
            OrderSide::Buy,
            TimeInForce::Ioc,
        );
        engine.get_mut(id).unwrap().arrives_before_bar = true;
        let fm = FillModel::default();

        engine.orders[0].submitted_idx = 1;
        assert!(engine.match_bar(0, &bar(0, 99.0, 100.0, 98.5, 99.5), &fm).is_empty());
        assert_eq!(
            engine.match_bar(1, &bar(1, 100.0, 102.0, 98.0, 100.5), &fm),
            vec![MatchOutcome::Cancel { order_id: id }]
        );
    }

    #[test]
    fn sell_stop_gap_through_fills_at_open() {
        let (mut engine, id) =
            engine_with(OrderKind::StopMarket { trigger: 95.0 }, OrderSide::Sell, TimeInForce::Gtc);
        let fm = FillModel::default();
        // Gap down through the trigger: fill at the (worse) open.
        let outcomes = engine.match_bar(1, &bar(1, 93.0, 94.0, 92.0, 93.5), &fm);
        assert_eq!(outcomes, vec![filled(id, 93.0)]);
    }

    #[test]
    fn stop_limit_triggers_then_fills_next_bar() {
        let (mut engine, id) = engine_with(
            OrderKind::StopLimit { trigger: 105.0, price: 104.5 },
            OrderSide::Buy,
            TimeInForce::Gtc,
        );
        let fm = FillModel::default();

        let outcomes = engine.match_bar(1, &bar(1, 104.0, 105.5, 103.5, 105.2), &fm);
        assert_eq!(outcomes, vec![MatchOutcome::Trigger { order_id: id }]);
        assert_eq!(engine.get(id).unwrap().status, OrderStatus::Triggered);

        // Now resting as a buy limit at 104.5.
        let outcomes = engine.match_bar(2, &bar(2, 105.0, 105.5, 104.0, 104.8), &fm);
        assert_eq!(outcomes, vec![filled(id, 104.5)]);
    }

    #[test]
    fn ioc_cancels_after_one_missed_bar() {
        let (mut engine, id) =
            engine_with(OrderKind::Limit { price: 90.0 }, OrderSide::Buy, TimeInForce::Ioc);
        let fm = FillModel::default();
        let outcomes = engine.match_bar(1, &bar(1, 100.0, 101.0, 99.0, 100.0), &fm);
        assert_eq!(outcomes, vec![MatchOutcome::Cancel { order_id: id }]);
        assert_eq!(engine.get(id).unwrap().status, OrderStatus::Canceled);
    }

    #[test]
    fn gtd_expires_at_timestamp() {
        let (mut engine, id) = engine_with(
            OrderKind::Limit { price: 90.0 },
            OrderSide::Buy,
            TimeInForce::Gtd { expire_ns: 100 },
        );
        let fm = FillModel::default();
        assert!(engine.match_bar(1, &bar(99, 100.0, 101.0, 99.0, 100.0), &fm).is_empty());
        let outcomes = engine.match_bar(2, &bar(100, 100.0, 101.0, 89.0, 100.0), &fm);
        // Expiry beats the would-be fill.
        assert_eq!(outcomes, vec![MatchOutcome::Expire { order_id: id }]);
    }

    #[test]
    fn day_expiry_follows_the_trading_date_not_the_utc_date() {
        // A session running past 05:30 IST crosses UTC midnight while the
        // IST trading date is unchanged. Under UTC-only expiry a DAY order
        // placed before that crossing dies mid-session.
        const IST: i64 = (5 * 3600 + 30 * 60) * 1_000_000_000;
        let day = 20_468i64 * NS_PER_DAY;
        // 22:30 UTC = 04:00 IST the next IST date.
        let submitted = day + 22 * 3_600_000_000_000 + 1_800_000_000_000;
        // 00:30 UTC the next UTC date = 06:00 IST, same IST trading date.
        let later = day + NS_PER_DAY + 1_800_000_000_000;

        let mut order = Order::plain(
            OrderSide::Buy,
            QtySpec::Units(1.0),
            OrderKind::Limit { price: 90.0 },
            TimeInForce::Day,
        );
        order.submitted_ts = submitted;
        let _ = order.transition(OrderStatus::Accepted);

        // UTC dates rolled, so the naive rule expires it.
        assert!(expired(&order, later, 0), "the UTC date rolled");
        // The IST trading date did not, so it must survive.
        assert!(!expired(&order, later, IST), "the trading date is unchanged");
    }

    #[test]
    fn day_expiry_honors_a_negative_offset() {
        // US Eastern is behind UTC: a 20:00 ET order and a 22:00 ET bar are
        // the same trading date but land on different UTC dates.
        const ET: i64 = -5 * 3600 * 1_000_000_000;
        let day = 20_468i64 * NS_PER_DAY;
        let submitted = day + 1_800_000_000_000; // 00:30 UTC = 19:30 ET prior day
        let later = day + 3 * 3_600_000_000_000; // 03:00 UTC = 22:00 ET same ET day

        let mut order = Order::plain(
            OrderSide::Buy,
            QtySpec::Units(1.0),
            OrderKind::Limit { price: 90.0 },
            TimeInForce::Day,
        );
        order.submitted_ts = submitted;
        let _ = order.transition(OrderStatus::Accepted);
        assert!(!expired(&order, later, 0), "same UTC date");
        assert!(!expired(&order, later, ET), "and the same ET trading date");
    }

    #[test]
    fn gtd_ignores_the_trading_date_offset() {
        // GTD names an absolute instant; the session calendar is irrelevant.
        let expire_ns = 20_468i64 * NS_PER_DAY + 3_600_000_000_000;
        let mut order = Order::plain(
            OrderSide::Buy,
            QtySpec::Units(1.0),
            OrderKind::Limit { price: 90.0 },
            TimeInForce::Gtd { expire_ns },
        );
        let _ = order.transition(OrderStatus::Accepted);
        const IST: i64 = (5 * 3600 + 30 * 60) * 1_000_000_000;
        assert_eq!(expired(&order, expire_ns, 0), expired(&order, expire_ns, IST));
        assert!(expired(&order, expire_ns, IST));
    }

    #[test]
    fn day_expires_on_utc_date_rollover() {
        let (mut engine, id) =
            engine_with(OrderKind::Limit { price: 90.0 }, OrderSide::Buy, TimeInForce::Day);
        let fm = FillModel::default();
        // Same UTC day: still working.
        assert!(engine
            .match_bar(1, &bar(NS_PER_DAY - 1, 100.0, 101.0, 99.0, 100.0), &fm)
            .is_empty());
        // Next UTC day: expired.
        let outcomes = engine.match_bar(2, &bar(NS_PER_DAY, 100.0, 101.0, 99.0, 100.0), &fm);
        assert_eq!(outcomes, vec![MatchOutcome::Expire { order_id: id }]);
    }

    #[test]
    fn expiries_precede_fills_in_outcome_order() {
        let mut engine = OrderEngine::new();
        let make = |tif| {
            let mut o = Order::plain(
                OrderSide::Buy,
                QtySpec::Units(1.0),
                OrderKind::Limit { price: 99.0 },
                tif,
            );
            let _ = o.transition(OrderStatus::Accepted);
            o
        };
        let fill_id = engine.submit(make(TimeInForce::Gtc));
        let expire_id = engine.submit(make(TimeInForce::Gtd { expire_ns: 50 }));
        let fm = FillModel::default();

        let outcomes = engine.match_bar(1, &bar(60, 100.0, 100.5, 98.0, 99.5), &fm);
        assert_eq!(
            outcomes,
            vec![
                MatchOutcome::Expire { order_id: expire_id },
                filled(fill_id, 99.0),
            ]
        );
    }

    #[test]
    fn mit_buy_triggers_on_favorable_touch() {
        // Buy MIT below the market: fills when price *falls* to the trigger.
        let (mut engine, id) = engine_with(
            OrderKind::MarketIfTouched { trigger: 98.0 },
            OrderSide::Buy,
            TimeInForce::Gtc,
        );
        let fm = FillModel::default();
        assert!(engine.match_bar(1, &bar(1, 100.0, 101.0, 99.0, 100.0), &fm).is_empty());
        let outcomes = engine.match_bar(2, &bar(2, 99.0, 99.5, 97.5, 98.5), &fm);
        assert_eq!(outcomes, vec![filled(id, 98.0)]);
    }

    #[test]
    fn market_to_limit_fills_at_open() {
        let (mut engine, id) =
            engine_with(OrderKind::MarketToLimit, OrderSide::Buy, TimeInForce::Gtc);
        let fm = FillModel::default();
        let outcomes = engine.match_bar(1, &bar(1, 101.5, 102.0, 100.5, 101.0), &fm);
        assert_eq!(outcomes, vec![filled(id, 101.5)]);
    }

    #[test]
    fn at_open_market_fills_at_next_open() {
        let (mut engine, id) = engine_with(OrderKind::Market, OrderSide::Buy, TimeInForce::AtOpen);
        let fm = FillModel::default();
        let outcomes = engine.match_bar(1, &bar(1, 102.5, 103.0, 101.0, 101.5), &fm);
        assert_eq!(outcomes, vec![filled(id, 102.5)]);
    }

    #[test]
    fn post_only_rejects_when_marketable_at_open() {
        let (mut engine, id) =
            engine_with(OrderKind::Limit { price: 101.0 }, OrderSide::Buy, TimeInForce::Gtc);
        engine.get_mut(id).unwrap().post_only = true;
        let fm = FillModel::default();
        // Open 100 < limit 101: the order would have crossed — rejected.
        let outcomes = engine.match_bar(1, &bar(1, 100.0, 102.0, 99.0, 101.5), &fm);
        assert_eq!(outcomes, vec![MatchOutcome::Reject { order_id: id, reason: "post_only" }]);
        assert_eq!(engine.get(id).unwrap().status, OrderStatus::Rejected);
    }

    #[test]
    fn trailing_sell_stop_ratchets_and_triggers() {
        use crate::execution::orders::TrailOffset;
        let (mut engine, id) = engine_with(
            OrderKind::TrailingStopMarket { offset: TrailOffset::Price(2.0) },
            OrderSide::Sell,
            TimeInForce::Gtc,
        );
        let fm = FillModel::default();

        // Bar 1: high 105 -> watermark 105, trigger 103; low 104 doesn't touch.
        assert!(engine.match_bar(1, &bar(1, 104.5, 105.0, 104.0, 104.8), &fm).is_empty());
        // Bar 2: high 108 ratchets trigger to 106; low 107 stays above.
        assert!(engine.match_bar(2, &bar(2, 105.0, 108.0, 107.0, 107.5), &fm).is_empty());
        assert!((engine.get(id).unwrap().working_price().unwrap() - 106.0).abs() < 1e-9);
        // Bar 3: low 105.5 crosses the 106 trigger.
        let outcomes = engine.match_bar(3, &bar(3, 107.0, 107.5, 105.5, 106.0), &fm);
        assert_eq!(outcomes, vec![filled(id, 106.0)]);
    }

    #[test]
    fn held_child_waits_for_parent_and_dies_with_it() {
        let mut engine = OrderEngine::new();
        let mut parent = Order::plain(
            OrderSide::Buy,
            QtySpec::Units(1.0),
            OrderKind::Limit { price: 99.0 },
            TimeInForce::Gtc,
        );
        let _ = parent.transition(OrderStatus::Accepted);
        let parent_id = engine.submit(parent);

        let mut child = Order::plain(
            OrderSide::Sell,
            QtySpec::FullPosition,
            OrderKind::Limit { price: 105.0 },
            TimeInForce::Gtc,
        );
        child.parent_id = Some(parent_id);
        let child_id = engine.submit(child);

        let fm = FillModel::default();
        // Child never matches while the parent works, even through its price.
        assert!(engine
            .match_bar(1, &bar(1, 106.0, 107.0, 99.5, 106.0), &fm)
            .iter()
            .all(|o| !matches!(o, MatchOutcome::Fill { order_id, .. } if *order_id == child_id)));

        // Parent cancels: the held child cancels with it.
        assert!(engine.cancel(parent_id));
        let outcomes = engine.match_bar(2, &bar(2, 106.0, 107.0, 105.5, 106.0), &fm);
        assert_eq!(outcomes, vec![MatchOutcome::Cancel { order_id: child_id }]);
    }

    #[test]
    fn modify_moves_limit_price() {
        let (mut engine, id) =
            engine_with(OrderKind::Limit { price: 90.0 }, OrderSide::Buy, TimeInForce::Gtc);
        assert!(engine.modify(id, None, Some(99.0), None));
        let fm = FillModel::default();
        let outcomes = engine.match_bar(1, &bar(1, 100.0, 100.5, 98.5, 99.5), &fm);
        assert_eq!(outcomes, vec![filled(id, 99.0)]);
    }

    #[test]
    fn cancel_stops_matching() {
        let (mut engine, id) =
            engine_with(OrderKind::Limit { price: 99.0 }, OrderSide::Buy, TimeInForce::Gtc);
        assert!(engine.cancel(id));
        assert!(!engine.cancel(id), "double-cancel must fail");
        let fm = FillModel::default();
        assert!(engine.match_bar(1, &bar(1, 100.0, 100.5, 98.0, 99.5), &fm).is_empty());
    }
}
