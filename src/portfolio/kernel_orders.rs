//! Order lifecycle on the kernel: submission, cancellation, modification,
//! and applying the matcher's outcomes.
//!
//! Split out of `kernel.rs` to keep that file reviewable; this is the same
//! `impl EngineKernel`, not a separate type.

use crate::core::types::{Direction, ExitReason, Price};
use crate::execution::algos::{AlgoError, ExecAlgorithm};
use crate::execution::fill::{FillDepth, NextPrint};
use crate::execution::orders::{
    MatchOutcome, Order, OrderEngine, OrderKind, OrderSide, OrderStatus, QtySpec, TimeInForce,
};
use crate::execution::queue::QueueVerdict;
use crate::portfolio::kernel::{
    EngineEvent, EngineKernel, FillTerms, KernelBar, OpenResult, ReduceResult,
};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::risk::RiskGate;

impl EngineKernel {
    /// Register an execution schedule that releases slices over time.
    ///
    /// Only `QtySpec::Units` is sliceable: `CapitalFrac` resolves against
    /// equity at fill time, so each slice would size against a different
    /// account, and `FullPosition` sliced N ways would close the whole
    /// position N times. Both are refused rather than guessed at.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_algo(
        &mut self,
        side: OrderSide,
        qty: QtySpec,
        kind: OrderKind,
        tif: TimeInForce,
        client_id: String,
        algo: ExecAlgorithm,
        reduce_only: bool,
        now_ns: i64,
        idx: usize,
    ) -> Result<u64, AlgoError> {
        let units = match qty {
            QtySpec::Units(units) => units,
            _ => return Err(AlgoError::InvalidUnits),
        };
        let algo_id = self.algos.submit(
            client_id.clone(),
            side,
            kind,
            tif,
            units,
            algo,
            reduce_only,
            now_ns,
        )?;
        self.pending_events.push(EngineEvent::AlgoStarted { idx, algo_id, client_id });
        Ok(algo_id)
    }

    /// Submit every slice due at this timestamp.
    ///
    /// Called from the step just before market orders sweep, so a released
    /// slice fills on the same step rather than trailing one behind.
    pub(crate) fn release_algo_slices(
        &mut self,
        idx: usize,
        now_ns: i64,
        events: &mut Vec<EngineEvent>,
    ) {
        if self.algos.is_empty() {
            return;
        }
        for slice in self.algos.release_due(now_ns) {
            let order_id = self.submit_order_full(
                slice.side,
                QtySpec::Units(slice.units),
                slice.kind,
                slice.tif,
                idx,
                now_ns,
                slice.client_id,
                None,
                None,
                false,
                slice.reduce_only,
                false,
                None,
            );
            self.orders.set_algo_id(order_id, Some(slice.algo_id));
        }
        events.append(&mut self.pending_events);
        for algo_id in self.algos.drain_completed() {
            events.push(EngineEvent::AlgoCompleted { idx, algo_id, client_id: String::new() });
        }
    }

    /// Stop a schedule and cancel the slices it has working.
    ///
    /// Slices that already filled stay filled: cancelling a schedule halts
    /// the remainder, it does not unwind what traded.
    pub fn cancel_algo(&mut self, algo_id: u64, idx: usize) -> bool {
        if !self.algos.cancel(algo_id) {
            return false;
        }
        for order_id in self.orders.algo_order_ids(algo_id) {
            self.cancel_order(idx, order_id);
        }
        true
    }

    /// Units still unreleased by a schedule, for diagnostics.
    pub fn algo_released(&self, algo_id: u64) -> Option<u32> {
        self.algos.get(algo_id).map(|s| s.released())
    }

    /// Submit an order from the class-based order API.
    ///
    /// `submitted_idx` is the bar the strategy was observing when it placed
    /// the order: market orders fill on that bar's step (matching the
    /// signal-entry contract), resting orders begin matching on the next
    /// bar. Returns the engine order id; acknowledgment events are
    /// delivered at the front of the next step's event list.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_order(
        &mut self,
        side: OrderSide,
        qty: QtySpec,
        kind: OrderKind,
        tif: TimeInForce,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: String,
        stop_price: Option<Price>,
        target_price: Option<Price>,
    ) -> u64 {
        self.submit_order_full(
            side,
            qty,
            kind,
            tif,
            submitted_idx,
            submitted_ts,
            client_id,
            stop_price,
            target_price,
            false,
            false,
            false,
            None,
        )
    }

    /// [`EngineKernel::submit_order`] with flags and one-triggers-other
    /// linkage. `parent_id` holds the order (unmatched, no expiry clock)
    /// until the parent fills; a dead parent cancels it.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_order_full(
        &mut self,
        side: OrderSide,
        qty: QtySpec,
        kind: OrderKind,
        tif: TimeInForce,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: String,
        stop_price: Option<Price>,
        target_price: Option<Price>,
        post_only: bool,
        reduce_only: bool,
        arrives_before_bar: bool,
        parent_id: Option<u64>,
    ) -> u64 {
        let mut order = Order::plain(side, qty, kind, tif);
        order.client_id = client_id.clone();
        order.submitted_idx = submitted_idx;
        order.submitted_ts = submitted_ts;
        order.stop_price = stop_price;
        order.target_price = target_price;
        order.post_only = post_only;
        order.reduce_only = reduce_only;
        order.arrives_before_bar = arrives_before_bar;
        order.parent_id = parent_id;
        let id = self.orders.submit(order);

        // Resting kinds start working immediately (held children stay
        // Submitted until their parent fills); plain market kinds are
        // acknowledged when their fill is processed in the step.
        let rests = !matches!(kind, OrderKind::Market)
            || matches!(tif, TimeInForce::AtOpen | TimeInForce::AtClose);
        if rests && parent_id.is_none() {
            if let Some(order) = self.orders.get_mut(id) {
                let _ = order.transition(OrderStatus::Accepted);
            }
            self.pending_events.push(EngineEvent::OrderAccepted {
                idx: submitted_idx,
                order_id: id,
                client_id,
            });
        }
        id
    }

    /// Put a set of working orders in one one-cancels-other group: the first
    /// fill among them cancels the rest. One-updates-other reduces to this
    /// while fills are all-or-nothing.
    pub fn link_oco(&mut self, ids: &[u64]) {
        let group = ids.iter().copied().min().unwrap_or(0);
        for id in ids {
            if let Some(order) = self.orders.get_mut(*id) {
                order.oco_group = Some(group);
            }
        }
    }

    /// Cancel a working order. Returns `false` for unknown/finished ids.
    pub fn cancel_order(&mut self, idx: usize, id: u64) -> bool {
        let client_id = match self.orders.get(id) {
            Some(order) => order.client_id.clone(),
            None => return false,
        };
        if self.orders.cancel(id) {
            self.pending_events.push(EngineEvent::OrderCanceled { idx, order_id: id, client_id });
            true
        } else {
            false
        }
    }

    /// Cancel every working order.
    pub fn cancel_all_orders(&mut self, idx: usize) -> Vec<u64> {
        let ids = self.orders.cancel_all();
        for id in &ids {
            let client_id = self.orders.get(*id).map(|o| o.client_id.clone()).unwrap_or_default();
            self.pending_events.push(EngineEvent::OrderCanceled { idx, order_id: *id, client_id });
        }
        ids
    }

    /// Replace a working order's prices/quantity. Returns `false` when the
    /// order is unknown, finished, or the modification is not applicable.
    pub fn modify_order(
        &mut self,
        id: u64,
        qty: Option<QtySpec>,
        limit_price: Option<Price>,
        trigger_price: Option<Price>,
    ) -> bool {
        self.orders.modify(id, qty, limit_price, trigger_price)
    }

    /// What an order still has outstanding, read after its fill was
    /// recorded. Zero once nothing is left -- including for an order the
    /// fill terminated, which no longer reports leaves at all.
    fn leaves_after_fill(&self, id: u64) -> f64 {
        self.orders.get(id).and_then(|order| order.leaves_qty()).unwrap_or(0.0).max(0.0)
    }

    /// Shared view of an order by engine id.
    pub fn order(&self, id: u64) -> Option<&Order> {
        self.orders.get(id)
    }

    /// One price step on this instrument; `0.0` when prices are a
    /// continuum, which is the case without either a spec or an explicit
    /// increment.
    ///
    /// An explicit instrument config wins over the spec, the same way it
    /// does for lot size.
    pub fn price_increment(&self) -> f64 {
        self.configured_price_increment
            .or_else(|| self.spec.as_ref().map(|spec| spec.price_increment))
            .filter(|increment| *increment > 0.0 && increment.is_finite())
            .unwrap_or(0.0)
    }

    /// All non-terminal orders, in submission order.
    pub fn open_orders(&self) -> Vec<&Order> {
        self.orders.working().collect()
    }
    /// Queue verdict for a resting limit, or `None` when the queue model is
    /// off or cannot see enough to judge — the caller then falls back to
    /// `fill_prob_limit`.
    ///
    /// Only trade prints carry the size the model consumes, so bar events
    /// always fall back: a bar's volume is not volume *at* the limit price.
    fn queue_verdict(
        &mut self,
        order_id: u64,
        kind: OrderKind,
        status: OrderStatus,
        side: OrderSide,
        bar: &KernelBar,
    ) -> Option<QueueVerdict> {
        if !self.config.queue_fill_model || !self.stepping_trade {
            return None;
        }
        let limit_price = match kind {
            OrderKind::Limit { price } => price,
            OrderKind::StopLimit { price, .. } if status == OrderStatus::Triggered => price,
            _ => return None,
        };
        let direction = match side {
            OrderSide::Buy => Direction::Long,
            OrderSide::Sell => Direction::Short,
        };
        let verdict = self.queue.observe_print(
            order_id,
            limit_price,
            direction,
            &self.book,
            bar.close,
            bar.volume,
        );
        (verdict != QueueVerdict::Unknown).then_some(verdict)
    }

    pub(crate) fn apply_match_outcome(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        outcome: MatchOutcome,
        events: &mut Vec<EngineEvent>,
    ) {
        let (id, matched_price, depth, on_arrival) = match outcome {
            MatchOutcome::Trigger { order_id } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderTriggered { idx, order_id, client_id });
                return;
            }
            MatchOutcome::Expire { order_id } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderExpired { idx, order_id, client_id });
                return;
            }
            MatchOutcome::Cancel { order_id } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderCanceled { idx, order_id, client_id });
                return;
            }
            MatchOutcome::Reject { order_id, reason } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderRejected { idx, order_id, client_id, reason });
                return;
            }
            MatchOutcome::Fill { order_id, price, depth, on_arrival } => {
                (order_id, price, depth, on_arrival)
            }
        };
        // A venue cannot fill a fraction of a lot, so the size a print
        // showed is quantized to the instrument's size grid before it can
        // bound anything. Truncating (rather than rounding) is what
        // Nautilus does, and the difference is observable: it decides
        // whether an order ends the bar filled or one increment short.
        let offered = depth.cap();
        let cap = match offered.is_finite() {
            true => self.round_size(offered),
            false => offered,
        };

        let Some(order) = self.orders.get(id) else { return };
        let side = order.side;
        let qty = order.qty;
        let kind = order.kind;
        let tif = order.tif;
        // An order resumed after a partial fill asks for the rest of the
        // size it already resolved to, not for a fresh sizing.
        let leaves = order.leaves_qty();
        let status = order.status;
        let client_id = order.client_id.clone();
        let stop_attach = order.stop_price;
        let target_attach = order.target_price;
        let reduce_only = order.reduce_only;
        // A fill taken from the book standing when the order reached the
        // venue happened at that instant, not when the bar it beat printed.
        // Anything the same order goes on to take from a print -- including
        // from the very bar it beat, which it rested through -- happened
        // when that print did, like any other fill.
        let arrived_at = on_arrival.then_some(order.submitted_ts);

        // Stochastic fills: a marketable resting limit may be passed over
        // (queue position, exhausted liquidity); it stays working. Stop and
        // market fills may instead slip one tick against the trader.
        let is_limit_fill = matches!(kind, OrderKind::Limit { .. })
            || (matches!(kind, OrderKind::StopLimit { .. }) && status == OrderStatus::Triggered);
        let mut queue_granted = false;
        if is_limit_fill {
            // The queue model reads the tape; it consumes no randomness, so
            // enabling it must not shift the RNG stream for other orders.
            let verdict = self.queue_verdict(id, kind, status, side, bar);
            match verdict {
                Some(QueueVerdict::Resting) => return,
                Some(_) => {
                    // The queue model earned this fill from volume observed
                    // trading ahead of the order, so it genuinely held the
                    // price: limit slippage would double-penalize it.
                    queue_granted = true;
                }
                None => {
                    if self.config.fill_prob_limit < 1.0
                        && self.fill_rng.next_f64() >= self.config.fill_prob_limit
                    {
                        return; // untouched: still Accepted/Triggered, retries next bar
                    }
                }
            }
        }
        let matched_price = match (queue_granted, kind) {
            (true, OrderKind::Limit { price }) => price,
            (true, OrderKind::StopLimit { price, .. }) => price,
            _ => matched_price,
        };
        let matched_price = if !is_limit_fill
            && !matched_price.is_nan()
            && self.config.fill_prob_slippage > 0.0
            && self.fill_rng.next_f64() < self.config.fill_prob_slippage
        {
            match &self.spec {
                Some(spec) if spec.price_increment > 0.0 => match side {
                    OrderSide::Buy => matched_price + spec.price_increment,
                    OrderSide::Sell => matched_price - spec.price_increment,
                },
                _ => matched_price,
            }
        } else {
            matched_price
        };

        // A refused order must always be observable. `reject` counts against
        // `rejected_entries` so a discarded order can never look like an
        // order that was never placed; `reject_uncounted` is the deliberate
        // exception, for refusals that are not constraint decisions (sizing
        // arithmetic, which `open_at` already reports, and failed closes,
        // which are not refused *entries*).
        fn reject_uncounted(
            orders: &mut OrderEngine,
            events: &mut Vec<EngineEvent>,
            idx: usize,
            id: u64,
            client_id: &str,
            reason: &'static str,
        ) {
            if let Some(order) = orders.get_mut(id) {
                let _ = order.transition(OrderStatus::Rejected);
            }
            events.push(EngineEvent::OrderRejected {
                idx,
                order_id: id,
                client_id: client_id.to_string(),
                reason,
            });
        }
        fn reject(
            orders: &mut OrderEngine,
            risk: &mut RiskGate,
            events: &mut Vec<EngineEvent>,
            idx: usize,
            id: u64,
            client_id: &str,
            reason: &'static str,
        ) {
            risk.record_rejection();
            reject_uncounted(orders, events, idx, id, client_id, reason);
        }

        // An order's side is authoritative for opening. Independent
        // (hedging) policy: every order opens in its own side's direction.
        // Net policy: an order opposing the open position closes it — that
        // is how bracket legs and take-profit orders exit — but with no
        // position the order opens in its *own* side's direction. The
        // kernel's `direction` field governs the signal path only and is
        // never consulted here, so one run can hold long and short legs and
        // a leg can flip side across rebalances once it goes flat.
        //
        // `reduce_only` short-circuits to the closing branch so a protective
        // leg left working after a stop already exited can never reverse
        // into a fresh position.
        let order_direction = match side {
            OrderSide::Buy => Direction::Long,
            OrderSide::Sell => Direction::Short,
        };
        let hedging = self.ledger.policy() == PositionPolicy::Independent;
        let held = self.ledger.first().map(|m| m.position.direction);
        let (opens, open_direction) = if hedging {
            (true, order_direction)
        } else if reduce_only {
            (false, order_direction)
        } else {
            match held {
                // Opposing an open position is a close.
                Some(dir) if dir != order_direction => (false, order_direction),
                // Agreeing with one: netting refuses to add (below).
                Some(_) => (true, order_direction),
                // Flat: the order's own side decides the direction.
                None => (true, order_direction),
            }
        };

        // The price the level actually traded at, which is what the next
        // level is measured from. A market order arrives priced `NAN` and
        // only resolves against the fill-price model inside the branches
        // below, so reading the request back would price its sweep off a
        // price that was never traded.
        let mut executed: Option<Price> = None;

        if opens {
            if reduce_only {
                // A reduce-only order must never increase exposure. Under
                // netting `reduce_only` already routed to the closing branch,
                // so this only fires for hedging, where every order opens.
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    "reduce_only",
                );
                return;
            }
            // Plain netting holds one position and refuses to add to it.
            // Netting-with-averaging is the same account model with the
            // refusal lifted: the fill grows the position instead, which is
            // what a re-sent remainder needs.
            //
            // An order that has already filled part of its size is exempt:
            // it is finishing the position it opened, not opening a second
            // one, and refusing it would strand every order a bar could
            // only partly absorb.
            let resuming = leaves.is_some();
            if !resuming
                && self.ledger.policy() == PositionPolicy::Net
                && self.ledger.is_in_position()
            {
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    "position_open",
                );
                return;
            }
            if self.margin.is_halted() {
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    "margin_call",
                );
                return;
            }
            if let Err(reason) = self.risk.check_entry(self.gating_open_count()) {
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    reason.as_str(),
                );
                return;
            }
            let raw_price = if matched_price.is_nan() {
                self.fill_price_for(bar, open_direction, true)
            } else {
                matched_price
            };
            let (size_mult, explicit_units) = match (leaves, qty) {
                (Some(left), _) => (None, Some(left)),
                (None, QtySpec::Units(u)) => (None, Some(u)),
                (None, QtySpec::CapitalFrac(f)) => (Some(f), None),
                (None, QtySpec::FullPosition) => {
                    reject(
                        &mut self.orders,
                        &mut self.risk,
                        events,
                        idx,
                        id,
                        &client_id,
                        "invalid_qty",
                    );
                    return;
                }
            };
            let terms =
                FillTerms { cap, all_or_none: tif == TimeInForce::Fok, resuming, at: arrived_at };
            match self.open_at(
                idx,
                bar,
                open_direction,
                raw_price,
                size_mult,
                explicit_units,
                // The most recent bar's ATR, so an order-path open honors an
                // ATR stop/target config exactly as a signal entry does.
                // Passing 0.0 here silently yielded no stop at all.
                self.last_atr,
                stop_attach,
                target_attach,
                terms,
            ) {
                Some(OpenResult {
                    event: EngineEvent::Entered { price, size, direction, .. },
                    requested,
                    fees,
                }) => {
                    executed = Some(price);
                    let filled =
                        self.orders.get_mut(id).map(|order| order.record_fill(size, requested));
                    if let Some(status) = filled {
                        if let Some(order) = self.orders.get_mut(id) {
                            let _ = order.transition(status);
                        }
                    }
                    events.push(EngineEvent::OrderFilled {
                        idx,
                        order_id: id,
                        client_id: client_id.clone(),
                        price,
                        size,
                        commission: fees,
                        leaves: self.leaves_after_fill(id),
                        // An opening fill realizes nothing; its cost is the
                        // commission, which the fill reports on its own.
                        gross_realized: 0.0,
                    });
                    events.push(EngineEvent::Entered { idx, price, size, direction });
                    match filled {
                        Some(OrderStatus::PartiallyFilled) => {
                            self.expire_remainder(idx, id, tif, &client_id, events)
                        }
                        _ => self.after_fill(idx, id, events),
                    }
                }
                Some(OpenResult { event: EngineEvent::EntryRejected { reason, .. }, .. }) => {
                    // `open_at` already reported this; sizing arithmetic is
                    // not a constraint refusal, and the signal path does not
                    // count it either.
                    reject_uncounted(
                        &mut self.orders,
                        events,
                        idx,
                        id,
                        &client_id,
                        reason.as_str(),
                    );
                }
                _ => reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    "unfillable",
                ),
            }
        } else {
            let Some(first) = self.ledger.first() else {
                // Nothing to close. A reduce-only order was routed here on
                // purpose and keeps reporting `reduce_only` (it must never
                // open, which is exactly what happened); any other order
                // wanted to open and was refused. Both are counted.
                let reason = if reduce_only { "reduce_only" } else { "no_position" };
                reject(&mut self.orders, &mut self.risk, events, idx, id, &client_id, reason);
                return;
            };
            let position_id = first.id;
            let direction = first.position.direction;
            let raw_price = if matched_price.is_nan() {
                self.fill_price_for(bar, direction, false)
            } else {
                matched_price
            };
            // An opposing order closes the position it meets; its own
            // quantity does not size the close (long-standing behavior).
            // What can size it down is the bar's liquidity.
            let open_size = first.position.size;
            match self.reduce_at(
                idx,
                bar,
                position_id,
                raw_price,
                ExitReason::Order,
                cap,
                arrived_at,
            ) {
                ReduceResult::Closed { size, price, fees, gross_realized, event } => {
                    executed = Some(price);
                    let filled =
                        self.orders.get_mut(id).map(|order| order.record_fill(size, open_size));
                    if let Some(order) = self.orders.get_mut(id) {
                        let _ = order.transition(filled.unwrap_or(OrderStatus::Filled));
                    }
                    events.push(EngineEvent::OrderFilled {
                        idx,
                        order_id: id,
                        client_id,
                        price,
                        size,
                        commission: fees,
                        leaves: self.leaves_after_fill(id),
                        gross_realized,
                    });
                    events.push(event);
                    self.after_fill(idx, id, events);
                }
                ReduceResult::Reduced { size, price, fees, gross_realized } => {
                    executed = Some(price);
                    if let Some(order) = self.orders.get_mut(id) {
                        let status = order.record_fill(size, open_size);
                        let _ = order.transition(status);
                    }
                    events.push(EngineEvent::OrderFilled {
                        idx,
                        order_id: id,
                        client_id: client_id.clone(),
                        price,
                        size,
                        commission: fees,
                        leaves: self.leaves_after_fill(id),
                        gross_realized,
                    });
                    self.expire_remainder(idx, id, tif, &client_id, events);
                }
                // A close that could not fill is not a refused *entry*, so it
                // stays out of the rejected-entries count.
                ReduceResult::None => {
                    reject_uncounted(&mut self.orders, events, idx, id, &client_id, "unfillable");
                    return;
                }
            }
        }

        if let Some(price) = executed {
            self.take_next_level(idx, bar, id, price, depth, on_arrival, events);
        }
    }

    /// Continue an order that the print it just took could not satisfy.
    ///
    /// `price` is the price that actually traded, not the price the match
    /// asked for: a sweep steps off the level it emptied. `on_arrival` is
    /// carried through unchanged: the continuation is the same fill walking
    /// the book it already met, so it happened when that fill did.
    ///
    /// The bar is not one price but a handful of prints, so an order that
    /// emptied one of them may still find size in the next. Re-entering
    /// [`Self::apply_match_outcome`] rather than looping here is deliberate:
    /// a second fill has to pass through exactly the same open/close, risk
    /// and bookkeeping paths as the first, and there is no way to forget one
    /// of them if there is only ever one path.
    fn take_next_level(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        id: u64,
        price: Price,
        depth: FillDepth,
        on_arrival: bool,
        events: &mut Vec<EngineEvent>,
    ) {
        let outstanding = self
            .orders
            .get(id)
            .filter(|order| !order.status.is_terminal())
            .and_then(|order| order.leaves_qty())
            .is_some_and(|leaves| leaves > 0.0);
        if !outstanding {
            return;
        }
        let (next_price, next_depth) = match depth.next() {
            // Another print at the order's own price, showing its own size.
            Some(NextPrint::Same(rest)) => (price, rest),
            // The market moved through an order that was already resting:
            // it was there first, so it is the side being traded against
            // and the whole remainder fills at its own price.
            Some(NextPrint::Through) => (price, FillDepth::UNLIMITED),
            // An aggressive order emptied the book at the price it took, so
            // the remainder crosses the next level, one increment worse. An
            // instrument with no price grid has no discrete levels, and the
            // sweep collapses onto the same price.
            Some(NextPrint::Sweep) => {
                let increment = self.price_increment();
                let swept = match self.orders.get(id).map(|order| order.side) {
                    Some(OrderSide::Buy) => price + increment,
                    Some(OrderSide::Sell) => price - increment,
                    None => return,
                };
                (swept, FillDepth::UNLIMITED)
            }
            None => return,
        };
        self.apply_match_outcome(
            idx,
            bar,
            MatchOutcome::Fill {
                order_id: id,
                price: next_price,
                depth: next_depth,
                on_arrival,
            },
            events,
        );
    }

    /// Decide what happens to the unfilled remainder of a partial fill.
    ///
    /// An immediate-or-cancel order lives for exactly one evaluation, so
    /// what the bar could not absorb is killed and the strategy sees a
    /// finished order that filled short. Anything else stays working and
    /// takes more size on later bars.
    ///
    /// Contingencies deliberately do not fire here: a one-cancels-other
    /// sibling must not be pulled, and a held child must not be activated,
    /// while the order that would trigger them is still partly working.
    /// Both happen through [`Self::after_fill`] when it completes -- or, for
    /// an IOC order, not at all, which is right: a bracket whose entry only
    /// half-filled never fully armed.
    fn expire_remainder(
        &mut self,
        idx: usize,
        id: u64,
        tif: TimeInForce,
        client_id: &str,
        events: &mut Vec<EngineEvent>,
    ) {
        if !matches!(tif, TimeInForce::Ioc | TimeInForce::Fok) {
            return;
        }
        if let Some(order) = self.orders.get_mut(id) {
            let _ = order.transition(OrderStatus::Canceled);
        }
        events.push(EngineEvent::OrderCanceled {
            idx,
            order_id: id,
            client_id: client_id.to_string(),
        });
    }

    /// Contingency consequences of a fill: activate held one-triggers-other
    /// children, then cancel one-cancels-other siblings.
    fn after_fill(&mut self, idx: usize, filled_id: u64, events: &mut Vec<EngineEvent>) {
        let children: Vec<(u64, String)> = self
            .orders
            .all()
            .iter()
            .filter(|o| o.parent_id == Some(filled_id) && o.status == OrderStatus::Submitted)
            .map(|o| (o.id, o.client_id.clone()))
            .collect();
        for (child_id, client_id) in children {
            if let Some(order) = self.orders.get_mut(child_id) {
                let _ = order.transition(OrderStatus::Accepted);
            }
            events.push(EngineEvent::OrderAccepted { idx, order_id: child_id, client_id });
        }

        let group = self.orders.get(filled_id).and_then(|o| o.oco_group);
        if let Some(group) = group {
            let siblings: Vec<(u64, String)> = self
                .orders
                .all()
                .iter()
                .filter(|o| {
                    o.oco_group == Some(group) && o.id != filled_id && !o.status.is_terminal()
                })
                .map(|o| (o.id, o.client_id.clone()))
                .collect();
            for (sibling_id, client_id) in siblings {
                if self.orders.cancel(sibling_id) {
                    events.push(EngineEvent::OrderCanceled {
                        idx,
                        order_id: sibling_id,
                        client_id,
                    });
                }
            }
        }
    }
}
