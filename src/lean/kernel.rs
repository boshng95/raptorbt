//! The kernel a library cell's replay drives, for its one shape: one
//! instrument, long only, netting-averaging, a margin account at infinite
//! leverage, limit and market orders, same-bar-close fills over the volume
//! slices of [`super::market`].
//!
//! It is a transcription, not a model: every branch mirrors a line of
//! `portfolio::kernel`, `kernel_orders` and `execution::orders::matching` as
//! that configuration runs them, and their margin, stop, expiry-settlement,
//! slippage and queue branches -- all no-ops there -- are left out. The
//! arithmetic is theirs: the fee schedule, the lot grid, the decimal price
//! step and the money quantization are called, not restated.

use super::market::{Market, Tape};
use crate::core::decimals::{decimal_sum, quantize_money};
use crate::core::types::Direction;
use crate::execution::{FeeModel, OrderBilling};

const DAY_NS: i64 = 86_400_000_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tif {
    Gtc,
    Day,
    Gtd(i64),
    Ioc,
}

impl Tif {
    pub fn name(self) -> &'static str {
        match self {
            Tif::Gtc => "GTC",
            Tif::Day => "DAY",
            Tif::Gtd(_) => "GTD",
            Tif::Ioc => "IOC",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Submitted,
    Accepted,
    PartiallyFilled,
    Filled,
    Canceled,
    Expired,
    Rejected,
}

impl Status {
    #[inline]
    fn terminal(self) -> bool {
        matches!(self, Status::Filled | Status::Canceled | Status::Expired | Status::Rejected)
    }
}

struct Order {
    buy: bool,
    units: f64,
    /// None for a market order.
    limit: Option<f64>,
    tif: Tif,
    at: usize,
    at_ts: i64,
    reduce_only: bool,
    status: Status,
    resolved: Option<f64>,
    filled: f64,
    billing: OrderBilling,
}

#[inline]
fn tolerance(total: f64) -> f64 {
    f64::EPSILON * total.abs().max(1.0) * 4.0
}

impl Order {
    /// `Order::leaves_qty`.
    #[inline]
    fn leaves(&self) -> Option<f64> {
        self.resolved.map(|total| {
            let left = total - self.filled;
            if left <= tolerance(total) {
                0.0
            } else {
                left
            }
        })
    }

    /// `Order::record_fill`.
    #[inline]
    fn record_fill(&mut self, units: f64, total: f64) -> Status {
        let total = *self.resolved.get_or_insert(total);
        self.filled += units;
        if total - self.filled <= tolerance(total) {
            Status::Filled
        } else {
            Status::PartiallyFilled
        }
    }

    #[inline]
    fn immediate(&self) -> bool {
        self.tif == Tif::Ioc
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Closing {
    size: f64,
    price: f64,
    gross: f64,
    fees: f64,
    last_ts: i64,
}

#[derive(Clone, Copy, Debug)]
struct Position {
    entry_price: f64,
    size: f64,
    entry_fees: f64,
    entry_ts: i64,
    realized: Option<f64>,
    closing: Closing,
}

/// A closed round trip, as the report reads it.
#[derive(Clone, Debug)]
pub struct Trade {
    pub entry_time: i64,
    pub exit_time: i64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size: f64,
    pub pnl: f64,
    pub entry_fees: f64,
    pub exit_fees: f64,
    pub end_of_data: bool,
}

/// What a step reports, in the kernel's order.
#[derive(Clone, Copy, Debug)]
pub enum Event {
    Accepted,
    Filled { id: usize, price: f64, size: f64, commission: f64, leaves: f64, gross: f64 },
    Entered,
    Exited,
    Canceled(usize),
    Expired(usize),
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Tail {
    Rests,
    Through,
    Sweep,
}

/// `FillDepth`: what the book offers an order, print by print.
#[derive(Clone, Copy, Debug)]
struct Depth {
    sizes: [f64; 4],
    prints: usize,
    tail: Tail,
}

enum Next {
    Same(Depth),
    Through,
    Sweep,
}

impl Depth {
    const NONE: Depth = Depth { sizes: [0.0; 4], prints: 0, tail: Tail::Rests };
    const UNLIMITED: Depth =
        Depth { sizes: [f64::INFINITY, 0.0, 0.0, 0.0], prints: 1, tail: Tail::Rests };

    fn single(size: f64, tail: Tail) -> Depth {
        Depth { sizes: [size, 0.0, 0.0, 0.0], prints: 1, tail }
    }

    fn repeated(size: f64, n: usize, tail: Tail) -> Depth {
        let mut sizes = [0.0; 4];
        for slot in sizes.iter_mut().take(n.min(4)) {
            *slot = size;
        }
        Depth { sizes, prints: n.min(4), tail }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.prints == 0
    }

    #[inline]
    fn cap(&self) -> f64 {
        if self.prints == 0 {
            0.0
        } else {
            self.sizes[0]
        }
    }

    fn next(&self) -> Option<Next> {
        match self.prints {
            0 => None,
            1 => match self.tail {
                Tail::Rests => None,
                Tail::Through => Some(Next::Through),
                Tail::Sweep => Some(Next::Sweep),
            },
            n => {
                let mut sizes = [0.0; 4];
                sizes[..n - 1].copy_from_slice(&self.sizes[1..n]);
                Some(Next::Same(Depth { sizes, prints: n - 1, tail: self.tail }))
            }
        }
    }
}

#[inline]
fn marketable(book_price: f64, price: f64, buy: bool) -> bool {
    if buy {
        book_price <= price
    } else {
        book_price >= price
    }
}

/// `BarTape::offered`: an arriving limit against the book the bar left.
fn standing_offered(book: (f64, f64), price: f64, buy: bool, immediate: bool) -> Depth {
    let (book_price, size) = book;
    if !marketable(book_price, price, buy) {
        return Depth::NONE;
    }
    if !size.is_finite() {
        return Depth::UNLIMITED;
    }
    match (immediate, book_price == price) {
        (true, _) => Depth::single(size, Tail::Rests),
        (false, true) => Depth::repeated(size, 2, Tail::Rests),
        (false, false) => Depth::single(size, Tail::Sweep),
    }
}

/// `Tape::offered`: a resting limit against this bar's prints.
fn tape_offered(tape: &Tape, price: f64, buy: bool, immediate: bool) -> Depth {
    if tape.unbounded {
        return Depth::UNLIMITED;
    }
    let mut depth = Depth::NONE;
    for &(print_price, print_size) in &tape.prints[..tape.n] {
        if !marketable(print_price, price, buy) {
            continue;
        }
        depth.sizes[depth.prints] = print_size;
        depth.prints += 1;
        if immediate {
            return depth;
        }
        if print_price != price {
            depth.tail = Tail::Through;
            return depth;
        }
    }
    depth
}

/// `BarTape::offered_once`: `walk_book`'s single bite.
fn offered_once(book: (f64, f64), price: f64, buy: bool) -> Depth {
    let (book_price, size) = book;
    if !marketable(book_price, price, buy) {
        return Depth::NONE;
    }
    if !size.is_finite() {
        return Depth::UNLIMITED;
    }
    Depth::single(size, if book_price == price { Tail::Rests } else { Tail::Sweep })
}

/// `price_step`: `price` moved by `step` along its grid.
#[inline]
fn price_step(price: f64, step: f64) -> f64 {
    if step == 0.0 {
        return price;
    }
    decimal_sum(price, step).unwrap_or(price + step)
}

enum Outcome {
    Expire(usize),
    Cancel(usize),
    Fill { id: usize, price: f64, depth: Depth },
}

enum Open {
    Rejected,
    Entered { price: f64, size: f64, requested: f64, fees: f64 },
}

/// The instrument and account terms a cell's kernel is opened on.
#[derive(Clone, Debug)]
pub struct Terms {
    pub initial_capital: f64,
    pub price_increment: f64,
    pub max_quantity: Option<f64>,
    pub currency_precision: Option<u32>,
    pub fees: f64,
    pub fee_per_share: f64,
    pub fee_minimum: f64,
    pub fee_max_pct: f64,
}

impl Terms {
    /// `BacktestConfig::fee_model` without a fee segment.
    pub fn fee_model(&self) -> FeeModel {
        if self.fee_per_share > 0.0 || self.fee_minimum > 0.0 || self.fee_max_pct > 0.0 {
            FeeModel::brokerage(self.fees, self.fee_per_share, self.fee_minimum, self.fee_max_pct)
        } else {
            FeeModel::percentage(self.fees)
        }
    }
}

pub struct Kernel<'m> {
    m: &'m Market,
    increment: f64,
    max_quantity: Option<f64>,
    precision: Option<u32>,
    fee: FeeModel,
    bills_per_order: bool,
    cash: f64,
    orders: Vec<Order>,
    working: Vec<usize>,
    pending: Vec<Event>,
    position: Option<Position>,
    pub trades: Vec<Trade>,
    last_bar: Option<usize>,
    /// A match's outcomes, kept between steps for their capacity.
    expiries: Vec<Outcome>,
    actions: Vec<Outcome>,
}

impl<'m> Kernel<'m> {
    pub fn new(m: &'m Market, terms: &Terms) -> Self {
        let fee = terms.fee_model();
        let increment = terms.price_increment;
        Self {
            m,
            increment: if increment > 0.0 && increment.is_finite() { increment } else { 0.0 },
            max_quantity: terms.max_quantity,
            precision: terms.currency_precision,
            bills_per_order: fee.bills_per_order(),
            fee,
            cash: terms.initial_capital,
            orders: Vec::new(),
            working: Vec::new(),
            pending: Vec::new(),
            position: None,
            trades: Vec::new(),
            last_bar: None,
            expiries: Vec::new(),
            actions: Vec::new(),
        }
    }

    // -- account reads ----------------------------------------------------

    #[inline]
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// `(has_position, net quantity)`.
    #[inline]
    pub fn net_position(&self) -> (bool, f64) {
        match &self.position {
            Some(p) => (true, p.size),
            None => (false, 0.0),
        }
    }

    /// `equity(close)`: cash plus the open position marked at `close`.
    #[inline]
    pub fn equity(&self, close: f64) -> f64 {
        match &self.position {
            Some(p) => self.cash + ((close - p.entry_price) * p.size * 1.0) * 1.0,
            None => self.cash + -0.0,
        }
    }

    /// Whether a step does nothing but mark: no event queued, no order working.
    #[inline]
    pub fn idle(&self) -> bool {
        self.pending.is_empty() && self.working.iter().all(|&id| self.orders[id].status.terminal())
    }

    /// Forget every order once none is live ([`Kernel::idle`]), handing
    /// their ids out again from 0. Nothing may still hold one: the caller's
    /// map from ids goes with them.
    pub fn reclaim(&mut self) -> bool {
        if !self.idle() {
            return false;
        }
        self.orders.clear();
        self.working.clear();
        true
    }

    // -- helpers ------------------------------------------------------------

    #[inline]
    fn q(&self, value: f64) -> f64 {
        quantize_money(value, self.precision)
    }

    #[inline]
    fn round_size(&self, raw: f64) -> f64 {
        if self.m.lot > 0.0 {
            self.m.grid.floor(raw)
        } else {
            raw
        }
    }

    #[inline]
    fn leaves_on_grid(&self, id: usize) -> Option<f64> {
        self.orders[id].leaves().map(|left| self.m.grid.snap(left))
    }

    #[inline]
    fn leaves_after_fill(&self, id: usize) -> f64 {
        self.leaves_on_grid(id).map_or(0.0, |left| left.max(0.0))
    }

    fn book_cash(&mut self, amounts: &[f64]) {
        let mut cash = self.cash;
        for &amount in amounts {
            cash += self.q(amount);
        }
        self.cash = self.q(cash);
    }

    /// `bill_fill`: the charge a fill adds to its order's billing.
    fn bill_fill(&self, id: usize, price: f64, size: f64) -> Option<(f64, OrderBilling)> {
        if !self.bills_per_order {
            return None;
        }
        let prior = self.orders[id].billing;
        let next = self.fee.accrue(&prior, price, size)?;
        let total = self.q(self.fee.order_charge(&next));
        let charge = self.q((total - prior.charged).max(0.0));
        Some((charge, OrderBilling { charged: prior.charged + charge, ..next }))
    }

    // -- orders ---------------------------------------------------------------

    /// `submit_order_on` for a limit (`limit` Some) or market order.
    pub fn submit(
        &mut self,
        i: usize,
        now: i64,
        buy: bool,
        units: f64,
        limit: Option<f64>,
        tif: Tif,
        reduce_only: bool,
    ) -> usize {
        let id = self.orders.len();
        let status = if limit.is_some() { Status::Accepted } else { Status::Submitted };
        self.orders.push(Order {
            buy,
            units,
            limit,
            tif,
            at: i,
            at_ts: now,
            reduce_only,
            status,
            resolved: None,
            filled: 0.0,
            billing: OrderBilling::default(),
        });
        self.working.push(id);
        if limit.is_some() {
            self.pending.push(Event::Accepted);
        }
        id
    }

    fn expired(&self, id: usize, ts: i64) -> bool {
        let order = &self.orders[id];
        match order.tif {
            Tif::Gtd(expire) => ts >= expire,
            Tif::Day => ts.div_euclid(DAY_NS) > order.at_ts.div_euclid(DAY_NS),
            _ => false,
        }
    }

    // -- the step -------------------------------------------------------------

    /// Idle steps up to bar `last`: they only mark, and the book is the bars'.
    #[inline]
    pub fn pass(&mut self, last: usize) {
        debug_assert!(self.idle());
        self.last_bar = Some(last);
    }

    /// `events.append(&mut self.pending)`. The caller's list is usually
    /// empty, and then trading the two buffers moves the same events in the
    /// same order without copying them.
    #[inline]
    fn take_pending(&mut self, events: &mut Vec<Event>) {
        if self.pending.is_empty() {
            return;
        }
        if events.is_empty() {
            std::mem::swap(events, &mut self.pending);
        } else {
            events.append(&mut self.pending);
        }
    }

    /// `EngineKernel::step` on bar `i`.
    pub fn step(&mut self, i: usize, events: &mut Vec<Event>) {
        self.last_bar = Some(i);
        if self.idle() {
            return;
        }
        self.take_pending(events);
        self.match_events(i, events);
        self.sweep_markets(i, self.m.ts[i], events);
    }

    /// `match_events`: every working order against bar `i`, all decided
    /// against the same tape before any is applied.
    fn match_events(&mut self, i: usize, events: &mut Vec<Event>) {
        let orders = &self.orders;
        self.working.retain(|&id| !orders[id].status.terminal());
        if self.working.is_empty() {
            return;
        }
        let m = self.m;
        // Replayed once a resting order's price is reached; an order on its
        // own bar meets the standing book.
        let mut tape = None;
        let standing = m.book_after(i);
        let ts = m.ts[i];
        let mut expiries = std::mem::take(&mut self.expiries);
        let mut actions = std::mem::take(&mut self.actions);
        for slot in 0..self.working.len() {
            let id = self.working[slot];
            if self.orders[id].status.terminal() || self.orders[id].at > i {
                continue;
            }
            if self.expired(id, ts) {
                self.orders[id].status = Status::Expired;
                expiries.push(Outcome::Expire(id));
                continue;
            }
            let order = &mut self.orders[id];
            let (buy, immediate) = (order.buy, order.immediate());
            if order.at == i {
                if let Some(price) = order.limit {
                    let depth = standing_offered(standing, price, buy, immediate);
                    if !depth.is_empty() {
                        actions.push(Outcome::Fill { id, price: standing.0, depth });
                        continue;
                    }
                }
                if immediate {
                    order.status = Status::Canceled;
                    actions.push(Outcome::Cancel(id));
                }
                continue;
            }
            let out = match order.limit {
                None => None,
                Some(price) => {
                    let reached = if buy { m.low[i] <= price } else { m.high[i] >= price };
                    if reached {
                        let tape = tape.get_or_insert_with(|| m.tape(i));
                        let depth = tape_offered(tape, price, buy, immediate);
                        (!depth.is_empty()).then_some(Outcome::Fill { id, price, depth })
                    } else {
                        None
                    }
                }
            };
            match out {
                Some(outcome) => actions.push(outcome),
                None if immediate && order.limit.is_some() => {
                    order.status = Status::Canceled;
                    actions.push(Outcome::Cancel(id));
                }
                None => {}
            }
        }
        // Two loops rather than a chained iterator, which moved each outcome
        // through the stack in pieces.
        for outcome in expiries.drain(..) {
            self.apply(i, ts, outcome, events);
        }
        for outcome in actions.drain(..) {
            self.apply(i, ts, outcome, events);
        }
        self.expiries = expiries;
        self.actions = actions;
    }

    /// `walk_book`: one more pass over the book the last bar left, for the
    /// orders its own fills provoked. No expiry, no replay.
    pub fn walk_book(&mut self, now: i64, events: &mut Vec<Event>) {
        let Some(i) = self.last_bar else { return };
        self.take_pending(events);
        let book = self.m.book_after(i);
        let orders = &self.orders;
        self.working.retain(|&id| !orders[id].status.terminal());
        let mut out = std::mem::take(&mut self.actions);
        for slot in 0..self.working.len() {
            let id = self.working[slot];
            let order = &mut self.orders[id];
            let Some(price) = order.limit else { continue };
            let depth = offered_once(book, price, order.buy);
            if depth.is_empty() {
                if order.immediate() {
                    order.status = Status::Canceled;
                    out.push(Outcome::Cancel(id));
                }
                continue;
            }
            out.push(Outcome::Fill { id, price: book.0, depth });
        }
        for outcome in out.drain(..) {
            self.apply(i, now, outcome, events);
        }
        self.actions = out;
        self.sweep_markets(i, now, events);
    }

    /// `sweep_markets`: market orders sent on bar `i` fill at its close,
    /// stamped `ts`.
    fn sweep_markets(&mut self, i: usize, ts: i64, events: &mut Vec<Event>) {
        for slot in 0..self.working.len() {
            let id = self.working[slot];
            let order = &mut self.orders[id];
            if order.limit.is_some() || order.status.terminal() || order.at != i {
                continue;
            }
            if order.status == Status::Submitted {
                order.status = Status::Accepted;
                events.push(Event::Accepted);
            }
            let tail = if order.immediate() { Tail::Rests } else { Tail::Sweep };
            let depth = Depth::single(self.m.book_after(i).1, tail);
            self.apply(i, ts, Outcome::Fill { id, price: f64::NAN, depth }, events);
        }
    }

    // -- outcomes -------------------------------------------------------------

    fn apply(&mut self, i: usize, ts: i64, outcome: Outcome, events: &mut Vec<Event>) {
        match outcome {
            Outcome::Expire(id) => events.push(Event::Expired(id)),
            Outcome::Cancel(id) => events.push(Event::Canceled(id)),
            Outcome::Fill { id, price, depth } => self.fill(i, ts, id, price, depth, events),
        }
    }

    /// `apply_match_outcome` for a fill, then `take_next_level`.
    fn fill(
        &mut self,
        i: usize,
        ts: i64,
        id: usize,
        matched: f64,
        depth: Depth,
        events: &mut Vec<Event>,
    ) {
        let cap = depth.cap();
        let cap = if cap.is_finite() { self.round_size(cap) } else { cap };
        let leaves = self.leaves_on_grid(id);
        let order = &self.orders[id];
        let (buy, immediate, units) = (order.buy, order.immediate(), order.units);
        let opens = !order.reduce_only && (self.position.is_none() || buy);
        let raw = if matched.is_nan() { self.m.close[i] } else { matched };
        let executed;
        if opens {
            let explicit = leaves.unwrap_or(units);
            match self.open_at(ts, id, raw, explicit, cap) {
                Open::Rejected => {
                    self.orders[id].status = Status::Rejected;
                    events.push(Event::Rejected);
                    return;
                }
                Open::Entered { price, size, requested, fees } => {
                    executed = price;
                    let status = self.orders[id].record_fill(size, requested);
                    self.orders[id].status = status;
                    let leaves = self.leaves_after_fill(id);
                    events.push(Event::Filled {
                        id,
                        price,
                        size,
                        commission: fees,
                        leaves,
                        gross: 0.0,
                    });
                    events.push(Event::Entered);
                    if status == Status::PartiallyFilled && immediate {
                        self.orders[id].status = Status::Canceled;
                        events.push(Event::Canceled(id));
                    }
                }
            }
        } else {
            let Some(mut p) = self.position else {
                self.orders[id].status = Status::Rejected;
                events.push(Event::Rejected);
                return;
            };
            let open = p.size;
            let requested = leaves.unwrap_or(units);
            let asked = requested.min(open);
            let taking = cap.min(asked);
            let size = open.min(taking);
            if !(size > 0.0) {
                self.orders[id].status = Status::Rejected;
                events.push(Event::Rejected);
                return;
            }
            let price = raw + 0.0;
            let fee_price = price * 1.0;
            let billed = self.bill_fill(id, fee_price, size);
            let fees = match billed {
                Some((charge, _)) => charge,
                None => self.q(self.fee.calculate(fee_price, size, Direction::Long)),
            };
            let filled = size.min(p.size);
            let gross = (price - p.entry_price) * filled * (1.0 * 1.0);
            let closing = &mut p.closing;
            closing.price = if closing.size > 0.0 {
                (closing.price * closing.size + price * filled) / (closing.size + filled)
            } else {
                price
            };
            closing.size += filled;
            closing.gross += gross;
            closing.fees += fees;
            closing.last_ts = ts;
            if let Some(realized) = p.realized {
                p.realized = Some(self.q(realized + (-fees + gross)));
            }
            p.size = self.m.grid.snap(p.size - filled);
            let closed = !(p.size > 0.0);
            if let Some((_, billing)) = billed {
                self.orders[id].billing = billing;
            }
            self.book_cash(&[gross, -fees]);
            executed = price;
            let status = self.orders[id].record_fill(filled, asked);
            self.orders[id].status = status;
            let leaves = self.leaves_after_fill(id);
            events.push(Event::Filled { id, price, size: filled, commission: fees, leaves, gross });
            if closed {
                let trade = self.trade(&p, ts, false);
                self.trades.push(trade);
                self.position = None;
                events.push(Event::Exited);
            } else {
                self.position = Some(p);
                if status == Status::PartiallyFilled && immediate {
                    self.orders[id].status = Status::Canceled;
                    events.push(Event::Canceled(id));
                }
            }
        }
        // `take_next_level`.
        if self.orders[id].status.terminal()
            || !self.leaves_on_grid(id).is_some_and(|left| left > 0.0)
        {
            return;
        }
        let (price, depth) = match depth.next() {
            Some(Next::Same(rest)) => (executed, rest),
            Some(Next::Through) => (executed, Depth::UNLIMITED),
            Some(Next::Sweep) => {
                let step = if buy { self.increment } else { -self.increment };
                (price_step(executed, step), Depth::UNLIMITED)
            }
            None => return,
        };
        self.fill(i, ts, id, price, depth, events);
    }

    /// `open_at` for a buy with explicit units, under this configuration.
    fn open_at(&mut self, ts: i64, id: usize, raw: f64, explicit: f64, cap: f64) -> Open {
        let price = raw + 0.0;
        let available = self.cash - 0.0;
        let contract_value = price * 1.0;
        let requested = self.round_size(explicit);
        if self.max_quantity.is_some_and(|max| requested > max) {
            return Open::Rejected;
        }
        let size = self.round_size(explicit.min(cap));
        if size <= 0.0 {
            return Open::Rejected;
        }
        let billed = self.bill_fill(id, contract_value, size);
        let fees = match billed {
            Some((charge, _)) => charge,
            None => self.q(self.fee.calculate(contract_value, size, Direction::Long)),
        };
        let funding = contract_value * size * 0.0;
        if funding + fees > available {
            return Open::Rejected;
        }
        match self.position.as_mut() {
            Some(p) => {
                if !(size > 0.0) || !(p.size > 0.0) {
                    return Open::Rejected;
                }
                p.entry_price = (p.entry_price * p.size + price * size) / (p.size + size);
                p.size = self.m.grid.snap(p.size + size);
                p.entry_fees += fees;
                if let Some(realized) = p.realized {
                    p.realized = Some(quantize_money(realized + (-fees + 0.0), self.precision));
                }
            }
            None => {
                self.position = Some(Position {
                    entry_price: price,
                    size,
                    entry_fees: fees,
                    entry_ts: ts,
                    realized: self.precision.map(|_| self.q(0.0 + (-fees + 0.0))),
                    closing: Closing::default(),
                });
            }
        }
        self.book_cash(&[-fees]);
        if let Some((_, billing)) = billed {
            self.orders[id].billing = billing;
        }
        Open::Entered { price, size, requested, fees }
    }

    /// `create_trade`.
    fn trade(&self, p: &Position, exit_ts: i64, end_of_data: bool) -> Trade {
        let total_fees = p.entry_fees + p.closing.fees;
        Trade {
            entry_time: p.entry_ts,
            exit_time: exit_ts,
            entry_price: p.entry_price,
            exit_price: p.closing.price,
            size: self.m.grid.snap(p.closing.size),
            pnl: p.realized.unwrap_or(p.closing.gross - total_fees),
            entry_fees: p.entry_fees,
            exit_fees: p.closing.fees,
            end_of_data,
        }
    }

    /// `finalize_all`: close what is still open at the last bar's close, for
    /// no fee and outside the fill ledger.
    pub fn finalize(&mut self, i: usize) {
        let Some(mut p) = self.position.take() else { return };
        let (ts, close) = (self.m.ts[i], self.m.close[i]);
        let filled = p.size;
        let gross = (close - p.entry_price) * filled * (1.0 * 1.0);
        let closing = &mut p.closing;
        closing.price = if closing.size > 0.0 {
            (closing.price * closing.size + close * filled) / (closing.size + filled)
        } else {
            close
        };
        closing.size += filled;
        closing.gross += gross;
        closing.fees += 0.0;
        closing.last_ts = ts;
        if let Some(realized) = p.realized {
            p.realized = Some(self.q(realized + (-0.0 + gross)));
        }
        p.size = self.m.grid.snap(p.size - filled);
        self.book_cash(&[gross, -0.0]);
        let trade = self.trade(&p, ts, true);
        self.trades.push(trade);
    }
}
