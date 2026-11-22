//! A library cell's rules, for the shape most of the library has: a legacy
//! spec with one direct crossover a side, every key on one resampled stream
//! that dispatches on the primary bar its composite closes on.
//!
//! The rule cursor is `python::rule_session`'s `advance`, `visit` and
//! `evaluate` for that shape, branch for branch. What makes it lean is the
//! entry side: its crossing only ever compares one resample bar's operands
//! with the one before (every resample bar is visited, in order, and a
//! primary visit repeats the operands it already holds), so whether a bar
//! enters is a bit computed once for the whole series -- once a call, for
//! every cell on the same keys ([`Signals`]). The exit side's clock is only
//! stepped while a position is held, so it stays a clock.
//!
//! A session filter (`slot_admits`) leaves a bar it does not admit unvisited:
//! its keys do not update and no clock steps, so the crossings are between
//! one admitted bar and the admitted bar before it.
//!
//! The band and stop families are level rules instead ([`Tree`]):
//! thresholds, zones and crossovers combined over the same frozen operands,
//! whose answer holds between composites, so a primary bar's visit can act
//! again. Their cursor is `python::rule_session`'s `advance_level` and its
//! condition program, branch for branch; every resample bar steps it.
//!
//! Level rules may read keys on more than one of the symbol's resampled
//! streams -- a filter on a second timeframe ([`Dispatch::of_legs`]). Each
//! instant of the frame then delivers its legs' bars in dispatch order
//! around the primary bar, each key holds its leg's last visit, and one not
//! ready yet holds no value.
//!
//! A leg may be one the rules read and never decide on -- another symbol's
//! reference leg (`DataParams.reference_legs`, [`Leg::decides`]). Its bar
//! moves its keys and evaluates nothing, and leaves the next primary bar a
//! visit owed (`rule_session`'s `owes_primary`): the first visit to read it.
//!
//! A cell may carry StrategyBase's risk exits ([`Policies`]). Nothing is
//! then deferred -- every primary bar is a callback, as on the full lane --
//! an entry decided from flat earlier in a callback rests through its later
//! events, the policies count the closes they read, and their check runs
//! after the events, its close taking over the rules' target.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};

use super::Declined;
use crate::python::rule_session::{
    delivery_step, Latch, PHASE_CLOSED_BY_GAP, PHASE_COMPLETED, PHASE_PRECLOSE, PHASE_PRIMARY,
};
use crate::stops::policies::{
    Account, AtrValues, BorrowView, OrderStyle, PolicyExit, PolicyState, Risk,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    Key(usize),
    Const(f64),
}

/// One compiled `CrossoverCondition`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cross {
    pub fast: Operand,
    pub slow: Operand,
    pub above: bool,
}

/// `ThresholdCondition.op`.
#[derive(Clone, Copy, Debug)]
pub enum Compare {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
}

impl Compare {
    #[inline]
    pub(crate) fn holds(self, a: f64, b: f64) -> bool {
        match self {
            Self::Gt => a > b,
            Self::Ge => a >= b,
            Self::Lt => a < b,
            Self::Le => a <= b,
            Self::Eq => a == b,
        }
    }
}

/// One node of a compiled condition, `DecisionProgram._compile`'s closure.
/// Children are earlier nodes of the same arena.
#[derive(Clone, Debug)]
pub enum Condition {
    All(Vec<usize>),
    Any(Vec<usize>),
    Not(usize),
    Zone { value: Operand, lower: f64, upper: f64 },
    Threshold { left: Operand, right: Operand, compare: Compare },
    Cross { clock: usize, cross: Cross },
}

/// What level rules read beyond crossover ones: StrategyBase's dust, on the
/// instrument contract's terms (`rule_session::Level`).
#[derive(Clone, Copy, Debug)]
pub struct Level {
    pub two_sided: bool,
    pub size_increment: f64,
    pub minimum_notional: f64,
}

impl Level {
    /// `InstrumentContract.is_dust`.
    #[inline]
    fn is_dust(&self, quantity: f64, price: f64) -> bool {
        let units = quantity.abs();
        units < self.size_increment
            || (self.minimum_notional > 0.0 && units * price < self.minimum_notional)
    }

    /// `NativeExecutionSession.reads_dust_as_flat`.
    #[inline]
    pub fn reads_dust_as_flat(&self, quantity: f64, pending: Option<f64>, close: f64) -> bool {
        quantity != 0.0
            && pending.is_none_or(|p| p != 0.0)
            && !self.two_sided
            && self.is_dust(quantity, close)
    }
}

/// A cell's level rules: a condition arena, the roots of its exit and entry
/// rules in rule order, how many crossover clocks they step, and the dust
/// they read.
#[derive(Clone, Debug)]
pub struct Tree {
    pub conditions: Vec<Condition>,
    pub exits: Vec<usize>,
    pub entries: Vec<usize>,
    pub clocks: usize,
    pub level: Level,
}

impl Tree {
    pub fn new(
        conditions: Vec<Condition>,
        exits: Vec<usize>,
        entries: Vec<usize>,
        clocks: usize,
        level: Level,
    ) -> Result<Self, Declined> {
        for (at, condition) in conditions.iter().enumerate() {
            // Children first: the arena is acyclic, and evaluation ends.
            let precede = |child: &usize| *child < at;
            let ok = match condition {
                Condition::All(children) | Condition::Any(children) => children.iter().all(precede),
                Condition::Not(child) => precede(child),
                Condition::Cross { clock, .. } => *clock < clocks,
                Condition::Zone { .. } | Condition::Threshold { .. } => true,
            };
            if !ok {
                return Err(Declined("a condition naming a later node or an unknown clock"));
            }
        }
        if exits.iter().chain(&entries).any(|&root| root >= conditions.len()) {
            return Err(Declined("a rule rooted outside its conditions"));
        }
        Ok(Self { conditions, exits, entries, clocks, level })
    }

    /// Each clock's crossing, in clock order: what `each_bar` steps on every
    /// evaluated bar (`rule_session::clock_crossings`).
    fn clock_crosses(&self) -> Result<Vec<Cross>, Declined> {
        let mut found: Vec<Option<Cross>> = vec![None; self.clocks];
        for condition in &self.conditions {
            let Condition::Cross { clock, cross } = condition else { continue };
            match found[*clock] {
                None => found[*clock] = Some(*cross),
                Some(seen) if seen == *cross => {}
                Some(_) => return Err(Declined("one clock names two crossings")),
            }
        }
        found.into_iter().map(|cross| cross.ok_or(Declined("a clock no crossing names"))).collect()
    }

    fn operands(&self) -> impl Iterator<Item = Operand> + '_ {
        self.conditions.iter().flat_map(|condition| match condition {
            Condition::Zone { value, .. } => vec![*value],
            Condition::Threshold { left, right, .. } => vec![*left, *right],
            Condition::Cross { cross, .. } => vec![cross.fast, cross.slow],
            _ => Vec::new(),
        })
    }
}

#[inline]
pub(crate) fn crossed(previous: (f64, f64), a: f64, b: f64, above: bool) -> bool {
    let (p0, p1) = previous;
    if above {
        p0 <= p1 && a > b
    } else {
        p0 >= p1 && a < b
    }
}

/// What a leg is built from: one of the traded symbol's `ScheduledStream`s
/// -- its bars' dispatch timestamps, labels and closes, its `dispatch_rank`,
/// whether the strategy builds it (`built_by_strategy`) and its
/// `stream_index` -- and whether the rules decide on it ([`Leg::decides`]).
pub struct LegTerms {
    pub ts: Vec<i64>,
    pub labels: Vec<i64>,
    pub close: Vec<f64>,
    pub rank: i64,
    pub built: bool,
    pub stream_index: i64,
    pub decides: bool,
}

/// One resampled stream of a frame: its bars' closes, their dispatch phases
/// and the instants they are dispatched at.
pub struct Leg {
    pub close: Vec<f64>,
    /// Each bar's dispatch phase (`rule_session::PHASE_*`), as
    /// `ScheduledStream.order` gives it: a bar labelled after its timestamp
    /// is a pre-close window, and one labelled before it on a leg the
    /// strategy builds a window a gap closed, both decided ahead of the
    /// primary bar at that timestamp ([`Leg::ahead`]); any other follows it.
    pub phase: Vec<i64>,
    /// Each bar's instant, a place in its frame's [`Dispatch::ts`].
    pub at: Vec<usize>,
    /// Whether the labels rise, so no visit repeats the label before it
    /// (`DecisionProgram.evaluate`'s repeat).
    pub labels_rise: bool,
    /// Whether the rules decide on its bars (`slot_decision`). One they only
    /// read -- a reference leg -- moves its keys, evaluates nothing and owes
    /// the next primary bar its visit.
    pub decides: bool,
}

impl Leg {
    /// Whether bar `r` is decided ahead of the primary bar it is dispatched
    /// with (`ScheduledBar.ahead_of_primary`).
    #[inline]
    pub fn ahead(&self, r: usize) -> bool {
        self.phase[r] < PHASE_PRIMARY
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.close.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

/// A frame's dispatches on its market, shared by every cell on it: each
/// instant a bar of any of its legs is dispatched at and the primary bar
/// that delivers it (`NativeExecutionSession._dispatch_timestamps` and
/// `_dispatch_positions`), the legs, and each instant's bars in the order
/// `NativeScheduleCursor._select` delivers them.
///
/// A frame of one leg -- the shape most of the library has -- has one bar an
/// instant, at the instant's own place.
pub struct Dispatch {
    pub ts: Vec<i64>,
    pub pos: Vec<usize>,
    pub legs: Vec<Leg>,
    /// The lowest close of any leg, NaN where any is.
    pub lowest_close: f64,
    /// `(leg, bar)` of every leg's bars, by instant and then in delivery
    /// order: instant `u`'s are `events[starts[u]..starts[u + 1]]`.
    events: Vec<(usize, usize)>,
    starts: Vec<usize>,
    /// How many of each instant's bars are decided ahead of its primary bar,
    /// which they lead.
    ahead: Vec<usize>,
}

impl Dispatch {
    /// A frame of one leg the strategy builds: its dispatches, the primary
    /// bar each reaches, its labels and closes.
    pub fn new(
        ts: Vec<i64>,
        pos: Vec<usize>,
        labels: &[i64],
        close: Vec<f64>,
    ) -> Result<Self, Declined> {
        let leg = LegTerms {
            ts: ts.clone(),
            labels: labels.to_vec(),
            close,
            rank: 0,
            built: true,
            stream_index: 0,
            decides: true,
        };
        Self::of_legs(ts, pos, vec![leg])
    }

    /// A frame of `legs`, dispatched at instants `ts` that reach primary
    /// bars `pos`: every instant one at least one leg dispatches at, and
    /// every leg's bar at one of them.
    ///
    /// An instant's bars sort as `ScheduledStream.order` sorts them: by
    /// phase, then by `stream_index` among pre-close windows and by
    /// `dispatch_rank` among the rest -- an order with no ties among one
    /// symbol's legs, which the leg's place breaks if it had -- and one
    /// leg's bars in their order. Only a leg the rules never decide on
    /// delivers more than one bar at an instant: another symbol's bars that
    /// closed while the traded symbol's market was shut, all handed over
    /// with its next bar. Each moves its leg's keys and evaluates nothing,
    /// so the last of them is what the rules read.
    pub fn of_legs(ts: Vec<i64>, pos: Vec<usize>, legs: Vec<LegTerms>) -> Result<Self, Declined> {
        let n = ts.len();
        if pos.len() != n {
            return Err(Declined("stream lengths differ"));
        }
        if !legs.iter().any(|leg| leg.decides) {
            return Err(Declined("a frame without a leg the rules decide on"));
        }
        if !ts.windows(2).all(|pair| pair[0] < pair[1])
            || !pos.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err(Declined("dispatches that do not rise"));
        }
        let mut counts = vec![0usize; n];
        // (instant, phase, rank, leg, bar)
        let mut sorted: Vec<(usize, i64, i64, usize, usize)> = Vec::new();
        let mut built = Vec::with_capacity(legs.len());
        for (j, terms) in legs.into_iter().enumerate() {
            let m = terms.ts.len();
            if terms.labels.len() != m || terms.close.len() != m {
                return Err(Declined("stream lengths differ"));
            }
            let rises = |pair: &[i64]| pair[0] < pair[1] || (!terms.decides && pair[0] == pair[1]);
            if !terms.ts.windows(2).all(rises) {
                return Err(Declined("dispatches that do not rise"));
            }
            let mut at = Vec::with_capacity(m);
            let mut phase = Vec::with_capacity(m);
            let mut u = 0;
            for (r, (&dispatched, &label)) in terms.ts.iter().zip(&terms.labels).enumerate() {
                while u < n && ts[u] < dispatched {
                    u += 1;
                }
                if u == n || ts[u] != dispatched {
                    return Err(Declined("a leg's bar at no instant of its frame"));
                }
                let (p, rank) = if label > dispatched {
                    (PHASE_PRECLOSE, terms.stream_index)
                } else if label < dispatched && terms.built {
                    (PHASE_CLOSED_BY_GAP, terms.rank)
                } else {
                    (PHASE_COMPLETED, terms.rank)
                };
                at.push(u);
                phase.push(p);
                counts[u] += 1;
                sorted.push((u, p, rank, j, r));
            }
            built.push(Leg {
                labels_rise: terms.labels.windows(2).all(|pair| pair[0] < pair[1]),
                close: terms.close,
                phase,
                at,
                decides: terms.decides,
            });
        }
        if counts.contains(&0) {
            return Err(Declined("an instant no leg dispatches at"));
        }
        sorted.sort_unstable();
        let mut starts = Vec::with_capacity(n + 1);
        let mut total = 0;
        starts.push(0);
        for &count in &counts {
            total += count;
            starts.push(total);
        }
        let mut ahead = vec![0usize; n];
        for &(u, p, ..) in &sorted {
            if p < PHASE_PRIMARY {
                ahead[u] += 1;
            }
        }
        let lowest_close = built.iter().fold(f64::INFINITY, |low, leg| {
            let leg_low = lowest(&leg.close);
            if low.is_nan() || leg_low.is_nan() {
                f64::NAN
            } else {
                low.min(leg_low)
            }
        });
        let events = sorted.into_iter().map(|(_, _, _, j, r)| (j, r)).collect();
        Ok(Self { ts, pos, legs: built, lowest_close, events, starts, ahead })
    }

    /// Instant `u`'s bars as `(leg, bar)` in delivery order, and how many of
    /// them lead its primary bar.
    #[inline]
    pub fn events(&self, u: usize) -> (&[(usize, usize)], usize) {
        (&self.events[self.starts[u]..self.starts[u + 1]], self.ahead[u])
    }

    /// The instants.
    #[inline]
    pub fn len(&self) -> usize {
        self.ts.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ts.is_empty()
    }
}

/// A slice's place in memory, standing for its contents within one call:
/// the caller's arrays neither move nor change while it runs.
type Place = (usize, usize);

fn place<T>(slice: &[T]) -> Place {
    (slice.as_ptr() as usize, slice.len())
}

/// A crossover operand as [`Signals`] keys it: a key's values by place, a
/// constant by its bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Side {
    Key(Place),
    Const(u64),
}

type Slot<T> = Arc<OnceLock<T>>;

/// Slots by key, each made for the first cell that asks for it.
type Slots<K, T> = Mutex<HashMap<K, Slot<T>>>;

/// A crossover under a session filter: its operands and direction.
type Crossover = (Option<Place>, Side, Side, bool);

/// What the cells of one call on a [`Dispatch`] share, worked out for the
/// first cell that reads it: each leg's admitted bars under each session
/// filter, each key's first ready bar under a filter, and each crossover's
/// crossings under a filter. A cell reads a crossing only after its own
/// first ready bar ([`Stream::entry_bit`]), so one series serves every cell,
/// whatever its other keys.
pub struct Signals<'a> {
    pub d: &'a Dispatch,
    admitted: Slots<(usize, Option<Place>), Arc<[bool]>>,
    ready: Slots<(usize, Option<Place>, Place), Result<usize, Declined>>,
    crossings: Slots<Crossover, Arc<[bool]>>,
}

impl<'a> Signals<'a> {
    pub fn new(d: &'a Dispatch) -> Self {
        Self { d, admitted: Mutex::default(), ready: Mutex::default(), crossings: Mutex::default() }
    }

    /// `key`'s slot in `map`, made where no cell has asked for it yet.
    fn slot<K: Eq + Hash, T>(map: &Slots<K, T>, key: K) -> Slot<T> {
        let mut map = map.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        Arc::clone(map.entry(key).or_default())
    }

    /// Per resample bar, whether `cross` on `values` crosses from the bar
    /// `admitted` admits before; on every admitted bar but the first.
    fn crossings(
        &self,
        filter: Option<Place>,
        admitted: &[bool],
        values: &[&[f64]],
        cross: &Cross,
    ) -> Arc<[bool]> {
        let side = |operand: Operand| match operand {
            Operand::Key(key) => Side::Key(place(values[key])),
            Operand::Const(value) => Side::Const(value.to_bits()),
        };
        let key = (filter, side(cross.fast), side(cross.slow), cross.above);
        let slot = Self::slot(&self.crossings, key);
        let crossings = slot.get_or_init(|| {
            let operand = |operand: Operand, r: usize| match operand {
                Operand::Key(key) => values[key][r],
                Operand::Const(value) => value,
            };
            let mut out = vec![false; admitted.len()];
            let mut previous = None;
            for r in (0..admitted.len()).filter(|&r| admitted[r]) {
                let (a, b) = (operand(cross.fast, r), operand(cross.slow, r));
                if let Some(previous) = previous {
                    out[r] = crossed(previous, a, b, cross.above);
                }
                previous = Some((a, b));
            }
            out.into()
        });
        Arc::clone(crossings)
    }
}

/// The first bar `admitted` admits on which `flags` is set, or the bar
/// count; only an admitted bar's flags are read.
fn first_ready(admitted: &[bool], flags: &[bool]) -> Result<usize, Declined> {
    let n = flags.len();
    let first = (0..n).find(|&r| admitted[r] && flags[r]).unwrap_or(n);
    if (first..n).any(|r| admitted[r] && !flags[r]) {
        return Err(Declined("a key that goes unready again"));
    }
    Ok(first)
}

/// StrategyBase's risk exits on a cell (`stops::policies`), the ATR's values
/// in its stream's bars, the style a close they decide is ordered in, and,
/// where the cell shorts with a borrow, the borrow's terms the checks charge.
///
/// A policy's slot is the full lane's: 0 the primary stream, `k` the
/// frame's leg `k - 1`.
#[derive(Clone, Copy, Debug)]
pub struct Policies<'a> {
    pub risk: Risk,
    pub atr: Option<AtrValues<'a>>,
    pub style: OrderStyle,
    pub borrow: Option<BorrowView<'a>>,
}

/// One cell's decision stream: its keys' values on their legs of a
/// [`Dispatch`], and the rules it trades.
///
/// A crossover a side reads one leg, whose bars are the frame's instants;
/// level rules read any of the frame's legs.
pub struct Stream<'a> {
    pub d: &'a Dispatch,
    /// Each key's values on its leg's bars.
    pub values: Vec<&'a [f64]>,
    /// Each key's leg.
    key_leg: Vec<usize>,
    /// Each key's first ready bar on its leg, or the leg's bar count.
    key_ready: Vec<usize>,
    /// Per leg and bar, whether the session filter admits its visit.
    pub admitted: Vec<Arc<[bool]>>,
    /// Per primary bar, the same; None where every bar is.
    pub primary_admits: Option<&'a [bool]>,
    /// Per leg, the first admitted bar on which every key on it is ready:
    /// a visit from there reaches its keys (`rule_session::reached`). None
    /// on a leg no key is on, whose visits reach nothing.
    leg_ready: Vec<Option<usize>>,
    /// The first leg's: a crossover's, whose keys are all on it.
    pub ready: usize,
    pub entry: Cross,
    pub exit: Cross,
    /// Level rules, in place of `entry` and `exit`.
    pub tree: Option<&'a Tree>,
    /// Per resample bar, whether `entry` crosses from the admitted bar
    /// before, as every cell on its keys reads it: this one's visit enters
    /// only after its first ready bar ([`Stream::entry_bit`]). Empty under
    /// level rules.
    entry_crossings: Arc<[bool]>,
    /// The same for `exit`: past the first ready bar, whether a visit exits
    /// a position the admitted bar before was visited holding.
    exit_crossings: Arc<[bool]>,
    pub dust: bool,
    pub position_required: bool,
    pub has_instrument: bool,
    pub size_increment: f64,
    pub minimum_notional: f64,
    /// The side the cell's entries open: 1 buys, -1 sells short
    /// (`ComposableStrategy` under SHORT_ONLY). Its exits close either.
    pub entry_side: i32,
    /// `StrategySpec.crossings` ``each_bar``: every crossing worked out once
    /// per evaluated bar, ahead of every rule ([`Stream::work_each_bar`]).
    pub each_bar: bool,
    /// Under level rules worked out each bar, each clock's crossing in clock
    /// order; empty otherwise.
    clock_crosses: Vec<Cross>,
    /// The cell's risk exits, if any ([`Stream::with_policies`]).
    pub policies: Option<Policies<'a>>,
}

/// Where a cell's keys are on its frame: each key's leg, and each leg's
/// session filter (None: every bar admitted).
pub struct Placing<'p> {
    pub key_leg: Vec<usize>,
    pub admits: Vec<Option<&'p [bool]>>,
}

impl<'p> Placing<'p> {
    /// Every key on a frame's one leg.
    pub fn one_leg(keys: usize, admits: Option<&'p [bool]>) -> Self {
        Self { key_leg: vec![0; keys], admits: vec![admits] }
    }
}

impl<'a> Stream<'a> {
    /// A cell on `signals`' dispatch, with what it shares with the other
    /// cells there taken from `signals`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        signals: &Signals<'a>,
        values: Vec<&'a [f64]>,
        init: &[&[bool]],
        placing: Placing<'_>,
        primary_admits: Option<&'a [bool]>,
        entry: Cross,
        exit: Cross,
        tree: Option<&'a Tree>,
        dust: bool,
        position_required: bool,
        has_instrument: bool,
        size_increment: f64,
        minimum_notional: f64,
    ) -> Result<Self, Declined> {
        let d = signals.d;
        let Placing { key_leg, admits } = placing;
        if admits.len() != d.legs.len() {
            return Err(Declined("admits for another count of legs"));
        }
        if tree.is_none() && d.legs.len() != 1 {
            return Err(Declined("a crossover a side over more than one leg"));
        }
        if tree.is_some() && d.legs.iter().any(|leg| !leg.labels_rise) {
            return Err(Declined("level rules on labels that do not rise"));
        }
        if values.is_empty() || init.len() != values.len() || key_leg.len() != values.len() {
            return Err(Declined("no keys, or keys without their flags or legs"));
        }
        if key_leg.iter().any(|&leg| leg >= d.legs.len()) {
            return Err(Declined("a key on no leg of its frame"));
        }
        let mut admitted = Vec::with_capacity(d.legs.len());
        for (leg, (bars, admits)) in d.legs.iter().zip(&admits).enumerate() {
            let n = bars.len();
            if admits.is_some_and(|admits| admits.len() != n) {
                return Err(Declined("admits of another length"));
            }
            let slot = Signals::slot(&signals.admitted, (leg, admits.map(place)));
            admitted.push(Arc::clone(
                slot.get_or_init(|| admits.map_or_else(|| vec![true; n].into(), Arc::from)),
            ));
        }
        let mut key_ready = Vec::with_capacity(values.len());
        let mut leg_ready: Vec<Option<usize>> = vec![None; d.legs.len()];
        for (key, flags) in init.iter().enumerate() {
            let leg = key_leg[key];
            let n = d.legs[leg].len();
            if flags.len() != n || values[key].len() != n {
                return Err(Declined("key lengths differ"));
            }
            let filter = admits[leg].map(place);
            let slot = Signals::slot(&signals.ready, (leg, filter, place(flags)));
            let ready = (*slot.get_or_init(|| first_ready(&admitted[leg], flags)))?;
            key_ready.push(ready);
            leg_ready[leg] = Some(leg_ready[leg].map_or(ready, |at| at.max(ready)));
        }
        // The first leg's ready bar, for a crossover: every key is on it.
        let ready = leg_ready[0].unwrap_or(d.legs[0].len());
        let operands: Vec<Operand> = match tree {
            Some(tree) => tree.operands().collect(),
            None => vec![entry.fast, entry.slow, exit.fast, exit.slow],
        };
        if operands
            .iter()
            .any(|operand| matches!(operand, Operand::Key(key) if *key >= values.len()))
        {
            return Err(Declined("an operand names no key"));
        }
        let (entry_crossings, exit_crossings) = match tree {
            Some(_) => (Arc::from([]), Arc::from([])),
            None => {
                let filter = admits[0].map(place);
                (
                    signals.crossings(filter, &admitted[0], &values, &entry),
                    signals.crossings(filter, &admitted[0], &values, &exit),
                )
            }
        };
        Ok(Self {
            d,
            values,
            key_leg,
            key_ready,
            admitted,
            primary_admits,
            leg_ready,
            ready,
            entry,
            exit,
            tree,
            entry_crossings,
            exit_crossings,
            dust,
            position_required,
            has_instrument,
            size_increment,
            minimum_notional,
            entry_side: 1,
            each_bar: false,
            clock_crosses: Vec::new(),
            policies: None,
        })
    }

    /// Check the cell's risk exits on a run of `bars` primary bars: each
    /// policy on a stream of the frame, the ATR's values one per bar of its
    /// stream, and the borrow's terms one per primary bar.
    pub fn with_policies(&mut self, policies: Policies<'a>, bars: usize) -> Result<(), Declined> {
        let legs = self.d.legs.len();
        let risk = &policies.risk;
        if risk.trade.is_none() && risk.atr.is_none() && risk.hold.is_none() {
            return Err(Declined("risk policies that name no check"));
        }
        if risk.hold.is_some_and(|hold| hold.slot == 0 || hold.slot > legs) {
            return Err(Declined("a time exit on no resample of the frame"));
        }
        match (risk.atr, policies.atr) {
            (None, None) => {}
            (Some(atr), Some((values, initialized))) => {
                if atr.slot > legs {
                    return Err(Declined("an ATR exit on no stream of the frame"));
                }
                let n = if atr.slot == 0 { bars } else { self.d.legs[atr.slot - 1].len() };
                if values.len() != n || initialized.len() != n {
                    return Err(Declined("an ATR's values not one per bar of its stream"));
                }
            }
            _ => return Err(Declined("an ATR exit without its values, or values without one")),
        }
        if let Some(borrow) = &policies.borrow {
            if borrow.validate(bars).is_err() {
                return Err(Declined("borrow terms not one per primary bar"));
            }
        }
        self.policies = Some(policies);
        Ok(())
    }

    /// Work every crossing out once per evaluated bar, before any rule reads
    /// one (`StrategySpec.crossings` ``each_bar``).
    ///
    /// A crossover a side then reads its exit as it reads its entry: each
    /// evaluated bar steps both clocks, flat or held, so the exit crosses
    /// where its operands cross from the admitted bar before -- the shared
    /// bit -- and a primary visit, which holds the operands its resample
    /// bar left, crosses nothing. Level rules step every clock first, in
    /// clock order, and their conditions read what each one did.
    pub fn work_each_bar(&mut self) -> Result<(), Declined> {
        if let Some(tree) = self.tree {
            self.clock_crosses = tree.clock_crosses()?;
        }
        self.each_bar = true;
        Ok(())
    }

    /// A crossover's operand on bar `r` of its one leg.
    #[inline]
    fn operand(&self, operand: Operand, r: usize) -> f64 {
        match operand {
            Operand::Key(key) => self.values[key][r],
            Operand::Const(value) => value,
        }
    }

    /// A level rule's operand with the legs' last visits at `current`
    /// (`rule_session`'s `operand`): a key's value there once it is ready,
    /// None before -- a key is ready on every admitted bar from its first
    /// ready one.
    #[inline]
    fn level_operand(&self, operand: Operand, current: &[Option<usize>]) -> Option<f64> {
        match operand {
            Operand::Key(key) => {
                let r = current[self.key_leg[key]]?;
                (r >= self.key_ready[key]).then(|| self.values[key][r])
            }
            Operand::Const(value) => Some(value),
        }
    }

    /// Whether a visit to `leg` at bar `r` reaches its keys.
    #[inline]
    fn leg_reached(&self, leg: usize, r: usize) -> bool {
        self.leg_ready[leg].is_some_and(|ready| r >= ready)
    }

    /// Whether a primary visit with the legs' last visits at `current`
    /// reaches its keys: every key is ready.
    #[inline]
    fn primary_reached(&self, current: &[Option<usize>]) -> bool {
        self.leg_ready
            .iter()
            .zip(current)
            .all(|(ready, r)| ready.is_none_or(|ready| r.is_some_and(|r| r >= ready)))
    }

    #[inline]
    pub fn operands(&self, cross: &Cross, r: usize) -> (f64, f64) {
        (self.operand(cross.fast, r), self.operand(cross.slow, r))
    }

    /// The first resample bar from `from` whose bit in `bits` is set, or the
    /// bar count.
    #[inline]
    pub fn next_set(bits: &[bool], from: usize) -> usize {
        bits.get(from..)
            .and_then(|rest| rest.iter().position(|&bit| bit))
            .map_or(bits.len(), |at| from + at)
    }

    /// Whether resample bar `r`'s visit enters: `entry` crosses there from
    /// the admitted bar before, on an admitted bar after the first ready one,
    /// whose clock starts there.
    #[inline]
    pub fn entry_bit(&self, r: usize) -> bool {
        r > self.ready && self.entry_crossings[r]
    }

    /// Whether resample bar `r`'s visit exits, its crossings worked out each
    /// bar: `exit` crosses there from the admitted bar before, on an admitted
    /// bar after the first ready one.
    #[inline]
    pub fn exit_bit(&self, r: usize) -> bool {
        r > self.ready && self.exit_crossings[r]
    }

    /// The first resample bar from `from` whose visit enters, or the bar
    /// count.
    #[inline]
    pub fn next_entry(&self, from: usize) -> usize {
        Self::next_set(&self.entry_crossings, from.max(self.ready + 1))
    }

    /// The first resample bar from `from` whose visit exits a position the
    /// admitted bar before was visited holding, or the bar count.
    #[inline]
    pub fn next_exit(&self, from: usize) -> usize {
        Self::next_set(&self.exit_crossings, from.max(self.ready + 1))
    }

    /// The last admitted resample bar in `from..to`.
    #[inline]
    pub fn last_admitted(&self, from: usize, to: usize) -> Option<usize> {
        (from..to).rev().find(|&r| self.admitted[0][r])
    }

    /// The one leg a crossover a side reads.
    #[inline]
    fn leg(&self) -> &Leg {
        &self.d.legs[0]
    }
}

/// What `advance` decided: a target side, and the price it was decided at.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub side: i32,
    pub reference: Option<f64>,
    /// The risk policy that decided the target, where one did
    /// (`policy_acted`).
    pub policy: Option<PolicyExit>,
    /// The rules' own close of the same callback, to send ahead of the
    /// policy's at this reference (`preceding_exit`).
    pub preceding: Option<f64>,
    /// How the target's order is placed (`NativeIntent.order_style`).
    pub style: OrderStyle,
}

impl Target {
    /// The rules' own target: no risk exit, in the venue's default order.
    fn model(side: i32, reference: Option<f64>) -> Self {
        Self { side, reference, policy: None, preceding: None, style: OrderStyle::Default }
    }
}

/// Level rules' crossover clocks.
#[derive(Clone, Debug)]
struct Clocks {
    /// Each clock's last operands.
    last: Vec<Option<(f64, f64)>>,
    /// Under `each_bar`, whether each clock crossed on the visit evaluating.
    fired: Vec<bool>,
    /// Whether a clock changed sides in the current visit.
    flipped: bool,
}

impl Clocks {
    /// Hold `other`'s operands and crossings, on clocks of its count.
    fn copy_from(&mut self, other: &Self) {
        self.last.copy_from_slice(&other.last);
        self.fired.copy_from_slice(&other.fired);
    }

    /// Whether these clocks hold `other`'s operands, to the bit, and its
    /// crossings.
    fn same(&self, other: &Self) -> bool {
        let bits = |last: Option<(f64, f64)>| last.map(|(a, b)| (a.to_bits(), b.to_bits()));
        self.fired == other.fired
            && self.last.iter().zip(&other.last).all(|(&a, &b)| bits(a) == bits(b))
    }
}

/// The rule cursor's state.
#[derive(Clone, Debug)]
pub struct Cursor {
    /// The next instant to dispatch.
    pub next_dispatch: usize,
    pub visit_primary_after_fill: bool,
    exit_clock: Option<(f64, f64)>,
    last_position: i32,
    /// Per leg, the last bar visited, whose values its keys hold.
    current: Vec<Option<usize>>,
    last_primary_label: Option<i64>,
    clocks: Clocks,
    // Level rules': whether the last visit reached its keys, and the last
    // evaluation's place in the dispatch order.
    last_reached: bool,
    last_sort_key: Option<(i64, i64)>,
    /// The pending-close latch's last bar, or -1 ([`Latch`]). A lean cell's
    /// exits are latched closes of the one side it opens, so it always keeps
    /// one.
    close_bar: i64,
    /// The risk policies' own state; untouched where there are none.
    policy: PolicyState,
    /// The clocks as a primary visit found them ([`Cursor::note_clocks`]).
    noted: Clocks,
}

impl Cursor {
    /// A fresh cursor for `s`'s rules.
    pub fn of(s: &Stream) -> Self {
        let clocks = s.tree.map_or(0, |tree| tree.clocks);
        let fresh = Clocks { last: vec![None; clocks], fired: vec![false; clocks], flipped: false };
        Self {
            next_dispatch: 0,
            visit_primary_after_fill: false,
            exit_clock: None,
            last_position: 0,
            current: vec![None; s.d.legs.len()],
            last_primary_label: None,
            clocks: fresh.clone(),
            last_reached: false,
            last_sort_key: None,
            close_bar: -1,
            policy: PolicyState::NEW,
            noted: fresh,
        }
    }

    /// `_DeclarativeNode.decision_position`.
    #[inline]
    fn decision_position(
        s: &Stream,
        side: i32,
        quantity: f64,
        pending: Option<f64>,
        close: f64,
    ) -> i32 {
        if side != 0 && s.has_instrument && s.position_required && pending.is_none_or(|p| p != 0.0)
        {
            let quantity = quantity.abs();
            if quantity < s.size_increment
                || (s.minimum_notional > 0.0 && quantity * close < s.minimum_notional)
            {
                return 0;
            }
        }
        side
    }

    /// `NativeExecutionSession._reopens_exits`.
    #[allow(clippy::too_many_arguments)]
    pub fn reopens_exits(
        &self,
        s: &Stream,
        close: &[f64],
        side: i32,
        quantity: f64,
        pending: Option<f64>,
        start: usize,
        stop: usize,
    ) -> bool {
        if !s.dust || side == 0 || stop <= start {
            return false;
        }
        if let Some(tree) = s.tree {
            return self.level_side_moves(
                s,
                &tree.level,
                close,
                side,
                quantity,
                pending,
                start,
                stop,
            );
        }
        if self.last_position != 0 {
            return false;
        }
        let price = if stop == start + 1 { close[start] } else { highest(&close[start..stop]) };
        Self::decision_position(s, side, quantity, pending, price) != 0
    }

    /// `NativeExecutionSession._level_side_moves`: whether a quiet primary
    /// bar in `[start, stop)` sees another side than the last evaluation, in
    /// the rules' reading or the executor's.
    #[allow(clippy::too_many_arguments)]
    fn level_side_moves(
        &self,
        s: &Stream,
        level: &Level,
        close: &[f64],
        side: i32,
        quantity: f64,
        pending: Option<f64>,
        start: usize,
        stop: usize,
    ) -> bool {
        let closes = &close[start..stop];
        let mut probes = [closes[0], f64::NAN, f64::NAN];
        let count = if stop == start + 1 {
            1
        } else {
            // numpy's fmin and fmax: NaN only when every close is NaN.
            probes[0] = closes.iter().fold(f64::NAN, |low, &close| low.min(close));
            probes[1] = closes.iter().fold(f64::NAN, |high, &close| high.max(close));
            if closes.iter().any(|close| close.is_nan()) {
                3
            } else {
                2
            }
        };
        let seen = self.last_position;
        for &close in &probes[..count] {
            if Self::decision_position(s, side, quantity, pending, close) != seen {
                return true;
            }
            if level.reads_dust_as_flat(quantity, pending, close) {
                if seen != 0 {
                    return true;
                }
            } else if seen != side {
                return true;
            }
        }
        false
    }

    /// From a flat, idle account that owes no visit: the first resample bar
    /// whose visit can act -- the next that enters. The visits before it
    /// see a flat book, step no exit clock and decide nothing.
    pub fn quiet_flat(&self, s: &Stream) -> usize {
        s.next_entry(self.next_dispatch)
    }

    /// Holding a lot no close reads as dust, with the exit clock on the bar
    /// last visited (under `each_bar`, always): the first resample bar whose
    /// visit can act -- the next that exits. None where the cursor is not in
    /// that steady state.
    pub fn quiet_held(&self, s: &Stream) -> Option<usize> {
        let current = self.current[0]?;
        if self.visit_primary_after_fill
            || self.last_position != s.entry_side
            || current >= self.next_dispatch
            || s.last_admitted(current + 1, self.next_dispatch).is_some()
            || current < s.ready
        {
            return None;
        }
        if s.each_bar {
            // Every visit steps the exit clock: each exits on its own bit.
            return Some(s.next_exit(self.next_dispatch));
        }
        let (a, b) = s.operands(&s.exit, current);
        let (p0, p1) = self.exit_clock?;
        if p0.to_bits() != a.to_bits() || p1.to_bits() != b.to_bits() {
            return None;
        }
        Some(s.next_exit(self.next_dispatch))
    }

    /// Take the quiet visits of resample bars `next_dispatch..to`, flat or
    /// held, as `visit` would: the cursor moves, the exit clock follows a
    /// held book, and the position the rules last saw is the book's.
    pub fn pass(&mut self, s: &Stream, to: usize, held: bool) {
        if to <= self.next_dispatch {
            return;
        }
        let from = std::mem::replace(&mut self.next_dispatch, to);
        if self.close_bar >= 0 {
            // Each dispatch's events step the latch as `advance` would: flat,
            // it clears; held, it carries to the dispatch's bar.
            let mut latch = Latch::new(self.close_bar);
            let side = if held { s.entry_side } else { 0 };
            for r in from..to {
                let bar = s.d.pos[r] as i64;
                if s.leg().ahead(r) {
                    latch.step(bar, true, side);
                }
                latch.step(bar, false, side);
                if !latch.holds() {
                    break;
                }
            }
            self.close_bar = latch.close_bar;
        }
        let Some(last) = s.last_admitted(from, to) else {
            return;
        };
        self.current[0] = Some(last);
        if last >= s.ready {
            if held {
                self.exit_clock = Some(s.operands(&s.exit, last));
                self.last_position = s.entry_side;
            } else {
                self.last_position = 0;
            }
        }
    }

    /// Level rules: keep the clocks as the primary visit about to run finds
    /// them, for [`Cursor::clocks_repeat`] to compare once it has run.
    pub fn note_clocks(&mut self) {
        self.noted.copy_from(&self.clocks);
    }

    /// Whether the visit since [`Cursor::note_clocks`] left the clocks as it
    /// found them.
    pub fn clocks_repeat(&self) -> bool {
        self.clocks.same(&self.noted)
    }

    /// `RustRuleSession::inert_run` on this lane: of the due bars `due`,
    /// every one before the next dispatch, how many a cell under risk
    /// policies takes as quiet from the venue's `account`, the rules reading
    /// side `side` on a lot no close from `low_close` up reads as dust. The
    /// policies are left as each bar's checks leave them, and the rules as
    /// each bar's visit leaves them.
    ///
    /// Between two dispatches a bar's only event is its own. Crossover rules
    /// skip it (no visit owed, no dust lot whose exits it could reopen).
    /// Level rules visit it, on the legs' bars the visit before read; where
    /// that visit -- of an admitted primary bar, `repeats` -- left the clocks
    /// as it found them and moved nothing, each later one finds what it
    /// found and repeats it. No latch reads them. What is left are the
    /// checks. Flat, they only clear what they keep, so once that is clear
    /// every bar is taken. Held, they run on each bar, and the run stops
    /// before the first they act on -- or fail on -- for that bar's own
    /// advance.
    #[allow(clippy::too_many_arguments)]
    pub fn inert_run(
        &mut self,
        s: &Stream,
        ts: &[i64],
        close: &[f64],
        low_close: f64,
        due: &[usize],
        account: Account,
        side: i32,
        repeats: bool,
    ) -> usize {
        let Some(policies) = &s.policies else {
            return 0;
        };
        if due.is_empty()
            || self.visit_primary_after_fill
            || self.close_bar >= 0
            || !self.policy.holds_nothing()
        {
            return 0;
        }
        let mut reached = false;
        if let Some(tree) = s.tree {
            if !repeats {
                return 0;
            }
            // Neither side the rules read moves with the close.
            if side != 0
                && (low_close.is_nan()
                    || tree.level.reads_dust_as_flat(account.quantity, None, low_close)
                    || Self::decision_position(s, side, account.quantity, None, low_close) != side)
            {
                return 0;
            }
            reached = s.primary_reached(&self.current);
            // A visit that reaches nothing keeps the side the rules last saw,
            // which must be the lot's (`held_reopens`).
            if side != 0 && !reached && self.last_position != side {
                return 0;
            }
        } else if s.dust && side != 0 && self.last_position == 0 {
            return 0;
        }
        let risk = &policies.risk;
        let taken = if side == 0 {
            let settled = account.quantity == 0.0
                && risk.flat_settled(
                    policies.atr.is_some(),
                    policies.borrow.is_some(),
                    &self.policy,
                    account,
                    false,
                );
            if settled {
                due.len()
            } else {
                0
            }
        } else {
            let mut p = self.policy;
            let mut taken = 0;
            for &at in due {
                let mut trial = p;
                if let Some(borrow) = &policies.borrow {
                    if borrow.observe(&mut trial, at, account.quantity).is_err() {
                        break;
                    }
                }
                if s.primary_admits.is_none_or(|admits| admits[at])
                    && risk.intent(policies.atr, &mut trial, account, at, close[at]).is_some()
                {
                    break;
                }
                trial.resampled = false;
                p = trial;
                taken += 1;
            }
            self.policy = p;
            taken
        };
        if taken > 0 && s.tree.is_some() {
            self.repeat_visits(s, ts, &due[..taken], reached, side);
        }
        taken
    }

    /// The level rules as the primary visits of `run`'s bars leave them,
    /// each a repeat of the one before: `level_visit`'s marks of the last
    /// admitted bar, and what the last bar reached.
    fn repeat_visits(&mut self, s: &Stream, ts: &[i64], run: &[usize], reached: bool, side: i32) {
        let admitted = |at: usize| s.primary_admits.is_none_or(|admits| admits[at]);
        if let Some(&last) = run.iter().rev().find(|&&at| admitted(at)) {
            self.last_primary_label = Some(ts[last]);
            self.last_sort_key = Some((ts[last], PHASE_PRIMARY));
            if reached {
                self.last_position = side;
            }
        }
        self.last_reached = reached && run.last().is_some_and(|&at| admitted(at));
        self.clocks.flipped = false;
    }

    pub fn next_dispatch_position(&self, s: &Stream, n_bars: usize) -> usize {
        if self.next_dispatch < s.d.len() {
            s.d.pos[self.next_dispatch]
        } else {
            n_bars
        }
    }

    pub fn held_reopens(
        &self,
        s: &Stream,
        close: &[f64],
        idx: usize,
        has_position: bool,
        net_position: f64,
        pending: Option<f64>,
    ) -> bool {
        if !has_position {
            return false;
        }
        let side = side_of(net_position);
        let stop = self.next_dispatch_position(s, close.len());
        self.reopens_exits(s, close, side, net_position, pending, idx + 1, stop)
    }

    /// `NativeExecutionSession.note_close`: a close the replay resent on
    /// primary bar `idx`, under a pending close that stands through it.
    pub fn note_close(&mut self, idx: usize) {
        self.close_bar = self.close_bar.max(idx as i64);
    }

    /// `visit` + `evaluate` of one event: the primary bar `bar`, or the
    /// resample bar `bar`. `(wants_entry, wants_exit, reference)`, or None
    /// where nothing was evaluated. `resting` is an entry an earlier event of
    /// the callback decided from flat, still resting: the rules read it as
    /// held where the account reads flat, or dust.
    #[allow(clippy::too_many_arguments)]
    fn visit(
        &mut self,
        s: &Stream,
        close: &[f64],
        primary: bool,
        bar: usize,
        now: i64,
        side: i32,
        quantity: f64,
        pending: Option<f64>,
        resting: i32,
    ) -> Option<(bool, bool, f64)> {
        let price = if primary { close[bar] } else { s.leg().close[bar] };
        let mut position = side;
        if position != 0 && s.position_required {
            position = Self::decision_position(s, position, quantity, pending, price);
        }
        if position == 0 {
            position = resting;
        }
        let admitted = if primary {
            s.primary_admits.is_none_or(|admits| admits[bar])
        } else {
            s.admitted[0][bar]
        };
        if !admitted {
            return None;
        }
        if let (false, Some(policies)) = (primary, &s.policies) {
            policies.risk.stream(&mut self.policy, 1, bar, s.leg().ahead(bar));
        }
        let (r, entry) = if primary {
            if self.last_primary_label == Some(now) {
                return None;
            }
            self.last_primary_label = Some(now);
            // Every key is ready once the resample bar they hold is.
            let r = self.current[0]?;
            if r < s.ready {
                return None;
            }
            // The entry clock already holds these operands.
            (r, false)
        } else {
            self.current[0] = Some(bar);
            if bar < s.ready {
                return None;
            }
            (bar, s.entry_bit(bar))
        };
        let mut exit = false;
        if s.each_bar {
            exit = position != 0 && !primary && s.exit_bit(r);
        } else if position != 0 {
            let (a, b) = s.operands(&s.exit, r);
            if let Some(previous) = self.exit_clock {
                exit = crossed(previous, a, b, s.exit.above);
            }
            self.exit_clock = Some((a, b));
        }
        self.last_position = position;
        let exiting = exit && position != 0;
        Some((entry && !exiting, exiting, price))
    }

    /// `NativeExecutionSession.advance` on primary bar `idx` at `now`:
    /// whether it got past the deferral, and the target it decided.
    ///
    /// `has_position` and `net_position` are the lot as the rules read it
    /// (dust as flat); `account` is the venue's, which the risk checks read.
    /// Under risk policies nothing is deferred; `apply_policies` false is a
    /// halted run's callback, which only moves the policies' stream counts.
    #[allow(clippy::too_many_arguments)]
    pub fn advance(
        &mut self,
        s: &Stream,
        close: &[f64],
        now: i64,
        idx: usize,
        has_position: bool,
        net_position: f64,
        pending: Option<f64>,
        account: Account,
        apply_policies: bool,
    ) -> Result<(bool, Option<Target>), Declined> {
        let held_side = if has_position { side_of(net_position) } else { 0 };
        let policies = s.policies.as_ref();
        if let Some(borrow) = policies.and_then(|policies| policies.borrow.as_ref()) {
            // `advance`'s `self._borrow.observe(ctx)`, ahead of the events.
            borrow.observe(&mut self.policy, idx, account.quantity).map_err(Declined)?;
        }
        let due = self.next_dispatch < s.d.len();
        let next_dispatch = if due { s.d.ts[self.next_dispatch] } else { now };
        if policies.is_none()
            && !self.visit_primary_after_fill
            && (!due || now < next_dispatch)
            && !(has_position
                && self.reopens_exits(s, close, held_side, net_position, pending, idx, idx + 1))
        {
            return Ok((false, None));
        }
        if due && next_dispatch < now {
            return Err(Declined("a dispatch passed unvisited"));
        }
        let quantity = if has_position { net_position } else { 0.0 };
        let start_side = held_side;
        let mut target = start_side;
        let mut reference: Option<f64> = None;
        let mut entry_timestamp: Option<i64> = None;
        let res = (due && next_dispatch == now).then(|| {
            self.next_dispatch += 1;
            self.next_dispatch - 1
        });
        if let Some(tree) = s.tree {
            let target = self.advance_level(
                s,
                tree,
                close,
                now,
                idx,
                res,
                start_side,
                quantity,
                pending,
                account,
                apply_policies,
            )?;
            return Ok((true, target));
        }
        // One leg: the instant is its bar.
        let events: [(bool, usize); 2];
        let events: &[(bool, usize)] = match res {
            Some(r) if s.leg().ahead(r) => {
                events = [(false, r), (true, idx)];
                &events
            }
            Some(r) => {
                events = [(true, idx), (false, r)];
                &events
            }
            None => {
                events = [(true, idx), (true, idx)];
                &events[..1]
            }
        };
        let mut latch = Latch::new(self.close_bar);
        // The rules' close this callback, and whether it was decided ahead of
        // both risk checks: the primary bar's decision, or a window decided
        // ahead of it.
        let mut model_exit: Option<(f64, bool)> = None;
        for &(primary, bar) in events {
            // Every event of a lean callback is on its clock, at `idx`.
            let early = !primary && s.leg().ahead(bar);
            latch.step(idx as i64, early, start_side);
            if primary
                && !self.visit_primary_after_fill
                && entry_timestamp != Some(now)
                && (!s.dust
                    || !self.reopens_exits(s, close, start_side, quantity, pending, bar, bar + 1))
            {
                continue;
            }
            if primary {
                self.visit_primary_after_fill = false;
            }
            // An entry an earlier event decided from flat is still resting:
            // the rules run their exit clock over it, and whatever they
            // decide finds nothing to act on. Only the lane that visits every
            // bar has the visit.
            let resting = if policies.is_some()
                && start_side == 0
                && target != 0
                && entry_timestamp == Some(now)
            {
                target
            } else {
                0
            };
            let intent =
                self.visit(s, close, primary, bar, now, start_side, quantity, pending, resting);
            let intent = if resting != 0 { None } else { intent };
            let Some((wants_entry, wants_exit, price)) = intent else { continue };
            let entering = entry_timestamp == Some(now);
            let Some(moved) = delivery_step(
                wants_entry,
                wants_exit,
                s.entry_side,
                target,
                start_side,
                entering,
                &latch,
            ) else {
                continue;
            };
            let previous = std::mem::replace(&mut target, moved);
            if previous == 0 && target != 0 {
                entry_timestamp = Some(now);
            }
            reference = Some(price);
            if start_side != 0 && previous == start_side && target == 0 {
                model_exit = Some((price, primary || early));
            }
            if previous != 0 && target == 0 {
                latch.close(idx as i64, early, idx as i64);
            }
        }
        let risk = policies.and_then(|policies| {
            self.policy_target(
                s,
                policies,
                &mut latch,
                close,
                account,
                idx,
                apply_policies,
                start_side,
                target,
                model_exit,
            )
        });
        self.close_bar = latch.close_bar;
        if risk.is_some() {
            return Ok((true, risk));
        }
        if target == start_side {
            return Ok((true, None));
        }
        Ok((true, Some(Target::model(target, reference))))
    }

    /// `advance`'s risk checks, once the callback's events are decided: the
    /// close a risk exit takes over the rules' `target`, if one acts.
    /// `model_exit` is the rules' own close this callback, if they decided
    /// one, and whether ahead of both checks. `apply_policies` false (a
    /// halted run) only tells the policies of the closes they were held
    /// from.
    #[allow(clippy::too_many_arguments)]
    fn policy_target(
        &mut self,
        s: &Stream,
        policies: &Policies,
        latch: &mut Latch,
        close: &[f64],
        account: Account,
        idx: usize,
        apply_policies: bool,
        start_side: i32,
        target: i32,
        model_exit: Option<(f64, bool)>,
    ) -> Option<Target> {
        let exit = if apply_policies {
            let primary = s.primary_admits.is_none_or(|admits| admits[idx]);
            policies.risk.apply(policies.atr, &mut self.policy, account, idx, close[idx], primary)
        } else {
            self.policy.release();
            None
        };
        // A close of a lot the rules read as dust: the bar's own decision
        // stands. A check acts only on a lot, so the close it takes moves the
        // target.
        let exit = exit.filter(|_| side_of(account.quantity) == start_side)?;
        let preceding = model_exit
            .filter(|&(_, before_check)| target == 0 && (before_check || self.policy.resampled))
            .map(|(model_reference, _)| model_reference);
        if start_side != 0 {
            // `handle_bar`'s risk check closes after the reconcile.
            latch.close_bar = latch.close_bar.max(idx as i64);
        }
        Some(Target {
            side: 0,
            reference: Some(close[idx]),
            policy: Some(exit),
            preceding,
            style: policies.style,
        })
    }

    /// The policies' `on_fill`: a fill of `quantity` on `side` (1 buys)
    /// charged `commission`, which left the account `flat` -- and, where the
    /// cell borrows, the venue holding signed `position` at primary bar
    /// `bar`. A cell without policies books nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn policy_fill(
        &mut self,
        s: &Stream,
        side: f64,
        quantity: f64,
        commission: f64,
        flat: bool,
        bar: usize,
        position: f64,
    ) -> Result<(), Declined> {
        let Some(policies) = &s.policies else {
            return Ok(());
        };
        if let Some(borrow) = &policies.borrow {
            // A fill on this lane is stamped by its bar's clock, so the loan
            // it opens settles as one opened on the bar.
            borrow.observe(&mut self.policy, bar, position).map_err(Declined)?;
        }
        policies.risk.on_fill(&mut self.policy, side, quantity, commission, flat);
        Ok(())
    }

    /// `rule_session::advance_level`: the events of one `advance` past its
    /// deferral, for level rules -- instant `res`'s bars where one is due,
    /// and primary bar `idx`, in dispatch order.
    ///
    /// Every primary event is visited -- a level rule's repeat can act.
    /// Deferred, the next primary bar is owed a visit after any event that
    /// flipped a clock, after a bar that acted, and after one whose rules saw
    /// another side than the one the bar moves from; only a primary bar the
    /// rules ran on settles that debt. Under risk policies nothing is
    /// deferred, and the checks run after the events as `advance`'s do.
    #[allow(clippy::too_many_arguments)]
    fn advance_level(
        &mut self,
        s: &Stream,
        tree: &Tree,
        close: &[f64],
        now: i64,
        idx: usize,
        res: Option<usize>,
        held_side: i32,
        quantity: f64,
        pending: Option<f64>,
        account: Account,
        apply_policies: bool,
    ) -> Result<Option<Target>, Declined> {
        let policies = s.policies.as_ref();
        let deferred = policies.is_none();
        // The side a decision moves from: the venue's, or flat for dust.
        let start_side =
            if held_side != 0 && tree.level.reads_dust_as_flat(quantity, pending, close[idx]) {
                0
            } else {
                held_side
            };
        let mut target = start_side;
        let mut reference: Option<f64> = None;
        let mut entry_timestamp: Option<i64> = None;
        // The instant's bars decided ahead of the primary bar, the primary
        // bar (None), and the rest.
        let (bars, ahead) = res.map_or((&[][..], 0), |u| s.d.events(u));
        let events = bars[..ahead]
            .iter()
            .map(|&event| Some(event))
            .chain(std::iter::once(None))
            .chain(bars[ahead..].iter().map(|&event| Some(event)));
        let mut latch = Latch::new(self.close_bar);
        let mut model_exit: Option<(f64, bool)> = None;
        for event in events {
            let primary = event.is_none();
            let early = event.is_some_and(|(leg, bar)| s.d.legs[leg].ahead(bar));
            latch.step(idx as i64, early, start_side);
            self.clocks.flipped = false;
            // An entry decided from flat earlier at this instant, still
            // resting: see `advance`.
            let resting =
                if !deferred && start_side == 0 && target != 0 && entry_timestamp == Some(now) {
                    target
                } else {
                    0
                };
            let intent = self.level_visit(
                s, tree, close, event, idx, now, held_side, quantity, pending, resting,
            )?;
            let intent = if resting != 0 { None } else { intent };
            // Only a primary bar the rules ran on settles a visit owed; a bar
            // of a leg they only read leaves one owed (`owes_primary`).
            if primary {
                if self.last_reached {
                    self.visit_primary_after_fill = false;
                }
            } else if event.is_some_and(|(leg, _)| !s.d.legs[leg].decides) {
                self.visit_primary_after_fill = true;
            }
            if deferred && self.clocks.flipped {
                self.visit_primary_after_fill = true;
            }
            let Some((wants_entry, wants_exit, price)) = intent else { continue };
            let entering = entry_timestamp == Some(now);
            let Some(moved) = delivery_step(
                wants_entry,
                wants_exit,
                s.entry_side,
                target,
                start_side,
                entering,
                &latch,
            ) else {
                continue;
            };
            let previous = std::mem::replace(&mut target, moved);
            if previous == 0 && target != 0 {
                entry_timestamp = Some(now);
            }
            reference = Some(price);
            if start_side != 0 && previous == start_side && target == 0 {
                model_exit = Some((price, primary || early));
            }
            if previous != 0 && target == 0 {
                latch.close(idx as i64, early, idx as i64);
            }
        }
        let risk = policies.and_then(|policies| {
            self.policy_target(
                s,
                policies,
                &mut latch,
                close,
                account,
                idx,
                apply_policies,
                start_side,
                target,
                model_exit,
            )
        });
        self.close_bar = latch.close_bar;
        if risk.is_some() {
            return Ok(risk);
        }
        if target == start_side {
            if deferred && self.last_position != start_side {
                self.visit_primary_after_fill = true;
            }
            return Ok(None);
        }
        if deferred {
            self.visit_primary_after_fill = true;
        }
        Ok(Some(Target::model(target, reference)))
    }

    /// `visit` + `DecisionProgram.evaluate` of one event under level rules:
    /// bar `bar` of leg `leg` (`event`), or primary bar `idx` (None).
    /// `(wants_entry, wants_exit, reference)`, or None where nothing was
    /// evaluated. `resting` as [`Cursor::visit`]'s.
    #[allow(clippy::too_many_arguments)]
    fn level_visit(
        &mut self,
        s: &Stream,
        tree: &Tree,
        close: &[f64],
        event: Option<(usize, usize)>,
        idx: usize,
        now: i64,
        side: i32,
        quantity: f64,
        pending: Option<f64>,
        resting: i32,
    ) -> Result<Option<(bool, bool, f64)>, Declined> {
        let (admitted, price) = match event {
            None => (s.primary_admits.is_none_or(|admits| admits[idx]), close[idx]),
            Some((leg, bar)) => (s.admitted[leg][bar], s.d.legs[leg].close[bar]),
        };
        if !admitted {
            self.last_reached = false;
            return Ok(None);
        }
        if let (Some((leg, bar)), Some(policies)) = (event, &s.policies) {
            policies.risk.stream(&mut self.policy, leg + 1, bar, s.d.legs[leg].ahead(bar));
        }
        let mut position = side;
        if position != 0 && s.position_required {
            position = Self::decision_position(s, position, quantity, pending, price);
        }
        if position == 0 {
            position = resting;
        }
        // `evaluate`: a repeated label evaluates nothing and reaches nothing;
        // the keys a visit updates are its leg's (the primary bar has none).
        let (reached, phase) = match event {
            None => {
                if self.last_primary_label == Some(now) {
                    self.last_reached = false;
                    return Ok(None);
                }
                self.last_primary_label = Some(now);
                (s.primary_reached(&self.current), PHASE_PRIMARY)
            }
            Some((leg, bar)) => {
                self.current[leg] = Some(bar);
                (s.leg_reached(leg, bar), s.d.legs[leg].phase[bar])
            }
        };
        let order = (now, phase);
        if self.last_sort_key.is_some_and(|key| order < key) {
            return Err(Declined("events out of dispatch order"));
        }
        self.last_sort_key = Some(order);
        self.last_reached = reached;
        // A leg the rules only read moves its keys and evaluates nothing.
        if !reached || event.is_some_and(|(leg, _)| !s.d.legs[leg].decides) {
            return Ok(None);
        }
        let current = &self.current;
        let clocks = &mut self.clocks;
        // Every crossing, once, before any rule reads one -- flat or not, and
        // whichever rules match first.
        for (clock, cross) in s.clock_crosses.iter().enumerate() {
            clocks.fired[clock] = clocks.step(s, clock, cross, current);
        }
        let exit_rule = if position != 0 {
            tree.exits.iter().position(|&root| clocks.holds(s, tree, current, root))
        } else {
            None
        };
        let entry_rule = tree.entries.iter().position(|&root| clocks.holds(s, tree, current, root));
        self.last_position = position;
        let exiting = exit_rule.is_some() && position != 0;
        Ok(Some((entry_rule.is_some() && !exiting, exiting, price)))
    }
}

impl Clocks {
    /// One compiled condition with the legs' last visits at `current`,
    /// short-circuiting as `all` and `any` do: which clocks a bar steps
    /// depends on it. A key not ready yet holds no value, on which no level
    /// holds.
    fn holds(&mut self, s: &Stream, tree: &Tree, current: &[Option<usize>], node: usize) -> bool {
        match &tree.conditions[node] {
            Condition::All(children) => {
                children.iter().all(|&child| self.holds(s, tree, current, child))
            }
            Condition::Any(children) => {
                children.iter().any(|&child| self.holds(s, tree, current, child))
            }
            Condition::Not(child) => !self.holds(s, tree, current, *child),
            Condition::Zone { value, lower, upper } => {
                s.level_operand(*value, current).is_some_and(|v| *lower <= v && v <= *upper)
            }
            Condition::Threshold { left, right, compare } => {
                match (s.level_operand(*left, current), s.level_operand(*right, current)) {
                    (Some(a), Some(b)) => compare.holds(a, b),
                    _ => false,
                }
            }
            Condition::Cross { clock, cross } => {
                if s.each_bar {
                    self.fired[*clock]
                } else {
                    self.step(s, *clock, cross, current)
                }
            }
        }
    }

    /// Step clock `clock` to `cross`'s operands with the legs' last visits
    /// at `current`: whether it crossed from where it last stepped. An
    /// operand not ready yet neither steps it nor crosses.
    #[inline]
    fn step(&mut self, s: &Stream, clock: usize, cross: &Cross, current: &[Option<usize>]) -> bool {
        let (Some(a), Some(b)) =
            (s.level_operand(cross.fast, current), s.level_operand(cross.slow, current))
        else {
            return false;
        };
        match self.last[clock].replace((a, b)) {
            None => false,
            Some((p0, p1)) => {
                if (p0 > p1) != (a > b) || (p0 < p1) != (a < b) {
                    self.flipped = true;
                }
                crossed((p0, p1), a, b, cross.above)
            }
        }
    }
}

#[inline]
pub fn side_of(quantity: f64) -> i32 {
    if quantity > 0.0 {
        1
    } else if quantity < 0.0 {
        -1
    } else {
        0
    }
}

/// numpy's `min`: NaN as soon as any element is NaN.
#[inline]
pub fn lowest(values: &[f64]) -> f64 {
    let mut best = f64::INFINITY;
    for &value in values {
        if value.is_nan() {
            return f64::NAN;
        }
        if value < best {
            best = value;
        }
    }
    best
}

/// [`highest`] of `values[start..stop]` for a `stop` that holds while
/// `start` rises: the window's suffix highs, computed on its first read.
#[derive(Default)]
pub struct Highs {
    from: usize,
    stop: usize,
    suffix: Vec<f64>,
}

impl Highs {
    /// `highest(&values[start..stop])`, `start < stop`.
    pub fn get(&mut self, values: &[f64], start: usize, stop: usize) -> f64 {
        if stop != self.stop || start < self.from || self.suffix.is_empty() {
            self.from = start;
            self.stop = stop;
            self.suffix.clear();
            self.suffix.resize(stop - start, f64::NAN);
            // Right to left, `>=` keeps the first of equal values, the one
            // `highest` keeps.
            let mut best = f64::NEG_INFINITY;
            for at in (start..stop).rev() {
                let value = values[at];
                if value.is_nan() || best.is_nan() {
                    best = f64::NAN;
                } else if value >= best {
                    best = value;
                }
                self.suffix[at - start] = best;
            }
        }
        self.suffix[start - self.from]
    }
}

/// Per bar `j`, [`highest`] of `values[j..q]`, `q` the first of the rising
/// `positions` after `j` or the bar count: the suffix highs of every window
/// between dispatches at once, as [`Highs`] computes one.
pub fn segment_highs(values: &[f64], positions: &[usize]) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    // `positions[next]` is the first position after the bar.
    let mut next = positions.len();
    let mut best = f64::NEG_INFINITY;
    for at in (0..values.len()).rev() {
        if next > 0 && positions[next - 1] > at {
            next -= 1;
            best = f64::NEG_INFINITY;
        }
        let value = values[at];
        if value.is_nan() || best.is_nan() {
            best = f64::NAN;
        } else if value >= best {
            best = value;
        }
        out[at] = best;
    }
    out
}

/// numpy's `max`: NaN as soon as any element is NaN.
#[inline]
pub fn highest(values: &[f64]) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for &value in values {
        if value.is_nan() {
            return f64::NAN;
        }
        if value > best {
            best = value;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_highs_are_each_windows_highest() {
        // Ties, signed zeros and NaNs, between dispatches dense and sparse.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut draw = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let pick = [1.0, 2.0, 2.0, 0.0, -0.0, -1.0, f64::NAN, 3.5];
        for case in 0..400 {
            let n = 1 + (draw() % 90) as usize;
            let nan_free = case % 2 == 0;
            let values: Vec<f64> =
                (0..n).map(|_| pick[(draw() % if nan_free { 6 } else { 8 }) as usize]).collect();
            let every = 1 + draw() % 9;
            let positions: Vec<usize> = (0..n).filter(|_| draw() % every == 0).collect();
            let table = segment_highs(&values, &positions);
            let mut highs = Highs::default();
            for start in 0..n {
                let stop = positions.iter().copied().find(|&at| at > start).unwrap_or(n);
                let want = highest(&values[start..stop]);
                assert_eq!(table[start].to_bits(), want.to_bits(), "case {case} start {start}");
                assert_eq!(highs.get(&values, start, stop).to_bits(), want.to_bits());
            }
        }
    }

    #[test]
    fn trees_decline_arenas_that_point_forward() {
        let level = Level { two_sided: false, size_increment: 1.0, minimum_notional: 0.0 };
        let cross = Cross { fast: Operand::Key(0), slow: Operand::Const(1.0), above: true };
        let zone = Condition::Zone { value: Operand::Key(0), lower: 0.0, upper: 1.0 };
        let ok =
            vec![zone.clone(), Condition::Cross { clock: 0, cross }, Condition::All(vec![0, 1])];
        assert!(Tree::new(ok.clone(), vec![0], vec![2], 1, level).is_ok());
        // A child after its parent, a clock past the count, a root past the arena.
        assert!(
            Tree::new(vec![Condition::Not(1), zone.clone()], vec![0], vec![], 0, level).is_err()
        );
        assert!(Tree::new(vec![Condition::Not(0)], vec![0], vec![], 0, level).is_err());
        assert!(Tree::new(ok.clone(), vec![0], vec![2], 0, level).is_err());
        assert!(Tree::new(ok, vec![3], vec![2], 1, level).is_err());
    }

    #[test]
    fn dust_reads_flat_only_on_one_sided_books() {
        let level = Level { two_sided: false, size_increment: 0.01, minimum_notional: 5.0 };
        assert!(level.reads_dust_as_flat(0.005, None, 100.0)); // under the increment
        assert!(level.reads_dust_as_flat(0.04, None, 100.0)); // under the notional
        assert!(level.reads_dust_as_flat(-0.04, Some(1.0), 100.0));
        assert!(!level.reads_dust_as_flat(0.05, None, 100.0)); // exactly the notional
        assert!(!level.reads_dust_as_flat(0.0, None, 100.0));
        assert!(!level.reads_dust_as_flat(0.005, Some(0.0), 100.0)); // flattening already
        assert!(!Level { two_sided: true, ..level }.reads_dust_as_flat(0.005, None, 100.0));
        let no_notional = Level { minimum_notional: 0.0, ..level };
        assert!(!no_notional.reads_dust_as_flat(0.04, None, 100.0));
    }

    fn leg(ts: &[i64], labels: &[i64], rank: i64, built: bool, stream_index: i64) -> LegTerms {
        LegTerms {
            ts: ts.to_vec(),
            labels: labels.to_vec(),
            close: ts.iter().map(|&t| t as f64).collect(),
            rank,
            built,
            stream_index,
            decides: true,
        }
    }

    #[test]
    fn of_legs_delivers_each_instant_in_dispatch_order() {
        // `ScheduledStream.order`: pre-close windows by stream index, then
        // windows a gap closed (on legs the strategy builds) and completed
        // bars by rank, the primary bar between the two.
        let hourly = leg(&[10, 20, 30, 40, 50], &[10, 20, 30, 40, 50], 1, true, 0);
        let daily = leg(&[20, 30, 40, 50], &[25, 28, 40, 45], 0, true, 1);
        let engine = leg(&[50], &[45], 2, false, 2);
        let d = Dispatch::of_legs(
            vec![10, 20, 30, 40, 50],
            vec![0, 2, 4, 6, 8],
            vec![hourly, daily, engine],
        )
        .unwrap();
        assert_eq!(d.len(), 5);
        assert_eq!(d.events(0), (&[(0, 0)][..], 0));
        // A pre-close window leads; a gap's window leads.
        assert_eq!(d.events(1), (&[(1, 0), (0, 1)][..], 1));
        assert_eq!(d.events(2), (&[(1, 1), (0, 2)][..], 1));
        // Both completed: the lower rank first.
        assert_eq!(d.events(3), (&[(1, 2), (0, 3)][..], 0));
        // The engine's leg follows the primary bar however it is labelled.
        assert_eq!(d.events(4), (&[(1, 3), (0, 4), (2, 0)][..], 1));
        assert_eq!(d.legs[1].at, vec![1, 2, 3, 4]);
        assert!(d.legs[1].ahead(0) && d.legs[1].ahead(1) && !d.legs[1].ahead(2));
        assert!(!d.legs[2].ahead(0));
        assert_eq!(d.lowest_close, 10.0);
    }

    #[test]
    fn one_leg_frames_are_their_bars() {
        let labels = [10, 25, 28, 40];
        let d = Dispatch::new(
            vec![10, 20, 30, 40],
            vec![1, 3, 5, 7],
            &labels,
            vec![4.0, 3.0, 2.0, 5.0],
        )
        .unwrap();
        // Labelled after its timestamp, a pre-close window; before it, a
        // window a gap closed.
        let phases = [PHASE_COMPLETED, PHASE_PRECLOSE, PHASE_CLOSED_BY_GAP, PHASE_COMPLETED];
        assert_eq!(d.legs[0].phase, phases);
        for u in 0..d.len() {
            let ahead = usize::from(d.legs[0].ahead(u));
            assert_eq!(d.events(u), (&[(0, u)][..], ahead));
        }
        assert_eq!(d.lowest_close, 2.0);
    }

    #[test]
    fn of_legs_declines_frames_off_their_instants() {
        let at = |ts: &[i64]| leg(ts, ts, 0, true, 0);
        let frame = |legs| Dispatch::of_legs(vec![10, 20, 30], vec![0, 1, 2], legs);
        assert!(frame(vec![at(&[10, 20, 30])]).is_ok());
        assert!(frame(vec![at(&[10, 30]), at(&[20])]).is_ok());
        // A bar at no instant, an instant no bar is at, bars that fall.
        assert!(frame(vec![at(&[10, 20, 30]), at(&[25])]).is_err());
        assert!(frame(vec![at(&[10, 30])]).is_err());
        assert!(frame(vec![at(&[10, 30, 20])]).is_err());
        assert!(frame(vec![]).is_err());
        // A leg the rules only read, beside one they decide on; never alone.
        let read = |ts: &[i64]| LegTerms { decides: false, ..at(ts) };
        let d = frame(vec![at(&[10, 30]), read(&[20])]).unwrap();
        assert!(d.legs[0].decides && !d.legs[1].decides);
        assert!(frame(vec![read(&[10, 20, 30])]).is_err());
        // Its bars may share an instant -- another market's, closed while
        // this one was shut -- delivered in their order; a decided leg's not.
        let shut = LegTerms { decides: false, ..leg(&[20, 20], &[15, 18], 0, true, 1) };
        let d = frame(vec![at(&[10, 20, 30]), shut]).unwrap();
        assert_eq!(d.events(1), (&[(1, 0), (1, 1), (0, 1)][..], 2));
        assert!(frame(vec![at(&[10, 20, 20, 30])]).is_err());
        let nan = LegTerms { close: vec![1.0, f64::NAN], ..at(&[10, 20]) };
        assert!(frame(vec![nan, at(&[30])]).unwrap().lowest_close.is_nan());
    }

    #[test]
    fn each_bar_steps_each_clock_once_in_clock_order() {
        // `rule_session::clock_crossings`: a clock checked twice is stepped
        // once; one naming two crossings, or none, declines.
        let level = Level { two_sided: false, size_increment: 1.0, minimum_notional: 0.0 };
        let up = Cross { fast: Operand::Key(0), slow: Operand::Const(1.0), above: true };
        let down = Cross { above: false, ..up };
        let other = Cross { fast: Operand::Key(1), ..up };
        let arena = |crosses: &[(usize, Cross)]| -> Vec<Condition> {
            crosses.iter().map(|&(clock, cross)| Condition::Cross { clock, cross }).collect()
        };
        let tree = Tree::new(arena(&[(1, other), (0, up), (0, up)]), vec![0], vec![1, 2], 2, level)
            .unwrap();
        assert_eq!(tree.clock_crosses().unwrap(), vec![up, other]);
        let two = Tree::new(arena(&[(0, up), (0, down)]), vec![0], vec![1], 1, level).unwrap();
        assert!(two.clock_crosses().is_err());
        let none = Tree::new(arena(&[(0, up)]), vec![0], vec![0], 2, level).unwrap();
        assert!(none.clock_crosses().is_err());
    }
}
