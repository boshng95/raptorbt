//! A library cell's rules, for the shape most of the library has: a legacy
//! spec with one direct crossover a side, every key on one resampled stream
//! that dispatches on the primary bar its composite closes on.
//!
//! The rule cursor is `python::rule_session`'s `advance`, `visit` and
//! `evaluate` for that shape, branch for branch. What makes it lean is the
//! entry side: its crossing only ever compares one resample bar's operands
//! with the one before (every resample bar is visited, in order, and a
//! primary visit repeats the operands it already holds), so whether a bar
//! enters is a bit computed once for the whole series. The exit side's clock
//! is only stepped while a position is held, so it stays a clock.
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

use super::Declined;

#[derive(Clone, Copy, Debug)]
pub enum Operand {
    Key(usize),
    Const(f64),
}

/// One compiled `CrossoverCondition`.
#[derive(Clone, Copy, Debug)]
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
    fn holds(self, a: f64, b: f64) -> bool {
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
fn crossed(previous: (f64, f64), a: f64, b: f64, above: bool) -> bool {
    let (p0, p1) = previous;
    if above {
        p0 <= p1 && a > b
    } else {
        p0 >= p1 && a < b
    }
}

/// A resampled stream's dispatches on its market: the resample bars, the
/// primary bar each dispatches on and their closes, shared by every cell
/// whose keys are on that stream.
pub struct Dispatch {
    pub ts: Vec<i64>,
    pub pos: Vec<usize>,
    pub close: Vec<f64>,
    /// A resample bar labelled after its timestamp sorts before the primary
    /// bar at that timestamp, otherwise after it.
    pub first: Vec<bool>,
    /// The lowest resample close, NaN where any is.
    pub lowest_close: f64,
    /// Whether the labels rise, so no resample visit repeats the label
    /// before it (`DecisionProgram.evaluate`'s repeat).
    pub labels_rise: bool,
}

impl Dispatch {
    pub fn new(
        ts: Vec<i64>,
        pos: Vec<usize>,
        labels: &[i64],
        close: Vec<f64>,
    ) -> Result<Self, Declined> {
        let n = ts.len();
        if pos.len() != n || labels.len() != n || close.len() != n {
            return Err(Declined("stream lengths differ"));
        }
        if !ts.windows(2).all(|pair| pair[0] < pair[1])
            || !pos.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err(Declined("dispatches that do not rise"));
        }
        let first = labels.iter().zip(&ts).map(|(&l, &t)| l > t).collect();
        let lowest_close = lowest(&close);
        let labels_rise = labels.windows(2).all(|pair| pair[0] < pair[1]);
        Ok(Self { ts, pos, close, first, lowest_close, labels_rise })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ts.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ts.is_empty()
    }
}

/// One cell's decision stream: its keys' values on a [`Dispatch`]'s
/// resample bars, and the crossings it trades.
pub struct Stream<'a> {
    pub d: &'a Dispatch,
    pub values: Vec<&'a [f64]>,
    /// Per resample bar, whether the session filter admits its visit.
    pub admitted: Vec<bool>,
    /// Per primary bar, the same; None where every bar is.
    pub primary_admits: Option<&'a [bool]>,
    /// The first admitted resample bar on which every key is ready.
    pub ready: usize,
    pub entry: Cross,
    pub exit: Cross,
    /// Level rules, in place of `entry` and `exit`.
    pub tree: Option<&'a Tree>,
    /// Per resample bar, whether its visit enters; empty under level rules.
    pub entry_bits: Vec<bool>,
    /// Per resample bar, whether its visit exits a position the admitted bar
    /// before was visited holding; empty under level rules.
    pub exit_bits: Vec<bool>,
    pub dust: bool,
    pub position_required: bool,
    pub has_instrument: bool,
    pub size_increment: f64,
    pub minimum_notional: f64,
}

impl<'a> Stream<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d: &'a Dispatch,
        values: Vec<&'a [f64]>,
        init: &[&[bool]],
        admits: Option<&[bool]>,
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
        let n = d.len();
        if tree.is_some() && !d.labels_rise {
            return Err(Declined("level rules on labels that do not rise"));
        }
        if values.is_empty() || init.len() != values.len() {
            return Err(Declined("no keys, or keys without their flags"));
        }
        let admitted = admits.map_or_else(|| vec![true; n], <[bool]>::to_vec);
        if admitted.len() != n {
            return Err(Declined("admits of another length"));
        }
        let mut ready = 0;
        for (key, flags) in init.iter().enumerate() {
            if flags.len() != n || values[key].len() != n {
                return Err(Declined("key lengths differ"));
            }
            // Only an admitted bar's flags are read.
            let first = (0..n).find(|&r| admitted[r] && flags[r]).unwrap_or(n);
            if (first..n).any(|r| admitted[r] && !flags[r]) {
                return Err(Declined("a key that goes unready again"));
            }
            ready = ready.max(first);
        }
        // The first admitted bar from there.
        let ready = Stream::next_set(&admitted, ready);
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
        let mut stream = Self {
            d,
            values,
            admitted,
            primary_admits,
            ready,
            entry,
            exit,
            tree,
            entry_bits: Vec::new(),
            exit_bits: Vec::new(),
            dust,
            position_required,
            has_instrument,
            size_increment,
            minimum_notional,
        };
        if tree.is_none() {
            stream.entry_bits = stream.bits(&entry);
            stream.exit_bits = stream.bits(&exit);
        }
        Ok(stream)
    }

    #[inline]
    fn operand(&self, operand: Operand, r: usize) -> f64 {
        match operand {
            Operand::Key(key) => self.values[key][r],
            Operand::Const(value) => value,
        }
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

    /// Per resample bar, whether `cross` crosses from the admitted bar
    /// before; only on admitted bars after the first ready one, whose clock
    /// starts there.
    fn bits(&self, cross: &Cross) -> Vec<bool> {
        let n = self.d.len();
        let mut out = vec![false; n];
        let mut previous = None;
        for r in self.ready..n {
            if !self.admitted[r] {
                continue;
            }
            let (a, b) = self.operands(cross, r);
            if let Some(previous) = previous {
                out[r] = crossed(previous, a, b, cross.above);
            }
            previous = Some((a, b));
        }
        out
    }

    /// The last admitted resample bar in `from..to`.
    #[inline]
    pub fn last_admitted(&self, from: usize, to: usize) -> Option<usize> {
        (from..to).rev().find(|&r| self.admitted[r])
    }
}

/// What `advance` decided: a target side, the price it was decided at, and
/// whether it was decided by a primary bar's visit after a fill.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub side: i32,
    pub reference: Option<f64>,
    pub policies_after_fill: bool,
}

/// The rule cursor's state.
#[derive(Clone, Debug)]
pub struct Cursor {
    pub next_dispatch: usize,
    pub visit_primary_after_fill: bool,
    exit_clock: Option<(f64, f64)>,
    last_position: i32,
    /// The last resample bar visited, whose values the keys hold.
    current: Option<usize>,
    last_primary_label: Option<i64>,
    // Level rules': each crossover clock's last operands, whether one
    // changed sides in the current visit, whether the last visit reached
    // its keys, and the last evaluation's place in the dispatch order.
    clocks: Vec<Option<(f64, f64)>>,
    flipped: bool,
    last_reached: bool,
    last_sort_key: Option<(i64, u8)>,
}

impl Default for Cursor {
    fn default() -> Self {
        Self::new()
    }
}

impl Cursor {
    pub fn new() -> Self {
        Self {
            next_dispatch: 0,
            visit_primary_after_fill: false,
            exit_clock: None,
            last_position: 0,
            current: None,
            last_primary_label: None,
            clocks: Vec::new(),
            flipped: false,
            last_reached: false,
            last_sort_key: None,
        }
    }

    /// A fresh cursor for `s`'s rules.
    pub fn of(s: &Stream) -> Self {
        Self { clocks: vec![None; s.tree.map_or(0, |tree| tree.clocks)], ..Self::new() }
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
        Stream::next_set(&s.entry_bits, self.next_dispatch)
    }

    /// Holding a lot no close reads as dust, with the exit clock on the bar
    /// last visited: the first resample bar whose visit can act -- the next
    /// that exits. None where the cursor is not in that steady state.
    pub fn quiet_held(&self, s: &Stream) -> Option<usize> {
        let current = self.current?;
        if self.visit_primary_after_fill
            || self.last_position != 1
            || current >= self.next_dispatch
            || s.last_admitted(current + 1, self.next_dispatch).is_some()
            || current < s.ready
        {
            return None;
        }
        let (a, b) = s.operands(&s.exit, current);
        let (p0, p1) = self.exit_clock?;
        if p0.to_bits() != a.to_bits() || p1.to_bits() != b.to_bits() {
            return None;
        }
        Some(Stream::next_set(&s.exit_bits, self.next_dispatch))
    }

    /// Take the quiet visits of resample bars `next_dispatch..to`, flat or
    /// held, as `visit` would: the cursor moves, the exit clock follows a
    /// held book, and the position the rules last saw is the book's.
    pub fn pass(&mut self, s: &Stream, to: usize, held: bool) {
        if to <= self.next_dispatch {
            return;
        }
        let from = std::mem::replace(&mut self.next_dispatch, to);
        let Some(last) = s.last_admitted(from, to) else {
            return;
        };
        self.current = Some(last);
        if last >= s.ready {
            if held {
                self.exit_clock = Some(s.operands(&s.exit, last));
                self.last_position = 1;
            } else {
                self.last_position = 0;
            }
        }
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

    /// `visit` + `evaluate` of one event: the primary bar `bar`, or the
    /// resample bar `bar`. `(wants_entry, wants_exit, reference)`, or None
    /// where nothing was evaluated.
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
    ) -> Option<(bool, bool, f64)> {
        let price = if primary { close[bar] } else { s.d.close[bar] };
        let mut position = side;
        if position != 0 && s.position_required {
            position = Self::decision_position(s, position, quantity, pending, price);
        }
        let admitted = if primary {
            s.primary_admits.is_none_or(|admits| admits[bar])
        } else {
            s.admitted[bar]
        };
        if !admitted {
            return None;
        }
        let (r, entry) = if primary {
            if self.last_primary_label == Some(now) {
                return None;
            }
            self.last_primary_label = Some(now);
            // Every key is ready once the resample bar they hold is.
            let r = self.current?;
            if r < s.ready {
                return None;
            }
            // The entry clock already holds these operands.
            (r, false)
        } else {
            self.current = Some(bar);
            if bar < s.ready {
                return None;
            }
            (bar, s.entry_bits[bar])
        };
        let mut exit = false;
        if position != 0 {
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
    ) -> Result<(bool, Option<Target>), Declined> {
        let held_side = if has_position { side_of(net_position) } else { 0 };
        let due = self.next_dispatch < s.d.len();
        let next_dispatch = if due { s.d.ts[self.next_dispatch] } else { now };
        if !self.visit_primary_after_fill
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
        let mut policies_after_fill = false;
        let res = (due && next_dispatch == now).then(|| {
            self.next_dispatch += 1;
            self.next_dispatch - 1
        });
        if let Some(tree) = s.tree {
            let target =
                self.advance_level(s, tree, close, now, idx, res, start_side, quantity, pending)?;
            return Ok((true, target));
        }
        let events: [(bool, usize); 2];
        let events: &[(bool, usize)] = match res {
            Some(r) if s.d.first[r] => {
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
        for &(primary, bar) in events {
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
            if entry_timestamp == Some(now) {
                policies_after_fill = true;
            }
            let intent = self.visit(s, close, primary, bar, now, start_side, quantity, pending);
            let previous = target;
            let Some((wants_entry, wants_exit, price)) = intent else { continue };
            if !(wants_entry || wants_exit) {
                continue;
            }
            if wants_exit && target != 0 {
                target = 0;
            }
            if wants_entry && target == 0 {
                target = 1;
            }
            if previous == 0 && target != 0 {
                entry_timestamp = Some(now);
            }
            reference = Some(price);
        }
        if target == start_side {
            return Ok((true, None));
        }
        Ok((true, Some(Target { side: target, reference, policies_after_fill })))
    }

    /// `rule_session::advance_level`: the events of one `advance` past its
    /// deferral, for level rules -- resample bar `res` where one is due, and
    /// primary bar `idx`, in dispatch order.
    ///
    /// Every primary event is visited -- a level rule's repeat can act --
    /// and the next primary bar is owed a visit after any event that flipped
    /// a clock, after a bar that acted, and after one whose rules saw another
    /// side than the one the bar moves from. Only a primary bar the rules
    /// ran on settles that debt.
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
    ) -> Result<Option<Target>, Declined> {
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
        let mut policies_after_fill = false;
        let events: [(bool, usize); 2];
        let events: &[(bool, usize)] = match res {
            Some(r) if s.d.first[r] => {
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
        for &(primary, bar) in events {
            if entry_timestamp == Some(now) {
                policies_after_fill = true;
            }
            self.flipped = false;
            let intent =
                self.level_visit(s, tree, close, primary, bar, now, held_side, quantity, pending)?;
            if primary && self.last_reached {
                self.visit_primary_after_fill = false;
            }
            if self.flipped {
                self.visit_primary_after_fill = true;
            }
            let previous = target;
            let Some((wants_entry, wants_exit, price)) = intent else { continue };
            if !(wants_entry || wants_exit) {
                continue;
            }
            if wants_exit && target != 0 {
                target = 0;
            }
            if wants_entry && target == 0 {
                target = 1;
            }
            if previous == 0 && target != 0 {
                entry_timestamp = Some(now);
            }
            reference = Some(price);
        }
        if target == start_side {
            if self.last_position != start_side {
                self.visit_primary_after_fill = true;
            }
            return Ok(None);
        }
        self.visit_primary_after_fill = true;
        Ok(Some(Target { side: target, reference, policies_after_fill }))
    }

    /// `visit` + `DecisionProgram.evaluate` of one event under level rules:
    /// the primary bar `bar`, or the resample bar `bar`. `(wants_entry,
    /// wants_exit, reference)`, or None where nothing was evaluated.
    #[allow(clippy::too_many_arguments)]
    fn level_visit(
        &mut self,
        s: &Stream,
        tree: &Tree,
        close: &[f64],
        primary: bool,
        bar: usize,
        now: i64,
        side: i32,
        quantity: f64,
        pending: Option<f64>,
    ) -> Result<Option<(bool, bool, f64)>, Declined> {
        let admitted = if primary {
            s.primary_admits.is_none_or(|admits| admits[bar])
        } else {
            s.admitted[bar]
        };
        if !admitted {
            self.last_reached = false;
            return Ok(None);
        }
        let price = if primary { close[bar] } else { s.d.close[bar] };
        let mut position = side;
        if position != 0 && s.position_required {
            position = Self::decision_position(s, position, quantity, pending, price);
        }
        // `evaluate`: a repeated label evaluates nothing and reaches nothing.
        let (r, phase) = if primary {
            if self.last_primary_label == Some(now) {
                self.last_reached = false;
                return Ok(None);
            }
            self.last_primary_label = Some(now);
            (self.current, 1)
        } else {
            self.current = Some(bar);
            (Some(bar), if s.d.first[bar] { 0 } else { 2 })
        };
        let order = (now, phase);
        if self.last_sort_key.is_some_and(|key| order < key) {
            return Err(Declined("events out of dispatch order"));
        }
        self.last_sort_key = Some(order);
        // Every key is ready once the resample bar they hold is.
        let Some(r) = r.filter(|&r| r >= s.ready) else {
            self.last_reached = false;
            return Ok(None);
        };
        self.last_reached = true;
        let exit_rule = if position != 0 {
            tree.exits.iter().position(|&root| self.holds(s, tree, r, root))
        } else {
            None
        };
        let entry_rule = tree.entries.iter().position(|&root| self.holds(s, tree, r, root));
        self.last_position = position;
        let exiting = exit_rule.is_some() && position != 0;
        Ok(Some((entry_rule.is_some() && !exiting, exiting, price)))
    }

    /// One compiled condition on resample bar `r`'s values, short-circuiting
    /// as `all` and `any` do: which clocks a bar steps depends on it.
    fn holds(&mut self, s: &Stream, tree: &Tree, r: usize, node: usize) -> bool {
        match &tree.conditions[node] {
            Condition::All(children) => children.iter().all(|&child| self.holds(s, tree, r, child)),
            Condition::Any(children) => children.iter().any(|&child| self.holds(s, tree, r, child)),
            Condition::Not(child) => !self.holds(s, tree, r, *child),
            Condition::Zone { value, lower, upper } => {
                let v = s.operand(*value, r);
                *lower <= v && v <= *upper
            }
            Condition::Threshold { left, right, compare } => {
                compare.holds(s.operand(*left, r), s.operand(*right, r))
            }
            Condition::Cross { clock, cross } => {
                let (a, b) = s.operands(cross, r);
                match self.clocks[*clock].replace((a, b)) {
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
}
