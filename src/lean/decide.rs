//! A traded series' decisions: what `ComposablePortfolio` intends at each
//! instant its members complete, from the bars alone.
//!
//! A pair or basket asks its rules with the position it *decided*, never the
//! book's, so its decisions are a function of its series and nothing a fill
//! does: one walk of them serves every engine and every account. The rules
//! are `DecisionProgram`'s, on one stream whose every bar is a decision event
//! (`ComposablePortfolio._decide`), compiled to the lean lane's [`Tree`]:
//!
//! * a bar is evaluated once every indicator key is ready -- a legacy spec
//!   with no indicator key never is (`DecisionProgram._reached`);
//! * ``each_bar`` steps every crossover clock first, in clock order; otherwise
//!   a clock steps where a rule's condition reaches it, short-circuiting as
//!   `all` and `any` do;
//! * the exits are asked, in rule order up to the first that holds, only
//!   while a position is intended, or always under an explicit schedule; the
//!   entries always are;
//! * an exit that holds while a position is intended flattens it; otherwise
//!   an entry that holds while flat takes its side (`DecisionProgram.advance`).
//!
//! A rank (`SeriesKind.RANK`) decides every member on its own bars, each with
//! its own keys, clocks and intended side, scores it, and holds the ``top``
//! candidates a side (`ComposablePortfolio._decide_rank`).

use std::cmp::Ordering;

use super::rules::{crossed, Condition, Operand, Tree};
use super::Declined;

/// One decider's keys over the bars it decides on: each key's value at every
/// bar and whether it was ready there. The first `gated` keys are the
/// indicators' -- the readiness gate counts those -- and any after them are
/// the bar's own prices, always ready.
pub struct Keys<'a> {
    pub values: Vec<&'a [f64]>,
    pub ready: Vec<&'a [bool]>,
    pub gated: usize,
}

impl Keys<'_> {
    fn len(&self) -> Option<usize> {
        let mut lens =
            self.values.iter().map(|v| v.len()).chain(self.ready.iter().map(|r| r.len()));
        let first = lens.next()?;
        lens.all(|n| n == first).then_some(first)
    }
}

/// How a rule set is asked: its compiled conditions, each entry rule's side
/// (+1 long, -1 short), whether its crossings are worked out on every bar
/// (``each_bar``), and whether it is a legacy spec (no ``decision_streams``),
/// whose exits are asked only while a position is intended.
pub struct Rules<'a> {
    pub tree: &'a Tree,
    pub entry_sides: &'a [i8],
    pub each_bar: bool,
    pub legacy: bool,
}

impl Rules<'_> {
    fn check(&self, keys: &Keys) -> Result<usize, Declined> {
        let tree = self.tree;
        if self.entry_sides.len() != tree.entries.len() {
            return Err(Declined("an entry rule without a side"));
        }
        if self.entry_sides.iter().any(|&side| side != 1 && side != -1) {
            return Err(Declined("an entry side that is neither long nor short"));
        }
        if keys.values.len() != keys.ready.len() || keys.gated > keys.values.len() {
            return Err(Declined("keys without their readiness"));
        }
        let reads = |operand: &Operand| match operand {
            Operand::Key(k) => *k < keys.values.len(),
            Operand::Const(_) => true,
        };
        let fits = tree.conditions.iter().all(|condition| match condition {
            Condition::Zone { value, .. } => reads(value),
            Condition::Threshold { left, right, .. } => reads(left) && reads(right),
            Condition::Cross { cross, .. } => reads(&cross.fast) && reads(&cross.slow),
            _ => true,
        });
        if !fits {
            return Err(Declined("a condition reading a key it was not given"));
        }
        match keys.len() {
            Some(n) => Ok(n),
            None if keys.values.is_empty() => Err(Declined("a series with no keys to read")),
            None => Err(Declined("keys of different lengths")),
        }
    }
}

/// One decider's walk: its clocks and the side it intends.
struct Decider<'r, 'k> {
    rules: &'r Rules<'r>,
    keys: &'k Keys<'k>,
    clocks: Vec<Option<(f64, f64)>>,
    fired: Vec<bool>,
    /// Each clock's crossing, for ``each_bar``.
    crossings: Vec<(Operand, Operand, bool)>,
    intended: i8,
}

impl<'r, 'k> Decider<'r, 'k> {
    fn new(rules: &'r Rules<'r>, keys: &'k Keys<'k>) -> Result<Self, Declined> {
        let tree = rules.tree;
        let mut crossings: Vec<Option<(Operand, Operand, bool)>> = vec![None; tree.clocks];
        for condition in &tree.conditions {
            let Condition::Cross { clock, cross } = condition else { continue };
            let found = (cross.fast, cross.slow, cross.above);
            match crossings[*clock] {
                None => crossings[*clock] = Some(found),
                Some(seen) if seen == found => {}
                Some(_) => return Err(Declined("one clock names two crossings")),
            }
        }
        let crossings = crossings
            .into_iter()
            .map(|c| c.ok_or(Declined("a clock no crossing names")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            rules,
            keys,
            clocks: vec![None; tree.clocks],
            fired: vec![false; tree.clocks],
            crossings,
            intended: 0,
        })
    }

    /// Whether bar `at` opens the gate: every indicator key ready. A legacy
    /// spec with none never opens it; an explicit schedule always does.
    #[inline]
    fn ready(&self, at: usize) -> bool {
        let gated = self.keys.gated;
        if gated == 0 {
            return !self.rules.legacy;
        }
        self.keys.ready[..gated].iter().all(|ready| ready[at])
    }

    #[inline]
    fn operand(&self, operand: Operand, at: usize) -> Option<f64> {
        match operand {
            Operand::Const(value) => Some(value),
            Operand::Key(k) => self.keys.ready[k][at].then(|| self.keys.values[k][at]),
        }
    }

    /// Step clock `clock` to its operands at `at`: whether it crossed. An
    /// operand without a value neither steps it nor crosses.
    #[inline]
    fn step(&mut self, clock: usize, fast: Operand, slow: Operand, above: bool, at: usize) -> bool {
        let (Some(a), Some(b)) = (self.operand(fast, at), self.operand(slow, at)) else {
            return false;
        };
        self.clocks[clock].replace((a, b)).is_some_and(|previous| crossed(previous, a, b, above))
    }

    fn holds(&mut self, node: usize, at: usize) -> bool {
        let tree = self.rules.tree;
        match &tree.conditions[node] {
            Condition::All(children) => children.iter().all(|&child| self.holds(child, at)),
            Condition::Any(children) => children.iter().any(|&child| self.holds(child, at)),
            Condition::Not(child) => !self.holds(*child, at),
            Condition::Zone { value, lower, upper } => {
                self.operand(*value, at).is_some_and(|v| *lower <= v && v <= *upper)
            }
            Condition::Threshold { left, right, compare } => {
                match (self.operand(*left, at), self.operand(*right, at)) {
                    (Some(a), Some(b)) => compare.holds(a, b),
                    _ => false,
                }
            }
            Condition::Cross { clock, cross } => {
                if self.rules.each_bar {
                    self.fired[*clock]
                } else {
                    self.step(*clock, cross.fast, cross.slow, cross.above, at)
                }
            }
        }
    }

    /// Bar `at` decided: the side now intended, if it changed.
    fn decide(&mut self, at: usize) -> Option<i8> {
        if !self.ready(at) {
            return None;
        }
        let rules = self.rules;
        if rules.each_bar {
            for clock in 0..self.crossings.len() {
                let (fast, slow, above) = self.crossings[clock];
                self.fired[clock] = self.step(clock, fast, slow, above, at);
            }
        }
        let tree = rules.tree;
        let exit = if self.intended != 0 || !rules.legacy {
            tree.exits.iter().any(|&root| self.holds(root, at))
        } else {
            false
        };
        let entry = tree.entries.iter().position(|&root| self.holds(root, at));
        if exit && self.intended != 0 {
            self.intended = 0;
            return Some(0);
        }
        match entry {
            Some(rule) if self.intended == 0 => {
                self.intended = rules.entry_sides[rule];
                Some(self.intended)
            }
            _ => None,
        }
    }
}

/// A series' decisions: each bar at which the side it intends changes, and
/// the side it changes to (+1, -1, or 0 for flat), from flat.
pub fn decide_series(rules: &Rules, keys: &Keys) -> Result<Vec<(u32, i8)>, Declined> {
    let bars = rules.check(keys)?;
    let mut decider = Decider::new(rules, keys)?;
    let mut changes = Vec::new();
    for at in 0..bars {
        if let Some(side) = decider.decide(at) {
            changes.push((at as u32, side));
        }
    }
    Ok(changes)
}

/// One member of a rank: its keys over the instants it is decided at -- the
/// ones it delivers a bar at -- and its score's value and readiness there.
pub struct Member<'a> {
    pub keys: Keys<'a>,
    pub score: &'a [f64],
    pub scored: &'a [bool],
}

/// A rank's book at every instant it changes: the instant, and each
/// member's side held (+1, -1, or 0 not held).
///
/// `delivered[m][i]` is member `m`'s bar at instant `i`, if it delivered one
/// (-1: carried, neither decided nor re-scored). `by_name` is the members'
/// order by instrument name, the tie-break of equal scores
/// (`rank_by_score`). Each side permitted holds its ``top`` candidates: the
/// members intending it with a score, best first -- highest for a long,
/// lowest for a short.
pub fn decide_rank(
    rules: &Rules,
    members: &[Member],
    delivered: &[&[i64]],
    by_name: &[usize],
    top: usize,
    permits: (bool, bool),
) -> Result<Vec<(u32, Vec<i8>)>, Declined> {
    let n = members.len();
    if delivered.len() != n || by_name.len() != n {
        return Err(Declined("a rank's members without their deliveries"));
    }
    let mut name_rank = vec![usize::MAX; n];
    for (rank, &member) in by_name.iter().enumerate() {
        if member >= n || name_rank[member] != usize::MAX {
            return Err(Declined("a name order that is not the members'"));
        }
        name_rank[member] = rank;
    }
    let instants = delivered.first().map_or(0, |d| d.len());
    if delivered.iter().any(|d| d.len() != instants) {
        return Err(Declined("deliveries of different lengths"));
    }
    let mut deciders = Vec::with_capacity(n);
    for (member, delivered) in members.iter().zip(delivered) {
        let bars = rules.check(&member.keys)?;
        if member.score.len() != bars || member.scored.len() != bars {
            return Err(Declined("a score off its member's bars"));
        }
        if delivered.iter().any(|&at| at >= bars as i64) {
            return Err(Declined("a delivery past its member's bars"));
        }
        deciders.push(Decider::new(rules, &member.keys)?);
    }
    let mut scores: Vec<Option<f64>> = vec![None; n];
    let mut held = vec![0i8; n];
    let mut changes = Vec::new();
    let mut candidates: Vec<(f64, usize)> = Vec::with_capacity(n);
    // Each instant reads every member's column of `delivered` at once.
    #[allow(clippy::needless_range_loop)]
    for instant in 0..instants {
        for (m, member) in members.iter().enumerate() {
            let at = delivered[m][instant];
            if at < 0 {
                continue;
            }
            let at = at as usize;
            deciders[m].decide(at);
            let value = member.score[at];
            scores[m] = (member.scored[at] && value.is_finite()).then_some(value);
        }
        let mut now = vec![0i8; n];
        for (side, permitted) in [(1i8, permits.0), (-1i8, permits.1)] {
            if !permitted {
                continue;
            }
            candidates.clear();
            for (m, decider) in deciders.iter().enumerate() {
                if decider.intended == side {
                    if let Some(score) = scores[m] {
                        candidates.push((if side > 0 { -score } else { score }, m));
                    }
                }
            }
            candidates.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(Ordering::Equal)
                    .then(name_rank[a.1].cmp(&name_rank[b.1]))
            });
            for &(_, m) in candidates.iter().take(top) {
                now[m] = side;
            }
        }
        if now != held {
            held.clone_from(&now);
            changes.push((instant as u32, now));
        }
    }
    Ok(changes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lean::rules::{Compare, Cross, Level};

    const LEVEL: Level = Level { two_sided: true, size_increment: 1.0, minimum_notional: 0.0 };

    fn keys<'a>(values: Vec<&'a [f64]>, ready: Vec<&'a [bool]>, gated: usize) -> Keys<'a> {
        Keys { values, ready, gated }
    }

    /// Enter long when key 0 crosses above key 1, exit when it crosses below.
    fn cross_tree() -> Tree {
        let up = Cross { fast: Operand::Key(0), slow: Operand::Key(1), above: true };
        let down = Cross { fast: Operand::Key(0), slow: Operand::Key(1), above: false };
        Tree::new(
            vec![
                Condition::Cross { clock: 0, cross: down },
                Condition::Cross { clock: 1, cross: up },
            ],
            vec![0],
            vec![1],
            2,
            LEVEL,
        )
        .unwrap()
    }

    #[test]
    fn a_crossing_pair_enters_and_exits_on_its_own_intended_state() {
        let tree = cross_tree();
        let rules = Rules { tree: &tree, entry_sides: &[1], each_bar: false, legacy: true };
        let fast = [1.0, 3.0, 1.0, 1.0, 3.0, 1.0];
        let slow = [2.0; 6];
        let ready = [true; 6];
        let k = keys(vec![&fast, &slow], vec![&ready, &ready], 2);
        // Legacy: the exit clock is not stepped while flat, so its first step
        // (bar 2) has nothing to cross from, and the long holds to bar 5.
        assert_eq!(decide_series(&rules, &k).unwrap(), vec![(1, 1), (5, 0)]);
        // Each bar: every clock steps on every bar, so the exit sees bar 1 -> 2.
        let rules = Rules { each_bar: true, ..rules };
        assert_eq!(decide_series(&rules, &k).unwrap(), vec![(1, 1), (2, 0), (4, 1), (5, 0)]);
    }

    #[test]
    fn an_unready_bar_steps_nothing_and_a_legacy_spec_without_indicators_never_decides() {
        let tree = cross_tree();
        let rules = Rules { tree: &tree, entry_sides: &[-1], each_bar: false, legacy: true };
        let fast = [1.0, 3.0, 1.0, 3.0];
        let slow = [2.0, 2.0, 2.0, 2.0];
        let ready = [true, false, true, true];
        let all = [true; 4];
        let k = keys(vec![&fast, &slow], vec![&ready, &all], 2);
        // Bar 1 is not evaluated: the entry clock steps 0 -> 2 -> 3.
        assert_eq!(decide_series(&rules, &k).unwrap(), vec![(3, -1)]);
        // The same keys as prices (none gated): a legacy spec never decides,
        // an explicit schedule always does.
        let k = keys(vec![&fast, &slow], vec![&all, &all], 0);
        assert!(decide_series(&rules, &k).unwrap().is_empty());
        let rules = Rules { legacy: false, ..rules };
        assert_eq!(decide_series(&rules, &k).unwrap(), vec![(1, -1), (2, 0), (3, -1)]);
    }

    #[test]
    fn the_first_entry_that_holds_takes_its_side() {
        let above = Condition::Threshold {
            left: Operand::Key(0),
            right: Operand::Const(1.0),
            compare: Compare::Gt,
        };
        let below = Condition::Threshold {
            left: Operand::Key(0),
            right: Operand::Const(-1.0),
            compare: Compare::Lt,
        };
        let inside = Condition::Zone { value: Operand::Key(0), lower: -0.5, upper: 0.5 };
        let tree = Tree::new(vec![above, below, inside], vec![2], vec![0, 1], 0, LEVEL).unwrap();
        let rules = Rules { tree: &tree, entry_sides: &[-1, 1], each_bar: false, legacy: true };
        let z = [2.0, 0.0, -2.0, -3.0, 0.1, f64::NAN];
        let ready = [true; 6];
        let k = keys(vec![&z], vec![&ready], 1);
        assert_eq!(decide_series(&rules, &k).unwrap(), vec![(0, -1), (1, 0), (2, 1), (4, 0)]);
    }

    #[test]
    fn a_rank_holds_its_best_candidates_a_side_ties_broken_by_name() {
        // Every member always intends long once ready (entry: key 0 > 0, no
        // exit that ever holds).
        let tree = Tree::new(
            vec![
                Condition::Threshold {
                    left: Operand::Key(0),
                    right: Operand::Const(0.0),
                    compare: Compare::Gt,
                },
                Condition::Threshold {
                    left: Operand::Key(0),
                    right: Operand::Const(1e9),
                    compare: Compare::Gt,
                },
            ],
            vec![1],
            vec![0],
            0,
            LEVEL,
        )
        .unwrap();
        let rules = Rules { tree: &tree, entry_sides: &[1], each_bar: false, legacy: true };
        let one = [1.0, 1.0, 1.0];
        let ready = [true; 3];
        let (a, b, c) = ([3.0, 1.0, 5.0], [3.0, 4.0, 5.0], [2.0, 9.0, 0.0]);
        let members: Vec<Member> = [&a, &b, &c]
            .iter()
            .map(|score| Member {
                keys: keys(vec![&one], vec![&ready], 1),
                score: &score[..],
                scored: &ready,
            })
            .collect();
        // Member 2 is carried at instant 1: it keeps its score.
        let (d0, d1, d2) = ([0i64, 1, 2], [0i64, 1, 2], [0i64, -1, 1]);
        let delivered: [&[i64]; 3] = [&d0, &d1, &d2];
        // Names order: member 1, then 0, then 2.
        let changes =
            decide_rank(&rules, &members, &delivered, &[1, 0, 2], 1, (true, false)).unwrap();
        // Instant 0: 0 and 1 tie at 3.0 -> by name, 1. Instant 1: 2 carried
        // with 2.0, 1 has 4.0 -> 1 again, no change. Instant 2: 2 scores 9.0.
        assert_eq!(changes, vec![(0, vec![0, 1, 0]), (2, vec![0, 0, 1])]);
    }
}
