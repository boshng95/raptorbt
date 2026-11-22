//! Quiet runs: the entries of a session nothing can happen on, taken in one
//! piece.
//!
//! Most of a book's schedule is bars that do nothing: no order is working,
//! no protective level stands, nothing reads the equity between instants,
//! and every member's kernel only follows the market. Each such entry still
//! costs a merge, a step and a curve check; across a minute book's members
//! that is nearly all of a run. A quiet run takes every entry before a time
//! at once, to exactly the state they leave one at a time: each member's
//! kernel steps its run of bars whole ([`EngineKernel::step_idle_rows`]),
//! the feed passes over them unread ([`EventFeed::skip_to`]), and the curve
//! samples only the instants either side of a day boundary
//! ([`InstantCurve::close_merged_run`]), each valued at every member's
//! close as of it -- the single-instrument runner's run, across members.

use super::{equity_of, EventSession};
use crate::accounts::AccountMode;
use crate::core::types::Timestamp;
use crate::portfolio::kernel::KernelBar;

impl EventSession {
    /// Whether nothing can happen on a bar the schedule holds, now and for
    /// as long as only bars are applied: the schedule is bars alone, no
    /// kernel can act or be called on any bar, lending a kernel the pool
    /// for its step moves no money, and nothing reads the equity an event
    /// leaves or stores each instant's.
    pub(super) fn runs_quiet(&self) -> bool {
        if !self.bars_only
            || !self.pending.is_empty()
            || self.curve.records()
            || self.watches_event_equity()
        {
            return false;
        }
        let margin = matches!(self.account.mode(), AccountMode::Margin { .. });
        self.kernels.iter().enumerate().all(|(i, kernel)| {
            let injected = if margin {
                self.account.balance() - (self.account.locked() - kernel.locked_margin())
            } else {
                self.account.balance()
            };
            kernel.idle_state()
                && kernel.idle_at_every_bar()
                && kernel.holds_cash_exactly(injected)
                // An open option leg re-prices its group after each step.
                && (!margin || kernel.open_option_legs(i).is_empty())
        })
    }

    /// Apply every entry before `until` in one piece, for a session that
    /// [`Self::runs_quiet`], once the instant standing open has had all its
    /// entries; `false`, having applied nothing, otherwise.
    pub(super) fn take_quiet_run(&mut self, until: Timestamp) -> bool {
        let Some(head) = self.head else { return false };
        let Some(open) = self.curve.open_instant() else { return false };
        let first = head.timestamp();
        // An entry of the open instant is applied as it comes, and an entry
        // pushed after the seal (`local_idx` past the series) is no bar of
        // a series read in place.
        if !(open < first && first < until) || head.local_idx >= self.bars[head.instrument].len() {
            return false;
        }

        let Self {
            kernels,
            bars,
            account,
            last_close,
            last_seen,
            curve,
            feed,
            batch_local_idx,
            cursor,
            ..
        } = self;

        // Each member's run: its bars from the first not yet applied up to
        // `until`. The head was delivered and is its member's first.
        let runs: Vec<(usize, usize)> = (0..kernels.len())
            .map(|i| {
                let from = if i == head.instrument { head.local_idx } else { batch_local_idx[i] };
                (from, from + bars[i][from..].partition_point(|bar| bar.timestamp < until))
            })
            .collect();
        let run = |i: usize| -> &[KernelBar] { &bars[i][runs[i].0..runs[i].1] };
        let newest = (0..kernels.len())
            .filter_map(|i| run(i).last().map(|bar| bar.timestamp))
            .max()
            .expect("the head is a bar before `until`");

        // The account at the close of `instant`: every member marked at its
        // newest close as of it, the one before the run where it has none.
        let close_at = |i: usize, instant: i64| {
            let bars = run(i);
            match bars.partition_point(|bar| bar.timestamp <= instant) {
                0 => last_close[i],
                taken => Some(bars[taken - 1].close),
            }
        };
        let equity = |instant: i64| equity_of(account, kernels, |i| close_at(i, instant));
        let first_from = |t: i64| {
            (0..runs.len())
                .filter_map(|i| {
                    let bars = run(i);
                    bars.get(bars.partition_point(|bar| bar.timestamp < t)).map(|bar| bar.timestamp)
                })
                .min()
                .map_or(newest, |found| found.min(newest))
        };
        let last_before = |t: i64| {
            (0..runs.len())
                .filter_map(|i| {
                    let bars = run(i);
                    match bars.partition_point(|bar| bar.timestamp < t) {
                        0 => None,
                        below => Some(bars[below - 1].timestamp),
                    }
                })
                .max()
                .map_or(open, |found| found.max(open))
        };
        curve.close_merged_run(newest, first_from, last_before, equity);

        let mut skips = Vec::with_capacity(runs.len());
        for (i, &(from, to)) in runs.iter().enumerate() {
            if to == from {
                continue;
            }
            kernels[i].step_idle_rows(&bars[i][from..to]);
            // Each step lent the pool and took it back unchanged.
            account.reconcile(0.0, 0.0);
            let last = bars[i][to - 1];
            last_close[i] = Some(last.close);
            last_seen[i] = Some((to - 1, last));
            if to > batch_local_idx[i] {
                skips.push((i, to));
            }
            batch_local_idx[i] = batch_local_idx[i].max(to);
            *cursor += to - from;
        }
        feed.skip_to(&skips);
        self.head = self.next_entry();
        #[cfg(test)]
        {
            self.quiet_runs_taken += 1;
        }
        true
    }
}
