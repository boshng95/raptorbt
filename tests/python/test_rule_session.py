"""A crossover rule session lets the kernel step a strategy's quiet masked bars.

The session's decision semantics are pinned where they are defined, against
the Python cursor they mirror (algotrade-nautilus's native runtime, answer for
answer in shadow mode). These cases hold the runner's side: which masked bars
the kernel takes, which it hands back, and that taking them changes nothing.
"""

import math

import numpy as np
import pytest

from raptorbt import CrossoverRuleSession, Strategy, run_strategy_backtest

N = 8
#: A cash-halt bound every flat bar clears: (currency precision, fraction,
#: cap, units, markup, slack, fee, need at highest).
CLEARS = (2, 1.0, math.inf, 0.0, 1.0, 0.0, 0.0, 0.0)
#: One no account clears, so every flat masked bar is the strategy's.
NEVER = (2, 0.0, 0.0, math.inf, math.inf, math.inf, math.inf, math.inf)


def _bars():
    timestamps = np.arange(N, dtype=np.int64) * 60_000_000_000
    close = 100.0 + np.arange(N, dtype=np.float64)
    return (timestamps, close, close, close, close, np.ones(N))


def _rules(**changes):
    timestamps, close = _bars()[:2]
    args = {
        "reg_slots": [0],
        "reg_stream_index": [0],
        "reg_rank": [0],
        "reg_timestamps": [timestamps],
        "reg_labels": [None],
        "slot_close": [close],
        "slot_admits": [None],
        "slot_decision": [True],
        "key_slot": [],
        "key_values": [],
        "key_init": [],
        "exits": [],
        "entries": [],
        "n_clocks": 0,
        "legacy": True,
        "dispatch_timestamps": np.empty(0, np.int64),
        "dispatch_positions": np.empty(0, np.int64),
        "primary_close": close,
        "dust": False,
        "position_required": False,
        "has_instrument": True,
        "size_increment": 1.0,
        "minimum_notional": 0.0,
    }
    args.update(changes)
    return CrossoverRuleSession(**args)


class Flat(Strategy):
    """Never trades; offers the kernel its masked bars through `quiet`."""

    def __init__(self, quiet, wake_at=()):
        super().__init__()
        self.rules = _rules()
        self.quiet = quiet
        self.wake_at = set(wake_at)
        self.calls = []

    def fast_forward_rules(self):
        return None if self.quiet is None else (self.rules, False, None, self.quiet)

    def on_bar(self, ctx):
        self.calls.append(ctx.idx)
        self.bar_callback_required = ctx.idx in self.wake_at


MASK = np.asarray([True, False, False, True, False, False, False, True])


def _run(strategy):
    return run_strategy_backtest(strategy, *_bars(), bar_callback_mask=MASK)


def test_quiet_masked_bars_are_stepped_without_a_callback() -> None:
    taken = Flat(CLEARS)
    result = _run(taken)
    ordinary = run_strategy_backtest(Flat(None), *_bars())
    # Flat, the bound cleared and nothing asking for a bar: every callback,
    # the first included, is the kernel's to step.
    assert taken.calls == []
    assert np.array_equal(result.equity_curve(), ordinary.equity_curve())


def test_a_bound_no_account_clears_hands_every_callback_back() -> None:
    handed_back = Flat(NEVER)
    _run(handed_back)
    assert handed_back.calls == [0, 3, 7]


def test_no_plan_is_the_sparse_mask_unchanged() -> None:
    plain = Flat(None)
    _run(plain)
    assert plain.calls == [0, 3, 7]


def test_a_callback_that_asks_for_the_next_bar_gets_it() -> None:
    woken = Flat(CLEARS, wake_at={0})
    woken.bar_callback_required = True
    _run(woken)
    # Bar 0 is asked for and asks for the next; bar 1 is masked out but
    # still the strategy's, and after it the kernel takes the quiet bars.
    assert woken.calls == [0, 1]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"reg_rank": []}, "inconsistent rule session layout"),
        ({"slot_close": []}, "inconsistent rule session layout"),
        (
            {
                "key_slot": [0],
                "key_values": [np.zeros(N)],
                "key_init": [np.ones(N, np.bool_)],
            },
            "reads no primary-stream indicator",
        ),
        ({"dispatch_positions": np.zeros(1, np.int64)}, "dispatch positions do not match"),
    ],
)
def test_an_inconsistent_layout_is_refused(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        _rules(**changes)
