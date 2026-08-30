# Nautilus parity branch

This branch is based on upstream `v0.10.1` and carries two narrowly scoped
compatibility corrections used by algotrade-nautilus backtest experiments.
Neither correction changes default matching behavior.

## Changes

1. `BacktestConfig.same_bar_marketable_limit_on_close` defaults to `false`.
   When explicitly enabled, a plain marketable limit submitted by a
   close-labelled composite callback may fill at the primary close with the
   same timestamp. The matcher deliberately does not inspect that bar's high
   or low, because those prices occurred before the decision.
2. Fractional lot flooring tolerates the few ULPs of error produced when an
   exact decimal-grid value is divided by its lot increment. For example,
   `0.10185 / 0.00001` no longer loses one lot by evaluating just below the
   integer boundary. Values genuinely below the boundary still floor.

## Evidence

- All 509 Rust library tests pass with `RAYON_NUM_THREADS=4`.
- The algotrade-nautilus strict BTC callback case matched 230 of 230 canonical
  data, indicator, decision, order, fill, fee, position, equity, and metric
  events over 2026-01-01 through 2026-02-01, twice per engine.
- The strict callback execution lane was 23.45x faster than Nautilus in that
  case.
- The separate Raptor array lane completed the exact 3,773-run BTC atlas in
  278.628 seconds with zero failures on this `v0.10.1` branch. The array API
  still uses MARKET/IOC
  semantics and is not claimed to have typed-order parity.

Paper and live trading remain Nautilus-only. The branch is intended for a
capability-gated hybrid backtest backend, with Nautilus fallback for unproven
strategy and execution families.
