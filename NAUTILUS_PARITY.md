# Nautilus parity branch

This branch is based on upstream `v0.10.4` and carries five narrowly scoped
compatibility corrections used by algotrade-nautilus backtest experiments.
The corrections are opt-in or default-neutral for existing callers.

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
3. `BacktestConfig` accepts optional `fee_per_share`, `fee_minimum`, and
   `fee_max_pct` fields. Together with the existing percentage `fees` field,
   these express the project's IB US and ASX commission schedules. All three
   default to zero, preserving stock Raptor behavior.
4. `InstrumentConfig.currency_precision` optionally quantizes fees and
   account arithmetic to the settlement-currency grid. `None` preserves the
   upstream floating-point path; the parity adapter supplies the precision
   already present in the Nautilus instrument metadata.
5. `InstrumentConfig.max_quantity` optionally rejects an opening quantity
   above the instrument's declared limit. `None` remains unlimited. This
   reproduces the existing Nautilus constraint which, for example, rejects a
   23,750-unit ADA order against a 9,000-unit maximum.

## Evidence

- All 518 Rust library tests pass with `RAYON_NUM_THREADS=4`.
- The algotrade-nautilus strict BTC callback case matched 230 of 230 canonical
  data, indicator, decision, order, fill, fee, position, equity, and metric
  events over 2026-01-01 through 2026-02-01, twice per engine.
- A nine-case venue certification over every downloaded Binance symbol plus
  representative IB US and ASX instruments passed six strict ledgers and
  identified three volume-limited partial-fill cases for Nautilus fallback.
  AAPL matched 67/67 events including US sessions and pre-close behavior; CBA
  matched 172/172 including ASX sessions, AUD settlement, and minimum fees.
- The separate Raptor array lane completed the exact 3,773-run BTC atlas in
  287.692 seconds with zero failures on this branch at eight workers: 13.11
  runs/s and 600.64x faster than the observed two-day run. The array API still
  uses MARKET/IOC semantics and is not claimed to have typed-order parity.

Paper and live trading remain Nautilus-only. The branch is intended for a
capability-gated hybrid backtest backend, with Nautilus fallback for unproven
strategy and execution families.
