# Nautilus parity branch

This branch is based on upstream `v0.11.0` and carries fourteen narrowly
scoped compatibility corrections used by algotrade-nautilus backtest
experiments, plus four fixes to pre-existing upstream bugs (see the end of
this section).
The corrections are opt-in or default-neutral for existing callers, except
where a stated one is simply a more accurate reading of the same arithmetic.

## Changes

1. `BacktestConfig.same_bar_marketable_limit_on_close` defaults to `false`.
   When explicitly enabled, a plain marketable limit submitted by a
   close-labelled composite callback may fill at the primary close with the
   same timestamp. The matcher deliberately does not inspect that bar's high
   or low, because those prices occurred before the decision.
2. Fractional lot flooring tolerates the few ULPs of error produced when an
   exact decimal-grid value is divided by its lot increment. For example,
   `0.10185 / 0.00001` no longer loses one lot by evaluating just below the
   integer boundary. Values genuinely below the boundary still floor. The
   surviving lot count is then snapped back onto the lot's decimal scale,
   because `lots.floor() * lot` is exact in decimal but not in binary:
   `10379 * 0.00001` evaluates to `0.10379000000000001`, one ULP above the
   `0.10379` Nautilus holds as a fixed-precision quantity.
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
6. A percentage fee is evaluated as one correctly-rounded `price * size *
   rate` product rather than left to right. Rounding the notional before
   applying the rate can miss the correctly-rounded value of the true product
   by an ULP, which decides the last decimal whenever that product lands on a
   tie at the settlement currency's precision. `0.10379 * 92104.5 * 0.001` is
   exactly `9.559526055`, a tie at USDT's 8 decimals: the naive chain drifts
   high and rounds to `9.55952606`, while Nautilus -- carrying the notional as
   an exact decimal and converting once -- reports `9.55952605`. Both round
   halves away from zero; only the value handed to the rounding differed.

   A tie is not decided by which way the rounding leans, then, but by which
   side of the exact decimal the nearest `f64` falls -- and that cuts both
   ways. `11.79 * 6.4125 * 0.001` is exactly `0.075603375`, also a tie, and
   its nearest `f64` sits just *above*, so the venue reports `0.07560338`.
   Nothing short of the exact product gets both cases right.

   `core::decimals::decimal_product` therefore recovers each factor as an
   integer and a decimal scale, multiplies the integers exactly in `i128`,
   and converts once by a single correctly-rounded division by a power of
   ten. A factor that is not a short decimal, or a product too large to stay
   exact, falls back to a `mul_add` chain that carries the error of the first
   multiplication -- which is all such a product ever was. This change is not
   opt-in: the previous result was simply the less accurate of the two.
7. `BacktestConfig.bar_volume_slices` bounds how much of a bar one aggressive
   order may take. `0.0` (the default) leaves fills unbounded, which is what
   the engine always did; `4.0` reproduces Nautilus's bar-execution model,
   where a bar is replayed as four synthetic ticks each carrying a quarter of
   its volume and an order consumes at most one of them. This is what closed
   the three "volume-limited partial fill" cases listed under Evidence, which
   had previously been routed to Nautilus fallback rather than modelled.

   The bound needed a position model that can be built and unwound in pieces,
   so three things came with it, all reachable only when the bound is on:

   - `OrderStatus::PartiallyFilled`, a *working* state, with `Order.filled_qty`
     and `Order.resolved_qty`. The resolved size is pinned at the first fill,
     so a capital-fraction order cannot silently change size between fills.
     An IOC or FOK order that fills short is canceled with its remainder;
     anything else keeps working. Contingencies (OCO siblings, held OTO
     children) deliberately do not fire until an order completes -- a bracket
     whose entry only half-filled never armed.
   - `PositionLedger::reduce_position`, which takes size off a position and
     accumulates the exit side until it goes flat. One round trip is still
     exactly one `Trade`, with a size-weighted average exit price. The first
     closing fill adopts its price outright rather than averaging against a
     zero-size history, so a position closed by a single fill keeps its exit
     price bit-for-bit.
   - `PositionPolicy::NetAveraging`, selected by the parity adapter. Nautilus's
     NETTING OMS grows the position it holds at a size-weighted average entry;
     Raptor's plain `Net` refuses the fill. That only matters when a strategy
     adds to a position -- and a re-sent partial-fill remainder is exactly
     that case. `FOK` is all-or-nothing under the bound: a bar that cannot
     absorb the whole order absorbs none of it.

   The dead `FillModel::fill_ratio` field, a placeholder for this feature that
   was never read, is replaced by `FillModel::bar_liquidity`.
8. `InstrumentConfig.price_increment` declares the instrument's price grid.
   An order that empties the book at its own price sweeps the level behind
   it, and that level only exists on a grid: without one the remainder fills
   at the price it just exhausted. Nautilus reports the two as separate fill
   events one increment apart, and so does this engine. `None` keeps the
   continuous book the engine had before. A sweep steps off the price that
   actually *traded*, not the price the match asked for: a market order
   arrives priced `NaN` and resolves only inside the fill-price model, so
   reading the request back would price the next level off a price that never
   existed.
9. `EngineEvent::OrderFilled` reports what the fill alone did: `commission`,
   `leaves` (what the order still has outstanding, `0.0` when the fill
   completed it) and `gross_realized` (PnL before this fill's commission,
   `0.0` for a fill that opened or grew a position). One rule then covers
   every fill of every kind -- an account moves by `gross_realized -
   commission` -- so a consumer can rebuild the account curve from the fill
   stream without re-deriving anything from positions afterwards, which no
   amount of care can do once two fills share an order or a timestamp.

   The engine settles the same way internally. Each closing fill credits the
   account for the units *it* sold; the fill that takes a position flat no
   longer settles the round trip, which paid the earlier fills' proceeds a
   second time. A position closed by a single fill is unaffected -- its fill
   is the round trip -- and the golden suite pins that.

10. The bar tape is persistent, because Nautilus's is. A bar is replayed as
    prints that move a running last price which carries *across* bars, so a
    bar whose OHLC never leaves the previous print prints nothing at all and
    leaves the L1 book showing an older print's size. Modelling each bar as a
    fresh book of its own overstates the liquidity a resting order meets on a
    quiet bar and understates how stale the book behind it is.

    An order submitted mid-bar meets only the book that bar left behind --
    and meets it *twice*. Nautilus drains new commands and then iterates
    every matching engine at the same instant, so an order priced exactly at
    the book takes a taker bite at drain time and a second, maker bite from
    that same-instant iterate, against a book nothing has moved in between.
    An order priced *through* the book takes one bite and sweeps. `IOC` and
    `FOK` are canceled before the sweep and so never see the second bite.

11. Realized PnL settles fill by fill when the settlement currency declares a
    precision. A venue does not carry a full-precision running total and
    round it at the end: it books each fill into the account in whole
    currency units, and the position's realized PnL is the sum of those
    bookings. The two agree on a position taken off in one fill and part
    company on one taken off in nine -- an AVAX position unwound over nine
    partial exits differed by 2e-8, which is 2000 times the tolerance the
    ledger comparison allows. `InstrumentConfig.currency_precision` (item 4)
    gates it; `None` keeps the raw floating-point round trip.

12. An order's outstanding quantity and its status agree on what "nothing
    left" means. Both now read one `residual_tolerance`, so a fill that
    `record_fill` already judged to have completed an order cannot leave
    `leaves_qty` reporting the 1e-13 of binary residue that made the same
    order `PARTIALLY_FILLED` to one caller and `FILLED` to another.
13. `Order.arrival_ns` (`arrives_before_bar` plus `submitted_ts` inside the
    engine) matches an order against the book the *previous* bar left behind
    rather than the one its own bar leaves, and dates what it fills there to
    the instant it arrived.

    A venue prices one instrument's bar at a time. A strategy trading a
    basket decides on the bar of whichever name printed first and sends
    orders for the rest at that same instant -- and for those names the bar
    has not reached the venue yet, so they are matched against a book one bar
    older than their own timestamp. Nautilus does this because its simulated
    exchange consumes the data stream element by element; nothing about it is
    a choice the strategy made, and a replay cannot tell the two cases apart
    without being told which is which. In a nine-name daily rebalance, nine
    of the ten orders met the older book and only the tenth -- the name whose
    bar triggered the rebalance -- met its own.

    The same happens on one instrument whenever a bar aggregated on a clock
    closes over a gap in the data: the window ends where nothing printed, and
    the finished bar reaches the strategy only when a later bar advances the
    clock past it. The order it sends is at the venue before that later bar
    is, and Nautilus fills it against the book the last bar before the gap
    left -- a 2021 Binance outage put two hours between the two, and the
    fill's price was the pre-gap one.

    Such a fill happened when the order arrived, not when the bar it beat
    printed, so the position it opens and the round trip it closes are dated
    to `arrival_ns`. Only a fill taken from that standing book is: whatever
    the same order goes on to take from a print is dated by that print, like
    any other fill.

    An order that does not cross the standing book is not finished with the
    bar it beat. It was working at the venue while that bar printed, so the
    range is ahead of it in time rather than behind it: the bar fills it if
    it trades through its limit, at its own limit, like any resting order the
    market comes to. Nothing here is look-ahead -- the order was there first.
    That is the ordinary case for a basket, where most names get a limit
    priced off a bar that is minutes old by the time the venue sees it: it
    does not cross on arrival, rests, and the bar it beat is usually the one
    that fills it. An immediate order is the exception, executed against the
    book it arrived at and killed; it never waits for a print. And an order
    arriving before the very first bar meets no book at all, then rests into
    that bar exactly as it would into any other.

    The field is per order and defaults to `None`, which is the behaviour the
    engine always had. It is independent of
    `same_bar_marketable_limit_on_close` (item 1): an order that reached the
    venue before its bar has already met a book, so it matches on the bar it
    was submitted on whether or not same-bar matching is enabled generally.
    An order submitted *from* its bar still never meets that bar's range --
    those prints came before the decision that sent it.

14. `AccountMode::Margin` accepts an infinite `leverage`: the venue that
    locks no initial margin at all. The per-position margin rate is the
    instrument's `margin_init` when it declares one and `1 / leverage`
    otherwise, so an infinite leverage makes that rate zero -- nothing is
    locked, no order is refused for want of capital, and the balance moves
    only with realized PnL and fees.

    That is not a degenerate configuration; it is what a Nautilus MARGIN
    account trades like for these instruments. Every instrument the project
    builds declares `margin_init = 0`, and Nautilus computes initial margin
    as `notional / leverage * margin_init`. A cash account is not a mirror of
    one: it debits the full notional of every entry, shorts included, and
    refuses orders such a venue filled. Both replay lanes hand Raptor sizes
    Nautilus already decided, so the account's only remaining job is to gate,
    and it has to gate the same way or the lane measures the harness instead
    of the engine. A cross-sectional long/short case failed exactly there,
    silently: Nautilus filled every short, Raptor's cash account rejected
    them with `insufficient_capital`, and the canonical projection kept only
    the submitted orders -- so the two ledgers agreed on every order and
    disagreed on everything after.

    A size given as a fraction of capital is refused against such an account
    with `RejectReason::UnfundedSizing` (`unfunded_sizing`). The fraction
    answers "how many units can this account afford", and an account that
    affords any size cannot answer it. The division previously fell through
    to the fee rate alone -- and to infinity where there was no fee -- sizing
    a position at hundreds of times the balance. Explicit unit sizes are
    unaffected.

## Upstream fixes

A closing order reduced by the whole position and ignored its own quantity,
so a one-lot trim of eleven held units flattened the book: ten units of
exposure a venue would still have been holding, and every later fill and mark
measured against a position that no longer existed. A venue sells what the
order says. The close is now bounded by the order's size as well as by the
position and the bar's liquidity, and an order asking for less than is held
reports itself filled rather than leaving a phantom remainder to expire.
Asking for the whole position is still spelled by naming no size at all
(`QtySpec::FullPosition`, which is what the Python API builds when a closing
order names neither `units` nor `size_frac`), and a capital fraction still
names units of an entry rather than of a reduction, so neither changes. This
is what a nine-name portfolio replay found: Nautilus trimmed one share of a
GOOGL position and Raptor sold all eleven.

A position held the float sum of the fills that opened it, and that sum need
not land where either fill did: 0.03835 + 0.04381 is 0.08216000000000001, so
selling the 0.08216 that was bought left 1.4e-17 of a coin open. Nothing
asks for a hundredth of a femto-lot, so the position never went flat and
every later entry averaged into it -- a resampled BTC replay of forty-odd
round trips reported one that never ended. The cash account was right to the
cent throughout; only the book was wrong. A venue holds a whole number of
lots, so the ledger now puts a position size back on the instrument's size
grid after every add and reduction, and reports a round trip's closed size
the same way.

The same grid was missing from an order's unfilled remainder. A total and
its fills are grid quantities, but their difference need not be: 0.07841
filled down to 0.06531 leaves 1309.9999999999986 lots, and flooring that
asked for a lot less than the venue still owed. The order finished short and
called itself partially filled with a sliver no bar would absorb. The
remainder is now read off the grid at the two points that consume it -- the
size a resumed order asks for, and the leaves a fill reports.

`portfolio::monte_carlo` rounds its chunk size up, so with more threads than
simulations the last chunks began past the end of the work and
`Vec::with_capacity(end - start)` underflowed -- five simulations over four
threads is a chunk size of two and a fourth chunk starting at six. The start
is now clamped, leaving those chunks empty. This is not a parity change; it
is a bug that predates this branch, and it is why the suite previously needed
a bounded `RAYON_NUM_THREADS` on a many-core machine.

## Evidence

- All 585 Rust library tests pass, at any thread count.
- All 432 of the fork's own Python tests pass, the bit-exact golden gate
  included. Its baselines were regenerated once, for the fee arithmetic of
  item 6; see the changelog for what moved and by how much.
- The algotrade-nautilus 81-case strategy matrix (27 strategies, three
  parameter variants each) passes 81 of 81, with no divergent case, no
  erroring case, and no case whose strategy never ordered: 25,033 Nautilus
  orders compared in total. All 15 strict independent cases pass a full
  independent ledger; all 24 single-name oracle cases pass, 17 replaying
  Nautilus's decisions as entry/exit signals and seven replaying a single
  name's typed order flow; all 42 portfolio cases pass on portfolio order
  replay. The 42 portfolio cases were unsupported before this branch replayed
  order flow; the two failures before that were a settlement row compared
  against Nautilus's unrounded mark; and the seven order-replay cases are the
  ones the signal lane cannot express -- a reduce-only order against a
  position it has already closed whole, or an order type the lane will not
  place -- which route to typed order replay instead. That fallback is what
  exposed the two lot-grid bugs fixed above.
- The algotrade-nautilus strict BTC callback case matched 230 of 230 canonical
  data, indicator, decision, order, fill, fee, position, equity, and metric
  events over 2026-01-01 through 2026-02-01, twice per engine.
- A nine-case venue certification over every downloaded Binance symbol plus
  representative IB US and ASX instruments passes all nine strict ledgers,
  with no case routed to Nautilus fallback. Every order, fill, fee, position
  and equity event matches, including the AVAX case that needed all of items
  10 through 12 to close: 21 orders and 21 fills over five positions, three
  of them opened or unwound in pieces. AAPL matched 67/67 events including US
  sessions and pre-close behavior; CBA matched 172/172 including ASX
  sessions, AUD settlement, and minimum fees.
- The separate Raptor array lane completed the exact 3,773-run BTC atlas in
  287.692 seconds with zero failures on this branch at eight workers: 13.11
  runs/s and 600.64x faster than the observed two-day run. The array API still
  uses MARKET/IOC semantics and is not claimed to have typed-order parity.

Paper and live trading remain Nautilus-only. The branch is intended for a
capability-gated hybrid backtest backend, with Nautilus fallback for unproven
strategy and execution families.
