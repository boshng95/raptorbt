//! Tests for the simulation kernel.
//!
//! Split out of `kernel.rs`, which the file-size rules cap; included back
//! into that module so `super::*` and private items still resolve.

use super::*;

fn make_kernel() -> EngineKernel {
    let config = BacktestConfig::default();
    let fee_model = config.fee_model();
    EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    )
}

#[test]
fn exact_fractional_lot_is_not_dropped_at_binary_boundary() {
    let rounded = floor_to_lot(0.10185, 0.00001);
    assert!((rounded - 0.10185).abs() < 1e-15, "rounded={rounded:.17}");

    let below = floor_to_lot(0.101849, 0.00001);
    assert!((below - 0.10184).abs() < 1e-15, "below={below:.17}");
}

#[test]
fn a_lot_grid_size_is_bit_exact_and_not_one_ulp_high() {
    // `lots.floor() * lot` reconstructs the size in binary, so an exact
    // decimal size can come back an ULP high: 0.10379 became
    // 0.10379000000000001. Nautilus holds the same quantity as a decimal, and
    // a percentage fee charged on `size * price` turns that ULP into a
    // last-decimal commission difference. Pin equality, not a tolerance.
    for (raw, lot, expected) in [
        (0.10379_f64, 0.00001_f64, 0.10379_f64),
        (0.10185, 0.00001, 0.10185),
        (0.101849, 0.00001, 0.10184),
        (1.5, 0.1, 1.5),
        (7.0, 1.0, 7.0),
    ] {
        let actual = floor_to_lot(raw, lot);
        assert_eq!(
            actual, expected,
            "floor_to_lot({raw}, {lot}) = {actual:.17}, want {expected:.17}"
        );
    }
}

#[test]
fn declared_currency_precision_quantizes_crypto_fees_and_pnl() {
    let config = BacktestConfig { fees: 0.001, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let instrument = InstrumentConfig {
        lot_size: Some(0.00001),
        currency_precision: Some(8),
        ..InstrumentConfig::default()
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "ETHUSDT".to_string(),
        Direction::Long,
        Some(&instrument),
    );
    let entered = kernel.open_at(
        0,
        &bar(0, 3223.91),
        Direction::Long,
        3223.91,
        None,
        Some(2.93546),
        0.0,
        None,
        None,
        FillTerms::WHOLE,
    );
    assert!(matches!(entered, Some(OpenResult { event: EngineEvent::Entered { .. }, .. })));
    let exited = kernel.close_at(1, &bar(1, 2923.12), 0, 2923.12, ExitReason::Signal, None);
    match exited {
        Some(EngineEvent::Exited { trade, .. }) => {
            assert_eq!(trade.entry_fees, 9.46365885);
            assert_eq!(trade.exit_fees, 8.58070184);
            // The round trip settles in the same units the fees do: the
            // entry fee is booked when it is charged and the close books
            // what it realized less what it cost. Raw float arithmetic over
            // the same terms lands an ULP below this, at
            // -901.0013740899999, which is not a number the account could
            // ever hold.
            assert_eq!(trade.pnl, -901.00137409);
        }
        other => panic!("expected exit, got {other:?}"),
    }
}

#[test]
fn declared_currency_precision_quantizes_cash_and_equity_to_cents() {
    let config = BacktestConfig { fees: 0.00088, fee_minimum: 6.60, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let instrument = InstrumentConfig {
        lot_size: Some(1.0),
        currency_precision: Some(2),
        ..InstrumentConfig::default()
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "CBA".to_string(),
        Direction::Long,
        Some(&instrument),
    );
    kernel.set_cash(10_117.7602928);
    assert_eq!(kernel.cash(), 10_117.76);
    assert_eq!(kernel.equity(150.123456), 10_117.76);
}

fn bar(idx: i64, price: Price) -> KernelBar {
    KernelBar {
        timestamp: idx,
        open: price,
        high: price + 1.0,
        low: price - 1.0,
        close: price,
        volume: 1000.0,
    }
}

fn enter(kernel: &mut EngineKernel, idx: usize, price: Price) {
    let events = kernel.step(
        idx,
        &bar(idx as i64, price),
        StepInput { entry: true, ..StepInput::default() },
    );
    assert!(
        matches!(events.as_slice(), [EngineEvent::Entered { .. }]),
        "expected entry, got {events:?}"
    );
}

#[test]
fn set_stop_price_is_noop_when_flat() {
    let mut kernel = make_kernel();
    kernel.set_stop_price(Some(90.0));
    assert!(kernel.position_snapshot().is_none());
}

fn trade(ts: i64, price: Price, size: f64) -> TradeTick {
    TradeTick { timestamp: ts, price, size, signed_size: 0.0 }
}

#[test]
fn step_trade_enters_and_exits_at_the_print() {
    let mut kernel = make_kernel();
    let events = kernel.step_trade(
        0,
        &trade(0, 100.0, 5.0),
        StepInput { entry: true, ..Default::default() },
    );
    assert!(matches!(events.as_slice(), [EngineEvent::Entered { price, .. }] if *price == 100.0));
    assert!(kernel.is_in_position());

    let events =
        kernel.step_trade(1, &trade(1, 110.0, 5.0), StepInput { exit: true, ..Default::default() });
    assert!(
        matches!(events.as_slice(), [EngineEvent::Exited { trade, .. }] if trade.exit_price == 110.0)
    );
}

#[test]
fn step_quote_does_not_move_the_trailing_watermark() {
    // A bid that never traded must not ratchet a position's trailing
    // stop — that would manufacture exits from an untraded price.
    let mut kernel = make_kernel();
    kernel.step_trade(0, &trade(0, 100.0, 1.0), StepInput { entry: true, ..Default::default() });
    let before = kernel.position_snapshot().expect("in position");

    let events = kernel.step_quote(&QuoteTick { timestamp: 1, bid: 500.0, ask: 501.0 });
    assert!(events.is_empty());
    let after = kernel.position_snapshot().expect("still in position");
    assert_eq!(before.stop_price, after.stop_price);
    assert_eq!(kernel.best_bid(), Some(500.0));
    assert_eq!(kernel.best_ask(), Some(501.0));
}

/// A kernel that replays each bar as four prints, like Nautilus does.
fn bounded_kernel(policy: PositionPolicy) -> EngineKernel {
    let config = BacktestConfig { bar_volume_slices: 4.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );
    kernel.set_position_policy(policy);
    kernel
}

/// A bounded kernel on a discrete price grid, so a sweep has a next level
/// to land on rather than collapsing onto the price it swept.
fn bounded_kernel_on_a_grid(policy: PositionPolicy, increment: f64) -> EngineKernel {
    let mut kernel = bounded_kernel(policy);
    kernel.configured_price_increment = Some(increment);
    kernel
}

/// A bounded kernel whose sizes sit on a whole-unit lot grid, so a bar's
/// prints have something to round onto.
fn bounded_kernel_on_a_lot(policy: PositionPolicy, lot: f64) -> EngineKernel {
    let config = BacktestConfig {
        bar_volume_slices: 4.0,
        same_bar_marketable_limit_on_close: true,
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let inst = InstrumentConfig { lot_size: Some(lot), ..InstrumentConfig::default() };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        Some(&inst),
    );
    kernel.set_position_policy(policy);
    kernel
}

/// A kernel funded with a known amount under a chosen account mode, so a
/// test can put an order against capital that does or does not fund it.
fn make_kernel_with_capital(capital: f64, account: AccountMode) -> EngineKernel {
    let config = BacktestConfig {
        initial_capital: capital,
        same_bar_marketable_limit_on_close: true,
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    )
    .with_account_mode(account);
    kernel.set_position_policy(PositionPolicy::Net);
    kernel
}

fn bar_with_volume(idx: i64, price: Price, volume: f64) -> KernelBar {
    KernelBar { volume, ..bar(idx, price) }
}

fn order_status(kernel: &EngineKernel, id: u64) -> OrderStatus {
    kernel.orders.get(id).expect("order").status
}

/// A margin account of infinite leverage locks nothing, so a sized order
/// fills however little cash is on hand. This is the Nautilus equity venue:
/// its instruments declare `margin_init = 0`, so its accounts refuse no
/// order for want of capital, and a cash account is not a mirror of one --
/// it would refuse the very orders that venue filled.
#[test]
fn an_unfunded_margin_account_never_refuses_a_sized_order() {
    let capital = 500.0;
    let submit = |kernel: &mut EngineKernel| {
        kernel.submit_order_full(
            OrderSide::Buy,
            QtySpec::Units(10.0),
            OrderKind::Limit { price: 105.0 },
            TimeInForce::Ioc,
            0,
            0,
            "entry".to_string(),
            None,
            None,
            false,
            false,
            false,
            None,
        );
        kernel.step(0, &bar(0, 100.0), StepInput::default())
    };

    // Fully funded, 1,000 of notional does not fit in 500 of cash.
    let mut cash = make_kernel_with_capital(capital, AccountMode::Cash);
    let refused = submit(&mut cash);
    assert!(
        refused.iter().any(|event| matches!(
            event,
            EngineEvent::OrderRejected { reason: "insufficient_capital", .. }
        )),
        "cash should refuse what it cannot fund: {refused:?}"
    );

    // The same order against a venue that locks no margin.
    let mut unfunded =
        make_kernel_with_capital(capital, AccountMode::Margin { leverage: f64::INFINITY });
    let filled = submit(&mut unfunded);
    assert!(
        filled.iter().any(|event| matches!(event, EngineEvent::Entered { .. })),
        "an unfunded venue posts nothing against the position: {filled:?}"
    );
    assert_eq!(unfunded.locked_margin(), 0.0, "an unfunded venue posts no margin");
}

/// The same account leaves a capital fraction with nothing to divide by, so
/// it names no size. Refusing says so; dividing would open a position of
/// infinite size.
#[test]
fn a_capital_fraction_is_refused_by_an_account_that_funds_nothing() {
    let mut kernel =
        make_kernel_with_capital(500.0, AccountMode::Margin { leverage: f64::INFINITY });
    kernel.submit_order_full(
        OrderSide::Buy,
        QtySpec::CapitalFrac(0.5),
        OrderKind::Limit { price: 105.0 },
        TimeInForce::Ioc,
        0,
        0,
        "entry".to_string(),
        None,
        None,
        false,
        false,
        false,
        None,
    );
    let events = kernel.step(0, &bar(0, 100.0), StepInput::default());
    assert!(
        events.iter().any(|event| matches!(
            event,
            EngineEvent::OrderRejected { reason: "unfunded_sizing", .. }
        )),
        "a fraction of capital names no size here: {events:?}"
    );
    assert!(kernel.position_snapshot().is_none(), "nothing should have opened");
}

/// An order the venue received before this bar's print transacted when it
/// arrived: it met the book standing then, and the round trip it opens and
/// closes is dated to those instants rather than to the bars it beat.
#[test]
fn a_fill_taken_before_a_bar_is_dated_when_the_order_arrived() {
    let mut kernel = make_kernel();
    kernel.set_position_policy(PositionPolicy::Net);
    let at = |ts: i64, idx: i64, price: Price| KernelBar { timestamp: ts, ..bar(idx, price) };

    // A first bar leaves a book at 100.
    kernel.step(0, &at(100, 0, 100.0), StepInput::default());

    // Sent at 150, between the bars: it crosses the standing book at 100.
    kernel.submit_order_full(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 105.0 },
        TimeInForce::Ioc,
        1,
        150,
        "entry".to_string(),
        None,
        None,
        false,
        false,
        true,
        None,
    );
    kernel.step(1, &at(200, 1, 110.0), StepInput::default());
    assert_eq!(kernel.position_snapshot().expect("open").entry_price, 100.0);

    // The close arrives the same way, ahead of the third bar.
    kernel.submit_order_full(
        OrderSide::Sell,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 95.0 },
        TimeInForce::Ioc,
        2,
        250,
        "exit".to_string(),
        None,
        None,
        false,
        true,
        true,
        None,
    );
    let events = kernel.step(2, &at(300, 2, 120.0), StepInput::default());
    let trade = events
        .iter()
        .find_map(|event| match event {
            EngineEvent::Exited { trade, .. } => Some(trade),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no round trip, got {events:?}"));
    assert_eq!((trade.entry_price, trade.exit_price), (100.0, 110.0));
    assert_eq!(
        (trade.entry_time, trade.exit_time),
        (150, 250),
        "a fill dated by the bar it beat would read (200, 300)"
    );
}

#[test]
fn a_bar_bounds_an_entry_to_one_print_of_its_volume() {
    let mut kernel = bounded_kernel(PositionPolicy::Net);
    // Resting on the bar's low: the market came down to the order and
    // turned, so only the print that touched it was ever on offer.
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "e".to_string(),
        None,
        None,
    );
    // 40 traded, four prints, so the one that touched the order shows 10.
    let events = kernel.step(1, &bar_with_volume(1, 100.0, 40.0), StepInput::default());
    let filled: Vec<f64> = events
        .iter()
        .filter_map(|e| match e {
            EngineEvent::OrderFilled { size, .. } => Some(*size),
            _ => None,
        })
        .collect();
    assert_eq!(filled, vec![10.0], "got {events:?}");
    assert_eq!(kernel.position_snapshot().unwrap().size, 10.0);
    assert_eq!(order_status(&kernel, id), OrderStatus::PartiallyFilled);
}

#[test]
fn an_ioc_entry_that_fills_short_dies_with_its_remainder() {
    let mut kernel = bounded_kernel(PositionPolicy::Net);
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Market,
        TimeInForce::Ioc,
        0,
        0,
        "ioc".to_string(),
        None,
        None,
    );
    // A market order would cross the book and fill whole -- but this one is
    // canceled the instant its first fill lands, so it gets one print.
    let events = kernel.step(0, &bar_with_volume(0, 100.0, 40.0), StepInput::default());
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::OrderCanceled { .. })),
        "the unfilled remainder must be canceled, got {events:?}"
    );
    assert_eq!(order_status(&kernel, id), OrderStatus::Canceled);
    assert_eq!(kernel.position_snapshot().unwrap().size, 10.0);
}

#[test]
fn a_working_entry_takes_more_size_on_the_next_bar() {
    // A resting order is not done when a bar runs out of size; it keeps
    // taking prints until it has the whole quantity it asked for.
    let mut kernel = bounded_kernel(PositionPolicy::NetAveraging);
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(20.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "g".to_string(),
        None,
        None,
    );
    // The bar bottoms out exactly on the order: 40 traded, so 10 of the 20
    // fill and the rest keeps resting.
    kernel.step(1, &bar_with_volume(1, 100.0, 40.0), StepInput::default());
    assert_eq!(order_status(&kernel, id), OrderStatus::PartiallyFilled);
    assert_eq!(kernel.position_snapshot().unwrap().size, 10.0);

    // The next bar trades *through* 99, emptying the book beneath the
    // order, and the remainder fills at once.
    kernel.step(2, &bar_with_volume(2, 99.0, 400.0), StepInput::default());
    assert_eq!(order_status(&kernel, id), OrderStatus::Filled);

    let snapshot = kernel.position_snapshot().unwrap();
    assert_eq!(snapshot.size, 20.0, "both fills belong to one position");
    // Both fills came off at the resting limit, so averaging them returns
    // that price exactly. The weighting across *differing* prices is pinned
    // in the ledger tests.
    assert_eq!(snapshot.entry_price, 99.0);
}

#[test]
fn an_exit_bounded_by_volume_leaves_the_position_open() {
    let mut kernel = bounded_kernel(PositionPolicy::Net);
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(20.0),
        OrderKind::Market,
        TimeInForce::Gtc,
        0,
        0,
        "in".to_string(),
        None,
        None,
    );
    kernel.step(0, &bar_with_volume(0, 100.0, 400.0), StepInput::default());
    assert_eq!(kernel.position_snapshot().unwrap().size, 20.0);

    let exit_id = kernel.submit_order(
        OrderSide::Sell,
        QtySpec::FullPosition,
        OrderKind::Limit { price: 111.0 },
        TimeInForce::Gtc,
        0,
        0,
        "out".to_string(),
        None,
        None,
    );
    // The bar tops out exactly on the exit and turns: 40 traded, so only
    // the 10 shown by the print that touched it can come off.
    let events = kernel.step(1, &bar_with_volume(1, 110.0, 40.0), StepInput::default());
    assert!(
        !events.iter().any(|e| matches!(e, EngineEvent::Exited { .. })),
        "a partial exit is not a round trip, got {events:?}"
    );
    assert_eq!(kernel.position_snapshot().unwrap().size, 10.0);
    assert_eq!(order_status(&kernel, exit_id), OrderStatus::PartiallyFilled);

    // The rest comes off on a bar that trades through the exit, and only
    // then is there a trade -- one trade, spanning both exit fills.
    let events = kernel.step(2, &bar_with_volume(2, 120.0, 4_000.0), StepInput::default());
    let trades: Vec<&Trade> = events
        .iter()
        .filter_map(|e| match e {
            EngineEvent::Exited { trade, .. } => Some(trade),
            _ => None,
        })
        .collect();
    assert_eq!(trades.len(), 1, "got {events:?}");
    assert_eq!(trades[0].size, 20.0, "one trade spanning both exit fills");
    // Both fills came off at the resting limit, so the size-weighted exit
    // is that price. The weighting itself is pinned in the ledger tests.
    assert_eq!(trades[0].exit_price, 111.0);
    assert!(!kernel.is_in_position());
}

#[test]
fn a_bar_that_trades_through_an_order_fills_it_whole() {
    // The counterpart to the bounded case: the bar did not stop at the
    // order, it went past it. There was nothing left resting underneath, so
    // the whole quantity fills even though the bar traded far less volume
    // than the order asked for.
    let mut kernel = bounded_kernel(PositionPolicy::Net);
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Limit { price: 99.5 },
        TimeInForce::Gtc,
        0,
        0,
        "thru".to_string(),
        None,
        None,
    );
    // 40 traded against an order for 100, and the low of 99 is through it.
    kernel.step(1, &bar_with_volume(1, 100.0, 40.0), StepInput::default());
    assert_eq!(order_status(&kernel, id), OrderStatus::Filled);
    assert_eq!(kernel.position_snapshot().unwrap().size, 100.0);
}

#[test]
fn a_resting_order_the_market_moves_through_fills_at_its_own_price() {
    // Two fills, not one: the print that reached the order, then the rest.
    // Both at 99.5, because the order was resting there before the market
    // came down -- it is the side being traded against, not the side
    // crossing, so it never pays up for its own remainder.
    let mut kernel = bounded_kernel_on_a_grid(PositionPolicy::Net, 0.25);
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Limit { price: 99.5 },
        TimeInForce::Gtc,
        0,
        0,
        "through".to_string(),
        None,
        None,
    );
    // A quarter of 40 prints at the low of 99, which is through the order.
    kernel.step(1, &bar_with_volume(1, 100.0, 40.0), StepInput::default());
    assert_eq!(order_status(&kernel, id), OrderStatus::Filled);
    let position = kernel.position_snapshot().expect("position");
    assert_eq!(position.size, 100.0);
    assert!(
        (position.entry_price - 99.5).abs() < 1e-9,
        "entry {} should be the resting limit",
        position.entry_price
    );
}

#[test]
fn a_sweep_pays_one_increment_worse_than_the_book_it_emptied() {
    // The aggressive counterpart: an order submitted while the bar was
    // being observed crosses the book it finds. It takes what the book was
    // showing, and the remainder pays one increment up for the level
    // behind it.
    let mut kernel = bounded_kernel_on_a_grid(PositionPolicy::Net, 0.25);
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Market,
        TimeInForce::Gtc,
        1,
        1,
        "sweep".to_string(),
        None,
        None,
    );
    // The bar closed at 100 showing a quarter of its 40 volume; the order
    // is priced through that, so the other 90 sweep the next level up.
    kernel.step(1, &bar_with_volume(1, 100.0, 40.0), StepInput::default());
    assert_eq!(order_status(&kernel, id), OrderStatus::Filled);
    let position = kernel.position_snapshot().expect("position");
    assert_eq!(position.size, 100.0);
    let expected = (10.0 * 100.0 + 90.0 * 100.25) / 100.0;
    assert!(
        (position.entry_price - expected).abs() < 1e-9,
        "swept entry {} should average {expected}",
        position.entry_price
    );
}

#[test]
fn the_closing_print_of_a_bar_carries_the_rounding_remainder() {
    // A quarter of 41 is 10.25, off a whole-unit lot grid, so the first
    // three prints show 10 and the close shows the 11 they left behind --
    // the four summing to the bar's volume exactly rather than to 40.
    let mut kernel = bounded_kernel_on_a_lot(PositionPolicy::Net, 1.0);
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Limit { price: 100.0 },
        TimeInForce::Ioc,
        1,
        1,
        "remainder".to_string(),
        None,
        None,
    );
    // Submitted while bar 1 was observed, so the close is the only print
    // still ahead of it -- and the close is where the remainder lives.
    // Reading the bar's range instead would be look-ahead.
    kernel.step(1, &bar_with_volume(1, 100.0, 41.0), StepInput::default());
    // Immediate-or-cancel, so what the close could not absorb is killed and
    // the size that did trade is the remainder alone.
    assert_eq!(order_status(&kernel, id), OrderStatus::Canceled);
    assert_eq!(kernel.position_snapshot().expect("position").size, 11.0);
}

#[test]
fn each_fill_reports_the_fees_it_paid_and_what_it_left_outstanding() {
    // A partial fill is only describable if the event says how much of the
    // order survived it and what that slice alone cost. Re-deriving either
    // from the position afterwards cannot separate the two fills.
    let config = BacktestConfig {
        bar_volume_slices: 4.0,
        fees: 0.001,
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );
    kernel.set_position_policy(PositionPolicy::Net);
    kernel.configured_price_increment = Some(0.25);
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Limit { price: 99.5 },
        TimeInForce::Gtc,
        0,
        0,
        "fees".to_string(),
        None,
        None,
    );
    let events = kernel.step(1, &bar_with_volume(1, 100.0, 40.0), StepInput::default());
    let fills: Vec<(f64, f64, f64)> = events
        .iter()
        .filter_map(|event| match event {
            EngineEvent::OrderFilled { order_id, size, commission, leaves, .. }
                if *order_id == id =>
            {
                Some((*size, *commission, *leaves))
            }
            _ => None,
        })
        .collect();
    assert_eq!(fills.len(), 2, "a swept order fills twice: {fills:?}");
    let (first_size, first_fee, first_leaves) = fills[0];
    let (second_size, second_fee, second_leaves) = fills[1];
    assert_eq!(first_size, 10.0);
    assert_eq!(first_leaves, 90.0);
    assert_eq!(second_size, 90.0);
    assert_eq!(second_leaves, 0.0);
    // Each fill pays for its own units at its own price, not a share of a
    // blended average.
    assert!((first_fee - 10.0 * 99.5 * 0.001).abs() < 1e-9, "first fee {first_fee}");
    assert!((second_fee - 90.0 * 99.5 * 0.001).abs() < 1e-9, "second fee {second_fee}");
}

#[test]
fn an_unbounded_kernel_still_fills_whole_orders() {
    // The default must be untouched: without a slice count configured,
    // volume says nothing about how much fills.
    let mut kernel = make_kernel();
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Market,
        TimeInForce::Ioc,
        0,
        0,
        "u".to_string(),
        None,
        None,
    );
    kernel.step(0, &bar_with_volume(0, 100.0, 4.0), StepInput::default());
    assert_eq!(order_status(&kernel, id), OrderStatus::Filled);
    assert_eq!(kernel.position_snapshot().unwrap().size, 100.0);
}

#[test]
fn step_quote_does_not_match_resting_orders() {
    let mut kernel = make_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 90.0 },
        TimeInForce::Gtc,
        0,
        0,
        "q".to_string(),
        None,
        None,
    );
    // A quote straddling the limit must not fill it.
    kernel.step_quote(&QuoteTick { timestamp: 1, bid: 80.0, ask: 81.0 });
    assert!(!kernel.is_in_position());

    // The print that follows is the evidence, and does fill it.
    let events = kernel.step_trade(1, &trade(2, 89.0, 10.0), StepInput::default());
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })),
        "the print should fill the resting limit, got {events:?}"
    );
}

#[test]
fn step_trade_does_not_fill_bar_phase_market_orders() {
    // AT_CLOSE queues for a bar phase a print does not have.
    let mut kernel = make_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Market,
        TimeInForce::AtClose,
        0,
        0,
        "atclose".to_string(),
        None,
        None,
    );
    let events = kernel.step_trade(1, &trade(1, 100.0, 1.0), StepInput::default());
    assert!(
        !events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })),
        "AT_CLOSE must keep resting on a print, got {events:?}"
    );
    assert!(!kernel.is_in_position());

    // It fills on the next bar event.
    let events = kernel.step(2, &bar(2, 100.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
}

#[test]
fn step_trade_skips_orders_submitted_on_the_same_event() {
    let mut kernel = make_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 200.0 },
        TimeInForce::Gtc,
        5,
        0,
        "same".to_string(),
        None,
        None,
    );
    // Submitted while observing event 5: cannot rest into event 5.
    let events = kernel.step_trade(5, &trade(5, 100.0, 1.0), StepInput::default());
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
    // Event 6 matches it.
    let events = kernel.step_trade(6, &trade(6, 100.0, 1.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
}

#[test]
fn queue_model_is_off_by_default() {
    // A resting limit fills on the first print that reaches it, exactly
    // as before the queue model existed.
    let mut kernel = make_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "q".to_string(),
        None,
        None,
    );
    let events = kernel.step_trade(1, &trade(1, 99.0, 1.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
}

#[test]
fn queue_model_holds_an_order_behind_displayed_size() {
    let config = BacktestConfig { queue_fill_model: true, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );
    // 300 lots displayed at our price: we join behind them.
    kernel.book.apply_depth(&crate::data::DepthTick::from_levels(
        0,
        &[crate::data::BookLevel { price: 99.0, size: 300.0 }],
        &[crate::data::BookLevel { price: 101.0, size: 100.0 }],
    ));
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "q".to_string(),
        None,
        None,
    );

    // A small print at our price does not reach us.
    let events = kernel.step_trade(1, &trade(1, 99.0, 100.0), StepInput::default());
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
    assert!(!kernel.is_in_position());

    // Enough volume prints through the queue ahead, and we fill.
    let events = kernel.step_trade(2, &trade(2, 99.0, 250.0), StepInput::default());
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })),
        "the queue ahead was exhausted, got {events:?}"
    );
}

#[test]
fn queue_model_fills_when_the_level_trades_through() {
    let config = BacktestConfig { queue_fill_model: true, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );
    kernel.book.apply_depth(&crate::data::DepthTick::from_levels(
        0,
        &[crate::data::BookLevel { price: 99.0, size: 100_000.0 }],
        &[crate::data::BookLevel { price: 101.0, size: 100.0 }],
    ));
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "q".to_string(),
        None,
        None,
    );
    // A huge queue ahead, but the print cleared the level entirely.
    let events = kernel.step_trade(1, &trade(1, 98.0, 1.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
}

#[test]
fn queue_model_falls_back_to_probability_on_bar_events() {
    // A bar's volume is not volume at the limit price, so the queue
    // model must not consume it; fill_prob_limit=1.0 fills as always.
    let config = BacktestConfig { queue_fill_model: true, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );
    kernel.book.apply_depth(&crate::data::DepthTick::from_levels(
        0,
        &[crate::data::BookLevel { price: 99.0, size: 100_000.0 }],
        &[crate::data::BookLevel { price: 101.0, size: 100.0 }],
    ));
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "q".to_string(),
        None,
        None,
    );
    let events = kernel.step(1, &bar(1, 98.5), StepInput::default());
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })),
        "bar events fall back to fill_prob_limit, got {events:?}"
    );
}

#[test]
fn external_open_count_overrides_the_ledger() {
    // A flat kernel would pass a max_positions=1 gate on its own ledger;
    // a portfolio that already holds a position elsewhere says otherwise.
    let mut kernel = make_kernel().with_risk_gate(RiskGate::new(Some(1), None));
    assert_eq!(kernel.open_count(), 0);

    kernel.set_external_open_count(Some(1));
    let events = kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
    assert!(
        matches!(
            events.as_slice(),
            [EngineEvent::EntryRejected { reason: RejectReason::MaxPositions, .. }]
        ),
        "expected a portfolio-wide rejection, got {events:?}"
    );
    assert!(!kernel.is_in_position());

    // Clearing it restores ledger-derived counting: the slot is free.
    kernel.set_external_open_count(None);
    enter(&mut kernel, 1, 100.0);
    assert!(kernel.is_in_position());
}

#[test]
fn set_stop_and_target_update_open_position() {
    let mut kernel = make_kernel();
    enter(&mut kernel, 0, 100.0);

    kernel.set_stop_price(Some(95.0));
    kernel.set_target_price(Some(110.0));

    let snap = kernel.position_snapshot().unwrap();
    assert_eq!(snap.stop_price, Some(95.0));
    assert_eq!(snap.target_price, Some(110.0));

    kernel.set_stop_price(None);
    assert_eq!(kernel.position_snapshot().unwrap().stop_price, None);
}

#[test]
fn programmatic_stop_triggers_exit() {
    let mut kernel = make_kernel();
    enter(&mut kernel, 0, 100.0);
    kernel.set_stop_price(Some(98.5));

    // Bar trades down through the stop.
    let events = kernel.step(1, &bar(1, 98.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert_eq!(trade.exit_reason, ExitReason::StopLoss);
        }
        other => panic!("expected stop exit, got {other:?}"),
    }
    assert!(!kernel.is_in_position());
}

#[test]
fn entry_stop_override_beats_config() {
    let config = BacktestConfig {
        stop: StopConfig::Fixed { percent: 0.05 },
        target: TargetConfig::Fixed { percent: 0.10 },
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );

    let events = kernel.step(
        0,
        &bar(0, 100.0),
        StepInput {
            entry: true,
            stop_price_override: Some(97.0),
            target_price_override: Some(104.0),
            ..StepInput::default()
        },
    );
    assert!(matches!(events.as_slice(), [EngineEvent::Entered { .. }]));

    let snap = kernel.position_snapshot().unwrap();
    assert_eq!(snap.stop_price, Some(97.0));
    assert_eq!(snap.target_price, Some(104.0));
}

#[test]
fn zero_size_entry_emits_rejection() {
    let config = BacktestConfig::default();
    let fee_model = config.fee_model();
    // Lot of 10,000 units at price 100 with 100k capital -> raw size
    // ~999 units floors to zero lots.
    let inst = InstrumentConfig {
        price_increment: None,
        lot_size: Some(10_000.0),
        alloted_capital: None,
        stop: None,
        target: None,
        existing_qty: None,
        avg_price: None,
        max_quantity: None,
        currency_precision: None,
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        Some(&inst),
    );

    let events = kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::EntryRejected { reason, .. }] => {
            assert_eq!(reason.as_str(), "zero_size");
        }
        other => panic!("expected zero-size rejection, got {other:?}"),
    }
    assert!(!kernel.is_in_position());
}

#[test]
fn instrument_maximum_quantity_rejects_an_oversized_entry() {
    let config = BacktestConfig::default();
    let fee_model = config.fee_model();
    let inst = InstrumentConfig {
        lot_size: Some(0.00001),
        max_quantity: Some(9_000.0),
        ..InstrumentConfig::default()
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "ADAUSDT".to_string(),
        Direction::Long,
        Some(&inst),
    );

    let result = kernel.open_at(
        0,
        &bar(0, 0.4),
        Direction::Long,
        0.4,
        None,
        Some(23_750.0),
        0.0,
        None,
        None,
        FillTerms::WHOLE,
    );
    assert!(matches!(
        result,
        Some(OpenResult {
            event: EngineEvent::EntryRejected { reason: RejectReason::MaxQuantity, .. },
            ..
        })
    ));
    assert!(!kernel.is_in_position());
}

#[test]
fn multiplier_scales_notional_and_pnl() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let spec = InstrumentSpec {
        multiplier: 50.0,
        lot_size: 1.0,
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    // 100k capital at price 100 with multiplier 50: 20 contracts.
    enter(&mut kernel, 0, 100.0);
    let snap = kernel.position_snapshot().unwrap();
    assert!((snap.size - 20.0).abs() < 1e-9, "size {}", snap.size);
    assert!(kernel.cash().abs() < 1e-6, "cash {}", kernel.cash());

    // Price to 102: equity = 102 * 20 * 50 = 102_000.
    assert!((kernel.equity(102.0) - 102_000.0).abs() < 1e-6);

    // Exit at 102: pnl = 2 * 20 * 50 = 2_000.
    let events = kernel.step(1, &bar(1, 102.0), StepInput { exit: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert!((trade.pnl - 2_000.0).abs() < 1e-6, "pnl {}", trade.pnl);
        }
        other => panic!("expected exit, got {other:?}"),
    }
    assert!((kernel.cash() - 102_000.0).abs() < 1e-6);
}

fn zero_fee_kernel() -> EngineKernel {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    )
}

#[test]
fn resting_limit_buy_fills_next_bar_and_opens() {
    let mut kernel = zero_fee_kernel();
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "ord-1".into(),
        Some(95.0),
        None,
    );

    // Bar 0 (submission bar): only the acknowledgment, no fill.
    let events = kernel.step(0, &bar(0, 100.0), StepInput::default());
    assert!(matches!(
        events.as_slice(),
        [EngineEvent::OrderAccepted { order_id, .. }] if *order_id == id
    ));
    assert!(!kernel.is_in_position());

    // Bar 1 trades down through the limit: fill at 99, position opens
    // with the attached protective stop.
    let events = kernel.step(1, &bar(1, 99.5), StepInput::default());
    match events.as_slice() {
        [EngineEvent::OrderFilled { order_id, price, size, .. }, EngineEvent::Entered { .. }] => {
            assert_eq!(*order_id, id);
            assert_eq!(*price, 99.0);
            assert_eq!(*size, 10.0);
        }
        other => panic!("expected fill + entered, got {other:?}"),
    }
    let snap = kernel.position_snapshot().unwrap();
    assert_eq!(snap.stop_price, Some(95.0));
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::Filled);
}

#[test]
fn market_order_fills_on_submission_bar() {
    let mut kernel = zero_fee_kernel();
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::CapitalFrac(0.5),
        OrderKind::Market,
        TimeInForce::Gtc,
        3,
        3,
        "mkt-1".into(),
        None,
        None,
    );

    let events = kernel.step(3, &bar(3, 100.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::OrderAccepted { .. }, EngineEvent::OrderFilled { order_id, price, .. }, EngineEvent::Entered { .. }] =>
        {
            assert_eq!(*order_id, id);
            // FillPrice::Close on the submission bar — same contract as
            // the signal-entry path.
            assert_eq!(*price, 100.0);
        }
        other => panic!("expected accept + fill + entered, got {other:?}"),
    }
    // Half the capital: 50k / 100 = 500 units.
    assert!((kernel.position_snapshot().unwrap().size - 500.0).abs() < 1e-9);
}

#[test]
fn sell_limit_closes_position_as_order_exit() {
    let mut kernel = zero_fee_kernel();
    enter(&mut kernel, 0, 100.0);

    kernel.submit_order(
        OrderSide::Sell,
        QtySpec::FullPosition,
        OrderKind::Limit { price: 105.0 },
        TimeInForce::Gtc,
        0,
        0,
        "tp-1".into(),
        None,
        None,
    );

    // Bar 1 stays below the limit.
    let events = kernel.step(1, &bar(1, 103.0), StepInput::default());
    assert!(matches!(events.as_slice(), [EngineEvent::OrderAccepted { .. }]));
    assert!(kernel.is_in_position());

    // Bar 2 trades through it.
    let events = kernel.step(2, &bar(2, 105.5), StepInput::default());
    match events.as_slice() {
        [EngineEvent::OrderFilled { price, .. }, EngineEvent::Exited { trade, .. }] => {
            assert_eq!(*price, 105.0);
            assert_eq!(trade.exit_reason, ExitReason::Order);
        }
        other => panic!("expected fill + exit, got {other:?}"),
    }
    assert!(!kernel.is_in_position());
}

#[test]
fn opening_order_rejected_while_in_position() {
    let mut kernel = zero_fee_kernel();
    enter(&mut kernel, 0, 100.0);

    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(1.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "dup-1".into(),
        None,
        None,
    );

    let events = kernel.step(1, &bar(1, 98.5), StepInput::default());
    assert!(events
        .iter()
        .any(|e| matches!(e, EngineEvent::OrderRejected { reason: "position_open", .. })));
    assert_eq!(kernel.position_snapshot().unwrap().entry_idx, 0);
}

#[test]
fn oversized_unit_order_rejects_for_capital() {
    let mut kernel = zero_fee_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10_000.0), // 10k * 100 = 1M >> 100k capital
        OrderKind::Limit { price: 100.0 },
        TimeInForce::Gtc,
        0,
        0,
        "big-1".into(),
        None,
        None,
    );
    let _ = kernel.step(0, &bar(0, 100.0), StepInput::default());
    let events = kernel.step(1, &bar(1, 99.0), StepInput::default());
    assert!(events
        .iter()
        .any(|e| matches!(e, EngineEvent::OrderRejected { reason: "insufficient_capital", .. })));
    assert!(!kernel.is_in_position());
}

#[test]
fn per_contract_fees_charge_on_contracts_not_notional() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    // 2.5 currency units per contract per side, IB-style.
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let spec = InstrumentSpec {
        multiplier: 50.0,
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        FeeModel::per_share(2.5),
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    enter(&mut kernel, 0, 100.0);
    let size = kernel.position_snapshot().unwrap().size;

    let events = kernel.step(1, &bar(1, 100.0), StepInput { exit: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            // Round trip: 2.5 per contract per side, NOT 2.5 * 50.
            let expected = 2.0 * 2.5 * size;
            assert!(
                (trade.fees - expected).abs() < 1e-9,
                "fees {} != {expected} (size {size})",
                trade.fees
            );
        }
        other => panic!("expected exit, got {other:?}"),
    }
}

#[test]
fn percentage_fees_charge_on_notional() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config = BacktestConfig { fees: 0.001, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let spec = InstrumentSpec {
        multiplier: 50.0,
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    enter(&mut kernel, 0, 100.0);
    let size = kernel.position_snapshot().unwrap().size;

    let events = kernel.step(1, &bar(1, 100.0), StepInput { exit: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            // 0.1% of true notional (price * size * multiplier), each side.
            let expected = 2.0 * 0.001 * 100.0 * size * 50.0;
            assert!(
                (trade.fees - expected).abs() < 1e-6,
                "fees {} != {expected} (size {size})",
                trade.fees
            );
        }
        other => panic!("expected exit, got {other:?}"),
    }
}

#[test]
fn expiry_settles_position_and_rejects_entries() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let spec = InstrumentSpec {
        expiration_ns: Some(5),
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    enter(&mut kernel, 0, 100.0);

    // Bar at the expiry timestamp: settle at close, no signal needed.
    let events = kernel.step(5, &bar(5, 103.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert_eq!(trade.exit_reason, ExitReason::Settlement);
            assert!((trade.exit_price - 103.0).abs() < 1e-9);
        }
        other => panic!("expected settlement, got {other:?}"),
    }
    assert!(!kernel.is_in_position());

    // Post-expiry entry is refused.
    let events = kernel.step(6, &bar(6, 103.0), StepInput { entry: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::EntryRejected { reason, .. }] => {
            assert_eq!(reason.as_str(), "expired");
        }
        other => panic!("expected expired rejection, got {other:?}"),
    }
}

#[test]
fn pre_activation_entry_is_rejected() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config = BacktestConfig::default();
    let fee_model = config.fee_model();
    let spec = InstrumentSpec {
        activation_ns: Some(10),
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    let events = kernel.step(0, &bar(5, 100.0), StepInput { entry: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::EntryRejected { reason, .. }] => {
            assert_eq!(reason.as_str(), "inactive");
        }
        other => panic!("expected inactive rejection, got {other:?}"),
    }

    let events = kernel.step(1, &bar(10, 100.0), StepInput { entry: true, ..StepInput::default() });
    assert!(matches!(events.as_slice(), [EngineEvent::Entered { .. }]));
}

#[test]
fn config_stop_quantizes_to_tick_grid() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config =
        BacktestConfig { stop: StopConfig::Fixed { percent: 0.033 }, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let spec =
        InstrumentSpec { price_increment: 0.05, ..InstrumentSpec::new("EQ", InstrumentKind::Cash) };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "EQ".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    enter(&mut kernel, 0, 100.0);
    // Raw stop = 96.7; on the 0.05 grid floored for a long -> 96.70
    // exactly (already on grid); use a messier percent to prove rounding:
    let stop = kernel.position_snapshot().unwrap().stop_price.unwrap();
    assert!((stop / 0.05 - (stop / 0.05).round()).abs() < 1e-9, "stop {stop} not on grid");
}

#[test]
fn spec_lot_size_defers_to_instrument_config() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let inst = InstrumentConfig { lot_size: Some(25.0), ..InstrumentConfig::default() };
    let spec = InstrumentSpec {
        lot_size: 50.0,
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        Some(&inst),
    )
    .with_instrument(spec);

    // 100k at price 100 -> 1000 raw; explicit config lot 25 wins over
    // the spec's 50, and 1000 is already a multiple of 25.
    enter(&mut kernel, 0, 100.0);
    let size = kernel.position_snapshot().unwrap().size;
    assert!((size - 1000.0).abs() < 1e-9, "size {size}");
}

#[test]
fn entry_without_override_uses_config_stop() {
    let config =
        BacktestConfig { stop: StopConfig::Fixed { percent: 0.05 }, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );

    let events = kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
    assert!(matches!(events.as_slice(), [EngineEvent::Entered { .. }]));

    let snap = kernel.position_snapshot().unwrap();
    assert_eq!(snap.stop_price, Some(95.0));
    assert_eq!(snap.direction, Direction::Long);
    assert_eq!(snap.entry_idx, 0);
}

const ALGO_SEC: i64 = 1_000_000_000;

/// A TWAP accumulates: under the default netting policy only the first
/// slice can open, so these tests use the hedging policy.
fn twap_kernel() -> EngineKernel {
    make_kernel().with_position_policy(crate::portfolio::ledger::PositionPolicy::Independent)
}

#[test]
fn a_twap_slice_fills_on_the_step_it_is_released_on() {
    // The whole design depends on this: slices are submitted inside the
    // step, just before the market sweep, so they do not trail a step
    // behind their own schedule.
    let mut kernel = twap_kernel();
    kernel
        .submit_algo(
            OrderSide::Buy,
            QtySpec::Units(30.0),
            OrderKind::Market,
            TimeInForce::Gtc,
            "tw".to_string(),
            crate::execution::algos::ExecAlgorithm::Twap { slices: 3, interval_ns: ALGO_SEC },
            false,
            0,
            0,
        )
        .expect("valid schedule");

    let events = kernel.step(0, &bar(0, 100.0), StepInput::default());
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })),
        "the first slice must fill on its release step, got {events:?}"
    );
    assert!(kernel.is_in_position());
}

#[test]
fn twap_releases_one_slice_per_interval_through_the_kernel() {
    let mut kernel = twap_kernel();
    kernel
        .submit_algo(
            OrderSide::Buy,
            QtySpec::Units(30.0),
            OrderKind::Market,
            TimeInForce::Gtc,
            "tw".to_string(),
            crate::execution::algos::ExecAlgorithm::Twap { slices: 3, interval_ns: ALGO_SEC },
            false,
            0,
            0,
        )
        .expect("valid schedule");

    let mut fills = 0;
    for i in 0..3i64 {
        let ts = i * ALGO_SEC;
        let events = kernel.step(
            i as usize,
            &KernelBar {
                timestamp: ts,
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 1.0,
            },
            StepInput::default(),
        );
        fills += events.iter().filter(|e| matches!(e, EngineEvent::OrderFilled { .. })).count();
    }
    assert_eq!(fills, 3, "one slice per interval");
}

#[test]
fn a_tick_session_slices_on_time_not_on_event_count() {
    // Many prints inside one interval must not accelerate the schedule.
    // This is why slicing is timed rather than counted in bars: `idx` is an
    // event ordinal on a tick feed.
    let mut kernel = twap_kernel();
    kernel
        .submit_algo(
            OrderSide::Buy,
            QtySpec::Units(20.0),
            OrderKind::Market,
            TimeInForce::Gtc,
            "tw".to_string(),
            crate::execution::algos::ExecAlgorithm::Twap { slices: 2, interval_ns: ALGO_SEC },
            false,
            0,
            0,
        )
        .expect("valid schedule");

    let mut fills = 0;
    // Ten prints, all inside the first interval.
    for i in 0..10i64 {
        let tick =
            TradeTick { timestamp: i * 1_000_000, price: 100.0, size: 1.0, signed_size: 0.0 };
        let events = kernel.step_trade(i as usize, &tick, StepInput::default());
        fills += events.iter().filter(|e| matches!(e, EngineEvent::OrderFilled { .. })).count();
    }
    assert_eq!(fills, 1, "only the first slice is due inside one interval");
}

#[test]
fn cancelling_a_schedule_stops_further_slices() {
    let mut kernel = twap_kernel();
    let algo_id = kernel
        .submit_algo(
            OrderSide::Buy,
            QtySpec::Units(40.0),
            OrderKind::Market,
            TimeInForce::Gtc,
            "tw".to_string(),
            crate::execution::algos::ExecAlgorithm::Twap { slices: 4, interval_ns: ALGO_SEC },
            false,
            0,
            0,
        )
        .expect("valid schedule");

    kernel.step(0, &bar(0, 100.0), StepInput::default());
    assert!(kernel.cancel_algo(algo_id, 1));

    let mut later_fills = 0;
    for i in 1..4i64 {
        let events = kernel.step(
            i as usize,
            &KernelBar {
                timestamp: i * ALGO_SEC,
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 1.0,
            },
            StepInput::default(),
        );
        later_fills +=
            events.iter().filter(|e| matches!(e, EngineEvent::OrderFilled { .. })).count();
    }
    assert_eq!(later_fills, 0, "a cancelled schedule releases nothing further");
}

#[test]
fn a_schedule_refuses_capital_fraction_sizing() {
    let mut kernel = twap_kernel();
    let result = kernel.submit_algo(
        OrderSide::Buy,
        QtySpec::CapitalFrac(0.5),
        OrderKind::Market,
        TimeInForce::Gtc,
        "tw".to_string(),
        crate::execution::algos::ExecAlgorithm::Twap { slices: 2, interval_ns: ALGO_SEC },
        false,
        0,
        0,
    );
    assert!(result.is_err(), "each slice would size against a different account");
}

fn expiring_kernel(spec: crate::instruments::InstrumentSpec) -> EngineKernel {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "OPT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec)
}

#[test]
fn settlement_is_free_by_default() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};
    let mut kernel = expiring_kernel(InstrumentSpec {
        expiration_ns: Some(5),
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    });
    enter(&mut kernel, 0, 100.0);
    let events = kernel.step(5, &bar(5, 110.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => assert_eq!(trade.fees, 0.0),
        other => panic!("expected settlement, got {other:?}"),
    }
}

#[test]
fn a_settlement_fee_is_charged_on_the_settled_notional() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};
    let mut kernel = expiring_kernel(InstrumentSpec {
        expiration_ns: Some(5),
        settlement_fee: 0.01,
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    });
    enter(&mut kernel, 0, 100.0);
    let events = kernel.step(5, &bar(5, 110.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            // 1% of the settled notional, not of the entry price.
            let expected = 110.0 * trade.size * 0.01;
            assert!(
                (trade.fees - expected).abs() < 1e-9,
                "fees {} should be {expected}",
                trade.fees
            );
        }
        other => panic!("expected settlement, got {other:?}"),
    }
}

#[test]
fn an_option_settles_at_its_own_close_without_an_underlying() {
    use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};
    let mut kernel = expiring_kernel(InstrumentSpec {
        expiration_ns: Some(5),
        ..InstrumentSpec::new(
            "CE",
            InstrumentKind::Option {
                underlying: None,
                strike: 100.0,
                right: OptionRight::Call,
                binary: false,
            },
        )
    });
    enter(&mut kernel, 0, 7.0);
    // No underlying supplied: the contract's own close is all we know.
    let events = kernel.step(5, &bar(5, 9.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert!((trade.exit_price - 9.0).abs() < 1e-9);
        }
        other => panic!("expected settlement, got {other:?}"),
    }
}

#[test]
fn an_option_settles_to_intrinsic_against_a_supplied_underlying() {
    use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};
    let mut kernel = expiring_kernel(InstrumentSpec {
        expiration_ns: Some(5),
        ..InstrumentSpec::new(
            "CE",
            InstrumentKind::Option {
                underlying: None,
                strike: 100.0,
                right: OptionRight::Call,
                binary: false,
            },
        )
    });
    enter(&mut kernel, 0, 7.0);
    // The underlying sits at 112, so a 100 call is worth 12 — regardless of
    // where the option's own last print happened to be.
    kernel.set_underlying_price(Some(112.0));
    let events = kernel.step(5, &bar(5, 9.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert!(
                (trade.exit_price - 12.0).abs() < 1e-9,
                "expected intrinsic 12.0, got {}",
                trade.exit_price
            );
        }
        other => panic!("expected settlement, got {other:?}"),
    }
}

#[test]
fn an_out_of_the_money_option_settles_worthless() {
    use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};
    let mut kernel = expiring_kernel(InstrumentSpec {
        expiration_ns: Some(5),
        ..InstrumentSpec::new(
            "CE",
            InstrumentKind::Option {
                underlying: None,
                strike: 100.0,
                right: OptionRight::Call,
                binary: false,
            },
        )
    });
    enter(&mut kernel, 0, 7.0);
    kernel.set_underlying_price(Some(95.0));
    let events = kernel.step(5, &bar(5, 0.5), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert_eq!(trade.exit_price, 0.0, "a 100 call with spot at 95 expires worthless");
        }
        other => panic!("expected settlement, got {other:?}"),
    }
}

fn margin_kernel(liquidate: bool) -> EngineKernel {
    let config = BacktestConfig {
        fees: 0.0,
        liquidate_on_margin_call: liquidate,
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    )
    .with_account_mode(AccountMode::Margin { leverage: 50.0 })
}

#[test]
fn a_margin_call_only_halts_by_default() {
    let mut kernel = margin_kernel(false);
    enter(&mut kernel, 0, 100.0);
    let events = kernel.step(1, &bar(1, 40.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::MarginCall { .. })));
    assert!(kernel.is_in_position(), "the position rides on; the strategy decides what to do");
}

#[test]
fn liquidation_closes_positions_on_the_call() {
    let mut kernel = margin_kernel(true);
    enter(&mut kernel, 0, 100.0);
    let events = kernel.step(1, &bar(1, 40.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::MarginCall { .. })));
    match events.iter().find(|e| matches!(e, EngineEvent::Exited { .. })) {
        Some(EngineEvent::Exited { trade, .. }) => {
            assert_eq!(trade.exit_reason, ExitReason::Liquidation);
        }
        _ => panic!("expected a liquidation, got {events:?}"),
    }
    assert!(!kernel.is_in_position(), "the broker closed it");
}

#[test]
fn liquidation_pays_exit_costs_unlike_settlement() {
    // A forced close crosses the spread; settlement does not.
    let config =
        BacktestConfig { fees: 0.001, liquidate_on_margin_call: true, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    )
    .with_account_mode(AccountMode::Margin { leverage: 50.0 });

    enter(&mut kernel, 0, 100.0);
    let events = kernel.step(1, &bar(1, 40.0), StepInput::default());
    match events.iter().find(|e| matches!(e, EngineEvent::Exited { .. })) {
        Some(EngineEvent::Exited { trade, .. }) => {
            assert!(trade.fees > 0.0, "a liquidation is a real trade-out");
        }
        _ => panic!("expected a liquidation, got {events:?}"),
    }
}

#[test]
fn liquidation_does_not_fire_without_a_margin_call() {
    let mut kernel = margin_kernel(true);
    enter(&mut kernel, 0, 100.0);
    // A gentle move keeps equity above the requirement.
    let events = kernel.step(1, &bar(1, 101.0), StepInput::default());
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::Exited { .. })));
    assert!(kernel.is_in_position());
}

// -- order side is authoritative for opening ---------------------------------

/// Submit a market order and step one bar so it fills.
fn market_order(
    kernel: &mut EngineKernel,
    idx: usize,
    price: Price,
    side: OrderSide,
    reduce_only: bool,
) -> Vec<EngineEvent> {
    kernel.submit_order_full(
        side,
        QtySpec::CapitalFrac(0.5),
        OrderKind::Market,
        TimeInForce::Gtc,
        idx,
        idx as i64,
        format!("o{idx}"),
        None,
        None,
        false,
        reduce_only,
        false,
        None,
    );
    kernel.step(idx, &bar(idx as i64, price), StepInput::default())
}

#[test]
fn sell_order_opens_short_on_a_default_long_kernel() {
    // The bug this fixes: the sell was reinterpreted as a close, found no
    // position, and vanished without a trade or a rejection.
    let mut kernel = make_kernel();
    let events = market_order(&mut kernel, 0, 100.0, OrderSide::Sell, false);
    match events.iter().find(|e| matches!(e, EngineEvent::Entered { .. })) {
        Some(EngineEvent::Entered { direction, .. }) => {
            assert_eq!(*direction, Direction::Short, "the order's side decides");
        }
        _ => panic!("expected a short entry, got {events:?}"),
    }
    assert!(kernel.is_in_position());
}

#[test]
fn short_opened_by_order_orients_stop_above_and_target_below() {
    let config = BacktestConfig {
        stop: StopConfig::Fixed { percent: 0.05 },
        target: TargetConfig::Fixed { percent: 0.10 },
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );
    market_order(&mut kernel, 0, 100.0, OrderSide::Sell, false);
    let snap = kernel.position_snapshot().expect("a short position");
    assert_eq!(snap.direction, Direction::Short);
    let stop = snap.stop_price.expect("a stop");
    let target = snap.target_price.expect("a target");
    assert!(stop > snap.entry_price, "a short's stop sits ABOVE entry: {stop}");
    assert!(target < snap.entry_price, "a short's target sits BELOW entry: {target}");
}

#[test]
fn a_short_opened_by_order_profits_when_price_falls() {
    let mut kernel = make_kernel();
    market_order(&mut kernel, 0, 100.0, OrderSide::Sell, false);
    // Buy back lower: the close is the opposing side while in position.
    let events = market_order(&mut kernel, 1, 90.0, OrderSide::Buy, false);
    match events.iter().find(|e| matches!(e, EngineEvent::Exited { .. })) {
        Some(EngineEvent::Exited { trade, .. }) => {
            assert!(trade.pnl > 0.0, "short profits on a fall, got {}", trade.pnl);
        }
        _ => panic!("expected the buy to close the short, got {events:?}"),
    }
    assert!(!kernel.is_in_position());
}

#[test]
fn an_opposing_order_still_closes_rather_than_reversing() {
    // Bracket legs and take-profit orders depend on this.
    let mut kernel = make_kernel();
    enter(&mut kernel, 0, 100.0);
    let events = market_order(&mut kernel, 1, 110.0, OrderSide::Sell, false);
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::Exited { .. })),
        "expected a close, got {events:?}"
    );
    assert!(!kernel.is_in_position(), "it must not reverse into a short");
}

#[test]
fn a_reduce_only_order_never_opens_and_is_counted() {
    let mut kernel = make_kernel();
    let events = market_order(&mut kernel, 0, 100.0, OrderSide::Sell, true);
    assert!(!kernel.is_in_position(), "reduce-only must never open");
    match events.iter().find(|e| matches!(e, EngineEvent::OrderRejected { .. })) {
        Some(EngineEvent::OrderRejected { reason, .. }) => {
            assert_eq!(*reason, "reduce_only");
        }
        _ => panic!("expected a rejection, got {events:?}"),
    }
    assert_eq!(kernel.rejected_entries(), 1, "a refusal must be observable");
}

/// Submit a market order for an explicit unit count and step one bar so it
/// fills.
fn sized_order(
    kernel: &mut EngineKernel,
    idx: usize,
    price: Price,
    side: OrderSide,
    units: f64,
) -> Vec<EngineEvent> {
    kernel.submit_order_full(
        side,
        QtySpec::Units(units),
        OrderKind::Market,
        TimeInForce::Gtc,
        idx,
        idx as i64,
        format!("o{idx}"),
        None,
        None,
        false,
        false,
        false,
        None,
    );
    kernel.step(idx, &bar(idx as i64, price), StepInput::default())
}

#[test]
fn a_closing_order_reduces_by_the_size_it_asks_for() {
    // The bug this fixes: the close ignored the order's own size and sold
    // the whole position, so a one-lot trim of eleven held units flattened
    // the book -- ten units of exposure a venue would still have been
    // holding.
    let mut kernel = make_kernel();
    sized_order(&mut kernel, 0, 100.0, OrderSide::Buy, 11.0);
    assert_eq!(kernel.position_snapshot().expect("a position").size, 11.0);

    let events = sized_order(&mut kernel, 1, 110.0, OrderSide::Sell, 1.0);

    match events.iter().find(|e| matches!(e, EngineEvent::OrderFilled { .. })) {
        Some(EngineEvent::OrderFilled { size, leaves, .. }) => {
            assert_eq!(*size, 1.0, "it asked for one unit");
            assert_eq!(*leaves, 0.0, "and it got all of what it asked for");
        }
        _ => panic!("expected a fill, got {events:?}"),
    }
    assert!(
        !events.iter().any(|e| matches!(e, EngineEvent::Exited { .. })),
        "the position survives a partial reduction: {events:?}"
    );
    assert_eq!(kernel.position_snapshot().expect("ten units left").size, 10.0);
}

#[test]
fn a_closing_order_larger_than_the_position_takes_what_is_there() {
    let mut kernel = make_kernel();
    sized_order(&mut kernel, 0, 100.0, OrderSide::Buy, 11.0);

    let events = sized_order(&mut kernel, 1, 110.0, OrderSide::Sell, 50.0);

    match events.iter().find(|e| matches!(e, EngineEvent::OrderFilled { .. })) {
        Some(EngineEvent::OrderFilled { size, leaves, .. }) => {
            assert_eq!(*size, 11.0);
            // A reduction can never take more than is held, so nothing is
            // left working to take the rest.
            assert_eq!(*leaves, 0.0);
        }
        _ => panic!("expected a fill, got {events:?}"),
    }
    assert!(!kernel.is_in_position());
}

#[test]
fn a_close_all_order_names_no_size_and_takes_the_whole_position() {
    let mut kernel = make_kernel();
    sized_order(&mut kernel, 0, 100.0, OrderSide::Buy, 11.0);
    kernel.submit_order_full(
        OrderSide::Sell,
        QtySpec::FullPosition,
        OrderKind::Market,
        TimeInForce::Gtc,
        1,
        1,
        "close-all".to_string(),
        None,
        None,
        false,
        true,
        false,
        None,
    );

    let events = kernel.step(1, &bar(1, 110.0), StepInput::default());

    match events.iter().find(|e| matches!(e, EngineEvent::OrderFilled { .. })) {
        Some(EngineEvent::OrderFilled { size, .. }) => assert_eq!(*size, 11.0),
        _ => panic!("expected a fill, got {events:?}"),
    }
    assert!(!kernel.is_in_position());
}

#[test]
fn a_leg_can_flip_side_within_one_run() {
    // The whole point: long, flat, then short on the same kernel.
    let mut kernel = make_kernel();
    market_order(&mut kernel, 0, 100.0, OrderSide::Buy, false);
    assert_eq!(kernel.position_snapshot().expect("a position").direction, Direction::Long);
    market_order(&mut kernel, 1, 110.0, OrderSide::Sell, false);
    assert!(!kernel.is_in_position(), "flat between the two sides");
    market_order(&mut kernel, 2, 110.0, OrderSide::Sell, false);
    assert_eq!(
        kernel.position_snapshot().expect("a position").direction,
        Direction::Short,
        "the same leg reopened on the other side"
    );
}

fn adoption_kernel(mode: AccountMode) -> EngineKernel {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    )
    .with_account_mode(mode)
}

#[test]
fn fully_funded_margin_adoption_locks_the_cost_basis() {
    // Margin funds an open by LOCKING the notional, not by debiting the
    // balance (see `open_at`). Adoption must fund the same way: margin
    // equity is `cash + unrealized`, with no position-value term, so a
    // cash-style debit would never be offset and would understate equity by
    // the cost basis for the entire run.
    let mut kernel = adoption_kernel(AccountMode::Margin { leverage: 1.0 });
    kernel.set_cash(100_000.0);

    kernel.adopt_position(0, 90.0, 100.0).expect("fully funded adoption must be allowed");

    assert_eq!(kernel.locked_margin(), 9_000.0, "the whole notional locks at leverage 1.0");
    assert_eq!(kernel.cash(), 100_000.0, "margin must not debit the balance");
    // Priced at the adoption price the holding is worth exactly what it cost,
    // so equity is back to the starting capital.
    assert_eq!(kernel.equity(90.0), 100_000.0);
    assert_eq!(kernel.free_capital(), 91_000.0, "free capital drops by the cost basis");
}

#[test]
fn cash_and_fully_funded_margin_adoption_agree() {
    // A fully funded book is economically identical to cash for a long
    // holding, so the two modes must report the same numbers. Divergence
    // means the funding arm and the mode's equity formula disagree.
    let mut cash = adoption_kernel(AccountMode::Cash);
    cash.set_cash(100_000.0);
    cash.adopt_position(0, 90.0, 100.0).unwrap();

    let mut margin = adoption_kernel(AccountMode::Margin { leverage: 1.0 });
    margin.set_cash(100_000.0);
    margin.adopt_position(0, 90.0, 100.0).unwrap();

    for close in [90.0, 95.0, 85.0] {
        assert_eq!(cash.equity(close), margin.equity(close), "equity must agree at close {close}");
    }
    assert_eq!(cash.free_capital(), margin.free_capital());
}

#[test]
fn short_adoption_stays_refused_by_construction() {
    // Ratified deferral (2026-08-06): seeding a backtest with an existing
    // SHORT position is not supported — the broker's posted collateral is
    // not derivable from quantity x average price, and the cash arm of
    // adoption debits price*size, which is wrong for a short. The API
    // encodes the deferral structurally: there is no direction parameter,
    // and a negative size (the only way to express a short here) is
    // refused. If shorts are ever adopted, this test must be replaced by
    // one that states the posted-margin convention.
    let mut kernel = adoption_kernel(AccountMode::Cash);
    kernel.set_cash(100_000.0);
    assert!(kernel.adopt_position(0, 90.0, -100.0).is_err());
    assert!(kernel.adopt_position(0, 90.0, 0.0).is_err());
}

#[test]
fn adoption_refuses_nan_and_infinite_inputs() {
    // The guard is written `!(price > 0.0)`, not `price <= 0.0`, precisely so
    // that NaN fails it: NaN compares false to everything, so the negated form
    // rejects it while the "simpler" form would wave it through.
    //
    // A NaN price here is not a cosmetic problem. It would be adopted as the
    // position's cost basis, and cash, equity, unrealized P&L and every
    // drawdown figure computed from it would all become NaN -- silently, with
    // no error anywhere. Clippy's neg_cmp_op_on_partial_ord lint suggests
    // exactly that rewrite, so this test is what stops someone accepting it.
    let mut kernel = adoption_kernel(AccountMode::Cash);
    kernel.set_cash(100_000.0);

    for (price, size, label) in [
        (f64::NAN, 100.0, "NaN price"),
        (90.0, f64::NAN, "NaN size"),
        (f64::INFINITY, 100.0, "infinite price"),
        (0.0, 100.0, "zero price"),
        (-1.0, 100.0, "negative price"),
        (90.0, 0.0, "zero size"),
    ] {
        let result = kernel.adopt_position(0, price, size);
        if label == "infinite price" {
            // Infinity passes the positivity guard; it is refused downstream by
            // the cash check rather than here. Pinned so the boundary is known.
            continue;
        }
        assert!(result.is_err(), "{label} must be refused, got {result:?}");
    }
}

#[test]
fn leveraged_adoption_is_refused_not_guessed() {
    // Above leverage 1.0 the broker's posted margin genuinely is not
    // derivable from quantity and average price. Guessing would misstate
    // free capital, which gates every later entry.
    let mut kernel = adoption_kernel(AccountMode::Margin { leverage: 2.0 });
    kernel.set_cash(100_000.0);

    let err = kernel.adopt_position(0, 90.0, 100.0).unwrap_err();
    assert!(err.contains("fully funded"), "expected a leverage refusal, got: {err}");
}

/// Under NextBarOpen a streaming caller sees the `Entered` event on the
/// step AFTER the signal, priced at that bar's open — never on the signal
/// bar itself.
#[test]
fn next_bar_open_defers_the_entered_event_to_the_next_step() {
    let config =
        BacktestConfig { fill_timing: Some(FillTiming::NextBarOpen), ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let fill_price = FillPrice::for_timing(config.resolved_fill_timing());
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        fill_price,
        "TEST".to_string(),
        Direction::Long,
        None,
    );

    // Signal bar: nothing fills, nothing opens.
    let events = kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
    assert!(events.is_empty(), "the signal bar must not fill: {events:?}");
    assert!(!kernel.is_in_position());

    // Next bar: the deferred entry fills at THIS bar's open, before this
    // bar's own signals could have been seen.
    let mut fill_bar = bar(1, 110.0);
    fill_bar.open = 104.0;
    let events = kernel.step(1, &fill_bar, StepInput::default());
    assert!(
        matches!(events.as_slice(), [EngineEvent::Entered { idx: 1, price, .. }] if *price == 104.0),
        "expected a deferred entry at open, got {events:?}"
    );
    assert!(kernel.is_in_position());

    // Exit signal defers the same way.
    let events = kernel.step(2, &bar(2, 111.0), StepInput { exit: true, ..StepInput::default() });
    assert!(events.is_empty(), "the exit-signal bar must not fill: {events:?}");
    let mut exit_bar = bar(3, 108.0);
    exit_bar.open = 109.0;
    let events = kernel.step(3, &exit_bar, StepInput::default());
    assert!(
        matches!(events.as_slice(), [EngineEvent::Exited { trade, .. }] if trade.exit_price == 109.0),
        "expected a deferred exit at open, got {events:?}"
    );
}

/// The deferred fill happens before the bar's own signal processing: a
/// same-step exit signal on the fill bar defers again rather than closing
/// the position that just opened.
#[test]
fn deferred_fill_precedes_the_fill_bars_own_signals() {
    let config =
        BacktestConfig { fill_timing: Some(FillTiming::NextBarOpen), ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Open,
        "TEST".to_string(),
        Direction::Long,
        None,
    );

    kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
    // Fill bar carries its own exit signal: the entry fills here, the exit
    // is an intent for the NEXT bar.
    let events = kernel.step(1, &bar(1, 101.0), StepInput { exit: true, ..StepInput::default() });
    assert!(
        matches!(events.as_slice(), [EngineEvent::Entered { .. }]),
        "only the entry may fill on this bar: {events:?}"
    );
    assert!(kernel.is_in_position(), "the exit must not act on the bar that signaled it");

    let events = kernel.step(2, &bar(2, 102.0), StepInput::default());
    assert!(
        matches!(events.as_slice(), [EngineEvent::Exited { .. }]),
        "the deferred exit fills one bar later: {events:?}"
    );
}

/// Under NextBarOpen an order-API market order submitted while observing
/// bar i is unreachable by bar i's sweep: it is acknowledged there and
/// fills at bar i+1's open — the same contract as a deferred signal.
#[test]
fn next_bar_open_market_order_fills_at_the_next_bars_open() {
    for tif in [TimeInForce::Gtc, TimeInForce::Ioc] {
        let config = BacktestConfig {
            fill_timing: Some(FillTiming::NextBarOpen),
            fees: 0.0,
            ..BacktestConfig::default()
        };
        let fee_model = config.fee_model();
        let mut kernel = EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Open,
            "TEST".to_string(),
            Direction::Long,
            None,
        );

        // Strategy observed bar 0 and placed a market order.
        kernel.submit_order(
            OrderSide::Buy,
            QtySpec::Units(10.0),
            OrderKind::Market,
            tif,
            0,
            0,
            "mkt-1".to_string(),
            None,
            None,
        );

        // Bar 0's step acknowledges but must not fill.
        let events = kernel.step(0, &bar(0, 100.0), StepInput::default());
        assert!(
            matches!(events.as_slice(), [EngineEvent::OrderAccepted { idx: 0, .. }]),
            "tif {tif:?}: submission bar acknowledges only, got {events:?}"
        );
        assert!(!kernel.is_in_position(), "tif {tif:?}: nothing may fill on the submission bar");

        // Bar 1: fills at THIS bar's open.
        let mut fill_bar = bar(1, 110.0);
        fill_bar.open = 104.0;
        let events = kernel.step(1, &fill_bar, StepInput::default());
        let filled = events.iter().any(
            |e| matches!(e, EngineEvent::OrderFilled { idx: 1, price, .. } if *price == 104.0),
        );
        assert!(filled, "tif {tif:?}: expected a fill at 104.0 on bar 1, got {events:?}");
        assert!(kernel.is_in_position());
    }
}

/// Settle one round trip, taking the exit off in the given pieces, and
/// report the cash it left behind.
fn cash_after_unwinding_in(pieces: &[f64]) -> f64 {
    let mut kernel = make_kernel();
    kernel.set_cash(10_000.0);
    let entered = kernel.open_at(
        0,
        &bar(0, 100.0),
        Direction::Long,
        100.0,
        None,
        Some(10.0),
        0.0,
        None,
        None,
        FillTerms::WHOLE,
    );
    assert!(matches!(entered, Some(OpenResult { event: EngineEvent::Entered { .. }, .. })));
    for cap in pieces {
        kernel.reduce_at(1, &bar(1, 110.0), 0, 110.0, ExitReason::Signal, *cap, None);
    }
    assert_eq!(kernel.ledger.open_count(), 0, "the position must end flat");
    kernel.cash()
}

#[test]
fn the_number_of_fills_an_exit_takes_does_not_change_the_account() {
    // The same ten units off at the same price, once in a single fill and
    // once split in two. Settling the round trip on the closing fill --
    // rather than each fill as it lands -- pays the first fill's proceeds a
    // second time and leaves the split kernel richer for nothing.
    let whole = cash_after_unwinding_in(&[10.0]);
    assert_eq!(cash_after_unwinding_in(&[4.0, 6.0]), whole);
    assert_eq!(cash_after_unwinding_in(&[1.0, 1.0, 8.0]), whole);
}

#[test]
fn each_closing_fill_reports_only_the_pnl_it_realized() {
    // A closing fill realizes its own units at its own price. Reporting the
    // round trip on the fill that goes flat would count the earlier fills a
    // second time -- the same error that once paid the account twice for
    // them.
    let mut kernel = make_kernel();
    kernel.set_cash(10_000.0);
    let entered = kernel.open_at(
        0,
        &bar(0, 100.0),
        Direction::Long,
        100.0,
        None,
        Some(10.0),
        0.0,
        None,
        None,
        FillTerms::WHOLE,
    );
    assert!(matches!(entered, Some(OpenResult { event: EngineEvent::Entered { .. }, .. })));

    let first = kernel.reduce_at(1, &bar(1, 110.0), 0, 110.0, ExitReason::Signal, 4.0, None);
    let second = kernel.reduce_at(1, &bar(1, 110.0), 0, 110.0, ExitReason::Signal, 6.0, None);
    let ReduceResult::Reduced { gross_realized: first_gross, fees: first_fees, .. } = &first else {
        panic!("expected a partial reduction, got {first:?}");
    };
    let ReduceResult::Closed { gross_realized: second_gross, fees: second_fees, event, .. } =
        &second
    else {
        panic!("expected a close, got {second:?}");
    };
    // Ten a unit gross on four units, then on six.
    assert!((first_gross - 40.0).abs() < 1e-9, "first {first_gross}");
    assert!((second_gross - 60.0).abs() < 1e-9, "second {second_gross}");

    let EngineEvent::Exited { trade, .. } = event else { panic!("expected an exit, got {event:?}") };
    // Netting each fill against its own fee accounts for the whole round
    // trip and nothing more.
    let fills = (first_gross - first_fees) + (second_gross - second_fees);
    let round_trip = trade.pnl + trade.entry_fees;
    assert!((fills - round_trip).abs() < 1e-9, "fills {fills} against round trip {round_trip}");
}

#[test]
fn a_position_closed_in_full_leaves_no_dust_behind() {
    // The bug this fixes: an entry filled in two pieces held
    // 0.03835 + 0.04381 = 0.08216000000000001 units, and selling the
    // 0.08216 it had bought left 1.4e-17 of a coin open. Nothing could
    // ever close that -- no order asks for a hundredth of a femto-lot --
    // so every later entry averaged into the same position and the run
    // reported one round trip that never ended in place of the forty-odd
    // it actually made.
    let inst = InstrumentConfig { lot_size: Some(0.00001), ..InstrumentConfig::default() };
    let config = BacktestConfig::default();
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "BTCUSDT".to_string(),
        Direction::Long,
        Some(&inst),
    );
    kernel.set_position_policy(PositionPolicy::NetAveraging);
    sized_order(&mut kernel, 0, 100.0, OrderSide::Buy, 0.03835);
    sized_order(&mut kernel, 1, 100.0, OrderSide::Buy, 0.04381);
    assert_eq!(
        kernel.position_snapshot().expect("a position").size,
        0.08216,
        "two fills on the grid hold a size on the grid"
    );

    let events = sized_order(&mut kernel, 2, 110.0, OrderSide::Sell, 0.08216);

    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::Exited { .. })),
        "selling all of it closes it, got {events:?}"
    );
    assert!(
        kernel.position_snapshot().is_none(),
        "nothing is left open: {:?}",
        kernel.position_snapshot().map(|p| p.size)
    );
}

#[test]
fn a_resumed_order_asks_for_the_remainder_the_venue_still_owes() {
    // The bug this fixes: 0.07841 units filled down to 0.06531 leave
    // 0.013099999999999987 -- 1309.9999999999986 lots -- and flooring that
    // asked for 0.01309, a lot less than the venue still owed. The order
    // never finished, and every later fill inherited the shortfall.
    let mut kernel = bounded_kernel_on_a_lot(PositionPolicy::NetAveraging, 0.00001);
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(0.07841),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "g".to_string(),
        None,
        None,
    );
    // A quarter of the bar's volume is all one aggressive order may take.
    kernel.step(1, &bar_with_volume(1, 100.0, 0.26126), StepInput::default());
    assert_eq!(order_status(&kernel, id), OrderStatus::PartiallyFilled);
    assert_eq!(kernel.position_snapshot().expect("a position").size, 0.06531);

    kernel.step(2, &bar_with_volume(2, 99.0, 400.0), StepInput::default());

    assert_eq!(order_status(&kernel, id), OrderStatus::Filled);
    assert_eq!(
        kernel.position_snapshot().expect("a position").size,
        0.07841,
        "the two fills add up to the size the order asked for"
    );
}
