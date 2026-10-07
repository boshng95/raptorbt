//! Partial fills: a typed order fills across prints, never past a print.
//!
//! Included from `kernel.rs` like `kernel_tests.rs`, so private kernel
//! items resolve.

use super::*;

fn partial_kernel() -> EngineKernel {
    let config = BacktestConfig { partial_fills: true, fees: 0.0, ..BacktestConfig::default() };
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

fn print(ts: i64, price: Price, size: f64) -> TradeTick {
    TradeTick { timestamp: ts, price, size, signed_size: 0.0, oi: 0.0 }
}

fn submit(
    kernel: &mut EngineKernel,
    side: OrderSide,
    qty: QtySpec,
    tif: TimeInForce,
    tag: &str,
) -> u64 {
    kernel.submit_order(side, qty, OrderKind::Market, tif, 0, 0, tag.into(), None, None)
}

fn fills(events: &[EngineEvent]) -> Vec<f64> {
    events
        .iter()
        .filter_map(|e| match e {
            EngineEvent::OrderFilled { size, .. } => Some(*size),
            _ => None,
        })
        .collect()
}

#[test]
fn an_opening_order_fills_across_prints_into_one_averaged_position() {
    let mut kernel = partial_kernel();
    let id = submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "o");
    let events = kernel.step_trade(1, &print(1, 100.0, 40.0), StepInput::default());
    assert_eq!(fills(&events), vec![40.0]);
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::PartiallyFilled);
    assert_eq!(kernel.order(id).unwrap().filled_qty, 40.0);
    // A second print of the SAME order is not "position_open": it adds.
    let events = kernel.step_trade(2, &print(2, 110.0, 60.0), StepInput::default());
    assert_eq!(fills(&events), vec![60.0]);
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::OrderRejected { .. })), "{events:?}");
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::Filled);
    let snap = kernel.position_snapshot().unwrap();
    assert!((snap.size - 100.0).abs() < 1e-9);
    // Size-weighted average: (40*100 + 60*110) / 100 = 106.
    assert!((snap.entry_price - 106.0).abs() < 1e-9, "{}", snap.entry_price);
    // Cash paid the two slices at their own prices.
    assert!((kernel.cash() - (100_000.0 - 4_000.0 - 6_600.0)).abs() < 1e-6);
    // A different order is still a second entry under netting.
    let other = submit(&mut kernel, OrderSide::Buy, QtySpec::Units(10.0), TimeInForce::Gtc, "p");
    let events = kernel.step_trade(3, &print(3, 110.0, 10.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderRejected { order_id, reason: "position_open", .. } if *order_id == other)), "{events:?}");
}

#[test]
fn a_derived_stop_follows_the_average_and_an_explicit_one_stays() {
    // Percent stop derived from the entry: shifts with the average.
    let config = BacktestConfig {
        partial_fills: true,
        fees: 0.0,
        stop: StopConfig::Fixed { percent: 0.10 },
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "T".into(),
        Direction::Long,
        None,
    );
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "o");
    kernel.step_trade(1, &print(1, 100.0, 50.0), StepInput::default());
    let stop_after_first = kernel.position_snapshot().unwrap().stop_price.unwrap();
    assert!((stop_after_first - 90.0).abs() < 1e-9);
    kernel.step_trade(2, &print(2, 120.0, 50.0), StepInput::default());
    let stop = kernel.position_snapshot().unwrap().stop_price.unwrap();
    // Average moved 100 -> 110, so the stop moved 90 -> 100.
    assert!((stop - 100.0).abs() < 1e-9, "{stop}");

    // Explicit stop attached to the order: untouched by the average.
    let mut kernel = partial_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Market,
        TimeInForce::Gtc,
        0,
        0,
        "e".into(),
        Some(80.0),
        None,
    );
    kernel.step_trade(1, &print(1, 100.0, 50.0), StepInput::default());
    kernel.step_trade(2, &print(2, 120.0, 50.0), StepInput::default());
    assert_eq!(kernel.position_snapshot().unwrap().stop_price, Some(80.0));
}

#[test]
fn a_closing_order_reduces_into_one_round_trip_and_never_reverses() {
    let mut kernel = partial_kernel();
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "o");
    kernel.step_trade(1, &print(1, 100.0, 100.0), StepInput::default());
    assert!(kernel.is_in_position());
    // Sell 150 against a 100 position: 40 prints, then 60 closes it; the
    // surplus 50 is canceled rather than opening a short.
    let id = submit(&mut kernel, OrderSide::Sell, QtySpec::Units(150.0), TimeInForce::Gtc, "x");
    let events = kernel.step_trade(2, &print(2, 105.0, 40.0), StepInput::default());
    assert_eq!(fills(&events), vec![40.0]);
    assert!((kernel.position_snapshot().unwrap().size - 60.0).abs() < 1e-9);
    assert!(
        !events.iter().any(|event| matches!(event, EngineEvent::Exited { .. })),
        "Nautilus reports one round trip when the position reaches flat: {events:?}"
    );
    let events = kernel.step_trade(3, &print(3, 106.0, 500.0), StepInput::default());
    assert_eq!(fills(&events), vec![60.0]);
    assert!(!kernel.is_in_position());
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::Filled);
    assert!((kernel.order(id).unwrap().filled_qty - 100.0).abs() < 1e-9);
    let trades: Vec<&Trade> = events
        .iter()
        .filter_map(|event| match event {
            EngineEvent::Exited { trade, .. } => Some(&**trade),
            _ => None,
        })
        .collect();
    assert_eq!(trades.len(), 1);
    assert!((trades[0].size - 100.0).abs() < 1e-9);
    assert!((trades[0].exit_price - 105.6).abs() < 1e-9);
    assert!((trades[0].pnl - 560.0).abs() < 1e-9, "got {}", trades[0].pnl);
    let events = kernel.step_trade(4, &print(4, 106.0, 500.0), StepInput::default());
    assert!(!kernel.is_in_position(), "the surplus must never reverse: {events:?}");
    // Cash: 100_000 - 10_000 + 40*105 + 60*106 = 100_560.
    assert!((kernel.cash() - 100_560.0).abs() < 1e-6, "{}", kernel.cash());
}

#[test]
fn ioc_keeps_one_slice_and_fok_takes_nothing_small() {
    let mut kernel = partial_kernel();
    let ioc = submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Ioc, "i");
    let events = kernel.step_trade(1, &print(1, 100.0, 30.0), StepInput::default());
    assert_eq!(fills(&events), vec![30.0]);
    assert!(events
        .iter()
        .any(|e| matches!(e, EngineEvent::OrderCanceled { order_id, .. } if *order_id == ioc)));
    assert_eq!(kernel.order(ioc).unwrap().status, OrderStatus::Canceled);
    assert!((kernel.order(ioc).unwrap().filled_qty - 30.0).abs() < 1e-9);
    assert!((kernel.position_snapshot().unwrap().size - 30.0).abs() < 1e-9);

    let mut kernel = partial_kernel();
    let fok = submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Fok, "f");
    let events = kernel.step_trade(1, &print(1, 100.0, 30.0), StepInput::default());
    assert!(fills(&events).is_empty(), "{events:?}");
    assert_eq!(kernel.order(fok).unwrap().status, OrderStatus::Canceled);
    assert!(!kernel.is_in_position());
    let mut kernel = partial_kernel();
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Fok, "g");
    let events = kernel.step_trade(1, &print(1, 100.0, 100.0), StepInput::default());
    assert_eq!(fills(&events), vec![100.0]);
}

#[test]
fn day_expiry_keeps_the_filled_part() {
    let mut kernel = partial_kernel();
    let day_ns = 86_400_000_000_000i64;
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Limit { price: 100.0 },
        TimeInForce::Day,
        0,
        0,
        "d".into(),
        None,
        None,
    );
    kernel.step_trade(1, &print(1, 100.0, 40.0), StepInput::default());
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::PartiallyFilled);
    let events = kernel.step_trade(2, &print(day_ns + 1, 100.0, 1.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderExpired { .. })), "{events:?}");
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::Expired);
    assert!((kernel.position_snapshot().unwrap().size - 40.0).abs() < 1e-9);
}

#[test]
fn whole_fills_when_the_flag_is_off_and_on_bars() {
    let mut kernel = zero_fee_kernel();
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "w");
    let events = kernel.step_trade(1, &print(1, 100.0, 5.0), StepInput::default());
    assert_eq!(fills(&events), vec![100.0], "flag off: whole");
    let mut kernel = partial_kernel();
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "b");
    let events = kernel.step(0, &bar(0, 100.0), StepInput::default());
    assert_eq!(fills(&events), vec![100.0], "a bar fills whole even with the flag on");
}

#[test]
fn an_order_in_flight_cannot_fill_before_its_latency_elapses() {
    let config =
        BacktestConfig { order_latency_ns: 250_000_000, fees: 0.0, ..BacktestConfig::default() };
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
    // A marketable limit placed at t=0 and a market order placed at t=0.
    let limit = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 100.0 },
        TimeInForce::Gtc,
        0,
        0,
        "l".into(),
        None,
        None,
    );
    let market = submit(&mut kernel, OrderSide::Buy, QtySpec::Units(10.0), TimeInForce::Gtc, "m");
    let _ = (limit, market);
    // 100 ms later: still in flight, nothing fills.
    let events = kernel.step_trade(1, &print(100_000_000, 100.0, 1_000.0), StepInput::default());
    assert!(fills(&events).is_empty(), "{events:?}");
    // 300 ms later: arrived, both fill on this print.
    let events = kernel.step_trade(2, &print(300_000_000, 100.0, 1_000.0), StepInput::default());
    assert_eq!(
        fills(&events).len(),
        1,
        "netting: the first fill opens, the second is refused: {events:?}"
    );
    assert!(kernel.is_in_position());
}

fn brokerage_kernel(fee_model: FeeModel) -> EngineKernel {
    let config = BacktestConfig { partial_fills: true, ..BacktestConfig::default() };
    let instrument =
        InstrumentConfig { currency_precision: Some(2), ..InstrumentConfig::default() };
    EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        Some(&instrument),
    )
}

fn commissions(events: &[EngineEvent]) -> Vec<f64> {
    events
        .iter()
        .filter_map(|e| match e {
            EngineEvent::OrderFilled { commission, .. } => Some(*commission),
            _ => None,
        })
        .collect()
}

/// IB US equities: $0.005 a share, $1 an order, at most 1% of its value.
fn ib_us() -> FeeModel {
    FeeModel::brokerage(0.0, 0.005, 1.0, 0.01)
}

#[test]
fn an_order_split_across_prints_pays_the_brokers_minimum_once() {
    // 100 shares is 50 cents at $0.005, so IB bills the $1 minimum -- once,
    // for the order. Billing each print would charge it twice.
    let mut kernel = brokerage_kernel(ib_us());
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "o");
    let first = kernel.step_trade(1, &print(1, 100.0, 40.0), StepInput::default());
    let second = kernel.step_trade(2, &print(2, 110.0, 60.0), StepInput::default());

    assert_eq!(commissions(&first), vec![1.0]);
    assert_eq!(commissions(&second), vec![0.0]);
    assert!((kernel.cash() - (100_000.0 - 4_000.0 - 6_600.0 - 1.0)).abs() < 1e-9);
}

#[test]
fn the_fills_of_an_order_together_pay_what_the_whole_order_costs() {
    // 300 shares cost $1.50 per share-rate: the first 100 pay the $1 floor
    // and the next 200 the 50 cents the order owes past it.
    let mut kernel = brokerage_kernel(ib_us());
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(300.0), TimeInForce::Gtc, "o");
    let first = kernel.step_trade(1, &print(1, 100.0, 100.0), StepInput::default());
    let second = kernel.step_trade(2, &print(2, 100.0, 200.0), StepInput::default());

    assert_eq!(commissions(&first), vec![1.0]);
    assert_eq!(commissions(&second), vec![0.5]);
    let whole = ib_us().calculate(100.0, 300.0, Direction::Long);
    assert!((1.0 + 0.5 - whole).abs() < 1e-12, "{whole}");
}

#[test]
fn a_percentage_schedule_with_a_floor_bills_the_order_on_its_whole_value() {
    // IB Australia: 0.088% with an AUD 6.60 floor. 40 at 100 alone is 3.52,
    // so the first print pays the floor; the order's 10,600 owes 9.33, and
    // the second print pays the 2.73 left.
    let asx = FeeModel::brokerage(0.00088, 0.0, 6.60, 0.0);
    let mut kernel = brokerage_kernel(asx);
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "o");
    let first = kernel.step_trade(1, &print(1, 100.0, 40.0), StepInput::default());
    let second = kernel.step_trade(2, &print(2, 110.0, 60.0), StepInput::default());

    assert_eq!(commissions(&first), vec![6.6]);
    assert_eq!(commissions(&second), vec![2.73]);
}

#[test]
fn a_closing_order_split_across_prints_pays_the_minimum_once() {
    let mut kernel = brokerage_kernel(ib_us());
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "o");
    kernel.step_trade(1, &print(1, 100.0, 100.0), StepInput::default());
    submit(&mut kernel, OrderSide::Sell, QtySpec::Units(100.0), TimeInForce::Gtc, "c");
    let first = kernel.step_trade(2, &print(2, 110.0, 40.0), StepInput::default());
    let second = kernel.step_trade(3, &print(3, 110.0, 60.0), StepInput::default());

    assert_eq!(commissions(&first), vec![1.0]);
    assert_eq!(commissions(&second), vec![0.0]);
    assert!(kernel.position_snapshot().is_none());
    // Two orders, two minimums: one to buy, one to sell.
    assert!((kernel.cash() - (100_000.0 + 1_000.0 - 2.0)).abs() < 1e-9);
}

#[test]
fn separate_orders_each_pay_their_own_minimum() {
    let mut kernel = brokerage_kernel(ib_us());
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(40.0), TimeInForce::Gtc, "a");
    let first = kernel.step_trade(1, &print(1, 100.0, 40.0), StepInput::default());
    submit(&mut kernel, OrderSide::Sell, QtySpec::Units(40.0), TimeInForce::Gtc, "b");
    let second = kernel.step_trade(2, &print(2, 100.0, 40.0), StepInput::default());

    assert_eq!(commissions(&first), vec![1.0]);
    assert_eq!(commissions(&second), vec![1.0]);
}
