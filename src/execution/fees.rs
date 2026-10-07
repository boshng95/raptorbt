//! Fee calculation models.

use serde::{Deserialize, Serialize};

use crate::core::decimals::decimal_product;
use crate::core::types::{Direction, Price};
use crate::execution::indian_costs::{self, FeeBreakdown, Segment};

/// What one order's fills have been billed so far.
///
/// A broker bills an order, not each print it took. IB's floor and notional
/// cap apply once to the whole order: an order for 300 shares that fills as
/// 100 and 200 pays one $1 minimum, not two. Pricing each fill on its own
/// size would charge the floor once per fill -- an amount the broker never
/// takes, and one the strategy sized its order without.
///
/// So an order carries the schedule's charge on everything it has filled,
/// before the floor and cap are applied, and what its fills were actually
/// charged. Each further fill pays the order's charge on its new total less
/// what is already paid. An order that fills in one piece pays exactly what
/// [`FeeModel::calculate`] charges for its size.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct OrderBilling {
    /// The schedule's base charge on every unit filled, before the floor and
    /// the cap.
    pub base: f64,
    /// The notional cap on every unit filled; zero for an uncapped schedule.
    pub cap: f64,
    /// What the order's fills were charged, each already settled money.
    pub charged: f64,
}

/// Fee model for calculating transaction costs.
#[derive(Debug, Clone)]
pub enum FeeModel {
    /// No fees.
    None,
    /// Fixed percentage of trade value.
    Percentage(f64),
    /// Fixed fee per trade.
    Fixed(f64),
    /// Per-share/contract fee.
    PerShare(f64),
    /// Tiered fee structure based on trade value.
    Tiered(Vec<(f64, f64)>), // (threshold, rate)
    /// Custom fee function (stored as percentage for simplicity).
    Custom { base: f64, per_share: f64 },
    /// Brokerage schedule with optional per-order floor and notional cap.
    ///
    /// A non-zero `per_share` replaces the percentage component. This covers
    /// brokers such as IB, where US equities are charged per share while ASX
    /// equities are charged as a percentage of notional. The minimum and cap
    /// are applied to either base calculation.
    Brokerage { percentage: f64, per_share: f64, minimum: f64, max_percentage: f64 },
    /// Itemized Indian regulatory costs for a market segment.
    ///
    /// Unlike the other variants this produces a per-component breakdown, so
    /// the equity curve and the reported costs are the same numbers.
    Indian { segment: Segment },
}

impl Default for FeeModel {
    fn default() -> Self {
        FeeModel::Percentage(0.001) // 0.1% default
    }
}

impl FeeModel {
    /// Create a new percentage fee model.
    pub fn percentage(rate: f64) -> Self {
        FeeModel::Percentage(rate)
    }

    /// Create a new fixed fee model.
    pub fn fixed(amount: f64) -> Self {
        FeeModel::Fixed(amount)
    }

    /// Create a new per-share fee model.
    pub fn per_share(rate: f64) -> Self {
        FeeModel::PerShare(rate)
    }

    /// Create a brokerage schedule with an optional minimum and notional cap.
    pub fn brokerage(percentage: f64, per_share: f64, minimum: f64, max_percentage: f64) -> Self {
        FeeModel::Brokerage { percentage, per_share, minimum, max_percentage }
    }

    /// Calculate fee for a trade.
    ///
    /// # Arguments
    /// * `price` - Trade price
    /// * `size` - Position size (shares/contracts)
    /// * `direction` - Trade direction (for asymmetric fees if needed)
    ///
    /// # Returns
    /// Fee amount
    pub fn calculate(&self, price: Price, size: f64, _direction: Direction) -> f64 {
        let trade_value = price * size.abs();

        match self {
            FeeModel::None => 0.0,
            FeeModel::Percentage(rate) => rate_on_notional(price, size.abs(), *rate),
            FeeModel::Fixed(amount) => *amount,
            FeeModel::PerShare(rate) => size.abs() * rate,
            FeeModel::Tiered(tiers) => {
                // Find applicable tier
                let mut applicable_rate = 0.0;
                for (threshold, rate) in tiers {
                    if trade_value >= *threshold {
                        applicable_rate = *rate;
                    } else {
                        break;
                    }
                }
                rate_on_notional(price, size.abs(), applicable_rate)
            }
            FeeModel::Custom { base, per_share } => base + size.abs() * per_share,
            FeeModel::Brokerage { .. } => {
                let (base, cap) = self.brokerage_parts(price, size);
                self.brokerage_charge(base, cap)
            }
            FeeModel::Indian { segment } => {
                indian_costs::calculate_side(*segment, trade_value, _direction, true).total()
            }
        }
    }

    /// Whether the schedule charges an order once rather than each fill.
    ///
    /// Only a floor or a notional cap makes the difference: without them the
    /// schedule is linear in size, and billing the order or its fills comes
    /// to the same charge.
    pub fn bills_per_order(&self) -> bool {
        matches!(
            self,
            FeeModel::Brokerage { minimum, max_percentage, .. }
                if *minimum > 0.0 || *max_percentage > 0.0
        )
    }

    /// The order's billing once a fill of `size` at `price` is added to it.
    ///
    /// The charge itself is left to the caller, which settles it in the
    /// account's currency: [`Self::order_charge`] of the result, less what
    /// the order already paid. `None` for a schedule billed per fill.
    pub fn accrue(&self, billed: &OrderBilling, price: Price, size: f64) -> Option<OrderBilling> {
        if !self.bills_per_order() {
            return None;
        }
        let (base, cap) = self.brokerage_parts(price, size);
        Some(OrderBilling { base: billed.base + base, cap: billed.cap + cap, ..*billed })
    }

    /// What the schedule charges an order with this billing, unsettled.
    pub fn order_charge(&self, billed: &OrderBilling) -> f64 {
        self.brokerage_charge(billed.base, billed.cap)
    }

    /// A brokerage fill's base charge and notional cap, before either the
    /// floor or the cap is applied. Zero for any other schedule.
    fn brokerage_parts(&self, price: Price, size: f64) -> (f64, f64) {
        let FeeModel::Brokerage { percentage, per_share, max_percentage, .. } = self else {
            return (0.0, 0.0);
        };
        let base = if *per_share > 0.0 {
            // Shares and the published rate are decimals. Recover their
            // exact product before currency quantization: for 203 shares at
            // $0.005, binary multiplication yields 1.0150000000000001 and
            // rounds a cent too high.
            decimal_product(&[size.abs(), *per_share]).unwrap_or_else(|| size.abs() * per_share)
        } else {
            rate_on_notional(price, size.abs(), *percentage)
        };
        let cap = if *max_percentage > 0.0 {
            rate_on_notional(price, size.abs(), *max_percentage)
        } else {
            0.0
        };
        (base, cap)
    }

    /// The floor and the cap applied to a brokerage base charge.
    fn brokerage_charge(&self, base: f64, cap: f64) -> f64 {
        let FeeModel::Brokerage { minimum, max_percentage, .. } = self else {
            return base;
        };
        let mut fee = base;
        if *minimum > 0.0 {
            fee = fee.max(*minimum);
        }
        if *max_percentage > 0.0 {
            fee = fee.min(cap);
        }
        fee
    }

    /// Itemized costs for one side of a trade.
    ///
    /// Returns `None` for the flat models, which have no component structure.
    /// Unlike [`FeeModel::calculate`], this distinguishes entry from exit,
    /// which matters because STT lands on the sell leg and stamp duty on the
    /// buy leg.
    pub fn breakdown(
        &self,
        price: Price,
        size: f64,
        direction: Direction,
        is_entry: bool,
    ) -> Option<FeeBreakdown> {
        match self {
            FeeModel::Indian { segment } => Some(indian_costs::calculate_side(
                *segment,
                price * size.abs(),
                direction,
                is_entry,
            )),
            _ => None,
        }
    }

    /// Fee for one side, honoring entry/exit asymmetry where the model has any.
    pub fn calculate_side(
        &self,
        price: Price,
        size: f64,
        direction: Direction,
        is_entry: bool,
    ) -> f64 {
        match self.breakdown(price, size, direction, is_entry) {
            Some(b) => b.total(),
            None => self.calculate(price, size, direction),
        }
    }

    /// Create an itemized Indian cost model for a segment.
    pub fn indian(segment: Segment) -> Self {
        FeeModel::Indian { segment }
    }

    /// Calculate round-trip fees (entry + exit).
    pub fn round_trip(
        &self,
        entry_price: Price,
        exit_price: Price,
        size: f64,
        direction: Direction,
    ) -> f64 {
        self.calculate(entry_price, size, direction) + self.calculate(exit_price, size, direction)
    }
}

/// `price * size * rate`, computed as the decimal it is.
///
/// A percentage fee is a rate applied to a notional, and the notional itself
/// is a product. All three factors are decimals -- a quoted price, a size on
/// a lot grid, a published rate -- and the venue multiplies them as such.
/// Evaluating the same expression in binary rounds three times, and can land
/// on the far side of a tie at the settlement currency's precision, which is
/// where a commission is quantized. [`decimal_product`] forms the product
/// exactly and converts it once; see its notes for the two real cases that
/// straddle a tie in opposite directions.
///
/// When a factor is not a short decimal there is no decimal product to
/// recover, and the best available answer is the correctly-rounded product
/// of the floats themselves. `mul_add` is specified to round once, so
/// `a.mul_add(b, -(a * b))` is the exact error of the first product, and
/// carrying that error through the second multiplication gives it without
/// pulling in a bignum type.
#[inline]
fn rate_on_notional(price: Price, size: f64, rate: f64) -> f64 {
    if let Some(exact) = decimal_product(&[price, size, rate]) {
        return exact;
    }
    let notional = price * size;
    let notional_err = price.mul_add(size, -notional);
    let fee = notional * rate;
    fee + notional_err.mul_add(rate, notional.mul_add(rate, -fee))
}

/// Broker-specific fee configurations.
pub struct BrokerFees;

impl BrokerFees {
    /// Interactive Brokers tiered pricing (approximate).
    pub fn interactive_brokers() -> FeeModel {
        FeeModel::Custom { base: 1.0, per_share: 0.005 }
    }

    /// Zero commission broker (like Robinhood).
    pub fn zero_commission() -> FeeModel {
        FeeModel::None
    }

    /// Indian broker (Zerodha-like).
    pub fn india_equity() -> FeeModel {
        // 0.03% or Rs 20 per trade, whichever is lower
        // Simplified as 0.03%
        FeeModel::Percentage(0.0003)
    }

    /// Crypto exchange (typical).
    pub fn crypto_exchange() -> FeeModel {
        FeeModel::Percentage(0.001) // 0.1% maker/taker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentage_fee() {
        let fee = FeeModel::percentage(0.001);
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 10.0).abs() < 1e-10); // 100 * 100 * 0.001 = 10
    }

    #[test]
    fn test_fixed_fee() {
        let fee = FeeModel::fixed(5.0);
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_per_share_fee() {
        let fee = FeeModel::per_share(0.01);
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 1.0).abs() < 1e-10); // 100 * 0.01 = 1
    }

    #[test]
    fn brokerage_models_ib_us_minimum_and_cap() {
        let fee = FeeModel::brokerage(0.0, 0.005, 1.0, 0.01);

        assert!((fee.calculate(258.26, 36.0, Direction::Long) - 1.0).abs() < 1e-10);
        assert!((fee.calculate(100.0, 1_000.0, Direction::Long) - 5.0).abs() < 1e-10);
        assert!((fee.calculate(1.0, 10.0, Direction::Long) - 0.10).abs() < 1e-10);
    }

    #[test]
    fn a_brokerage_order_billed_in_pieces_owes_what_it_owes_whole() {
        let fee = FeeModel::brokerage(0.0, 0.005, 1.0, 0.01);
        assert!(fee.bills_per_order());

        let first = fee.accrue(&OrderBilling::default(), 100.0, 40.0).expect("per order");
        assert_eq!(fee.order_charge(&first), fee.calculate(100.0, 40.0, Direction::Long));
        let whole = fee.accrue(&first, 100.0, 260.0).expect("per order");
        assert!(
            (fee.order_charge(&whole) - fee.calculate(100.0, 300.0, Direction::Long)).abs() < 1e-12
        );
        // The cap is the whole order's too: 10 shares at $1 cap at 10 cents.
        let tiny = fee.accrue(&OrderBilling::default(), 1.0, 4.0).expect("per order");
        let tiny = fee.accrue(&tiny, 1.0, 6.0).expect("per order");
        assert!((fee.order_charge(&tiny) - 0.10).abs() < 1e-12);
    }

    #[test]
    fn a_linear_schedule_bills_each_fill() {
        assert!(!FeeModel::Percentage(0.001).bills_per_order());
        assert!(!FeeModel::brokerage(0.001, 0.0, 0.0, 0.0).bills_per_order());
        assert!(FeeModel::Percentage(0.001).accrue(&OrderBilling::default(), 100.0, 1.0).is_none());
    }

    #[test]
    fn brokerage_per_share_half_cent_matches_decimal_venue_fee() {
        let fee = FeeModel::brokerage(0.0, 0.005, 1.0, 0.01);
        let raw = fee.calculate(227.4, 203.0, Direction::Long);
        assert_eq!(raw, 1.015);
        assert_eq!(crate::core::decimals::quantize_money(raw, Some(2)), 1.01);
    }

    #[test]
    fn brokerage_models_ib_asx_percentage_and_minimum() {
        let fee = FeeModel::brokerage(0.00088, 0.0, 6.60, 0.0);

        assert!((fee.calculate(150.0, 10.0, Direction::Long) - 6.60).abs() < 1e-10);
        assert!((fee.calculate(150.0, 100.0, Direction::Long) - 13.20).abs() < 1e-10);
    }

    #[test]
    fn test_round_trip() {
        let fee = FeeModel::percentage(0.001);
        let result = fee.round_trip(100.0, 110.0, 100.0, Direction::Long);
        // Entry: 100 * 100 * 0.001 = 10
        // Exit: 110 * 100 * 0.001 = 11
        // Total: 21
        assert!((result - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_fee() {
        let fee = FeeModel::None;
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 0.0).abs() < 1e-10);
    }

    /// The itemized model splits a round trip across its two sides.
    ///
    /// `calculate` alone always prices the entry side, so an exit priced
    /// through it would carry the wrong side-specific charges. These cover
    /// `breakdown`/`calculate_side`, which the flat-rate tests above cannot
    /// reach.
    #[test]
    fn indian_model_charges_each_side_separately() {
        use crate::execution::indian_costs::Segment;
        let model = FeeModel::indian(Segment::OptionsNfo);

        let entry = model.breakdown(100.0, 75.0, Direction::Long, true).unwrap();
        let exit = model.breakdown(100.0, 75.0, Direction::Long, false).unwrap();

        // A long buys to open: stamp duty on entry, transaction tax on exit.
        assert!(entry.stamp_duty > 0.0);
        assert_eq!(entry.stt, 0.0);
        assert_eq!(exit.stamp_duty, 0.0);
        assert!(exit.stt > 0.0);

        // Per-order brokerage lands on both sides.
        assert_eq!(entry.brokerage, exit.brokerage);
        assert!(entry.brokerage > 0.0);
    }

    /// `calculate_side` returns the itemized total, or the flat fee when the
    /// model has no component structure.
    #[test]
    fn a_percentage_fee_rounds_the_notional_and_rate_together() {
        // Both of these are exact ties at USDT's 8 decimals, and the venue
        // settles them in opposite directions -- the first down, the second
        // up. Only the exact decimal product reproduces both. See
        // `rate_on_notional`.
        let model = FeeModel::Percentage(0.001);

        let btc = model.calculate(92104.5, 0.10379, Direction::Long);
        assert_eq!((btc * 1e8).round() / 1e8, 9.55952605);
        let naive: f64 = 92104.5 * 0.10379 * 0.001;
        assert_ne!((naive * 1e8).round() / 1e8, 9.55952605);

        let avax = model.calculate(11.79, 6.4125, Direction::Long);
        assert_eq!((avax * 1e8).round() / 1e8, 0.07560338);
        let naive: f64 = 11.79 * 6.4125 * 0.001;
        assert_ne!((naive * 1e8).round() / 1e8, 0.07560338);
    }

    #[test]
    fn calculate_side_matches_the_breakdown_it_reports() {
        use crate::execution::indian_costs::Segment;
        let indian = FeeModel::indian(Segment::OptionsNfo);
        let side = indian.calculate_side(100.0, 75.0, Direction::Short, true);
        let breakdown = indian.breakdown(100.0, 75.0, Direction::Short, true).unwrap();
        assert!((side - breakdown.total()).abs() < 1e-9);

        // A flat model has no breakdown to report, and falls back cleanly.
        let flat = FeeModel::percentage(0.001);
        assert!(flat.breakdown(100.0, 75.0, Direction::Long, true).is_none());
        assert_eq!(flat.calculate_side(100.0, 75.0, Direction::Long, true), 7.5);
    }
}
