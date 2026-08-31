//! Fee calculation models.

use crate::core::decimals::decimal_product;
use crate::core::types::{Direction, Price};
use crate::execution::indian_costs::{self, FeeBreakdown, Segment};

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
            FeeModel::Brokerage { percentage, per_share, minimum, max_percentage } => {
                let mut fee = if *per_share > 0.0 {
                    size.abs() * per_share
                } else {
                    rate_on_notional(price, size.abs(), *percentage)
                };
                if *minimum > 0.0 {
                    fee = fee.max(*minimum);
                }
                if *max_percentage > 0.0 {
                    fee = fee.min(rate_on_notional(price, size.abs(), *max_percentage));
                }
                fee
            }
            FeeModel::Indian { segment } => {
                indian_costs::calculate_side(*segment, trade_value, _direction, true).total()
            }
        }
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
