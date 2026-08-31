//! Multiplying decimal quantities without leaving the decimal.
//!
//! Prices, sizes and fee rates are decimals: a venue quotes them on a
//! declared increment and computes with them as fixed-precision numbers.
//! Binary floats are not decimals, and a chain of `f64` multiplications
//! therefore computes the product of three *approximations*. The error is
//! around an ULP and normally invisible -- but a commission is quantized to
//! the settlement currency's precision, and where the true product lands
//! exactly on a half at that precision, an ULP decides the last decimal.
//!
//! Both real cases from the Nautilus parity work sit on that tie and fall
//! opposite ways, which rules out any decimal rounding rule as the
//! explanation: `0.10379 * 92104.5 * 0.001` is exactly `9.559526055` and
//! settles at `9.55952605`, while `6.4125 * 11.79 * 0.001` is exactly
//! `0.075603375` and settles at `0.07560338`. What decides each is which
//! side of the decimal its nearest `f64` falls on -- so reproducing the
//! venue means producing that `f64`, and only an exact product can.

/// Largest decimal scale a factor is searched for.
///
/// Nine digits covers every price, size and rate a venue quotes, and keeps
/// the scaled integers far inside the range `f64` represents exactly.
const MAX_SCALE: u32 = 9;

/// Largest total scale the reconstruction may carry.
///
/// Powers of ten are exact in `f64` to `1e22`; past that the divisor itself
/// is approximate and the final conversion would round twice.
const MAX_TOTAL_SCALE: u32 = 22;

/// Integers above this are no longer exactly representable in `f64`.
const MAX_EXACT_INT: i128 = 1i128 << 53;

/// The scale that expresses `value` as an integer, if a short decimal does.
///
/// Searched smallest-first so `0.001` is found at three digits rather than
/// nine, which keeps the scaled integers small.
///
/// The tolerance is a handful of ULPs of the scaled value, not a fixed
/// epsilon and not a fixed relative slack. A size that reached us through
/// binary arithmetic sits a few ULPs off the decimal it means, and reading
/// it back as that decimal is the whole point; anything further away is not
/// a decimal that drifted but a number that never was one, and a slack
/// loose enough to swallow it would call `pi` a nine-digit decimal.
fn decimal_scale(value: f64) -> Option<(i128, u32)> {
    if !value.is_finite() {
        return None;
    }
    for scale in 0..=MAX_SCALE {
        let scaled = value * 10f64.powi(scale as i32);
        if scaled.abs() >= MAX_EXACT_INT as f64 {
            return None;
        }
        let nearest = scaled.round();
        let drift = scaled.abs().max(1.0) * f64::EPSILON * 16.0;
        if (scaled - nearest).abs() <= drift {
            return Some((nearest as i128, scale));
        }
    }
    None
}

/// The product of decimal `factors`, as the `f64` nearest to it.
///
/// Every factor is recovered as an integer and a scale, so the product is
/// formed exactly and converted once, by a single correctly-rounded
/// division by an exact power of ten. Returns `None` when a factor is not a
/// short decimal, or when the exact product no longer fits where the
/// arithmetic stays exact; the caller then falls back to floating point,
/// which is all such a product ever was.
pub(crate) fn decimal_product(factors: &[f64]) -> Option<f64> {
    let mut mantissa: i128 = 1;
    let mut total_scale: u32 = 0;
    for &factor in factors {
        let (scaled, scale) = decimal_scale(factor)?;
        mantissa = mantissa.checked_mul(scaled)?;
        total_scale = total_scale.checked_add(scale)?;
        if mantissa.abs() >= MAX_EXACT_INT || total_scale > MAX_TOTAL_SCALE {
            return None;
        }
    }
    Some(mantissa as f64 / 10f64.powi(total_scale as i32))
}

/// Round a settlement-currency amount to the currency's own precision.
///
/// Money is counted in whole units of the smallest one the currency has, so
/// an amount that carries a fraction of that unit is not yet money. An
/// instrument that declares no `precision` declares no such unit: the
/// amount is left as the raw float it was, which is what stock Raptor has
/// always reported.
#[inline]
pub(crate) fn quantize_money(value: f64, precision: Option<u32>) -> f64 {
    let Some(precision) = precision else {
        return value;
    };
    let factor = 10_f64.powi(precision.min(15) as i32);
    (value * factor).round() / factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decimal_product_lands_on_the_side_of_the_tie_the_venue_lands_on() {
        // Both are exact ties at eight decimals, and they settle opposite
        // ways: what decides is where the nearest `f64` falls.
        let btc = decimal_product(&[0.10379, 92104.5, 0.001]).expect("decimal");
        assert_eq!((btc * 1e8).round() / 1e8, 9.55952605);

        let avax = decimal_product(&[6.4125, 11.79, 0.001]).expect("decimal");
        assert_eq!((avax * 1e8).round() / 1e8, 0.07560338);
    }

    #[test]
    fn a_product_of_binary_approximations_misses_both() {
        // The naive chain rounds three times and lands on the wrong side of
        // one tie; the point of the exact product is that it never does.
        let btc: f64 = 0.10379 * 92104.5 * 0.001;
        assert_ne!((btc * 1e8).round() / 1e8, 9.55952605);
        let avax: f64 = 6.4125 * 11.79 * 0.001;
        assert_ne!((avax * 1e8).round() / 1e8, 0.07560338);
    }

    #[test]
    fn a_factor_that_drifted_through_binary_is_read_as_the_decimal_it_means() {
        // 496.19175 comes back from a partial reduce an ULP light. It is
        // still the size the venue holds, and the fee is charged on that.
        let drifted = decimal_product(&[496.19174999999996, 13.88, 0.001]).expect("decimal");
        let exact = decimal_product(&[496.19175, 13.88, 0.001]).expect("decimal");
        assert_eq!(drifted, exact);
    }

    #[test]
    fn nothing_is_claimed_for_a_value_that_is_not_a_short_decimal() {
        assert_eq!(decimal_product(&[std::f64::consts::PI, 2.0]), None);
        assert_eq!(decimal_product(&[1.0, f64::INFINITY]), None);
        assert_eq!(decimal_product(&[1e300, 1e300]), None);
    }

    #[test]
    fn an_exact_product_is_the_plain_product_wherever_binary_can_hold_it() {
        for (a, b) in [(2.5_f64, 4.0_f64), (0.1, 0.2), (123.45, 6.0), (0.0, 99.9)] {
            let exact = decimal_product(&[a, b]).expect("decimal");
            assert!(
                (exact - a * b).abs() <= (a * b).abs() * 1e-12,
                "{a} * {b}: exact={exact}, plain={}",
                a * b
            );
        }
    }
}
