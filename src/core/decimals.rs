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

/// `10**k` for `k` in `0..=22`, each exact in a float.
const EXACT_POWERS: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// `10_f64.powi(k)`. Through `1e22` every product `powi` forms is exact, so
/// the table holds the same float without the call.
#[inline]
pub(crate) fn pow10_f64(k: u32) -> f64 {
    match EXACT_POWERS.get(k as usize) {
        Some(&power) => power,
        None => 10_f64.powi(k as i32),
    }
}

/// `x.round()`, half away from zero. On x86-64 `round` is a software
/// routine, while SSE4.1 truncates in hardware; `x - x.trunc()` is exact,
/// so comparing it with a half decides the same float.
#[inline]
pub(crate) fn round_half_away(x: f64) -> f64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
    {
        let whole = x.trunc();
        if (x - whole).abs() >= 0.5 {
            whole + x.signum()
        } else {
            whole
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse4.1")))]
    {
        x.round()
    }
}

/// `a.checked_mul(b)`, one machine multiply where both fit 64 bits: their
/// product always fits 128, and the checked `i128` multiply is a call.
#[inline]
pub(crate) fn mul_i128(a: i128, b: i128) -> Option<i128> {
    match (i64::try_from(a), i64::try_from(b)) {
        (Ok(a), Ok(b)) => Some(i128::from(a) * i128::from(b)),
        _ => a.checked_mul(b),
    }
}

/// `c as f64`: a 64-bit conversion where `c` fits, which rounds the same
/// and is one instruction rather than a call.
#[inline]
pub(crate) fn i128_to_f64(c: i128) -> f64 {
    match i64::try_from(c) {
        Ok(c) => c as f64,
        Err(_) => c as f64,
    }
}

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
        let scaled = value * pow10_f64(scale);
        if scaled.abs() >= MAX_EXACT_INT as f64 {
            return None;
        }
        let nearest = round_half_away(scaled);
        let drift = scaled.abs().max(1.0) * f64::EPSILON * 16.0;
        if (scaled - nearest).abs() <= drift {
            // At most `2**53`, so the 64-bit conversion is exact.
            return Some((i128::from(nearest as i64), scale));
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
        mantissa = mul_i128(mantissa, scaled)?;
        total_scale = total_scale.checked_add(scale)?;
        if mantissa.abs() >= MAX_EXACT_INT || total_scale > MAX_TOTAL_SCALE {
            return None;
        }
    }
    Some(i128_to_f64(mantissa) / pow10_f64(total_scale))
}

/// The sum of two decimals, as the `f64` nearest to it.
///
/// A venue steps a price by its increment in fixed point, so one level
/// below `295.04` on a cent grid is exactly `295.03`; the binary
/// `295.04 - 0.01` is `295.03000000000003`, and a realized PnL formed from
/// it can settle a cent away. Both terms are recovered at a common scale,
/// added as integers and converted once. Returns `None` when either is not
/// a short decimal; the caller then falls back to the plain float sum.
pub(crate) fn decimal_sum(a: f64, b: f64) -> Option<f64> {
    let (a_scaled, a_scale) = decimal_scale(a)?;
    let (b_scaled, b_scale) = decimal_scale(b)?;
    let scale = a_scale.max(b_scale);
    let lift = |scaled: i128, from: u32| scaled.checked_mul(10i128.pow(scale - from));
    let total = lift(a_scaled, a_scale)?.checked_add(lift(b_scaled, b_scale)?)?;
    if total.abs() >= MAX_EXACT_INT {
        return None;
    }
    Some(i128_to_f64(total) / pow10_f64(scale))
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
    let factor = pow10_f64(precision.min(15));
    round_half_away(value * factor) / factor
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
    fn a_step_down_the_grid_lands_on_the_level_the_venue_quotes() {
        assert_ne!(295.04_f64 - 0.01, 295.03);
        assert_eq!(decimal_sum(295.04, -0.01), Some(295.03));
        assert_eq!(decimal_sum(0.1, 0.2), Some(0.3));
        assert_eq!(decimal_sum(92104.5, 0.01), Some(92104.51));
        assert_eq!(decimal_sum(1.2345, -0.0001), Some(1.2344));
    }

    #[test]
    fn a_sum_that_is_not_of_short_decimals_is_not_claimed() {
        assert_eq!(decimal_sum(std::f64::consts::PI, 0.01), None);
        assert_eq!(decimal_sum(f64::NAN, 0.01), None);
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

    /// Bit patterns spread over every exponent, and the halves, ties and
    /// specials a rounding rule gets wrong.
    fn bit_floats() -> Vec<f64> {
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut out: Vec<f64> = (0..200_000)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                f64::from_bits(z ^ (z >> 31))
            })
            .collect();
        for k in -3000..3000 {
            let half = k as f64 + 0.5;
            out.extend([half, -half, half / 1024.0, k as f64 * 1e-3, k as f64 * 0.1 + 0.05]);
        }
        let big = (1_u64 << 52) as f64;
        out.extend([
            0.0, -0.0, 0.5, -0.5, 1.5, 2.5, 0.49999999999999994, -0.49999999999999994,
            big - 0.5, big + 1.0, 2.0 * big - 1.0, -(big - 0.5), f64::MIN_POSITIVE,
            5e-324, f64::MAX, f64::MIN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
        ]);
        out
    }

    #[test]
    fn round_half_away_is_round() {
        for x in bit_floats() {
            let (ours, std) = (round_half_away(x), x.round());
            assert!(ours.to_bits() == std.to_bits() || (ours.is_nan() && std.is_nan()), "{x:e}");
        }
    }

    #[test]
    fn the_power_table_is_powi() {
        for k in 0..=40 {
            assert_eq!(pow10_f64(k).to_bits(), 10_f64.powi(k as i32).to_bits(), "{k}");
        }
    }

    #[test]
    fn the_fast_paths_answer_as_i128_does() {
        let edges = [
            0_i128, 1, -1, 10, i64::MAX as i128, i64::MIN as i128, i64::MAX as i128 + 1,
            i64::MIN as i128 - 1, 1 << 53, (1 << 53) + 1, 1 << 100, -(1 << 100), i128::MAX,
            i128::MIN, 10_i128.pow(28) - 1, 123_456_789_012_345_678,
        ];
        for &a in &edges {
            assert_eq!(i128_to_f64(a).to_bits(), (a as f64).to_bits(), "{a}");
            for &b in &edges {
                assert_eq!(mul_i128(a, b), a.checked_mul(b), "{a} * {b}");
            }
        }
        for x in bit_floats() {
            let c = x as i128;
            assert_eq!(i128_to_f64(c).to_bits(), (c as f64).to_bits(), "{c}");
        }
    }
}
