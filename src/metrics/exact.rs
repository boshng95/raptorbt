//! Python's and numpy's float arithmetic, to the bit.
//!
//! The canonical daily report (`src/services/performance_report.py` in the
//! engine) sums, averages and roots its numbers with `math.fsum`, the builtin
//! `sum`, numpy's `add.reduce`, `statistics` and `pow`, and each of them
//! rounds in its own way. Each is transcribed here as CPython 3.12 and numpy 2
//! compute it, so a report built from these is Python's to the last bit.
//!
//! `None` wherever Python would raise, or take a branch not transcribed here;
//! the caller then leaves the report to Python.

use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};

/// `math.fsum` of finite floats: Shewchuk's exact partials, then CPython's
/// correction that makes a half-way sum round to even across them. `None`
/// where it raises: an intermediate overflow.
pub fn fsum(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut partials: Vec<f64> = Vec::new();
    for value in values {
        if !value.is_finite() {
            return None;
        }
        let mut x = value;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        if x != 0.0 {
            if !x.is_finite() {
                return None;
            }
            partials.push(x);
        }
    }
    let mut n = partials.len();
    if n == 0 {
        return Some(0.0);
    }
    n -= 1;
    let mut hi = partials[n];
    let mut lo = 0.0;
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = partials[n];
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 {
            break;
        }
    }
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    Some(hi)
}

/// `statistics.fmean`: `fsum` over the count. `None` for no values, where it
/// raises.
pub fn fmean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(fsum(values.iter().copied())? / values.len() as f64)
}

/// The builtin `sum` of a list of floats: CPython 3.12 adds the first to the
/// int `0`, then the rest with Neumaier's compensation, which it adds back
/// at the end when it is a finite non-zero. An empty list sums to the int
/// `0`, which reads as `0.0` wherever the report divides by or into it.
pub fn py_sum(values: &[f64]) -> f64 {
    let Some((&first, rest)) = values.split_first() else {
        return 0.0;
    };
    let mut total = 0.0 + first;
    let mut compensation = 0.0;
    for &x in rest {
        let t = total + x;
        if total.abs() >= x.abs() {
            compensation += (total - t) + x;
        } else {
            compensation += (x - t) + total;
        }
        total = t;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

/// What numpy's iterator hands its add loop at once: one buffer.
const NUMPY_BUFFER: usize = 8192;

/// `np.add.reduce` of a contiguous float64 array, which is also what
/// pandas' `Series.sum` comes to without bottleneck: from the identity,
/// one buffer at a time, each summed pairwise (`DOUBLE_pairwise_sum`).
pub fn np_sum(values: &[f64]) -> f64 {
    let mut total = 0.0;
    for chunk in values.chunks(NUMPY_BUFFER) {
        total += pairwise_sum(chunk);
    }
    total
}

/// numpy's `pairwise_sum`: blocks of 128 summed with eight accumulators,
/// halved on a multiple of eight above that.
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - n % 8 {
            for (j, slot) in r.iter_mut().enumerate() {
                *slot += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `statistics.median` of finite floats: the middle of the stably sorted
/// values, or the mean of the two middle ones. `None` for no values.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| value.is_nan()) {
        return None;
    }
    let mut data = values.to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let n = data.len();
    Some(if n % 2 == 1 { data[n / 2] } else { (data[n / 2 - 1] + data[n / 2]) / 2.0 })
}

/// The builtin `max` of a non-empty list: the first of the largest.
pub fn py_max(values: &[f64]) -> Option<f64> {
    let (&first, rest) = values.split_first()?;
    Some(rest.iter().fold(first, |best, &x| if x > best { x } else { best }))
}

/// The builtin `min` of a non-empty list: the first of the smallest.
pub fn py_min(values: &[f64]) -> Option<f64> {
    let (&first, rest) = values.split_first()?;
    Some(rest.iter().fold(first, |best, &x| if x < best { x } else { best }))
}

/// `statistics.stdev` of at least two finite floats, as the report's
/// `_stdev` computes it: the exact sample variance -- every float an integer
/// times a power of two -- and its correctly rounded square root
/// (`statistics._float_sqrt_of_frac`). `None` for fewer than two values, a
/// non-finite one, or a variance too large for that root to be a float.
pub fn stdev(values: &[f64]) -> Option<f64> {
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    // Each non-zero value as `mantissa * 2**exponent`, grouped by exponent:
    // within one, the mantissas and their squares sum exactly in 128 bits
    // (below `2**53` and `2**106`, over fewer than `2**21` values).
    let mut terms: Vec<(i32, i64)> = values
        .iter()
        .filter(|value| **value != 0.0)
        .map(|&value| {
            let (mantissa, exponent) = decompose(value);
            (exponent, mantissa)
        })
        .collect();
    if terms.len() >= 1 << 21 {
        return None;
    }
    terms.sort_unstable_by_key(|&(exponent, _)| exponent);
    let base = terms.first().map_or(0, |&(exponent, _)| exponent);
    let mut total = BigInt::zero();
    let mut squares = BigUint::zero();
    let mut start = 0;
    while start < terms.len() {
        let exponent = terms[start].0;
        let mut stop = start;
        let (mut sum, mut sum_sq) = (0_i128, 0_u128);
        while stop < terms.len() && terms[stop].0 == exponent {
            let mantissa = terms[stop].1;
            sum += i128::from(mantissa);
            sum_sq += (i128::from(mantissa) * i128::from(mantissa)) as u128;
            stop += 1;
        }
        let shift = (exponent - base) as usize;
        total += BigInt::from(sum) << shift;
        squares += BigUint::from(sum_sq) << (2 * shift);
        start = stop;
    }
    // The variance, `(n * S2 - S1**2) / (n * (n - 1))`, over `2**(2 * base)`.
    let count = values.len() as u64;
    let total_sq = (&total * &total).to_biguint()?;
    let mut numerator = BigUint::from(count) * squares;
    if numerator < total_sq {
        return None;
    }
    numerator -= total_sq;
    let mut denominator = BigUint::from(count) * BigUint::from(count - 1);
    if base >= 0 {
        numerator <<= 2 * base as usize;
    } else {
        denominator <<= 2 * (-base) as usize;
    }
    float_sqrt_of_frac(&numerator, &denominator)
}

/// A finite non-zero float as `mantissa * 2**exponent`, the mantissa below
/// `2**53` in magnitude and signed as the float is.
fn decompose(value: f64) -> (i64, i32) {
    let bits = value.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction = (bits & ((1 << 52) - 1)) as i64;
    let (mantissa, exponent) =
        if biased == 0 { (fraction, -1074) } else { (fraction | 1 << 52, biased - 1075) };
    (if value < 0.0 { -mantissa } else { mantissa }, exponent)
}

/// `statistics._float_sqrt_of_frac(n, m)`: the square root of `n / m`,
/// correctly rounded. Computed as Python does, to an integer of at least 55
/// bits rounded to odd, so the one rounding to a float is the correct one.
/// `None` where that integer would not be below `2**64` (a variance of
/// `2**108` or more) or the root would not be a normal float.
fn float_sqrt_of_frac(n: &BigUint, m: &BigUint) -> Option<f64> {
    if m.is_zero() {
        return None;
    }
    if n.is_zero() {
        return Some(0.0);
    }
    const SQRT_BIT_WIDTH: i64 = 2 * 53 + 3;
    let q = (n.bits() as i64 - m.bits() as i64 - SQRT_BIT_WIDTH).div_euclid(2);
    if q >= 0 {
        return None;
    }
    let scaled = n << (2 * (-q) as usize);
    let root = sqrt_round_to_odd(&scaled, m);
    let integer = root.to_u64()?;
    // `integer / 2**-q`: the integer rounds once, to nearest even, and a
    // power of two scales it exactly while the result stays normal.
    let scale = power_of_two(q)?;
    let value = integer as f64 * scale;
    (value >= f64::MIN_POSITIVE).then_some(value)
}

/// `statistics._integer_sqrt_of_frac_rto`: `isqrt(n // m)`, its last bit set
/// when that is not exact.
fn sqrt_round_to_odd(n: &BigUint, m: &BigUint) -> BigUint {
    let root = (n / m).sqrt();
    let inexact = &root * &root * m != *n;
    if inexact {
        root | BigUint::from(1_u8)
    } else {
        root
    }
}

/// `2.0**exponent` for a normal power of two.
fn power_of_two(exponent: i64) -> Option<f64> {
    (-1022..=1023).contains(&exponent).then(|| f64::from_bits(((exponent + 1023) as u64) << 52))
}

extern "C" {
    /// The C library's `pow`, which Python's float `**` and `pow` call.
    #[link_name = "pow"]
    fn c_pow(x: f64, y: f64) -> f64;
}

/// Why `py_pow` gave no float.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowError {
    /// Python raises `OverflowError`.
    Overflow,
    /// A case Python decides by `errno` or its own rules not transcribed
    /// here: a NaN or infinite exponent, a negative base to a fractional
    /// power, zero to a negative power, or an underflow to a subnormal.
    Unsupported,
}

/// Python's `float ** float` (`float_pow`) for a finite exponent: its own
/// special cases, then the platform `pow`, an infinite result raising
/// `OverflowError`.
///
/// The exponent passes through `black_box`: told it is `2.0`, the compiler
/// would square instead, and `pow(x, 2.0)` is not always `x * x`.
pub fn py_pow(base: f64, exponent: f64) -> Result<f64, PowError> {
    if exponent == 0.0 {
        return Ok(1.0);
    }
    if base.is_nan() {
        return Ok(base);
    }
    if !exponent.is_finite() {
        return Err(PowError::Unsupported);
    }
    let odd = is_odd_integer(exponent);
    if base.is_infinite() {
        return Ok(if exponent > 0.0 {
            if odd {
                base
            } else {
                base.abs()
            }
        } else if odd {
            0.0_f64.copysign(base)
        } else {
            0.0
        });
    }
    if base == 0.0 {
        if exponent < 0.0 {
            return Err(PowError::Unsupported);
        }
        return Ok(if odd { base } else { 0.0 });
    }
    let (magnitude, negate) = if base < 0.0 {
        if exponent != exponent.floor() {
            return Err(PowError::Unsupported);
        }
        (-base, odd)
    } else {
        (base, false)
    };
    if magnitude == 1.0 {
        return Ok(if negate { -1.0 } else { 1.0 });
    }
    // SAFETY: `pow` is a pure function of two doubles.
    let value = unsafe { c_pow(std::hint::black_box(magnitude), std::hint::black_box(exponent)) };
    if value.is_infinite() {
        return Err(PowError::Overflow);
    }
    if value != 0.0 && value.abs() < f64::MIN_POSITIVE {
        return Err(PowError::Unsupported);
    }
    Ok(if negate { -value } else { value })
}

/// `DOUBLE_IS_ODD_INTEGER`.
fn is_odd_integer(x: f64) -> bool {
    (x % 2.0).abs() == 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsum_rounds_half_even_across_partials() {
        assert_eq!(fsum([1e-16, 1.0, 1e16]), Some(10000000000000002.0));
        assert_eq!(fsum([0.1; 10]), Some(1.0));
        assert_eq!(fsum([-0.0]).map(f64::is_sign_negative), Some(false));
        assert_eq!(fsum([1e308, 1e308]), None);
        assert_eq!(fsum([]), Some(0.0));
    }

    #[test]
    fn builtin_sum_compensates() {
        // Neumaier keeps the 1.0 a plain left fold loses.
        assert_eq!(py_sum(&[1e100, 1.0, -1e100]), 1.0);
        assert_eq!(py_sum(&[0.1; 10]), 1.0);
        // `0 + -0.0` is `0.0`.
        assert!(!py_sum(&[-0.0]).is_sign_negative());
    }

    #[test]
    fn numpy_sum_is_pairwise_per_buffer() {
        let values: Vec<f64> = (0..20_000).map(|i| 1.0 / f64::from(i + 1)).collect();
        let mut expected = 0.0;
        for chunk in values.chunks(8192) {
            expected += pairwise_sum(chunk);
        }
        assert_eq!(np_sum(&values), expected);
        assert_eq!(np_sum(&[]), 0.0);
    }

    #[test]
    fn stdev_is_correctly_rounded() {
        assert_eq!(stdev(&[1.0, 2.0, 3.0, 4.0]), Some(1.2909944487358056));
        assert_eq!(stdev(&[2.5, 2.5]), Some(0.0));
        assert_eq!(stdev(&[1.0]), None);
        assert_eq!(stdev(&[1.0, f64::NAN]), None);
    }

    #[test]
    fn pow_keeps_python_special_cases() {
        assert_eq!(py_pow(1.0, 1e-3), Ok(1.0));
        assert_eq!(py_pow(-2.0, 2.0), Ok(4.0));
        assert_eq!(py_pow(-2.0, 3.0), Ok(-8.0));
        assert_eq!(py_pow(0.0, 2.0), Ok(0.0));
        assert_eq!(py_pow(-0.0, 2.0).map(f64::is_sign_negative), Ok(false));
        assert_eq!(py_pow(1e300, 2.0), Err(PowError::Overflow));
        assert_eq!(py_pow(-2.0, 0.5), Err(PowError::Unsupported));
        assert_eq!(py_pow(f64::INFINITY, 0.5), Ok(f64::INFINITY));
    }

    #[test]
    fn extremes_are_first_wins() {
        assert!(!py_max(&[0.0, -0.0]).unwrap().is_sign_negative());
        assert!(py_max(&[-0.0, 0.0]).unwrap().is_sign_negative());
        assert!(py_min(&[-0.0, 0.0]).unwrap().is_sign_negative());
        assert_eq!(py_max(&[]), None);
    }
}
