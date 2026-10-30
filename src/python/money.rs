//! A replay's money, in the kernel: the caller's sizing, price grid, cash
//! check and cash halt, computed exactly as the caller computes them.
//!
//! `SignalReplay` answers four questions through Python functions -- what an
//! entry is sized at (`sized`), what price grid an order lands on
//! (`snapped`), whether cash pays for an order (`TradeCost.can_afford`) and
//! whether the cash halt holds (`cash_is_exhausted`) -- and a replay asks
//! them hundreds of times a run. Each is `Decimal` arithmetic over floats
//! read through their shortest repr (`src/sizing/cost.py`), a venue's tick
//! table and Nautilus's `Price` grid. This module is that arithmetic on exact
//! decimals.
//!
//! Python's `Decimal` rounds every result to 28 significant digits. Every
//! value here is computed exactly, and an answer is given only when each
//! value the Python computation forms provably fits in those digits, so the
//! two agree by construction. Anything else -- a NaN, a value too large to
//! prove, a sizer this does not know -- answers `None`, and the caller asks
//! Python.

use std::cmp::Ordering;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `10**28`: one past the largest coefficient `Decimal` keeps exact.
const DIGITS: i128 = 10_000_000_000_000_000_000_000_000_000;

fn pow10(n: u32) -> Option<i128> {
    10_i128.checked_pow(n)
}

/// `10**k` for `k` in `0..=22`, each exact in a float.
const EXACT: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// `|x| * 10**precision` rounded half to even on its exact binary value, or
/// `None` where that integer is not below `2**53` or would take more than a
/// `u128` to find. The digits `round(x, precision)` and
/// `f"{x:.{precision}f}"` both take, without spelling them.
fn scaled_half_even(x: f64, precision: u32) -> Option<u64> {
    scaled_half_even_clear(x, precision).map(|(units, _)| units)
}

/// [`scaled_half_even`], and whether every value within one float step of
/// `x` rounds to the same units: true where the tie lies further off than
/// that step, so `x`'s shortest repr -- within half a step -- rounds there
/// too.
fn scaled_half_even_clear(x: f64, precision: u32) -> Option<(u64, bool)> {
    if !x.is_finite() || precision as usize >= EXACT.len() {
        return None;
    }
    let bits = x.abs().to_bits();
    let (exponent, fraction) = ((bits >> 52) as i32, bits & ((1 << 52) - 1));
    // `|x| = mantissa * 2**-shift`, the mantissa below `2**53`.
    let (mantissa, shift) =
        if exponent == 0 { (fraction, 1074) } else { (fraction | 1 << 52, 1075 - exponent) };
    if shift <= 0 {
        // A whole number of `2**52` or more: no fraction to round.
        return None;
    }
    // Below `2**53 * 10**22 < 2**127`.
    let step = 10_u128.pow(precision);
    let product = u128::from(mantissa) * step;
    let shift = shift as u32;
    // A float step of `x` is one unit of the mantissa, so `step` of these.
    let (units, clear) = if shift >= 128 {
        // Below a quarter, its steps far finer than another quarter.
        (0, true)
    } else {
        let whole = product >> shift;
        let rest = product & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let up = rest > half || (rest == half && whole & 1 == 1);
        (whole + u128::from(up), rest.abs_diff(half) > step)
    };
    (units < 1 << 53).then_some((units as u64, clear))
}

/// `round(x, precision)` on a finite float: the decimal nearest its exact
/// value, ties to even, then the float nearest that, sign kept.
pub(crate) fn round_to(x: f64, precision: u32) -> Option<f64> {
    if let Some(units) = scaled_half_even(x, precision) {
        // Both operands exact, so the one division rounds once.
        let value = units as f64 / EXACT[precision as usize];
        return Some(if x.is_sign_negative() { -value } else { value });
    }
    read_spelled(format_args!("{:.*}", precision as usize, x), |text| text.parse().ok())
}

/// `read(&format!(...))`, spelled on the stack: the text this module reads
/// back is short, and a replay spells some of it a few times an order.
fn read_spelled<T>(args: std::fmt::Arguments<'_>, read: impl FnOnce(&str) -> T) -> T {
    let mut spelled = Spelled { bytes: [0; 64], len: 0 };
    match std::fmt::Write::write_fmt(&mut spelled, args) {
        // Only whole `str`s are written, so the bytes are one too.
        Ok(()) => read(std::str::from_utf8(&spelled.bytes[..spelled.len]).unwrap_or_default()),
        Err(_) => read(&std::fmt::format(args)),
    }
}

/// A fixed buffer of text, which refuses what does not fit.
struct Spelled {
    bytes: [u8; 64],
    len: usize,
}

impl std::fmt::Write for Spelled {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let end = self.len + text.len();
        let slot = self.bytes.get_mut(self.len..end).ok_or(std::fmt::Error)?;
        slot.copy_from_slice(text.as_bytes());
        self.len = end;
        Ok(())
    }
}

/// `f"{x:.{precision}f}"` for a float that is not a NaN.
pub(crate) fn fixed(x: f64, precision: usize) -> String {
    if let Some(units) = u32::try_from(precision).ok().and_then(|p| scaled_half_even(x, p)) {
        let digits = units.to_string();
        let mut text = String::with_capacity(digits.len() + precision + 3);
        if x.is_sign_negative() {
            text.push('-');
        }
        if precision == 0 {
            text.push_str(&digits);
        } else if digits.len() > precision {
            let (whole, fraction) = digits.split_at(digits.len() - precision);
            text.push_str(whole);
            text.push('.');
            text.push_str(fraction);
        } else {
            text.push_str("0.");
            text.extend(std::iter::repeat_n('0', precision - digits.len()));
            text.push_str(&digits);
        }
        return text;
    }
    format!("{:.*}", precision, x)
}

/// `c * 10**e`, exact.
#[derive(Clone, Copy, Debug)]
struct Dec {
    c: i128,
    e: i32,
}

#[derive(Clone, Copy)]
enum Rounding {
    Ceiling,
    Floor,
    HalfEven,
}

impl Dec {
    const ZERO: Dec = Dec { c: 0, e: 0 };

    /// The value, or `None` once its coefficient leaves `Decimal`'s digits.
    fn checked(c: i128, e: i32) -> Option<Dec> {
        (c.unsigned_abs() < DIGITS.unsigned_abs()).then_some(Dec { c, e })
    }

    /// `_dec`: a float through its shortest repr, which Rust's `{:e}` writes
    /// with the same digits Python's `repr` does.
    fn of(x: f64) -> Option<Dec> {
        if !x.is_finite() {
            return None;
        }
        Dec::shortest(x).or_else(|| read_spelled(format_args!("{x:e}"), Dec::parse))
    }

    /// `Dec::of` without spelling the float, wherever its digits number at
    /// most fifteen or so: the fewest decimal places `k` at which an integer
    /// `c` gives `c / 10**k == |x|`.
    ///
    /// Below `2**51` at scale `k`, `|x|`'s neighbours lie closer together
    /// than half a unit of that scale, so at most one such `c` exists, it is
    /// the integer nearest the scaled float, and the one division -- of two
    /// exact operands -- is the correctly rounded read of `c`e`-k`. The
    /// fewest places with an answer are the fewest digits, which is the
    /// shortest repr. `None` past that range, for the spelling to answer.
    fn shortest(x: f64) -> Option<Dec> {
        const LIMIT: f64 = (1_u64 << 51) as f64;
        let magnitude = x.abs();
        if magnitude == 0.0 {
            return Some(Dec::ZERO);
        }
        for (places, scale) in EXACT.iter().enumerate() {
            let scaled = magnitude * scale;
            if scaled >= LIMIT {
                return None;
            }
            let units = scaled.round();
            if units != 0.0 && units / scale == magnitude {
                let (mut c, mut e) = (units as i128, -(places as i32));
                while c % 10 == 0 {
                    c /= 10;
                    e += 1;
                }
                return Some(Dec { c: if x < 0.0 { -c } else { c }, e });
            }
        }
        None
    }

    /// A decimal literal, as `Decimal(text)` reads it, trailing zeros dropped.
    fn parse(text: &str) -> Option<Dec> {
        let (mantissa, exponent) = match text.find(['e', 'E']) {
            Some(at) => (&text[..at], text[at + 1..].parse::<i32>().ok()?),
            None => (text, 0),
        };
        let (negative, mantissa) = match mantissa.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, mantissa.strip_prefix('+').unwrap_or(mantissa)),
        };
        let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        if whole.is_empty() && fraction.is_empty() {
            return None;
        }
        let mut c: i128 = 0;
        for digit in whole.bytes().chain(fraction.bytes()) {
            if !digit.is_ascii_digit() {
                return None;
            }
            c = c.checked_mul(10)?.checked_add(i128::from(digit - b'0'))?;
        }
        let mut e = exponent.checked_sub(i32::try_from(fraction.len()).ok()?)?;
        if c == 0 {
            return Some(Dec::ZERO);
        }
        while c % 10 == 0 {
            c /= 10;
            e += 1;
        }
        Dec::checked(if negative { -c } else { c }, e)
    }

    fn signum(self) -> i128 {
        self.c.signum()
    }

    /// The nearest float, as `float(Decimal)` rounds it.
    fn to_f64(self) -> f64 {
        // Both operands exact, so the one operation rounds once.
        if self.c.unsigned_abs() < (1_u128 << 53) && self.e.unsigned_abs() <= 22 {
            let c = self.c as f64;
            let scale = EXACT[self.e.unsigned_abs() as usize];
            return if self.e >= 0 { c * scale } else { c / scale };
        }
        read_spelled(format_args!("{}e{}", self.c, self.e), |text| text.parse().unwrap_or(f64::NAN))
    }

    /// The approximate value, for a search's first guess only.
    fn approx(self) -> f64 {
        self.c as f64 * 10_f64.powi(self.e)
    }

    fn mul(self, other: Dec) -> Option<Dec> {
        Dec::checked(self.c.checked_mul(other.c)?, self.e.checked_add(other.e)?)
    }

    /// Both coefficients on the finer exponent.
    fn align(self, other: Dec) -> Option<(i128, i128, i32)> {
        let e = self.e.min(other.e);
        let a = self.c.checked_mul(pow10((self.e - e) as u32)?)?;
        let b = other.c.checked_mul(pow10((other.e - e) as u32)?)?;
        Some((a, b, e))
    }

    fn add(self, other: Dec) -> Option<Dec> {
        if other.c == 0 {
            return Some(self);
        }
        if self.c == 0 {
            return Some(other);
        }
        let (a, b, e) = self.align(other)?;
        Dec::checked(a.checked_add(b)?, e)
    }

    fn compare(self, other: Dec) -> Option<Ordering> {
        let (a, b) = (self.signum(), other.signum());
        if a != b || a == 0 {
            return Some(a.cmp(&b));
        }
        let (a, b, _) = self.align(other)?;
        Some(a.cmp(&b))
    }

    /// `quantize(Decimal(1).scaleb(e), rounding)`.
    fn quantize(self, e: i32, rounding: Rounding) -> Option<Dec> {
        if self.e >= e {
            return Dec::checked(self.c.checked_mul(pow10((self.e - e) as u32)?)?, e);
        }
        let shift = (self.e as i64 - e as i64).unsigned_abs();
        // A coefficient below `10**28` is under half of `10**38`'s step.
        let (q, r, half) = match u32::try_from(shift).ok().and_then(pow10) {
            Some(d) => (self.c / d, self.c % d, (2 * (self.c % d).abs()).cmp(&d)),
            None => (0, self.c, Ordering::Less),
        };
        let q = match rounding {
            Rounding::Ceiling if r > 0 => q + 1,
            Rounding::Floor if r < 0 => q - 1,
            Rounding::HalfEven
                if half == Ordering::Greater || (half == Ordering::Equal && q % 2 != 0) =>
            {
                q + self.signum()
            }
            _ => q,
        };
        Dec::checked(q, e)
    }

    /// `self / other`, given only when the quotient terminates within
    /// `Decimal`'s digits, so its 28-digit division rounds nothing.
    fn div_exact(self, other: Dec) -> Option<Dec> {
        if other.c == 0 {
            return None;
        }
        let mut numerator = self.c;
        for shift in 0..=28 {
            if numerator % other.c == 0 {
                return Dec::checked(numerator / other.c, self.e - other.e - shift);
            }
            numerator = numerator.checked_mul(10)?;
        }
        None
    }

    /// `floor(self / other)` for two positive values.
    fn floor_div(self, other: Dec) -> Option<i128> {
        if self.e >= other.e {
            let numerator = self.c.checked_mul(pow10((self.e - other.e) as u32)?)?;
            Some(numerator / other.c)
        } else {
            match pow10((other.e - self.e) as u32).and_then(|d| other.c.checked_mul(d)) {
                Some(denominator) => Some(self.c / denominator),
                // A denominator past `i128` exceeds every coefficient.
                None => Some(0),
            }
        }
    }
}

/// `TradeCost` with its `CommissionSchedule`.
struct Cost {
    lot: Dec,
    /// `-cash_precision`: the exponent of one cent.
    cent: i32,
    taker_fee: Dec,
    minimum: Dec,
    per_share: Dec,
    max_pct: Dec,
}

impl Cost {
    /// `_cash`.
    fn cash(&self, cash: f64) -> Option<Dec> {
        // Clear of a tie, the shortest repr rounds where the float does.
        if let Some((units, true)) =
            u32::try_from(-self.cent).ok().and_then(|places| scaled_half_even_clear(cash, places))
        {
            let c = i128::from(units);
            return Dec::checked(if cash.is_sign_negative() { -c } else { c }, self.cent);
        }
        Dec::of(cash)?.quantize(self.cent, Rounding::HalfEven)
    }

    /// `_up`.
    fn up(&self, amount: Dec) -> Option<Dec> {
        amount.quantize(self.cent, Rounding::Ceiling)
    }

    /// `CommissionSchedule.commission_for`.
    fn commission(&self, quantity: Dec, price: Dec) -> Option<Dec> {
        let units = Dec { c: quantity.c.abs(), e: quantity.e };
        let notional = units.mul(price)?;
        let mut fee = if self.per_share.c > 0 {
            units.mul(self.per_share)?
        } else {
            notional.mul(self.taker_fee)?
        };
        if self.minimum.c > 0 && self.minimum.compare(fee)? == Ordering::Greater {
            fee = self.minimum;
        }
        if self.max_pct.c > 0 {
            let cap = notional.mul(self.max_pct)?;
            if cap.compare(fee)? == Ordering::Less {
                fee = cap;
            }
        }
        Some(fee)
    }

    /// `buy_cost`.
    fn buy_cost(&self, quantity: Dec, price: Dec) -> Option<Dec> {
        if quantity.c <= 0 {
            return Some(Dec::ZERO);
        }
        let notional = self.up(quantity.mul(price)?)?;
        notional.add(self.up(self.commission(quantity, price)?)?)
    }

    /// `can_afford`.
    fn can_afford(&self, cash: f64, quantity: Dec, price: Dec) -> Option<bool> {
        let cost = self.buy_cost(quantity, price)?;
        Some(cost.compare(self.cash(cash)?)? != Ordering::Greater)
    }

    fn lots(&self, count: i128) -> Option<Dec> {
        Dec::checked(count.checked_mul(self.lot.c)?, self.lot.e)
    }

    /// `affordable_quantity`: the most whole lots `cash` buys at `price`.
    ///
    /// Cost never falls as lots rise, so the answer is the one lot count
    /// that is affordable with the next one not, whichever probes find it;
    /// `TradeCost` interpolates and bisects, this brackets from a guess. Its
    /// probes are at most `floor(cash / (price * lot)) + 1` lots, and the
    /// cost there fitting `Decimal`'s digits means every probe's does.
    fn affordable_quantity(&self, cash: f64, price: f64) -> Option<Dec> {
        let c = self.cash(cash)?;
        let p = Dec::of(price)?;
        if c.c <= 0 || p.c <= 0 {
            return Some(Dec::ZERO);
        }
        let most = c.floor_div(p.mul(self.lot)?)?;
        if most >= 100_000_000_000_000_000_000 {
            return None;
        }
        self.buy_cost(self.lots(most + 1)?, p)?;
        if most == 0 {
            return Some(Dec::ZERO);
        }
        let affordable = |count: i128| -> Option<bool> {
            let cost = self.buy_cost(self.lots(count)?, p)?;
            Some(cost.compare(c)? != Ordering::Greater)
        };
        let cost_most = self.buy_cost(self.lots(most)?, p)?;
        if cost_most.compare(c)? != Ordering::Greater {
            return self.lots(most);
        }
        // The last affordable count is below `most`; guess it from the
        // average cost of a lot there, then bracket it.
        let guess = (most as f64 * (c.approx() / cost_most.approx())).floor();
        let guess = if guess.is_finite() { (guess as i128).clamp(0, most - 1) } else { 0 };
        let (mut lo, mut hi) = (0, most);
        if affordable(guess)? {
            lo = guess;
            let mut step = 1;
            while lo + step < hi {
                if affordable(lo + step)? {
                    lo += step;
                    step *= 2;
                } else {
                    hi = lo + step;
                    break;
                }
            }
        } else {
            hi = guess;
            let mut step = 1;
            while hi - step > 0 {
                if affordable(hi - step)? {
                    lo = hi - step;
                    break;
                }
                hi -= step;
                step *= 2;
            }
        }
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if affordable(mid)? {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        self.lots(lo)
    }
}

/// `snapped_order_price`: a marketable limit on Nautilus's price grid.
struct Grid {
    /// `fallback_buffer_for_venue`.
    buffer: f64,
    /// `venue_price_tick`'s table: ascending `(below, tick)`, `None` on top.
    tiers: Vec<(Option<Dec>, Dec)>,
    /// `_increment_precision(price_increment)`.
    increment_precision: usize,
    price_precision: i32,
}

impl Grid {
    fn snap(&self, reference: f64, buy: bool) -> Option<f64> {
        // `marketable_limit_price` prices nothing at or below zero; such a
        // reference, and one too large for a `Price`, are Python's to answer.
        if !(reference > 0.0 && reference < 1e12) {
            return None;
        }
        let signed = if buy { self.buffer } else { -self.buffer };
        let mut price = reference * (1.0 + signed);
        if !self.tiers.is_empty() {
            let at = Dec::of(price)?;
            let mut tick = self.tiers.last()?.1;
            for (below, step) in &self.tiers {
                match below {
                    Some(bound) if at.compare(*bound)? != Ordering::Less => {}
                    _ => {
                        tick = *step;
                        break;
                    }
                }
            }
            // `snap_price_to_tick`, rounding in the aggressive direction.
            let rounding = if buy { Rounding::Ceiling } else { Rounding::Floor };
            let steps = at.div_exact(tick)?.quantize(0, rounding)?;
            price = steps.mul(tick)?.to_f64();
        }
        // `round(price, increment_precision)`: Rust's fixed formatting rounds
        // the float's exact value half to even, as Python's `round` does.
        let rounded = round_to(price, u32::try_from(self.increment_precision).ok()?)?;
        // `float(Price(rounded, price_precision))`: Nautilus scales by the
        // precision in floats, rounds half away from zero and keeps the
        // fixed-point raw, which `float` divides back out.
        let units = (rounded * 10_f64.powi(self.price_precision)).round();
        if !(units.abs() < 1e17) {
            return None;
        }
        let raw = (units as i128).checked_mul(pow10(16 - self.price_precision as u32)?)?;
        Some(raw as f64 / 1e16)
    }
}

/// The run's sizer.
enum Sizer {
    /// `FixedQuantitySizer`: its quantity on the lot grid, as a float and as
    /// the decimal `can_fund` costs.
    Fixed { units: f64, lots: Dec },
    /// `EquityPercentSizer`.
    Equity { percent: f64 },
    /// `FixedBudgetSizer`.
    Budget { budget: f64, percent: f64 },
}

/// A run's money rules: its sizer, cost model and price grid.
///
/// Every method answers as the caller's function of the same name would,
/// or `None` where only that function can.
#[pyclass(name = "MoneyModel", module = "raptorbt")]
pub struct PyMoneyModel {
    sizer: Sizer,
    cost: Cost,
    grid: Grid,
    /// `limit_entries`: an entry is sized at its marketable limit, else at
    /// the last close times `market_markup`.
    limit_entries: bool,
    market_markup: f64,
}

impl PyMoneyModel {
    /// `entry_order_price`.
    fn entry_price(&self, reference: f64, last_close: f64) -> Option<f64> {
        if self.limit_entries {
            return self.grid.snap(reference, true);
        }
        let price = last_close * self.market_markup;
        price.is_finite().then_some(price)
    }

    /// `SpendingSizer.spend`; `None` for a sizer that spends no share.
    fn spend(&self, cash: f64) -> Option<f64> {
        match self.sizer {
            Sizer::Fixed { .. } => None,
            Sizer::Equity { percent } => Some(cash * (percent / 100.0)),
            Sizer::Budget { budget, percent } => {
                let share = budget * (percent / 100.0);
                Some(if cash < share { cash } else { share })
            }
        }
    }

    pub(crate) fn sized(&self, cash: f64, close: f64, reference: f64) -> Option<f64> {
        if !cash.is_finite() {
            return None;
        }
        let price = self.entry_price(reference, close)?;
        if let Sizer::Fixed { units, .. } = self.sizer {
            return Some(units);
        }
        if price <= 0.0 {
            return Some(0.0);
        }
        Some(self.cost.affordable_quantity(self.spend(cash)?, price)?.to_f64())
    }

    pub(crate) fn snap(&self, reference: f64, buy: bool) -> Option<f64> {
        self.grid.snap(reference, buy)
    }

    pub(crate) fn affordable(&self, cash: f64, units: f64, price: f64) -> Option<bool> {
        self.cost.can_afford(cash, Dec::of(units)?, Dec::of(price)?)
    }

    pub(crate) fn exhausted(&self, cash: f64, price: f64) -> Option<bool> {
        if !cash.is_finite() {
            return None;
        }
        if !(0.0 < price && price < f64::INFINITY) || cash <= 0.0 {
            return Some(false);
        }
        let entry = self.entry_price(price, price)?;
        let funded = match self.sizer {
            Sizer::Fixed { lots, .. } => {
                entry > 0.0 && lots.c > 0 && self.cost.can_afford(cash, lots, Dec::of(entry)?)?
            }
            _ => {
                entry > 0.0
                    && self.cost.can_afford(self.spend(cash)?, self.cost.lot, Dec::of(entry)?)?
            }
        };
        Some(!funded)
    }
}

fn decimal(name: &str, text: &str) -> PyResult<Dec> {
    Dec::parse(text).ok_or_else(|| PyValueError::new_err(format!("{name}: not a decimal: {text}")))
}

#[pymethods]
impl PyMoneyModel {
    /// `sizer` is `"fixed_quantity"` (with `quantity`, already on the lot
    /// grid), `"equity_percent"` (with `percent`) or `"fixed_budget"` (with
    /// `budget` and `percent`). Decimals arrive as their `str`.
    #[new]
    #[pyo3(signature = (
        *, sizer, percent, budget, quantity, lot, cash_precision, taker_fee, minimum,
        per_share, max_pct, limit_entries, market_markup, buffer, tiers,
        increment_precision, price_precision,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        sizer: &str,
        percent: f64,
        budget: f64,
        quantity: &str,
        lot: &str,
        cash_precision: i32,
        taker_fee: &str,
        minimum: &str,
        per_share: &str,
        max_pct: &str,
        limit_entries: bool,
        market_markup: f64,
        buffer: f64,
        tiers: Vec<(Option<String>, String)>,
        increment_precision: usize,
        price_precision: i32,
    ) -> PyResult<Self> {
        let lots = decimal("quantity", quantity)?;
        let sizer = match sizer {
            "fixed_quantity" => Sizer::Fixed { units: lots.to_f64(), lots },
            "equity_percent" => Sizer::Equity { percent },
            "fixed_budget" => Sizer::Budget { budget, percent },
            other => return Err(PyValueError::new_err(format!("unknown sizer {other:?}"))),
        };
        let lot = decimal("lot", lot)?;
        if lot.c <= 0 {
            return Err(PyValueError::new_err("lot must be positive"));
        }
        if !(0..=16).contains(&price_precision) || increment_precision > 16 {
            return Err(PyValueError::new_err("price precision must be within 0..=16"));
        }
        let tiers = tiers
            .iter()
            .map(|(below, tick)| {
                let below = below.as_deref().map(|text| decimal("tier", text)).transpose()?;
                let tick = decimal("tick", tick)?;
                if tick.c <= 0 {
                    return Err(PyValueError::new_err("tick must be positive"));
                }
                Ok((below, tick))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            sizer,
            cost: Cost {
                lot,
                cent: -cash_precision,
                taker_fee: decimal("taker_fee", taker_fee)?,
                minimum: decimal("minimum", minimum)?,
                per_share: decimal("per_share", per_share)?,
                max_pct: decimal("max_pct", max_pct)?,
            },
            grid: Grid { buffer, tiers, increment_precision, price_precision },
            limit_entries,
            market_markup,
        })
    }

    /// `sized(cash, close, reference)`, or `None`.
    #[pyo3(name = "sized")]
    fn py_sized(&self, cash: f64, close: f64, reference: f64) -> Option<f64> {
        self.sized(cash, close, reference)
    }

    /// `snapped(reference, buy=buy)`, or `None`.
    #[pyo3(name = "snap")]
    fn py_snap(&self, reference: f64, buy: bool) -> Option<f64> {
        self.snap(reference, buy)
    }

    /// `entry_cost.can_afford(cash, units, price)`, or `None`.
    #[pyo3(name = "affordable")]
    fn py_affordable(&self, cash: f64, units: f64, price: f64) -> Option<bool> {
        self.affordable(cash, units, price)
    }

    /// `cash_is_exhausted(None, price, cash)`, or `None`.
    #[pyo3(name = "exhausted")]
    fn py_exhausted(&self, cash: f64, price: f64) -> Option<bool> {
        self.exhausted(cash, price)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(text: &str) -> Dec {
        Dec::parse(text).unwrap()
    }

    fn same(a: Dec, b: &str) -> bool {
        a.compare(dec(b)) == Some(Ordering::Equal)
    }

    /// Floats a run meets -- prices, lots, cash -- and every awkward kind:
    /// ties, subnormals, the largest and smallest, both zeros.
    fn awkward_floats() -> Vec<f64> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut values = vec![
            0.0,
            -0.0,
            0.5,
            1.5,
            2.5,
            0.125,
            0.375,
            2.675,
            1e-5,
            1e-8,
            0.1,
            0.3,
            92767.66,
            -320.315,
            1e15,
            1e16,
            9007199254740991.0,
            4503599627370496.5,
            1e22,
            1e23,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
            0.999_999_999_999_999_9,
            123456.789e-3,
        ];
        for _ in 0..200_000 {
            let draw = next();
            let value = match draw % 4 {
                // A decimal of up to sixteen digits at some scale.
                0 => {
                    (next() % 10_u64.pow(1 + (draw >> 8) as u32 % 16)) as f64
                        / 10_f64.powi((draw >> 16) as i32 % 18)
                }
                // A tie at some precision.
                1 => ((next() % 1_000_000) as f64 + 0.5) / 10_f64.powi((draw >> 16) as i32 % 6),
                // Any finite float.
                2 => f64::from_bits(next()),
                // A float near a typical price, lot or balance.
                _ => f64::from_bits(0x3e00_0000_0000_0000 + next() % 0x0600_0000_0000_0000),
            };
            if value.is_finite() {
                values.push(if draw & 1 << 40 != 0 { -value } else { value });
            }
        }
        values
    }

    #[test]
    fn the_shortest_digits_are_the_spelled_ones() {
        for x in awkward_floats() {
            let spelled = Dec::parse(&format!("{x:e}")).unwrap();
            if let Some(fast) = Dec::shortest(x) {
                assert_eq!((fast.c, fast.e), (spelled.c, spelled.e), "{x:e}");
            }
        }
        // The fast path answers what a run meets.
        assert!(Dec::shortest(92767.66).is_some() && Dec::shortest(1e-8).is_some());
    }

    #[test]
    fn rounding_and_fixed_digits_match_their_spelling() {
        for x in awkward_floats() {
            for precision in [0, 1, 2, 3, 5, 8, 10, 16, 22, 23] {
                let want: f64 = format!("{:.*}", precision as usize, x).parse().unwrap();
                let got = round_to(x, precision).unwrap();
                assert_eq!(got.to_bits(), want.to_bits(), "round({x:e}, {precision})");
                assert_eq!(fixed(x, precision as usize), format!("{:.*}", precision as usize, x));
            }
        }
        // The fast path answers what a run meets.
        assert!(scaled_half_even(92767.66, 2).is_some() && scaled_half_even(0.00123, 8).is_some());
        assert_eq!(fixed(-0.0, 2), "-0.00");
        assert_eq!(fixed(2.5, 0), "2");
        assert_eq!(fixed(0.000123, 3), "0.000");
    }

    #[test]
    fn reads_floats_through_their_shortest_repr() {
        assert!(same(Dec::of(92767.66).unwrap(), "92767.66"));
        assert!(same(Dec::of(0.1).unwrap(), "0.1"));
        assert!(same(Dec::of(1e-5).unwrap(), "0.00001"));
        assert!(same(Dec::of(100.0).unwrap(), "100"));
        assert!(Dec::of(f64::NAN).is_none());
    }

    #[test]
    fn quantizes_as_decimal_does() {
        assert!(same(dec("1.005").quantize(-2, Rounding::HalfEven).unwrap(), "1.00"));
        assert!(same(dec("1.015").quantize(-2, Rounding::HalfEven).unwrap(), "1.02"));
        assert!(same(dec("1.001").quantize(-2, Rounding::Ceiling).unwrap(), "1.01"));
        assert!(same(dec("-1.001").quantize(-2, Rounding::Ceiling).unwrap(), "-1.00"));
        assert!(same(dec("-1.001").quantize(-2, Rounding::Floor).unwrap(), "-1.01"));
        assert!(same(dec("7").quantize(-2, Rounding::Ceiling).unwrap(), "7"));
    }

    #[test]
    fn declines_past_twenty_eight_digits() {
        // 9999999999999 squared has 26 digits; 99999999999999 times
        // 999999999999999 has 29.
        assert!(dec("9999999999999").mul(dec("9999999999999")).is_some());
        assert!(dec("99999999999999").mul(dec("999999999999999")).is_none());
    }

    fn us_cost() -> Cost {
        Cost {
            lot: dec("1"),
            cent: -2,
            taker_fee: dec("0"),
            minimum: dec("1.00"),
            per_share: dec("0.005"),
            max_pct: dec("0.01"),
        }
    }

    #[test]
    fn cash_rounds_as_its_shortest_repr_does() {
        for x in awkward_floats() {
            for places in [0, 2, 3, 8, 22, 23] {
                let cost = Cost { cent: -places, ..us_cost() };
                let want = Dec::of(x).and_then(|d| d.quantize(-places, Rounding::HalfEven));
                let got = cost.cash(x);
                assert_eq!(got.map(|d| (d.c, d.e)), want.map(|d| (d.c, d.e)), "{x:e} at {places}");
            }
        }
        // The fast path answers what a run meets, and leaves a tie alone.
        assert!(matches!(scaled_half_even_clear(98_765.432_109_876_54, 2), Some((_, true))));
        assert!(matches!(scaled_half_even_clear(1.005, 2), Some((_, false))));
    }

    #[test]
    fn affordable_quantity_is_the_last_affordable_lot() {
        let cost = us_cost();
        for (cash, price) in [(10_000.0, 187.43), (1_000_000.0, 3.21), (5.0, 2.0), (1.0, 2.0)] {
            let units = cost.affordable_quantity(cash, price).unwrap();
            let p = Dec::of(price).unwrap();
            let c = cost.cash(cash).unwrap();
            let fits =
                |u: Dec| cost.buy_cost(u, p).unwrap().compare(c).unwrap() != Ordering::Greater;
            assert!(units.c == 0 || fits(units));
            let next = units.add(cost.lot).unwrap();
            assert!(!fits(next), "{cash} at {price}");
        }
    }

    #[test]
    fn snaps_onto_the_price_grid() {
        let grid =
            Grid { buffer: 0.002, tiers: Vec::new(), increment_precision: 2, price_precision: 2 };
        // 92582.5 * 1.002 is 92767.665: rounded first, it lands on .66.
        assert_eq!(grid.snap(92582.5, true), Some(92767.66));
        let asx = Grid {
            buffer: 0.001,
            tiers: vec![
                (Some(dec("0.10")), dec("0.001")),
                (Some(dec("2.00")), dec("0.005")),
                (None, dec("0.01")),
            ],
            increment_precision: 3,
            price_precision: 3,
        };
        assert_eq!(asx.snap(1.234, true), Some(1.24));
        assert_eq!(asx.snap(1.234, false), Some(1.23));
        assert_eq!(asx.snap(4.79, true), Some(4.80));
    }
}
