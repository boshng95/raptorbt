//! Putting a size onto an instrument's lot grid.
//!
//! A lot increment is a decimal quantity, and every arithmetic step that
//! reaches one has to survive the trip through binary. Both operations here
//! exist because a plain `(size / lot).floor() * lot` does not: the quotient
//! can land a few ULPs under an integer boundary and lose a whole lot, and
//! the product back up can land an ULP above the decimal size a venue would
//! have quoted.

/// A lot increment together with the decimal scale derived from it.
///
/// The increment is immutable for an instrument but the old helpers derived
/// its scale with `log10` and `powi` on every bar and every fill. Keeping that
/// invariant here makes the hot path arithmetic-only without changing a
/// single floating-point operation applied to the actual size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LotGrid {
    lot: f64,
    scale: Option<f64>,
}

impl Default for LotGrid {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl LotGrid {
    pub(crate) fn new(lot: f64) -> Self {
        let scale = if lot > 0.0 && lot.is_finite() {
            let decimals = (-lot.log10()).ceil().max(0.0);
            // Beyond ~15 significant decimals the scaling itself loses
            // precision, so leave values alone rather than corrupting them.
            (decimals <= 15.0).then(|| 10f64.powi(decimals as i32))
        } else {
            None
        };
        Self { lot, scale }
    }

    #[inline]
    pub(crate) fn lot(self) -> f64 {
        self.lot
    }

    #[inline]
    pub(crate) fn floor(self, raw_size: f64) -> f64 {
        let lots = raw_size / self.lot;
        let boundary_tolerance = f64::EPSILON * lots.abs().max(1.0) * 4.0;
        self.snap((lots + boundary_tolerance).floor() * self.lot)
    }

    #[inline]
    pub(crate) fn snap(self, value: f64) -> f64 {
        match self.scale {
            Some(scale) if value.is_finite() => (value * scale).round() / scale,
            _ => value,
        }
    }
}

/// Floor to the lot grid without dropping an exact decimal-grid value
/// because its binary quotient landed a few ULPs below the integer boundary.
///
/// The tolerance is relative to the lot count, not absolute: a size of
/// 481.335 on a `0.00001` grid is 48,133,500 lots, where one ULP of the
/// quotient is already larger than any fixed epsilon small enough to be safe
/// on a size of 0.1.
pub(crate) fn floor_to_lot(raw_size: f64, lot: f64) -> f64 {
    LotGrid::new(lot).floor(raw_size)
}

/// Return `value` expressed on the lot increment's own decimal scale.
///
/// Flooring already tolerates a quotient that lands just under the boundary,
/// but multiplying the whole lot count back up is exact only in decimal, not
/// in binary: `10379 * 0.00001` evaluates to `0.10379000000000001`, one ULP
/// above the `0.10379` that Nautilus holds as a fixed-precision quantity.
/// The stray ULP is invisible in the size itself and becomes visible in the
/// ledger, because a percentage fee is charged on `size * price`: it moves
/// the true product off the value the decimal size would have produced, and
/// where that product sits on a rounding tie it decides the last decimal of
/// the commission. Rounding onto the increment's decimal scale recovers the
/// size that was intended.
pub(crate) fn snap_to_lot_grid(value: f64, lot: f64) -> f64 {
    LotGrid::new(lot).snap(value)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // 0.10379000000000001. Nautilus holds the same quantity as a
        // decimal, and a percentage fee charged on `size * price` turns that
        // ULP into a last-decimal commission difference. Pin equality, not a
        // tolerance.
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
    fn a_large_lot_count_keeps_the_lot_an_absolute_epsilon_would_lose() {
        // 1925.34 / 4 is 481.335 to the decimal and a hair under it in
        // binary -- but at 48,133,500 lots the shortfall is thousands of
        // times a fixed 1e-9 on the lot count. Flooring it away costs a lot,
        // and downstream that lot lands on whatever print carries the
        // rounding remainder.
        assert_eq!(floor_to_lot(1925.34 / 4.0, 0.00001), 481.335);
    }

    #[test]
    fn cached_grid_is_bit_identical_to_uncached_arithmetic() {
        fn uncached_snap(value: f64, lot: f64) -> f64 {
            if !matches!(lot.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater))
                || !value.is_finite()
            {
                return value;
            }
            let decimals = (-lot.log10()).ceil().max(0.0);
            if decimals > 15.0 {
                return value;
            }
            let scale = 10f64.powi(decimals as i32);
            (value * scale).round() / scale
        }

        fn uncached_floor(raw_size: f64, lot: f64) -> f64 {
            let lots = raw_size / lot;
            let boundary_tolerance = f64::EPSILON * lots.abs().max(1.0) * 4.0;
            uncached_snap((lots + boundary_tolerance).floor() * lot, lot)
        }

        for lot in [0.00001, 0.001, 0.1, 1.0, 25.0, 1e-16, 0.0, -1.0] {
            let grid = LotGrid::new(lot);
            for value in [0.0, 0.101849, 0.10379, 481.335, 1925.34, f64::INFINITY] {
                assert_eq!(grid.snap(value).to_bits(), uncached_snap(value, lot).to_bits());
                assert_eq!(grid.floor(value).to_bits(), uncached_floor(value, lot).to_bits());
            }
        }
    }
}
