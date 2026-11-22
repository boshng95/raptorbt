//! StrategyBase's risk exits (`native_policies`), as both lanes that replay a
//! rule set on the account check them: the full native lane
//! (`python::rule_session`) and the lean lane (`lean::rules`).
//!
//! Three checks, each optional, run in this order on the executing primary
//! bar's close, the last to act deciding (`_policy_intent`):
//!
//! * the trade exit (`NativeTradeExitPolicy`): a stop and take-profit on the
//!   position's return net of its round trip's commission and its borrow;
//! * the ATR exit (`NativeAtrExitPolicy`): a stop -- fixed or trailing the
//!   best close -- and a take-profit, ATRs from the entry, the ATR read on
//!   its own stream's last closed bar;
//! * the time exit (`NativeHoldExitPolicy`): a close once the position has
//!   been held a number of a resample's bars.
//!
//! Neither lane's own state lives here: a lane hands each check the account
//! (`Account`), the bar's close, and the ATR's values, and keeps the
//! checks' memory in a [`PolicyState`].

/// The primary stream's slot: an ATR on it is read on the executing bar.
pub const PRIMARY: usize = 0;

/// The risk policy that closed a position, by its reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyExit {
    TradeStopLoss,
    TradeTakeProfit,
    AtrStop,
    AtrTakeProfit,
    MaxHold,
}

impl PolicyExit {
    pub fn reason(self) -> &'static str {
        match self {
            Self::TradeStopLoss => "trade_stop_loss",
            Self::TradeTakeProfit => "trade_take_profit",
            Self::AtrStop => "atr_stop",
            Self::AtrTakeProfit => "atr_take_profit",
            Self::MaxHold => "max_hold",
        }
    }
}

/// `NativeOrderStyle`: how an exit is ordered. `Default` is the venue's
/// configured close.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderStyle {
    Default,
    Market,
    Limit,
}

impl OrderStyle {
    pub fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Market => "market",
            Self::Limit => "limit",
        }
    }
}

/// The account the risk policies read: the venue's signed lot -- dust and
/// all -- and the open position's average entry (`NativeAccountView`'s
/// `position_quantity` and `average_entry_price`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Account {
    pub quantity: f64,
    pub entry: f64,
}

impl Account {
    pub const FLAT: Self = Self { quantity: 0.0, entry: 0.0 };
}

/// The side a signed quantity holds.
pub fn side_of(quantity: f64) -> i32 {
    if quantity > 0.0 {
        1
    } else if quantity < 0.0 {
        -1
    } else {
        0
    }
}

/// `NativeTradeExitPolicy`: the percentage stop and take-profit on the
/// position's net return.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TradeExit {
    pub stop: f64,
    pub take_profit: f64,
}

/// `NativeAtrExitPolicy`'s terms. Its ATR is indexed in its own stream's
/// bars: the executing primary bar's for `slot` [`PRIMARY`],
/// else the last bar of that resample to close.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtrExit {
    pub slot: usize,
    pub stop: f64,
    pub take_profit: f64,
    pub trailing: bool,
}

/// `NativeHoldExitPolicy`: a time exit counted in bars of resample `slot`,
/// or of the primary ([`PRIMARY`]) on a session visiting each of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoldExit {
    pub slot: usize,
    pub bars: i64,
}

/// The ATR's values and initialized flags, in its stream's bars.
pub type AtrValues<'a> = (&'a [f64], &'a [bool]);

/// StrategyBase's risk checks, as `_POLICY_COMPILERS` builds them: each
/// optional, checked in this order, the last to act deciding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Risk {
    pub trade: Option<TradeExit>,
    pub atr: Option<AtrExit>,
    pub hold: Option<HoldExit>,
    /// The contract's taker fee where commissions are on, else 0.
    pub fee_rate: f64,
}

/// `_PositionCharges`: the commission the open position's fills were
/// charged, and the signed quantity those fills add up to.
#[derive(Clone, Copy, Debug)]
pub struct Charges {
    pub charged: f64,
    pub held: f64,
}

impl Charges {
    /// `round_trip`.
    pub fn round_trip(&self, fee_rate: f64, units: f64, entry: f64, current: f64) -> f64 {
        if self.charged > 0.0 {
            return self.charged * 2.0;
        }
        fee_rate * units * (entry + current)
    }

    /// `on_fill`: book a fill of `quantity` on `side` (1 buys) charged
    /// `commission`, which left the account `flat` -- and say so.
    pub fn on_fill(&mut self, side: f64, quantity: f64, commission: f64, flat: bool) -> bool {
        let held = self.held;
        self.held = held + side * quantity;
        if flat {
            self.charged = 0.0;
            self.held = 0.0;
            return true;
        }
        let commission = commission.abs();
        if held == 0.0 {
            self.charged = commission;
        } else if side * held < 0.0 && quantity >= held.abs() * (1.0 - 1e-9) {
            // The close leg of a replacement.
            self.charged = 0.0;
            self.held = 0.0;
        } else {
            self.charged += commission;
        }
        false
    }
}

/// `_NativeBorrowTracker` on a lane that observes every primary bar
/// (`open_short_borrow`): per primary bar its market date and the date a
/// short opened on it settles, both in days from the epoch, and one share's
/// fee for each day after `first_day` -- the first bar's date -- summed:
/// `charged[k]` through `first_day + k`, in `10**-BORROW_FEE_DIGITS` of the
/// currency.
#[derive(Clone, Copy, Debug)]
pub struct BorrowView<'a> {
    pub days: &'a [i32],
    pub settles: &'a [i32],
    pub first_day: i32,
    pub charged: &'a [i64],
}

/// `BORROW_FEE_DIGITS`: a fee is a whole number of `10**-10` of the currency.
pub const BORROW_FEE_SCALE: f64 = 1e10;

/// `float(Decimal(units).scaleb(-BORROW_FEE_DIGITS))`: the double nearest the
/// fee. Below 2**53 both operands are exact and the division rounds once;
/// past it the decimal spelling is parsed, which rounds once too.
pub fn borrow_amount(units: i128) -> f64 {
    if units.unsigned_abs() < 1 << 53 {
        units as f64 / BORROW_FEE_SCALE
    } else {
        format!("{units}e-10").parse().expect("an integer's decimal spelling parses")
    }
}

impl BorrowView<'_> {
    /// `_NativeBorrowTracker.observe` on primary bar `idx`, the venue holding
    /// signed `quantity`: anything but a short ends the loan; a short with
    /// none opens one, settling as a short opened on the bar does; and each
    /// date the bar reaches past the one the loan is charged through is
    /// charged on the shares held now, at one share's fee for those days.
    pub fn observe(
        &self,
        p: &mut PolicyState,
        idx: usize,
        quantity: f64,
    ) -> Result<(), &'static str> {
        if quantity >= 0.0 {
            p.loan = None;
            p.borrow_units = 0;
            p.borrow = 0.0;
            return Ok(());
        }
        let marked = *p.loan.get_or_insert(self.settles[idx] - 1);
        let day = self.days[idx];
        if day <= marked {
            return Ok(());
        }
        let shares = -quantity;
        if shares.fract() != 0.0 || shares >= (1u64 << 53) as f64 {
            return Err("a short's borrow is charged on whole shares");
        }
        // A loan is charged from no earlier than the day after the first
        // bar's date, and through no later than the last bar's.
        let since = (marked - self.first_day) as usize;
        let through = (day - self.first_day) as usize;
        let fees = i128::from(self.charged[through] - self.charged[since]);
        p.borrow_units += shares as i128 * fees;
        p.loan = Some(day);
        p.borrow = borrow_amount(p.borrow_units);
        Ok(())
    }

    /// Refuse arrays [`Self::observe`] could read past, over `bars` primary
    /// bars: a date and a settlement for every bar, dates that never fall,
    /// from `first_day` on, each loan settling after the date it opens, and
    /// a sum for every date from `first_day` through the last bar's.
    pub fn validate(&self, bars: usize) -> Result<(), String> {
        let refuse = |what: &str| Err(format!("borrow terms: {what}"));
        if self.days.len() != bars || self.settles.len() != bars {
            return refuse("a date and a settlement for every primary bar");
        }
        let (Some(&first), Some(&last)) = (self.days.first(), self.days.last()) else {
            return Ok(());
        };
        if first != self.first_day || self.days.windows(2).any(|w| w[1] < w[0]) {
            return refuse("dates rising from the first bar's");
        }
        if self.days.iter().zip(self.settles).any(|(&day, &settles)| settles <= day) {
            return refuse("a loan settles after the date it opens");
        }
        if self.charged.len() != (last - first) as usize + 1 || self.charged[0] != 0 {
            return refuse("a sum for every date through the last bar's");
        }
        Ok(())
    }
}

/// What the risk policies keep between checks.
#[derive(Clone, Copy, Debug)]
pub struct PolicyState {
    /// `NativeTradeExitPolicy._charges`.
    pub trade_charges: Charges,
    /// `NativeAtrExitPolicy`'s `_charges`, `_atr_index`, `_entry_price` and
    /// `_extreme`.
    pub atr_charges: Charges,
    pub atr_index: i64,
    pub atr_entry: f64,
    pub atr_extreme: f64,
    /// `NativeHoldExitPolicy`'s `_counted` and `_mark`.
    pub hold_counted: i64,
    pub hold_mark: i64,
    /// `_held_policy_streams` over the call in progress, as releasing it
    /// moves the policies: whether it holds any close, the last of the
    /// ATR's resample among them (-1 for none), and how many of the time
    /// exit's.
    pub held: bool,
    pub held_atr: i64,
    pub held_counts: i64,
    /// `policy_resampled`.
    pub resampled: bool,
    /// `_NativeBorrowTracker._accrual`: the date the open short's loan is
    /// charged through, `None` with no loan; what it has charged, in
    /// `10**-BORROW_FEE_DIGITS` of the currency; and that as the account's
    /// `accrued_borrow`. Untouched on a run that borrows nothing.
    pub loan: Option<i32>,
    pub borrow_units: i128,
    pub borrow: f64,
}

impl PolicyState {
    pub const NEW: Self = Self {
        trade_charges: Charges { charged: 0.0, held: 0.0 },
        atr_charges: Charges { charged: 0.0, held: 0.0 },
        atr_index: -1,
        atr_entry: 0.0,
        atr_extreme: 0.0,
        hold_counted: 0,
        hold_mark: -1,
        held: false,
        held_atr: -1,
        held_counts: 0,
        resampled: false,
        loan: None,
        borrow_units: 0,
        borrow: 0.0,
    };

    /// `release_policy_streams`.
    pub fn release(&mut self) {
        if self.held_atr >= 0 {
            self.atr_index = self.held_atr;
        }
        self.hold_counted += self.held_counts;
        self.held = false;
        self.held_atr = -1;
        self.held_counts = 0;
    }

    /// Whether no resample close is held from the checks.
    pub fn holds_nothing(&self) -> bool {
        !self.held && self.held_atr < 0 && self.held_counts == 0
    }
}

/// `trade_roi_pct`.
pub fn trade_roi_pct(entry: f64, current: f64, position: f64, costs: f64) -> f64 {
    let quantity = position.abs();
    let notional = entry * quantity;
    if notional <= 0.0 {
        return 0.0;
    }
    let gross =
        if position > 0.0 { (current - entry) * quantity } else { (entry - current) * quantity };
    ((gross - costs) / notional) * 100.0
}

impl Risk {
    /// `NativeModelRunner._advance_readiness` for the policies: a close of
    /// resample `slot`'s bar `bar`, told now where it is decided ahead of the
    /// primary bar's risk check (`ahead`), else held back until that check
    /// has run (`_held_policy_streams`).
    pub fn stream(&self, p: &mut PolicyState, slot: usize, bar: usize, ahead: bool) {
        let atr = self.atr.is_some_and(|atr| atr.slot == slot);
        let hold = self.hold.is_some_and(|hold| hold.slot == slot);
        if ahead {
            if atr {
                p.atr_index = bar as i64;
            }
            if hold {
                p.hold_counted += 1;
            }
            return;
        }
        p.held = true;
        if atr {
            p.held_atr = bar as i64;
        }
        if hold {
            p.held_counts += 1;
        }
    }

    /// `_advance_readiness` of admitted primary bar, for the policies: a
    /// time exit counting the primary bars (`NativeHoldExitPolicy`'s
    /// `_stream` None) counts it at once, ahead of the bar's own check. The
    /// ATR on the primary reads the executing bar itself.
    pub fn primary(&self, p: &mut PolicyState) {
        if self.hold.is_some_and(|hold| hold.slot == PRIMARY) {
            p.hold_counted += 1;
        }
    }

    /// `NativeModelRunner.apply_policies` for the executing primary bar
    /// `idx`, closing at `current`: checked on the bar -- unless the session
    /// refuses it (`primary`) -- and again once the resample closes it held
    /// back have reached the policies.
    pub fn apply(
        &self,
        atr: Option<AtrValues>,
        p: &mut PolicyState,
        account: Account,
        idx: usize,
        current: f64,
        primary: bool,
    ) -> Option<PolicyExit> {
        let mut selected = if primary { self.intent(atr, p, account, idx, current) } else { None };
        p.resampled = false;
        let resampled = p.held;
        p.release();
        if selected.is_none() && resampled {
            selected = self.intent(atr, p, account, idx, current);
            p.resampled = selected.is_some();
        }
        selected
    }

    /// `_policy_intent`: every policy's `on_bar`, in order, the last to act
    /// deciding.
    pub fn intent(
        &self,
        atr: Option<AtrValues>,
        p: &mut PolicyState,
        account: Account,
        idx: usize,
        current: f64,
    ) -> Option<PolicyExit> {
        let quantity = account.quantity;
        let mut selected = None;
        if let Some(trade) = &self.trade {
            p.trade_charges.held = quantity;
            let entry = account.entry;
            if quantity != 0.0 && entry > 0.0 {
                let costs =
                    p.trade_charges.round_trip(self.fee_rate, quantity.abs(), entry, current)
                        + p.borrow;
                let roi = trade_roi_pct(entry, current, quantity, costs);
                if trade.stop > 0.0 && roi <= -trade.stop {
                    selected = Some(PolicyExit::TradeStopLoss);
                } else if trade.take_profit > 0.0 && roi >= trade.take_profit {
                    selected = Some(PolicyExit::TradeTakeProfit);
                }
            }
        }
        if let (Some(exit), Some((values, initialized))) = (&self.atr, atr) {
            if let Some(exit) = self.atr_exit(exit, values, initialized, p, account, idx, current) {
                selected = Some(exit);
            }
        }
        if let Some(hold) = &self.hold {
            if quantity == 0.0 {
                p.hold_mark = -1;
            } else {
                if p.hold_mark < 0 {
                    p.hold_mark = p.hold_counted;
                }
                if p.hold_counted - p.hold_mark >= hold.bars {
                    selected = Some(PolicyExit::MaxHold);
                }
            }
        }
        selected
    }

    /// `NativeAtrExitPolicy.on_bar`.
    #[allow(clippy::too_many_arguments)]
    pub fn atr_exit(
        &self,
        atr: &AtrExit,
        values: &[f64],
        initialized: &[bool],
        p: &mut PolicyState,
        account: Account,
        idx: usize,
        current: f64,
    ) -> Option<PolicyExit> {
        let quantity = account.quantity;
        p.atr_charges.held = quantity;
        let index = if atr.slot == PRIMARY { idx as i64 } else { p.atr_index };
        if index < 0 || !initialized[index as usize] {
            return None;
        }
        if quantity == 0.0 {
            p.atr_entry = 0.0;
            p.atr_charges.charged = 0.0;
            p.atr_extreme = 0.0;
            return None;
        }
        if p.atr_entry == 0.0 {
            p.atr_entry = account.entry;
            if p.atr_entry <= 0.0 {
                return None;
            }
            p.atr_extreme = p.atr_entry;
        }
        let long = quantity > 0.0;
        if atr.trailing {
            // Python's `max` and `min`: the first argument unless the second
            // compares past it.
            let extreme = p.atr_extreme;
            p.atr_extreme = if long {
                if current > extreme {
                    current
                } else {
                    extreme
                }
            } else if current < extreme {
                current
            } else {
                extreme
            };
        }
        let anchor = if atr.trailing { p.atr_extreme } else { p.atr_entry };
        let units = quantity.abs();
        let commission = p.atr_charges.round_trip(self.fee_rate, units, p.atr_entry, current);
        let per_share = if units > 0.0 { (commission + p.borrow) / units } else { 0.0 };
        let direction = if long { 1.0 } else { -1.0 };
        let value = values[index as usize];
        if atr.stop > 0.0 {
            let stop = anchor - direction * (value * atr.stop - per_share);
            if if long { current <= stop } else { current >= stop } {
                p.atr_entry = 0.0;
                p.atr_extreme = 0.0;
                return Some(PolicyExit::AtrStop);
            }
        }
        if atr.take_profit > 0.0 {
            let take_profit = p.atr_entry + direction * (value * atr.take_profit + per_share);
            if if long { current >= take_profit } else { current <= take_profit } {
                p.atr_entry = 0.0;
                p.atr_extreme = 0.0;
                return Some(PolicyExit::AtrTakeProfit);
            }
        }
        None
    }

    /// The policies' `on_fill` past the borrow tracker's: a fill of
    /// `quantity` on `side` (1 buys) charged `commission`, which left the
    /// account `flat`.
    pub fn on_fill(
        &self,
        p: &mut PolicyState,
        side: f64,
        quantity: f64,
        commission: f64,
        flat: bool,
    ) {
        if self.trade.is_some() {
            p.trade_charges.on_fill(side, quantity, commission, flat);
        }
        if self.atr.is_some() && p.atr_charges.on_fill(side, quantity, commission, flat) {
            p.atr_entry = 0.0;
            p.atr_extreme = 0.0;
        }
    }

    /// Whether a flat bar's risk checks leave `p` as they find it: each
    /// clears what it keeps once nothing is held, and the borrow tracker --
    /// where the run `borrows` -- ends the loan. `atr` is whether the ATR
    /// check has its values.
    pub fn flat_settled(
        &self,
        atr: bool,
        borrows: bool,
        p: &PolicyState,
        account: Account,
        halted: bool,
    ) -> bool {
        if borrows && (p.loan.is_some() || p.borrow_units != 0 || p.borrow.to_bits() != 0) {
            return false;
        }
        if halted {
            // A halted run's checks only release the closes held from them.
            return true;
        }
        let held = account.quantity.to_bits();
        if p.resampled
            || (self.trade.is_some() && p.trade_charges.held.to_bits() != held)
            || (self.hold.is_some() && p.hold_mark != -1)
        {
            return false;
        }
        if self.atr.is_some() && atr {
            return p.atr_charges.held.to_bits() == held
                && p.atr_entry.to_bits() == 0
                && p.atr_charges.charged.to_bits() == 0
                && p.atr_extreme.to_bits() == 0;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{
        borrow_amount, trade_roi_pct, Account, AtrExit, BorrowView, Charges, HoldExit, PolicyExit,
        PolicyState, Risk, PRIMARY,
    };

    #[test]
    fn a_time_exit_on_the_primary_counts_each_bar_ahead_of_its_check() {
        // `NativeHoldExitPolicy` with nothing resampled: each primary bar's
        // visit counts it before the bar's check, which marks the position
        // at the first check that sees it and closes once held two bars.
        let hold = HoldExit { slot: PRIMARY, bars: 2 };
        let risk = Risk { trade: None, atr: None, hold: Some(hold), fee_rate: 0.0 };
        let held = Account { quantity: 1.0, entry: 100.0 };
        let mut p = PolicyState::NEW;
        let bar = |p: &mut PolicyState, account, idx| {
            risk.primary(p);
            risk.apply(None, p, account, idx, 100.0, true)
        };
        assert_eq!(bar(&mut p, Account::FLAT, 0), None);
        assert_eq!(bar(&mut p, held, 1), None);
        assert_eq!(p.hold_mark, 2);
        assert_eq!(bar(&mut p, held, 2), None);
        assert_eq!(bar(&mut p, held, 3), Some(PolicyExit::MaxHold));
        assert_eq!(bar(&mut p, Account::FLAT, 4), None);
        assert_eq!(p.hold_mark, -1);
        // A resample's time exit counts nothing on a primary bar.
        let resampled = Risk { hold: Some(HoldExit { slot: 1, ..hold }), ..risk };
        let mut p = PolicyState::NEW;
        resampled.primary(&mut p);
        assert_eq!(p.hold_counted, 0);
    }

    #[test]
    fn charges_book_what_the_open_position_paid() {
        // `_PositionCharges`: the opening fill's commission, plus every fill
        // after it that reduced the position; flat starts over, and so does
        // the close leg of a replacement.
        let mut charges = Charges { charged: 0.0, held: 0.0 };
        assert!(!charges.on_fill(1.0, 10.0, -1.0, false));
        assert_eq!((charges.charged, charges.held), (1.0, 10.0));
        assert!(!charges.on_fill(-1.0, 4.0, 0.4, false));
        assert_eq!((charges.charged, charges.held), (1.4, 6.0));
        assert_eq!(charges.round_trip(0.01, 6.0, 100.0, 110.0), 2.8);
        assert!(charges.on_fill(-1.0, 6.0, 0.6, true));
        assert_eq!((charges.charged, charges.held), (0.0, 0.0));
        // Uncharged, the fee rate across both notionals.
        assert_eq!(charges.round_trip(0.5, 2.0, 100.0, 110.0), 210.0);
        charges.on_fill(1.0, 10.0, 1.0, false);
        assert!(!charges.on_fill(-1.0, 10.0, 1.0, false));
        assert_eq!((charges.charged, charges.held), (0.0, 0.0));
    }

    #[test]
    fn a_trade_returns_net_of_costs_on_either_side() {
        assert_eq!(trade_roi_pct(100.0, 110.0, 2.0, 2.0), 9.0);
        assert_eq!(trade_roi_pct(100.0, 90.0, -2.0, 2.0), 9.0);
        assert_eq!(trade_roi_pct(100.0, 90.0, 2.0, 0.0), -10.0);
        assert_eq!(trade_roi_pct(0.0, 90.0, 2.0, 0.0), 0.0);
    }

    #[test]
    fn a_trailing_atr_stop_hangs_off_the_best_close() {
        // Long one unit from 100, ATR 2, a 2-ATR trailing stop: 96 at entry,
        // 101 once 105 prints, hit at 101. The hit clears the entry.
        let risk = Risk { trade: None, atr: None, hold: None, fee_rate: 0.0 };
        let atr = AtrExit { slot: PRIMARY, stop: 2.0, take_profit: 0.0, trailing: true };
        let (values, initialized) = ([2.0; 4], [false, true, true, true]);
        let account = Account { quantity: 1.0, entry: 100.0 };
        let mut p = PolicyState::NEW;
        let mut step =
            |idx, close| risk.atr_exit(&atr, &values, &initialized, &mut p, account, idx, close);
        // Not ready yet: nothing anchors.
        assert_eq!(step(0, 100.0), None);
        assert_eq!(step(1, 100.0), None);
        assert_eq!(step(2, 105.0), None);
        assert_eq!(step(3, 101.0), Some(PolicyExit::AtrStop));
        assert_eq!((p.atr_entry, p.atr_extreme), (0.0, 0.0));
        // A short's stop sits above, and trails the lowest close.
        let short = Account { quantity: -1.0, entry: 100.0 };
        let mut p = PolicyState::NEW;
        let fixed = AtrExit { trailing: false, take_profit: 1.0, ..atr };
        let mut step =
            |idx, close| risk.atr_exit(&fixed, &values, &initialized, &mut p, short, idx, close);
        assert_eq!(step(1, 100.0), None);
        assert_eq!(step(2, 98.0), Some(PolicyExit::AtrTakeProfit));
    }

    #[test]
    fn a_fee_is_the_double_nearest_its_decimal() {
        assert_eq!(borrow_amount(0), 0.0);
        assert_eq!(borrow_amount(12_345), 12_345e-10);
        assert_eq!(borrow_amount(-7), -7e-10);
        // Past 2**53 the integer is no longer exact as a double: the
        // decimal spelling still rounds once.
        let big: i128 = (1 << 60) + 1;
        assert_eq!(borrow_amount(big), "1152921504606846977e-10".parse::<f64>().unwrap());
        assert_eq!(borrow_amount(1 << 53), (1u64 << 53) as f64 / 1e10);
    }

    #[test]
    fn a_loan_is_charged_from_settlement_on_the_shares_held() {
        // Five bars over dates 10, 10, 11, 14, 15; a short opened on date 10
        // settles on 12, one opened on 14 on 16. One share costs 3, 5, 7, 11
        // and 13 units on dates 11..=15.
        let days = [10, 10, 11, 14, 15];
        let settles = [12, 12, 13, 16, 17];
        let charged = [0, 3, 8, 15, 26, 39];
        let borrow =
            BorrowView { days: &days, settles: &settles, first_day: 10, charged: &charged };
        let mut p = PolicyState::NEW;
        // Long or flat: no loan.
        borrow.observe(&mut p, 0, 4.0).unwrap();
        assert_eq!((p.loan, p.borrow_units), (None, 0));
        // Opened on bar 1: charged through date 11, the day before it settles.
        borrow.observe(&mut p, 1, -2.0).unwrap();
        assert_eq!((p.loan, p.borrow_units), (Some(11), 0));
        borrow.observe(&mut p, 2, -2.0).unwrap();
        assert_eq!((p.loan, p.borrow_units), (Some(11), 0));
        // Date 14 charges 12, 13 and 14 -- 5 + 7 + 11 units a share -- on
        // the three shares held when it gets there.
        borrow.observe(&mut p, 3, -3.0).unwrap();
        assert_eq!((p.loan, p.borrow_units), (Some(14), 3 * 23));
        assert_eq!(p.borrow, 69.0 / 1e10);
        borrow.observe(&mut p, 4, -1.0).unwrap();
        assert_eq!((p.loan, p.borrow_units), (Some(15), 69 + 13));
        // Covered: the loan ends, and a new one starts from its own bar.
        borrow.observe(&mut p, 4, 0.0).unwrap();
        assert_eq!((p.loan, p.borrow_units, p.borrow), (None, 0, 0.0));
        borrow.observe(&mut p, 3, -1.0).unwrap();
        assert_eq!(p.loan, Some(15));
        // Fractional shares are not charged here.
        let mut p = PolicyState::NEW;
        borrow.observe(&mut p, 1, -0.5).unwrap();
        assert!(borrow.observe(&mut p, 3, -0.5).is_err());
    }
}
