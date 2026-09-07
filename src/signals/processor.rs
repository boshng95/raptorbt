//! Signal processor for cleaning entry/exit signals.
//!
//! Ensures proper alternation between entries and exits to prevent
//! overlapping positions or orphaned signals.

use crate::core::types::Direction;

/// Signal processor for cleaning raw entry/exit signals.
#[derive(Debug, Clone)]
pub struct SignalProcessor {
    /// Whether to allow multiple entries before an exit (pyramiding).
    pub allow_pyramiding: bool,
    /// Maximum number of pyramid entries.
    pub max_pyramid_entries: usize,
}

impl Default for SignalProcessor {
    fn default() -> Self {
        Self { allow_pyramiding: false, max_pyramid_entries: 1 }
    }
}

impl SignalProcessor {
    /// Create a new signal processor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable pyramiding with a maximum number of entries.
    pub fn with_pyramiding(mut self, max_entries: usize) -> Self {
        self.allow_pyramiding = max_entries > 1;
        self.max_pyramid_entries = max_entries;
        self
    }

    /// Clean entry/exit signals to ensure proper alternation.
    ///
    /// Rules:
    /// 1. First signal must be an entry
    /// 2. After an entry, ignore further entries (unless pyramiding)
    /// 3. After an exit, ignore further exits
    /// 4. Entries and exits must alternate properly
    /// 5. Same-bar conflict: If both entry AND exit signals are True on the same bar
    ///    when in position, entry takes priority — stay in position (ignore the exit).
    ///
    /// # Arguments
    /// * `entries` - Raw entry signals
    /// * `exits` - Raw exit signals
    ///
    /// # Returns
    /// Tuple of (cleaned_entries, cleaned_exits)
    pub fn clean_signals(&self, entries: &[bool], exits: &[bool]) -> (Vec<bool>, Vec<bool>) {
        let n = entries.len();
        assert_eq!(n, exits.len(), "Entry and exit arrays must have same length");

        let mut clean_entries = vec![false; n];
        let mut clean_exits = vec![false; n];

        if n == 0 {
            return (clean_entries, clean_exits);
        }

        let mut in_position = false;
        let mut position_count = 0;

        for i in 0..n {
            if !in_position {
                // Not in position - looking for entry
                if entries[i] {
                    clean_entries[i] = true;
                    in_position = true;
                    position_count = 1;
                }
                // Ignore exits when not in position
            } else {
                // In position - looking for exit (or pyramid entry)
                // Same-bar conflict: entry takes priority — stay in position
                if exits[i] && !entries[i] {
                    // Only exit if there's no conflicting entry signal
                    clean_exits[i] = true;
                    if self.allow_pyramiding {
                        position_count -= 1;
                        if position_count == 0 {
                            in_position = false;
                        }
                    } else {
                        in_position = false;
                        position_count = 0;
                    }
                } else if entries[i]
                    && self.allow_pyramiding
                    && position_count < self.max_pyramid_entries
                {
                    // Pyramid entry
                    clean_entries[i] = true;
                    position_count += 1;
                }
                // If both entry and exit are True, we stay in position (ignore both)
                // If only entry is True and not pyramiding, ignore entry (already in position)
            }
        }

        (clean_entries, clean_exits)
    }

    /// Clean entry/exit signals whose entries name their own direction.
    ///
    /// This is [`Self::clean_signals`] for a two-sided run. It keeps that
    /// method's alternation rules and adds the one case a single-sided run
    /// cannot express: an entry arriving against the open position's
    /// direction is a **reversal**, and closes the position and opens the
    /// other side on the same bar rather than being ignored as a duplicate
    /// entry. That is what a stop-and-reverse strategy does, and what the
    /// kernel's own step order already allows -- an exit and a re-entry may
    /// both land on one bar.
    ///
    /// `directions` carries `1`/`-1` per bar and is read only where `entries`
    /// is set; a bar whose direction names neither side takes `fallback`, so
    /// a caller that leaves the array zeroed gets `clean_signals` back.
    ///
    /// Returns the cleaned entries, the cleaned exits, and the direction each
    /// surviving entry opens in.
    pub fn clean_signals_signed(
        &self,
        entries: &[bool],
        exits: &[bool],
        directions: &[i8],
        fallback: Direction,
    ) -> (Vec<bool>, Vec<bool>, Vec<i8>) {
        let n = entries.len();
        assert_eq!(n, exits.len(), "Entry and exit arrays must have same length");
        assert_eq!(n, directions.len(), "Entry and direction arrays must have same length");

        let mut clean_entries = vec![false; n];
        let mut clean_exits = vec![false; n];
        let mut clean_directions = vec![0i8; n];

        let mut held: Option<Direction> = None;

        for i in 0..n {
            let wanted = Direction::from_int(i32::from(directions[i])).unwrap_or(fallback);
            match held {
                None => {
                    // Flat: an entry opens, and an exit has nothing to close.
                    if entries[i] {
                        clean_entries[i] = true;
                        clean_directions[i] = wanted as i8;
                        held = Some(wanted);
                    }
                }
                Some(open) if entries[i] && wanted != open => {
                    // Reversal: close this side and open the other, on one bar.
                    clean_exits[i] = true;
                    clean_entries[i] = true;
                    clean_directions[i] = wanted as i8;
                    held = Some(wanted);
                }
                Some(_) => {
                    // Same-side entry keeps the position, exactly as
                    // `clean_signals` does -- including when an exit shares
                    // the bar, where the entry still takes priority.
                    if exits[i] && !entries[i] {
                        clean_exits[i] = true;
                        held = None;
                    }
                }
            }
        }

        (clean_entries, clean_exits, clean_directions)
    }

    /// Clean signals with direction awareness (for strategies that can go long/short).
    ///
    /// # Arguments
    /// * `long_entries` - Long entry signals
    /// * `long_exits` - Long exit signals
    /// * `short_entries` - Short entry signals
    /// * `short_exits` - Short exit signals
    ///
    /// # Returns
    /// Tuple of (clean_long_entries, clean_long_exits, clean_short_entries, clean_short_exits)
    pub fn clean_signals_bidirectional(
        &self,
        long_entries: &[bool],
        long_exits: &[bool],
        short_entries: &[bool],
        short_exits: &[bool],
    ) -> (Vec<bool>, Vec<bool>, Vec<bool>, Vec<bool>) {
        let n = long_entries.len();
        assert_eq!(n, long_exits.len());
        assert_eq!(n, short_entries.len());
        assert_eq!(n, short_exits.len());

        let mut clean_long_entries = vec![false; n];
        let mut clean_long_exits = vec![false; n];
        let mut clean_short_entries = vec![false; n];
        let mut clean_short_exits = vec![false; n];

        if n == 0 {
            return (clean_long_entries, clean_long_exits, clean_short_entries, clean_short_exits);
        }

        let mut current_direction: Option<Direction> = None;

        for i in 0..n {
            match current_direction {
                None => {
                    // Not in any position - look for entry
                    if long_entries[i] {
                        clean_long_entries[i] = true;
                        current_direction = Some(Direction::Long);
                    } else if short_entries[i] {
                        clean_short_entries[i] = true;
                        current_direction = Some(Direction::Short);
                    }
                }
                Some(Direction::Long) => {
                    // In long position - look for exit or reversal
                    if long_exits[i] {
                        clean_long_exits[i] = true;
                        current_direction = None;
                    } else if short_entries[i] {
                        // Reversal: exit long and enter short
                        clean_long_exits[i] = true;
                        clean_short_entries[i] = true;
                        current_direction = Some(Direction::Short);
                    }
                }
                Some(Direction::Short) => {
                    // In short position - look for exit or reversal
                    if short_exits[i] {
                        clean_short_exits[i] = true;
                        current_direction = None;
                    } else if long_entries[i] {
                        // Reversal: exit short and enter long
                        clean_short_exits[i] = true;
                        clean_long_entries[i] = true;
                        current_direction = Some(Direction::Long);
                    }
                }
            }
        }

        (clean_long_entries, clean_long_exits, clean_short_entries, clean_short_exits)
    }

    /// Generate exit-on-opposite-entry signals.
    ///
    /// Useful for strategies where an entry in opposite direction
    /// should automatically close the current position.
    ///
    /// # Arguments
    /// * `entries` - Entry signals
    /// * `direction` - Current position direction
    ///
    /// # Returns
    /// Modified exit signals that include opposite-direction entries as exits
    pub fn exits_from_opposite_entries(
        &self,
        long_entries: &[bool],
        short_entries: &[bool],
    ) -> (Vec<bool>, Vec<bool>) {
        let n = long_entries.len();
        assert_eq!(n, short_entries.len());

        // Long exits when short entry
        // Short exits when long entry
        (short_entries.to_vec(), long_entries.to_vec())
    }

    /// Count the number of trades that would be generated from signals.
    ///
    /// # Arguments
    /// * `entries` - Entry signals (already cleaned)
    /// * `exits` - Exit signals (already cleaned)
    ///
    /// # Returns
    /// Number of complete trades (entry + exit pairs)
    pub fn count_trades(_entries: &[bool], exits: &[bool]) -> usize {
        exits.iter().filter(|&&e| e).count()
    }

    /// Get indices of entries and exits.
    ///
    /// # Arguments
    /// * `entries` - Entry signals
    /// * `exits` - Exit signals
    ///
    /// # Returns
    /// Tuple of (entry_indices, exit_indices)
    pub fn get_trade_indices(entries: &[bool], exits: &[bool]) -> (Vec<usize>, Vec<usize>) {
        let entry_indices: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter_map(|(i, &e)| if e { Some(i) } else { None })
            .collect();

        let exit_indices: Vec<usize> =
            exits.iter().enumerate().filter_map(|(i, &e)| if e { Some(i) } else { None }).collect();

        (entry_indices, exit_indices)
    }
}

/// Shift signals forward by n bars (delays execution).
///
/// Note: boolean-array signal combinators serve the legacy array-based
/// strategy path; strategies written against the class-based strategy
/// contract express this logic directly in code. Kept for backward
/// compatibility.
pub fn shift_signals(signals: &[bool], n: usize) -> Vec<bool> {
    let len = signals.len();
    let mut result = vec![false; len];

    if n >= len {
        return result;
    }

    result[n..len].copy_from_slice(&signals[..len - n]);

    result
}

/// Combine multiple signal arrays with AND logic.
pub fn combine_signals_and(signals: &[&[bool]]) -> Vec<bool> {
    if signals.is_empty() {
        return vec![];
    }

    let n = signals[0].len();
    for sig in signals.iter() {
        assert_eq!(sig.len(), n, "All signal arrays must have same length");
    }

    let mut result = vec![true; n];
    for sig in signals.iter() {
        for i in 0..n {
            result[i] = result[i] && sig[i];
        }
    }

    result
}

/// Combine multiple signal arrays with OR logic.
pub fn combine_signals_or(signals: &[&[bool]]) -> Vec<bool> {
    if signals.is_empty() {
        return vec![];
    }

    let n = signals[0].len();
    for sig in signals.iter() {
        assert_eq!(sig.len(), n, "All signal arrays must have same length");
    }

    let mut result = vec![false; n];
    for sig in signals.iter() {
        for i in 0..n {
            result[i] = result[i] || sig[i];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_signals_basic() {
        let processor = SignalProcessor::new();

        let entries = vec![true, false, true, false, true, false];
        let exits = vec![false, true, false, true, false, true];

        let (clean_e, clean_x) = processor.clean_signals(&entries, &exits);

        // First entry should be kept
        assert!(clean_e[0]);
        // First exit should be kept
        assert!(clean_x[1]);
        // Second entry should be kept
        assert!(clean_e[2]);
        // Second exit should be kept
        assert!(clean_x[3]);
    }

    #[test]
    fn test_clean_signals_consecutive_entries() {
        let processor = SignalProcessor::new();

        let entries = vec![true, true, true, false, false];
        let exits = vec![false, false, false, true, false];

        let (clean_e, clean_x) = processor.clean_signals(&entries, &exits);

        // Only first entry should be kept
        assert!(clean_e[0]);
        assert!(!clean_e[1]);
        assert!(!clean_e[2]);
        // Exit should be kept
        assert!(clean_x[3]);
    }

    #[test]
    fn test_clean_signals_consecutive_exits() {
        let processor = SignalProcessor::new();

        let entries = vec![true, false, false, false, false];
        let exits = vec![false, true, true, true, false];

        let (clean_e, clean_x) = processor.clean_signals(&entries, &exits);

        // Entry should be kept
        assert!(clean_e[0]);
        // Only first exit should be kept
        assert!(clean_x[1]);
        assert!(!clean_x[2]);
        assert!(!clean_x[3]);
    }

    #[test]
    fn test_clean_signals_exit_before_entry() {
        let processor = SignalProcessor::new();

        let entries = vec![false, false, true, false, false];
        let exits = vec![true, true, false, true, false];

        let (clean_e, clean_x) = processor.clean_signals(&entries, &exits);

        // Exits before first entry should be ignored
        assert!(!clean_x[0]);
        assert!(!clean_x[1]);
        // Entry should be kept
        assert!(clean_e[2]);
        // Exit after entry should be kept
        assert!(clean_x[3]);
    }

    #[test]
    fn test_pyramiding() {
        let processor = SignalProcessor::new().with_pyramiding(3);

        let entries = vec![true, true, true, false, false];
        let exits = vec![false, false, false, true, true];

        let (clean_e, clean_x) = processor.clean_signals(&entries, &exits);

        // All three entries should be kept (pyramiding)
        assert!(clean_e[0]);
        assert!(clean_e[1]);
        assert!(clean_e[2]);
        // Both exits should be kept
        assert!(clean_x[3]);
        assert!(clean_x[4]);
    }

    #[test]
    fn test_shift_signals() {
        let signals = vec![true, false, true, false, true];
        let shifted = shift_signals(&signals, 2);

        assert!(!shifted[0]);
        assert!(!shifted[1]);
        assert!(shifted[2]); // Original [0]
        assert!(!shifted[3]); // Original [1]
        assert!(shifted[4]); // Original [2]
    }

    #[test]
    fn test_combine_signals_and() {
        let sig1 = vec![true, true, false, false];
        let sig2 = vec![true, false, true, false];

        let combined = combine_signals_and(&[&sig1, &sig2]);

        assert!(combined[0]); // true && true
        assert!(!combined[1]); // true && false
        assert!(!combined[2]); // false && true
        assert!(!combined[3]); // false && false
    }

    #[test]
    fn test_combine_signals_or() {
        let sig1 = vec![true, true, false, false];
        let sig2 = vec![true, false, true, false];

        let combined = combine_signals_or(&[&sig1, &sig2]);

        assert!(combined[0]); // true || true
        assert!(combined[1]); // true || false
        assert!(combined[2]); // false || true
        assert!(!combined[3]); // false || false
    }

    // ------------------------------------------------------------------
    // Two-sided signal cleaning
    //
    // The rule these pin down: an entry against the open position is a
    // reversal and lands on one bar, while everything else behaves exactly
    // as the single-sided cleaner does.
    // ------------------------------------------------------------------

    #[test]
    fn signed_cleaning_opens_in_the_direction_the_bar_names() {
        let processor = SignalProcessor::new();
        let (entries, exits, directions) = processor.clean_signals_signed(
            &[true, false, false],
            &[false, false, true],
            &[-1, 0, 0],
            Direction::Long,
        );
        assert_eq!(entries, vec![true, false, false]);
        assert_eq!(exits, vec![false, false, true]);
        assert_eq!(directions, vec![-1, 0, 0]);
    }

    #[test]
    fn signed_cleaning_reverses_on_one_bar() {
        let processor = SignalProcessor::new();
        let (entries, exits, directions) = processor.clean_signals_signed(
            &[true, true, false],
            &[false, false, true],
            &[1, -1, 0],
            Direction::Long,
        );
        // Bar 1 both closes the long and opens the short.
        assert_eq!(entries, vec![true, true, false]);
        assert_eq!(exits, vec![false, true, true]);
        assert_eq!(directions, vec![1, -1, 0]);
    }

    #[test]
    fn signed_cleaning_ignores_a_same_side_entry() {
        let processor = SignalProcessor::new();
        let (entries, exits, _) = processor.clean_signals_signed(
            &[true, true, false],
            &[false, false, true],
            &[-1, -1, 0],
            Direction::Long,
        );
        assert_eq!(entries, vec![true, false, false]);
        assert_eq!(exits, vec![false, false, true]);
    }

    #[test]
    fn signed_cleaning_lets_a_same_side_entry_outrank_its_bar_s_exit() {
        let processor = SignalProcessor::new();
        let (entries, exits, _) = processor.clean_signals_signed(
            &[true, true, false],
            &[false, true, true],
            &[1, 1, 0],
            Direction::Long,
        );
        // Exactly `clean_signals`: the bar-1 exit is dropped, not taken.
        assert_eq!(entries, vec![true, false, false]);
        assert_eq!(exits, vec![false, false, true]);
    }

    #[test]
    fn signed_cleaning_falls_back_to_the_run_direction() {
        let processor = SignalProcessor::new();
        let (entries, exits, directions) = processor.clean_signals_signed(
            &[true, false, false],
            &[false, false, true],
            &[0, 0, 0],
            Direction::Short,
        );
        assert_eq!(directions, vec![-1, 0, 0]);
        // And the alternation is the single-sided cleaner's, unchanged.
        let (plain_entries, plain_exits) =
            processor.clean_signals(&[true, false, false], &[false, false, true]);
        assert_eq!(entries, plain_entries);
        assert_eq!(exits, plain_exits);
    }

    #[test]
    fn signed_cleaning_ignores_an_exit_taken_while_flat() {
        let processor = SignalProcessor::new();
        let (entries, exits, _) = processor.clean_signals_signed(
            &[false, true],
            &[true, false],
            &[0, -1],
            Direction::Long,
        );
        assert_eq!(entries, vec![false, true]);
        assert_eq!(exits, vec![false, false]);
    }
}
