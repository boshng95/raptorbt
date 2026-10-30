//! `collections.deque(maxlen=...)` of floats, readable as one slice.

/// A deque that drops its oldest item once full, kept contiguous so that it
/// can be read as numpy reads `np.asarray(deque)`: oldest first.
#[derive(Clone, Debug)]
pub(crate) struct Window {
    data: Vec<f64>,
    start: usize,
    capacity: Option<usize>,
}

impl Window {
    /// `deque(maxlen=capacity)`.
    pub fn new(capacity: usize) -> Self {
        Window { data: Vec::with_capacity(2 * capacity + 1), start: 0, capacity: Some(capacity) }
    }

    /// `append`: past the capacity, the oldest item leaves.
    pub fn push(&mut self, value: f64) {
        if let Some(capacity) = self.capacity {
            if self.len() == capacity {
                self.start += 1;
            }
            if self.start > capacity {
                self.data.drain(..self.start);
                self.start = 0;
            }
        }
        self.data.push(value);
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data[self.start..]
    }

    pub fn len(&self) -> usize {
        self.data.len() - self.start
    }

    /// `deque[0]`.
    pub fn first(&self) -> f64 {
        self.as_slice()[0]
    }

    /// `deque[-1]`.
    pub fn last(&self) -> f64 {
        self.as_slice()[self.len() - 1]
    }

    /// `deque[-k]`, for `k >= 1`.
    pub fn back(&self, k: usize) -> f64 {
        self.as_slice()[self.len() - k]
    }
}

#[cfg(test)]
mod tests {
    use super::Window;

    #[test]
    fn keeps_the_newest_capacity_items_in_order() {
        let mut window = Window::new(3);
        for value in 0..20 {
            window.push(f64::from(value));
            let low = (value - 2).max(0);
            let expected: Vec<f64> = (low..=value).map(f64::from).collect();
            assert_eq!(window.as_slice(), expected.as_slice());
        }
        assert_eq!(window.first(), 17.0);
        assert_eq!(window.last(), 19.0);
        assert_eq!(window.back(2), 18.0);
    }
}
