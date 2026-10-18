//! Zero-copy multidimensional slicing descriptor.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceIndex {
    /// Single index at a dimension: drops the rank of that dimension
    Point(usize),
    /// Range slice `[start, end, step]`
    Range(usize, usize, usize),
    /// Full dimension slice `:`
    All,
}

impl SliceIndex {
    pub fn range(start: usize, end: usize) -> Self {
        SliceIndex::Range(start, end, 1)
    }

    pub fn stepped_range(start: usize, end: usize, step: usize) -> Self {
        assert!(step > 0, "Step must be strictly positive");
        SliceIndex::Range(start, end, step)
    }
}

/// Slicing plan representing sub-view creation
#[derive(Debug, Clone)]
pub struct SliceSpec {
    pub indices: Vec<SliceIndex>,
}

impl SliceSpec {
    pub fn new(indices: Vec<SliceIndex>) -> Self {
        Self { indices }
    }
}
