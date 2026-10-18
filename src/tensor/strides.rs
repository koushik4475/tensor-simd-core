//! N-dimensional striding, indexing, and dimension utilities.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strides {
    strides: Vec<isize>,
}

impl Strides {
    /// Computes default contiguous row-major (C-contiguous) strides for a given shape.
    pub fn from_shape_row_major(shape: &[usize]) -> Self {
        if shape.is_empty() {
            return Self { strides: Vec::new() };
        }
        let mut strides = vec![1isize; shape.len()];
        let mut running = 1isize;
        for i in (0..shape.len()).rev() {
            strides[i] = running;
            running *= shape[i] as isize;
        }
        Self { strides }
    }

    /// Creates custom strides from an explicit slice.
    pub fn new(strides: Vec<isize>) -> Self {
        Self { strides }
    }

    #[inline]
    pub fn as_slice(&self) -> &[isize] {
        &self.strides
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.strides.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strides.is_empty()
    }

    /// Computes the flat linear offset given an N-dimensional index coordinate.
    #[inline]
    pub fn offset(&self, indices: &[usize]) -> isize {
        assert_eq!(
            indices.len(),
            self.strides.len(),
            "Indices length must match tensor dimension rank"
        );
        let mut off = 0isize;
        for (&idx, &stride) in indices.iter().zip(self.strides.iter()) {
            off += (idx as isize) * stride;
        }
        off
    }

    /// Returns true if this stride layout is contiguous in row-major memory order.
    pub fn is_contiguous(&self, shape: &[usize]) -> bool {
        if shape.is_empty() || self.strides.is_empty() {
            return true;
        }
        let mut expected = 1isize;
        for i in (0..shape.len()).rev() {
            if shape[i] == 1 {
                continue;
            }
            if self.strides[i] != expected {
                return false;
            }
            expected *= shape[i] as isize;
        }
        true
    }
}

impl fmt::Display for Strides {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.strides)
    }
}
