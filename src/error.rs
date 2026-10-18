//! Error types for tensor allocations, striding, slicing, and SIMD operations.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, TensorError>;

#[derive(Error, Debug, PartialEq)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch {
        expected: String,
        got: String,
    },

    #[error("Broadcast error: shapes {shape_a:?} and {shape_b:?} cannot be broadcast together")]
    BroadcastError {
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
    },

    #[error("Dimension index out of bounds: index {dim} on tensor with rank {rank}")]
    InvalidDimension {
        dim: usize,
        rank: usize,
    },

    #[error("Index out of bounds: index {index} exceeds dimension size {size}")]
    IndexOutOfBounds {
        index: usize,
        size: usize,
    },

    #[error("Invalid slice range: start={start}, end={end} for dimension size {size}")]
    InvalidSlice {
        start: usize,
        end: usize,
        size: usize,
    },

    #[error("Kernel execution error: {0}")]
    KernelError(String),
}
