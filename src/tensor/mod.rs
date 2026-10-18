//! N-dimensional Tensor struct with strided indexing, broadcasting, zero-copy slicing, and SIMD execution.

pub mod broadcasting;
pub mod slicing;
pub mod strides;

use std::sync::Arc;
use crate::error::{Result, TensorError};
use crate::simd;
use broadcasting::{broadcast_shapes, broadcast_strides};
use slicing::{SliceIndex, SliceSpec};
use strides::Strides;

/// An N-dimensional Tensor backed by either owned heap storage or shared reference storage.
#[derive(Debug, Clone)]
pub struct Tensor<T = f32> {
    data: Arc<Vec<T>>,
    offset: usize,
    shape: Vec<usize>,
    strides: Strides,
}

impl<T: Clone> Tensor<T> {
    /// Creates a new contiguous Tensor initialized with provided data.
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(TensorError::ShapeMismatch {
                expected: format!("total elements {}", expected_len),
                got: format!("{} elements", data.len()),
            });
        }
        let strides = Strides::from_shape_row_major(&shape);
        Ok(Self {
            data: Arc::new(data),
            offset: 0,
            shape,
            strides,
        })
    }

    /// Creates an empty or default initialized Tensor of given shape.
    pub fn from_elem(elem: T, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let data = vec![elem; numel];
        let strides = Strides::from_shape_row_major(&shape);
        Self {
            data: Arc::new(data),
            offset: 0,
            shape,
            strides,
        }
    }

    /// Tensor shape slice.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Tensor rank (number of dimensions).
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Total number of logical elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Internal striding layout.
    #[inline]
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Memory offset to the first element in shared buffer.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Check if tensor elements are laid out contiguously in memory.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }

    /// Compute the flat data index for an N-dimensional coordinate.
    #[inline]
    pub fn linear_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::InvalidDimension {
                dim: indices.len(),
                rank: self.shape.len(),
            });
        }
        for (&idx, &dim_size) in indices.iter().zip(self.shape.iter()) {
            if idx >= dim_size {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx,
                    size: dim_size,
                });
            }
        }
        let rel_offset = self.strides.offset(indices);
        let actual_idx = (self.offset as isize + rel_offset) as usize;
        Ok(actual_idx)
    }

    /// Get a reference to the element at the specified coordinate.
    #[inline]
    pub fn get(&self, indices: &[usize]) -> Result<&T> {
        let idx = self.linear_index(indices)?;
        Ok(&self.data[idx])
    }

    /// Reshapes the tensor if it is contiguous.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if self.numel() != new_numel {
            return Err(TensorError::ShapeMismatch {
                expected: format!("elements {}", self.numel()),
                got: format!("elements {}", new_numel),
            });
        }
        if self.is_contiguous() {
            let new_strides = Strides::from_shape_row_major(&new_shape);
            Ok(Self {
                data: Arc::clone(&self.data),
                offset: self.offset,
                shape: new_shape,
                strides: new_strides,
            })
        } else {
            // Contiguate data first
            let contiguous = self.to_contiguous();
            let new_strides = Strides::from_shape_row_major(&new_shape);
            Ok(Self {
                data: contiguous.data,
                offset: 0,
                shape: new_shape,
                strides: new_strides,
            })
        }
    }

    /// Creates a zero-copy sliced sub-view of the tensor.
    pub fn slice(&self, spec: &SliceSpec) -> Result<Self> {
        if spec.indices.len() > self.rank() {
            return Err(TensorError::InvalidDimension {
                dim: spec.indices.len(),
                rank: self.rank(),
            });
        }

        let mut new_offset = self.offset as isize;
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();

        for (d, index) in spec.indices.iter().enumerate() {
            let dim_len = self.shape[d];
            let dim_stride = self.strides.as_slice()[d];

            match *index {
                SliceIndex::Point(pt) => {
                    if pt >= dim_len {
                        return Err(TensorError::IndexOutOfBounds {
                            index: pt,
                            size: dim_len,
                        });
                    }
                    new_offset += (pt as isize) * dim_stride;
                    // Point drops dimension from shape
                }
                SliceIndex::Range(start, end, step) => {
                    let end = end.min(dim_len);
                    if start >= dim_len || start >= end {
                        return Err(TensorError::InvalidSlice {
                            start,
                            end,
                            size: dim_len,
                        });
                    }
                    let count = (end - start + step - 1) / step;
                    new_offset += (start as isize) * dim_stride;
                    new_shape.push(count);
                    new_strides.push(dim_stride * (step as isize));
                }
                SliceIndex::All => {
                    new_shape.push(dim_len);
                    new_strides.push(dim_stride);
                }
            }
        }

        // Remaining un-sliced dimensions are copied as-is
        for d in spec.indices.len()..self.rank() {
            new_shape.push(self.shape[d]);
            new_strides.push(self.strides.as_slice()[d]);
        }

        Ok(Self {
            data: Arc::clone(&self.data),
            offset: new_offset as usize,
            shape: new_shape,
            strides: Strides::new(new_strides),
        })
    }

    /// Broadcast this tensor to a target shape with zero memory copies using stride 0.
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self> {
        let new_strides = broadcast_strides(&self.shape, &self.strides, target_shape)
            .ok_or_else(|| TensorError::BroadcastError {
                shape_a: self.shape.clone(),
                shape_b: target_shape.to_vec(),
            })?;

        Ok(Self {
            data: Arc::clone(&self.data),
            offset: self.offset,
            shape: target_shape.to_vec(),
            strides: new_strides,
        })
    }

    /// Materialize a contiguous copy in row-major layout.
    pub fn to_contiguous(&self) -> Self {
        if self.is_contiguous() && self.offset == 0 && self.data.len() == self.numel() {
            return self.clone();
        }

        let numel = self.numel();
        let mut out_data = Vec::with_capacity(numel);

        let mut coord = vec![0usize; self.rank()];
        for _ in 0..numel {
            let idx = (self.offset as isize + self.strides.offset(&coord)) as usize;
            out_data.push(self.data[idx].clone());

            // Increment N-D coordinate
            for r in (0..self.rank()).rev() {
                coord[r] += 1;
                if coord[r] < self.shape[r] {
                    break;
                }
                coord[r] = 0;
            }
        }

        let new_strides = Strides::from_shape_row_major(&self.shape);
        Self {
            data: Arc::new(out_data),
            offset: 0,
            shape: self.shape.clone(),
            strides: new_strides,
        }
    }

    /// Read raw data as contiguous slice (materializes if non-contiguous).
    pub fn as_slice(&self) -> std::borrow::Cow<'_, [T]> {
        if self.is_contiguous() {
            let start = self.offset;
            let end = start + self.numel();
            std::borrow::Cow::Borrowed(&self.data[start..end])
        } else {
            let contiguous = self.to_contiguous();
            std::borrow::Cow::Owned(Arc::try_unwrap(contiguous.data).unwrap_or_else(|a| (*a).clone()))
        }
    }
}

impl Tensor<f32> {
    /// Creates a tensor of zeros with given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::from_elem(0.0, shape)
    }

    /// Creates a tensor of ones with given shape.
    pub fn ones(shape: Vec<usize>) -> Self {
        Self::from_elem(1.0, shape)
    }

    /// High performance SIMD elementwise addition with broadcasting.
    pub fn add(&self, other: &Tensor<f32>) -> Result<Tensor<f32>> {
        let target_shape = broadcast_shapes(&self.shape, &other.shape).ok_or_else(|| {
            TensorError::BroadcastError {
                shape_a: self.shape.clone(),
                shape_b: other.shape.clone(),
            }
        })?;

        let numel: usize = target_shape.iter().product();

        // Fast contiguous path without broadcasting
        if self.shape == other.shape && self.is_contiguous() && other.is_contiguous() {
            let mut out = vec![0.0f32; numel];
            let a_slice = &self.data[self.offset..self.offset + numel];
            let b_slice = &other.data[other.offset..other.offset + numel];
            simd::add_f32(a_slice, b_slice, &mut out);
            return Tensor::from_vec(out, target_shape);
        }

        let t1 = self.broadcast_to(&target_shape)?;
        let t2 = other.broadcast_to(&target_shape)?;

        let mut out = Vec::with_capacity(numel);
        let mut coord = vec![0usize; target_shape.len()];

        for _ in 0..numel {
            let idx1 = (t1.offset as isize + t1.strides.offset(&coord)) as usize;
            let idx2 = (t2.offset as isize + t2.strides.offset(&coord)) as usize;
            out.push(t1.data[idx1] + t2.data[idx2]);

            for r in (0..target_shape.len()).rev() {
                coord[r] += 1;
                if coord[r] < target_shape[r] {
                    break;
                }
                coord[r] = 0;
            }
        }

        Tensor::from_vec(out, target_shape)
    }

    /// High performance SIMD elementwise multiplication with broadcasting.
    pub fn mul(&self, other: &Tensor<f32>) -> Result<Tensor<f32>> {
        let target_shape = broadcast_shapes(&self.shape, &other.shape).ok_or_else(|| {
            TensorError::BroadcastError {
                shape_a: self.shape.clone(),
                shape_b: other.shape.clone(),
            }
        })?;

        let numel: usize = target_shape.iter().product();

        if self.shape == other.shape && self.is_contiguous() && other.is_contiguous() {
            let mut out = vec![0.0f32; numel];
            let a_slice = &self.data[self.offset..self.offset + numel];
            let b_slice = &other.data[other.offset..other.offset + numel];
            simd::mul_f32(a_slice, b_slice, &mut out);
            return Tensor::from_vec(out, target_shape);
        }

        let t1 = self.broadcast_to(&target_shape)?;
        let t2 = other.broadcast_to(&target_shape)?;

        let mut out = Vec::with_capacity(numel);
        let mut coord = vec![0usize; target_shape.len()];

        for _ in 0..numel {
            let idx1 = (t1.offset as isize + t1.strides.offset(&coord)) as usize;
            let idx2 = (t2.offset as isize + t2.strides.offset(&coord)) as usize;
            out.push(t1.data[idx1] * t2.data[idx2]);

            for r in (0..target_shape.len()).rev() {
                coord[r] += 1;
                if coord[r] < target_shape[r] {
                    break;
                }
                coord[r] = 0;
            }
        }

        Tensor::from_vec(out, target_shape)
    }

    /// Matrix multiplication (2D GEMM or Batched GEMM).
    ///
    /// For 2D matrices: `[M, K] x [K, N] -> [M, N]`.
    pub fn matmul(&self, other: &Tensor<f32>) -> Result<Tensor<f32>> {
        if self.rank() < 2 || other.rank() < 2 {
            return Err(TensorError::InvalidDimension {
                dim: self.rank().min(other.rank()),
                rank: 2,
            });
        }

        let m = self.shape[self.rank() - 2];
        let k1 = self.shape[self.rank() - 1];
        let k2 = other.shape[other.rank() - 2];
        let n = other.shape[other.rank() - 1];

        if k1 != k2 {
            return Err(TensorError::ShapeMismatch {
                expected: format!("inner dimension {}", k1),
                got: format!("inner dimension {}", k2),
            });
        }

        // 2D Matrix multiplication
        if self.rank() == 2 && other.rank() == 2 {
            let a_contig = self.to_contiguous();
            let b_contig = other.to_contiguous();
            let mut c = vec![0.0f32; m * n];

            simd::gemm_f32(
                m,
                n,
                k1,
                1.0,
                &a_contig.data[a_contig.offset..a_contig.offset + m * k1],
                &b_contig.data[b_contig.offset..b_contig.offset + k1 * n],
                0.0,
                &mut c,
            );

            return Tensor::from_vec(c, vec![m, n]);
        }

        // Batched Matrix Multiplication
        let batch_a = &self.shape[..self.rank() - 2];
        let batch_b = &other.shape[..other.rank() - 2];
        let batch_shape = broadcast_shapes(batch_a, batch_b).ok_or_else(|| {
            TensorError::BroadcastError {
                shape_a: batch_a.to_vec(),
                shape_b: batch_b.to_vec(),
            }
        })?;

        let batch_count: usize = batch_shape.iter().product();
        let mut final_shape = batch_shape.clone();
        final_shape.push(m);
        final_shape.push(n);

        let mut out = vec![0.0f32; batch_count * m * n];

        let a_bcast = self.to_contiguous();
        let b_bcast = other.to_contiguous();

        let a_matrix_size = m * k1;
        let b_matrix_size = k1 * n;
        let c_matrix_size = m * n;

        for b_idx in 0..batch_count {
            let a_offset = (b_idx % (a_bcast.numel() / a_matrix_size)) * a_matrix_size;
            let b_offset = (b_idx % (b_bcast.numel() / b_matrix_size)) * b_matrix_size;
            let c_offset = b_idx * c_matrix_size;

            let a_sub = &a_bcast.data[a_offset..a_offset + a_matrix_size];
            let b_sub = &b_bcast.data[b_offset..b_offset + b_matrix_size];
            let c_sub = &mut out[c_offset..c_offset + c_matrix_size];

            simd::gemm_f32(m, n, k1, 1.0, a_sub, b_sub, 0.0, c_sub);
        }

        Tensor::from_vec(out, final_shape)
    }
}
