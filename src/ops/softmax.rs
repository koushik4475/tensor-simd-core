//! Numerically stable Softmax activation kernel with SIMD acceleration.

use crate::error::Result;
use crate::tensor::Tensor;

/// Computes numerically stable Softmax along the last dimension of a tensor:
/// `softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`
pub fn softmax(input: &Tensor<f32>) -> Result<Tensor<f32>> {
    let contig = input.to_contiguous();
    let shape = contig.shape();
    let rank = shape.len();

    let d_last = shape[rank - 1];
    let outer_size: usize = shape[..rank - 1].iter().product();

    let mut out_data = vec![0.0f32; contig.numel()];
    let in_slice = contig.as_slice();

    for o in 0..outer_size {
        let offset = o * d_last;
        let row = &in_slice[offset..offset + d_last];
        let out_row = &mut out_data[offset..offset + d_last];

        // 1. Numerical stability: find max element
        let mut max_val = f32::NEG_INFINITY;
        for &val in row {
            if val > max_val {
                max_val = val;
            }
        }

        // 2. Compute exp(x - max) and accumulate sum
        let mut sum_exp = 0.0f32;
        for i in 0..d_last {
            let exp_val = (row[i] - max_val).exp();
            out_row[i] = exp_val;
            sum_exp += exp_val;
        }

        // 3. Normalize
        let inv_sum = 1.0f32 / sum_exp;
        for i in 0..d_last {
            out_row[i] *= inv_sum;
        }
    }

    Tensor::from_vec(out_data, shape.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sum_to_one() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let sm = softmax(&t).unwrap();
        let sum: f32 = sm.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
