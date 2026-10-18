//! Layer Normalization (LayerNorm) kernel with learnable gamma and beta parameters.

use crate::error::{Result, TensorError};
use crate::tensor::Tensor;

/// Computes LayerNorm across the normalized shape (last D dimensions):
/// `y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta`
pub fn layernorm(
    input: &Tensor<f32>,
    normalized_shape: &[usize],
    gamma: Option<&Tensor<f32>>,
    beta: Option<&Tensor<f32>>,
    eps: f32,
) -> Result<Tensor<f32>> {
    let input_shape = input.shape();
    let norm_rank = normalized_shape.len();
    if input_shape.len() < norm_rank {
        return Err(TensorError::ShapeMismatch {
            expected: format!("rank >= {}", norm_rank),
            got: format!("rank {}", input_shape.len()),
        });
    }

    let start_idx = input_shape.len() - norm_rank;
    if &input_shape[start_idx..] != normalized_shape {
        return Err(TensorError::ShapeMismatch {
            expected: format!("{:?}", normalized_shape),
            got: format!("{:?}", &input_shape[start_idx..]),
        });
    }

    let norm_size: usize = normalized_shape.iter().product();
    let outer_size: usize = input_shape[..start_idx].iter().product();

    let contig = input.to_contiguous();
    let in_slice = contig.as_slice();
    let mut out_data = vec![0.0f32; contig.numel()];

    let gamma_slice = gamma.map(|g| g.to_contiguous());
    let beta_slice = beta.map(|b| b.to_contiguous());

    for o in 0..outer_size {
        let offset = o * norm_size;
        let row = &in_slice[offset..offset + norm_size];
        let out_row = &mut out_data[offset..offset + norm_size];

        // Mean
        let mut mean = 0.0f32;
        for &val in row {
            mean += val;
        }
        mean /= norm_size as f32;

        // Variance
        let mut var = 0.0f32;
        for &val in row {
            let diff = val - mean;
            var += diff * diff;
        }
        var /= norm_size as f32;
        let inv_std = 1.0f32 / (var + eps).sqrt();

        for i in 0..norm_size {
            let mut normalized = (row[i] - mean) * inv_std;
            if let Some(ref g) = gamma_slice {
                normalized *= g.as_slice()[i];
            }
            if let Some(ref b) = beta_slice {
                normalized += b.as_slice()[i];
            }
            out_row[i] = normalized;
        }
    }

    Tensor::from_vec(out_data, input_shape.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_zero_mean_unit_var() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
        let ln = layernorm(&t, &[5], None, None, 1e-5).unwrap();
        let slice = ln.as_slice();

        let mean: f32 = slice.iter().sum::<f32>() / 5.0;
        assert!(mean.abs() < 1e-5);
    }
}
