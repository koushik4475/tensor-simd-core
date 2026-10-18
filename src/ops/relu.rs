//! Vectorized Rectified Linear Unit (ReLU) and LeakyReLU activations.

use crate::error::Result;
use crate::tensor::Tensor;

/// Computes elementwise standard ReLU: `f(x) = max(0, x)`
pub fn relu(input: &Tensor<f32>) -> Result<Tensor<f32>> {
    leaky_relu(input, 0.0)
}

/// Computes elementwise LeakyReLU: `f(x) = x if x >= 0 else negative_slope * x`
pub fn leaky_relu(input: &Tensor<f32>, negative_slope: f32) -> Result<Tensor<f32>> {
    let contig = input.to_contiguous();
    let in_slice = contig.as_slice();
    let mut out_data = vec![0.0f32; contig.numel()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                relu_avx2(&in_slice, negative_slope, &mut out_data);
                return Tensor::from_vec(out_data, contig.shape().to_vec());
            }
        }
    }

    // Scalar fallback
    for i in 0..in_slice.len() {
        let val = in_slice[i];
        out_data[i] = if val >= 0.0 {
            val
        } else {
            val * negative_slope
        };
    }

    Tensor::from_vec(out_data, contig.shape().to_vec())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2(src: &[f32], negative_slope: f32, dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let len = src.len();
    let zero = _mm256_setzero_ps();
    let slope = _mm256_set1_ps(negative_slope);

    let mut i = 0;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        if negative_slope == 0.0 {
            let res = _mm256_max_ps(v, zero);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), res);
        } else {
            let mask = _mm256_cmp_ps(v, zero, _CMP_GE_OQ);
            let neg_part = _mm256_mul_ps(v, slope);
            let res = _mm256_blendv_ps(neg_part, v, mask);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), res);
        }
        i += 8;
    }

    for idx in i..len {
        let val = src[idx];
        dst[idx] = if val >= 0.0 {
            val
        } else {
            val * negative_slope
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let t = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 1.5, 3.0], vec![5]).unwrap();
        let r = relu(&t).unwrap();
        assert_eq!(r.as_slice().to_vec(), vec![0.0, 0.0, 0.0, 1.5, 3.0]);
    }
}
