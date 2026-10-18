//! High-performance 2D direct & im2col-based convolution kernels.

use crate::error::{Result, TensorError};
use crate::simd;
use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct Conv2dParams {
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
}

impl Default for Conv2dParams {
    fn default() -> Self {
        Self {
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
        }
    }
}

/// 2D Convolution: `input [N, C_in, H, W]` * `weight [C_out, C_in, kH, kW]` -> `output [N, C_out, outH, outW]`
/// Optional `bias [C_out]` vector.
pub fn conv2d(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    params: Conv2dParams,
) -> Result<Tensor<f32>> {
    if input.rank() != 4 || weight.rank() != 4 {
        return Err(TensorError::KernelError(
            "Conv2D requires 4D tensors: [N, C, H, W] and [C_out, C_in, kH, kW]".to_string(),
        ));
    }

    let batch_size = input.shape()[0];
    let in_channels = input.shape()[1];
    let in_h = input.shape()[2];
    let in_w = input.shape()[3];

    let out_channels = weight.shape()[0];
    let weight_in_channels = weight.shape()[1];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];

    if in_channels != weight_in_channels {
        return Err(TensorError::ShapeMismatch {
            expected: format!("in_channels={}", in_channels),
            got: format!("weight_in_channels={}", weight_in_channels),
        });
    }

    let out_h = (in_h + 2 * params.pad_h - params.dilation_h * (kh - 1) - 1) / params.stride_h + 1;
    let out_w = (in_w + 2 * params.pad_w - params.dilation_w * (kw - 1) - 1) / params.stride_w + 1;

    let k_size = in_channels * kh * kw;
    let spatial_out = out_h * out_w;

    let input_contig = input.to_contiguous();
    let weight_contig = weight.to_contiguous();
    let in_data = input_contig.as_slice();
    let weight_data = weight_contig.as_slice();

    let bias_contig = bias.map(|b| b.to_contiguous());

    let mut out_data = vec![0.0f32; batch_size * out_channels * out_h * out_w];

    // im2col scratch buffer per image
    let mut col_buffer = vec![0.0f32; k_size * spatial_out];

    for n in 0..batch_size {
        let in_batch_offset = n * (in_channels * in_h * in_w);
        let out_batch_offset = n * (out_channels * spatial_out);

        // Perform im2col transformation
        im2col(
            &in_data[in_batch_offset..in_batch_offset + in_channels * in_h * in_w],
            in_channels,
            in_h,
            in_w,
            kh,
            kw,
            params.pad_h,
            params.pad_w,
            params.stride_h,
            params.stride_w,
            params.dilation_h,
            params.dilation_w,
            out_h,
            out_w,
            &mut col_buffer,
        );

        // GEMM: Weight [C_out, k_size] x ColBuffer [k_size, spatial_out] -> Out [C_out, spatial_out]
        let out_slice = &mut out_data[out_batch_offset..out_batch_offset + out_channels * spatial_out];

        simd::gemm_f32(
            out_channels,
            spatial_out,
            k_size,
            1.0,
            &weight_data,
            &col_buffer,
            0.0,
            out_slice,
        );

        // Add bias if provided
        if let Some(ref b) = bias_contig {
            let b_slice = b.as_slice();
            for c in 0..out_channels {
                let bias_val = b_slice[c];
                let row_offset = c * spatial_out;
                for s in 0..spatial_out {
                    out_slice[row_offset + s] += bias_val;
                }
            }
        }
    }

    Tensor::from_vec(out_data, vec![batch_size, out_channels, out_h, out_w])
}

fn im2col(
    data_im: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    out_h: usize,
    out_w: usize,
    data_col: &mut [f32],
) {
    let num_patches = out_h * out_w;
    let mut col_row = 0;

    for c in 0..channels {
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                for oh in 0..out_h {
                    let ih = (oh * stride_h) as isize - pad_h as isize + (kh * dilation_h) as isize;
                    for ow in 0..out_w {
                        let iw = (ow * stride_w) as isize - pad_w as isize + (kw * dilation_w) as isize;
                        let patch_idx = oh * out_w + ow;
                        let col_idx = col_row * num_patches + patch_idx;

                        if ih >= 0 && ih < height as isize && iw >= 0 && iw < width as isize {
                            let im_idx = (c * height + ih as usize) * width + iw as usize;
                            data_col[col_idx] = data_im[im_idx];
                        } else {
                            data_col[col_idx] = 0.0;
                        }
                    }
                }
                col_row += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_identity_kernel() {
        // 1x1 image with 1x1 filter
        let input = Tensor::from_vec(vec![3.0], vec![1, 1, 1, 1]).unwrap();
        let weight = Tensor::from_vec(vec![2.0], vec![1, 1, 1, 1]).unwrap();
        let out = conv2d(&input, &weight, None, Conv2dParams::default()).unwrap();
        assert_eq!(out.as_slice().to_vec(), vec![6.0]);
    }
}
