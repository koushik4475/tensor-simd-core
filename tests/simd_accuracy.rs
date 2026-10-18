//! Accuracy verification suite: compares AVX-512, AVX2, and ARM NEON SIMD kernels
//! against the golden unrolled scalar baseline across varying vector and matrix dimensions.

use tensor_simd_core::simd::{dot_f32, gemm_f32, scalar};
use tensor_simd_core::tensor::Tensor;
use tensor_simd_core::ops::{softmax, layernorm, relu, conv2d, Conv2dParams};

#[test]
fn test_dot_product_accuracy_against_scalar() {
    let test_sizes = [1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 255, 512, 1024, 4096];

    for &size in &test_sizes {
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.125).collect();
        let b: Vec<f32> = (0..size).map(|i| ((size - i) as f32) * 0.25).collect();

        let scalar_result = scalar::dot_f32(&a, &b);
        let simd_result = dot_f32(&a, &b);

        let diff = (simd_result - scalar_result).abs();
        let max_val = scalar_result.abs().max(1.0);
        let rel_err = diff / max_val;

        assert!(
            rel_err < 1e-4,
            "Dot product failed at size {}: scalar={}, simd={}, rel_err={}",
            size,
            scalar_result,
            simd_result,
            rel_err
        );
    }
}

#[test]
fn test_gemm_accuracy_against_scalar() {
    let test_configs = [
        (2, 2, 2),
        (4, 4, 4),
        (7, 9, 11),
        (16, 16, 16),
        (32, 32, 32),
        (48, 64, 32),
        (64, 64, 64),
    ];

    for &(m, n, k) in &test_configs {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) - 8.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) - 6.0).collect();

        let mut c_scalar = vec![1.5f32; m * n];
        let mut c_simd = vec![1.5f32; m * n];

        let alpha = 1.25f32;
        let beta = 0.5f32;

        scalar::gemm_f32(m, n, k, alpha, &a, &b, beta, &mut c_scalar);
        gemm_f32(m, n, k, alpha, &a, &b, beta, &mut c_simd);

        for idx in 0..m * n {
            let s = c_scalar[idx];
            let d = c_simd[idx];
            let diff = (s - d).abs();
            let denom = s.abs().max(1.0);
            assert!(
                diff / denom < 1e-4,
                "GEMM mismatch at index {} for shape ({}, {}, {}): scalar={}, simd={}",
                idx,
                m,
                n,
                k,
                s,
                d
            );
        }
    }
}

#[test]
fn test_tensor_broadcasting_and_matmul() {
    // A: [2, 1, 3] -> broadcast to [2, 4, 3]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::from_vec(a_data, vec![2, 1, 3]).unwrap();

    // B: [3, 2] -> matrix multiplication [2, 1, 3] x [3, 2] -> [2, 1, 2]
    let b_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let b = Tensor::from_vec(b_data, vec![3, 2]).unwrap();

    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 1, 2]);

    // Row 0: [1, 2, 3] @ [[1, 0], [0, 1], [1, 1]] = [1 + 3, 2 + 3] = [4, 5]
    // Row 1: [4, 5, 6] @ [[1, 0], [0, 1], [1, 1]] = [4 + 6, 5 + 6] = [10, 11]
    let out = c.as_slice().to_vec();
    assert_eq!(out, vec![4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn test_full_neural_network_layer_pipeline() {
    // Input: [1, 4]
    let x = Tensor::from_vec(vec![-2.0, 1.0, 3.0, -0.5], vec![1, 4]).unwrap();

    // ReLU activation
    let act = relu(&x).unwrap();
    assert_eq!(act.as_slice().to_vec(), vec![0.0, 1.0, 3.0, 0.0]);

    // LayerNorm
    let norm = layernorm(&act, &[4], None, None, 1e-5).unwrap();
    let norm_slice = norm.as_slice();
    let mean: f32 = norm_slice.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4);

    // Softmax
    let sm = softmax(&norm).unwrap();
    let sum: f32 = sm.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}
