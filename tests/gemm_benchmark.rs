//! GEMM throughput and benchmark harness comparing SIMD kernels vs baseline.

use std::time::Instant;
use tensor_simd_core::simd::{gemm_f32, scalar, SimdArchitecture};

fn run_gemm_benchmark(m: usize, n: usize, k: usize, iterations: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 100) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 100) as f32) * 0.01).collect();
    let mut c_simd = vec![0.0f32; m * n];
    let mut c_scalar = vec![0.0f32; m * n];

    // Warmup
    gemm_f32(m, n, k, 1.0, &a, &b, 0.0, &mut c_simd);
    scalar::gemm_f32(m, n, k, 1.0, &a, &b, 0.0, &mut c_scalar);

    let gflops_per_iter = (2.0 * m as f64 * n as f64 * k as f64) / 1e9;

    // SIMD benchmark
    let start_simd = Instant::now();
    for _ in 0..iterations {
        gemm_f32(m, n, k, 1.0, &a, &b, 0.0, &mut c_simd);
    }
    let simd_duration = start_simd.elapsed();
    let simd_gflops = (gflops_per_iter * iterations as f64) / simd_duration.as_secs_f64();

    // Scalar benchmark
    let start_scalar = Instant::now();
    for _ in 0..iterations {
        scalar::gemm_f32(m, n, k, 1.0, &a, &b, 0.0, &mut c_scalar);
    }
    let scalar_duration = start_scalar.elapsed();
    let scalar_gflops = (gflops_per_iter * iterations as f64) / scalar_duration.as_secs_f64();

    println!("--------------------------------------------------");
    println!("Matrix Dimensions: M={}, N={}, K={} ({} iterations)", m, n, k, iterations);
    println!("Active SIMD Backend: {:?}", SimdArchitecture::detect());
    println!("Scalar Baseline: {:?} ({:.2} GFLOPS)", scalar_duration, scalar_gflops);
    println!("SIMD Kernel:     {:?} ({:.2} GFLOPS)", simd_duration, simd_gflops);
    println!("Speedup Factor:  {:.2}x", scalar_duration.as_secs_f64() / simd_duration.as_secs_f64());
    println!("--------------------------------------------------");
}

#[test]
fn test_gemm_benchmark_execution() {
    println!("\n=== TENSOR-SIMD-CORE BENCHMARK HARNESS ===");
    run_gemm_benchmark(32, 32, 32, 100);
    run_gemm_benchmark(64, 64, 64, 50);
    run_gemm_benchmark(128, 128, 128, 10);
}
