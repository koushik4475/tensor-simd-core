//! Cross-platform SIMD abstraction module for x86_64 and aarch64.
//!
//! Provides runtime CPU feature detection, vectorized dot products,
//! elementwise arithmetic, and high-performance blocked GEMM routines
//! with AVX-512, AVX2/FMA, and ARM NEON backends with scalar fallbacks.

pub mod avx2;
pub mod avx512;
pub mod neon;
pub mod scalar;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdArchitecture {
    Scalar,
    Neon,
    Avx2,
    Avx512,
}

impl SimdArchitecture {
    /// Detect the highest supported SIMD instruction set at runtime on the current CPU.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512cd") {
                return SimdArchitecture::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return SimdArchitecture::Avx2;
            }
            SimdArchitecture::Scalar
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return SimdArchitecture::Neon;
            }
            SimdArchitecture::Scalar
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdArchitecture::Scalar
        }
    }
}

/// Computes the dot product of two single-precision floating point slices.
///
/// Dispatches dynamically to the best available instruction set on the target machine.
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Dot product slices must have equal length");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { avx512::dot_f32(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::dot_f32(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon::dot_f32(a, b) };
        }
    }

    scalar::dot_f32(a, b)
}

/// Elementwise vector addition: `out[i] = a[i] + b[i]`.
#[inline]
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx512::add_f32(a, b, out) };
            return;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::add_f32(a, b, out) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon::add_f32(a, b, out) };
            return;
        }
    }

    scalar::add_f32(a, b, out);
}

/// Elementwise vector multiplication: `out[i] = a[i] * b[i]`.
#[inline]
pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx512::mul_f32(a, b, out) };
            return;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::mul_f32(a, b, out) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon::mul_f32(a, b, out) };
            return;
        }
    }

    scalar::mul_f32(a, b, out);
}

/// General Matrix Multiply (GEMM) for row-major matrices:
/// `C = alpha * (A @ B) + beta * C`
///
/// Dimensions:
/// - A: `m x k`
/// - B: `k x n`
/// - C: `m x n`
pub fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    assert_eq!(a.len(), m * k, "Matrix A size mismatch");
    assert_eq!(b.len(), k * n, "Matrix B size mismatch");
    assert_eq!(c.len(), m * n, "Matrix C size mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                avx512::gemm_f32(m, n, k, alpha, a, b, beta, c);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                avx2::gemm_f32(m, n, k, alpha, a, b, beta, c);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon::gemm_f32(m, n, k, alpha, a, b, beta, c);
            }
            return;
        }
    }

    scalar::gemm_f32(m, n, k, alpha, a, b, beta, c);
}
