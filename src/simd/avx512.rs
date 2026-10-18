//! AVX-512 512-bit vectorized operations (16xf32 per register).

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::*;

/// AVX-512 accelerated dot product for f32 slices.
///
/// Processes 16 single-precision floats per AVX-512 register.
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        let len = a.len();
        let mut ptr_a = a.as_ptr();
        let mut ptr_b = b.as_ptr();

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        let mut chunks = len / 32;
        while chunks > 0 {
            let va0 = _mm512_loadu_ps(ptr_a);
            let vb0 = _mm512_loadu_ps(ptr_b);
            acc0 = _mm512_fmadd_ps(va0, vb0, acc0);

            let va1 = _mm512_loadu_ps(ptr_a.add(16));
            let vb1 = _mm512_loadu_ps(ptr_b.add(16));
            acc1 = _mm512_fmadd_ps(va1, vb1, acc1);

            ptr_a = ptr_a.add(32);
            ptr_b = ptr_b.add(32);
            chunks -= 1;
        }

        let mut rem16 = (len % 32) / 16;
        while rem16 > 0 {
            let va = _mm512_loadu_ps(ptr_a);
            let vb = _mm512_loadu_ps(ptr_b);
            acc0 = _mm512_fmadd_ps(va, vb, acc0);

            ptr_a = ptr_a.add(16);
            ptr_b = ptr_b.add(16);
            rem16 -= 1;
        }

        let acc = _mm512_add_ps(acc0, acc1);
        let mut res = _mm512_reduce_add_ps(acc);

        let tail = len - (len % 16);
        for i in tail..len {
            res += a[i] * b[i];
        }

        res
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        // Safe runtime AVX2/scalar fallback when compiling without direct -C target-feature=+avx512f
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return crate::simd::avx2::dot_f32(a, b);
            }
        }
        crate::simd::scalar::dot_f32(a, b)
    }
}

/// AVX-512 elementwise addition: `out = a + b`
#[target_feature(enable = "avx512f")]
pub unsafe fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let vo = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(out.as_mut_ptr().add(i), vo);
            i += 16;
        }
        for idx in i..len {
            out[idx] = a[idx] + b[idx];
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                crate::simd::avx2::add_f32(a, b, out);
                return;
            }
        }
        crate::simd::scalar::add_f32(a, b, out);
    }
}

/// AVX-512 elementwise multiplication: `out = a * b`
#[target_feature(enable = "avx512f")]
pub unsafe fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let vo = _mm512_mul_ps(va, vb);
            _mm512_storeu_ps(out.as_mut_ptr().add(i), vo);
            i += 16;
        }
        for idx in i..len {
            out[idx] = a[idx] * b[idx];
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                crate::simd::avx2::mul_f32(a, b, out);
                return;
            }
        }
        crate::simd::scalar::mul_f32(a, b, out);
    }
}

/// High-throughput AVX-512 matrix multiplication kernel: `C = alpha * (A @ B) + beta * C`
///
/// Uses 16-element ZMM SIMD vectors with cache blocking and register tiling.
#[target_feature(enable = "avx512f")]
pub unsafe fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            for val in c.iter_mut() {
                *val *= beta;
            }
        }

        const NR: usize = 16;
        const KC: usize = 128;

        for jc in (0..n).step_by(NR) {
            let j_len = (n - jc).min(NR);

            for pc in (0..k).step_by(KC) {
                let p_len = (k - pc).min(KC);

                for i in 0..m {
                    if j_len == NR {
                        let mut c_vec = _mm512_loadu_ps(c.as_ptr().add(i * n + jc));
                        for p in pc..(pc + p_len) {
                            let a_val = _mm512_set1_ps(alpha * a[i * k + p]);
                            let b_vec = _mm512_loadu_ps(b.as_ptr().add(p * n + jc));
                            c_vec = _mm512_fmadd_ps(a_val, b_vec, c_vec);
                        }
                        _mm512_storeu_ps(c.as_mut_ptr().add(i * n + jc), c_vec);
                    } else {
                        for p in pc..(pc + p_len) {
                            let a_val = alpha * a[i * k + p];
                            for col in 0..j_len {
                                c[i * n + jc + col] += a_val * b[p * n + jc + col];
                            }
                        }
                    }
                }
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                crate::simd::avx2::gemm_f32(m, n, k, alpha, a, b, beta, c);
                return;
            }
        }
        crate::simd::scalar::gemm_f32(m, n, k, alpha, a, b, beta, c);
    }
}
