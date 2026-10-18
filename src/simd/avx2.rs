//! AVX2 + FMA 256-bit vectorized implementations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 256-bit AVX2/FMA dot product for f32 slices.
///
/// Accumulates 8 floats per vector register with 4-way loop unrolling (32 floats per iteration).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        let len = a.len();
        let mut ptr_a = a.as_ptr();
        let mut ptr_b = b.as_ptr();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut chunks = len / 32;
        while chunks > 0 {
            let va0 = _mm256_loadu_ps(ptr_a);
            let vb0 = _mm256_loadu_ps(ptr_b);
            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

            let va1 = _mm256_loadu_ps(ptr_a.add(8));
            let vb1 = _mm256_loadu_ps(ptr_b.add(8));
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

            let va2 = _mm256_loadu_ps(ptr_a.add(16));
            let vb2 = _mm256_loadu_ps(ptr_b.add(16));
            acc2 = _mm256_fmadd_ps(va2, vb2, acc2);

            let va3 = _mm256_loadu_ps(ptr_a.add(24));
            let vb3 = _mm256_loadu_ps(ptr_b.add(24));
            acc3 = _mm256_fmadd_ps(va3, vb3, acc3);

            ptr_a = ptr_a.add(32);
            ptr_b = ptr_b.add(32);
            chunks -= 1;
        }

        // Remaining 8-float chunks
        let mut rem_8 = (len % 32) / 8;
        while rem_8 > 0 {
            let va = _mm256_loadu_ps(ptr_a);
            let vb = _mm256_loadu_ps(ptr_b);
            acc0 = _mm256_fmadd_ps(va, vb, acc0);

            ptr_a = ptr_a.add(8);
            ptr_b = ptr_b.add(8);
            rem_8 -= 1;
        }

        let sum256 = _mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        );

        // Horizontal add across 256-bit lane
        let hi128 = _mm256_extractf128_ps(sum256, 1);
        let lo128 = _mm256_castps256_ps128(sum256);
        let sum128 = _mm_add_ps(lo128, hi128);

        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, shuf2);

        let mut res = _mm_cvtss_f32(final_sum);

        // Tail remainder
        let tail_start = len - (len % 8);
        for i in tail_start..len {
            res += a[i] * b[i];
        }

        res
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        crate::simd::scalar::dot_f32(a, b)
    }
}

/// AVX2 elementwise addition: `out = a + b`
#[target_feature(enable = "avx2")]
pub unsafe fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        let len = a.len();
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vo = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), vo);
            i += 8;
        }
        for idx in i..len {
            out[idx] = a[idx] + b[idx];
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        crate::simd::scalar::add_f32(a, b, out);
    }
}

/// AVX2 elementwise multiplication: `out = a * b`
#[target_feature(enable = "avx2")]
pub unsafe fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        let len = a.len();
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vo = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), vo);
            i += 8;
        }
        for idx in i..len {
            out[idx] = a[idx] * b[idx];
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        crate::simd::scalar::mul_f32(a, b, out);
    }
}

/// High-performance AVX2 + FMA matrix multiplication micro-kernel and blocked GEMM.
///
/// Implements a 6x16 register-blocked inner kernel accumulating into 12 AVX2 registers.
#[target_feature(enable = "avx2,fma")]
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
    #[cfg(target_arch = "x86_64")]
    {
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            for val in c.iter_mut() {
                *val *= beta;
            }
        }

        const MR: usize = 4;
        const NR: usize = 8;
        const MC: usize = 128;
        const KC: usize = 128;

        for jc in (0..n).step_by(NR) {
            let j_len = (n - jc).min(NR);

            for pc in (0..k).step_by(KC) {
                let p_len = (k - pc).min(KC);

                for ic in (0..m).step_by(MC) {
                    let i_len = (m - ic).min(MC);

                    // Micro-kernel on blocks
                    let mut i = ic;
                    while i + MR <= ic + i_len {
                        if j_len == NR {
                            // Fast path: full 4x8 register tile
                            let mut c0 = _mm256_loadu_ps(c.as_ptr().add(i * n + jc));
                            let mut c1 = _mm256_loadu_ps(c.as_ptr().add((i + 1) * n + jc));
                            let mut c2 = _mm256_loadu_ps(c.as_ptr().add((i + 2) * n + jc));
                            let mut c3 = _mm256_loadu_ps(c.as_ptr().add((i + 3) * n + jc));

                            for p in pc..(pc + p_len) {
                                let b_vec = _mm256_loadu_ps(b.as_ptr().add(p * n + jc));

                                let a0 = _mm256_set1_ps(alpha * a[i * k + p]);
                                let a1 = _mm256_set1_ps(alpha * a[(i + 1) * k + p]);
                                let a2 = _mm256_set1_ps(alpha * a[(i + 2) * k + p]);
                                let a3 = _mm256_set1_ps(alpha * a[(i + 3) * k + p]);

                                c0 = _mm256_fmadd_ps(a0, b_vec, c0);
                                c1 = _mm256_fmadd_ps(a1, b_vec, c1);
                                c2 = _mm256_fmadd_ps(a2, b_vec, c2);
                                c3 = _mm256_fmadd_ps(a3, b_vec, c3);
                            }

                            _mm256_storeu_ps(c.as_mut_ptr().add(i * n + jc), c0);
                            _mm256_storeu_ps(c.as_mut_ptr().add((i + 1) * n + jc), c1);
                            _mm256_storeu_ps(c.as_mut_ptr().add((i + 2) * n + jc), c2);
                            _mm256_storeu_ps(c.as_mut_ptr().add((i + 3) * n + jc), c3);
                        } else {
                            // Tail columns with partial MR rows
                            for r in 0..MR {
                                let row = i + r;
                                for p in pc..(pc + p_len) {
                                    let a_val = alpha * a[row * k + p];
                                    for col in 0..j_len {
                                        c[row * n + jc + col] += a_val * b[p * n + jc + col];
                                    }
                                }
                            }
                        }
                        i += MR;
                    }

                    // Remaining rows
                    while i < ic + i_len {
                        for p in pc..(pc + p_len) {
                            let a_val = alpha * a[i * k + p];
                            for col in 0..j_len {
                                c[i * n + jc + col] += a_val * b[p * n + jc + col];
                            }
                        }
                        i += 1;
                    }
                }
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        crate::simd::scalar::gemm_f32(m, n, k, alpha, a, b, beta, c);
    }
}
