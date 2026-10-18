//! ARM NEON 128-bit SIMD implementation for aarch64 (Apple Silicon M1/M2/M3/M4, AWS Graviton, etc.).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// ARM NEON 128-bit vector dot product for f32 slices.
///
/// Uses 4x32-bit float vector registers with 4-way loop unrolling (16 floats per iteration).
#[target_feature(enable = "neon")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        let len = a.len();
        let mut ptr_a = a.as_ptr();
        let mut ptr_b = b.as_ptr();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut chunks = len / 16;
        while chunks > 0 {
            let va0 = vld1q_f32(ptr_a);
            let vb0 = vld1q_f32(ptr_b);
            acc0 = vfmaq_f32(acc0, va0, vb0);

            let va1 = vld1q_f32(ptr_a.add(4));
            let vb1 = vld1q_f32(ptr_b.add(4));
            acc1 = vfmaq_f32(acc1, va1, vb1);

            let va2 = vld1q_f32(ptr_a.add(8));
            let vb2 = vld1q_f32(ptr_b.add(8));
            acc2 = vfmaq_f32(acc2, va2, vb2);

            let va3 = vld1q_f32(ptr_a.add(12));
            let vb3 = vld1q_f32(ptr_b.add(12));
            acc3 = vfmaq_f32(acc3, va3, vb3);

            ptr_a = ptr_a.add(16);
            ptr_b = ptr_b.add(16);
            chunks -= 1;
        }

        let mut rem4 = (len % 16) / 4;
        while rem4 > 0 {
            let va = vld1q_f32(ptr_a);
            let vb = vld1q_f32(ptr_b);
            acc0 = vfmaq_f32(acc0, va, vb);

            ptr_a = ptr_a.add(4);
            ptr_b = ptr_b.add(4);
            rem4 -= 1;
        }

        let sum01 = vaddq_f32(acc0, acc1);
        let sum23 = vaddq_f32(acc2, acc3);
        let total = vaddq_f32(sum01, sum23);

        let mut res = vaddvq_f32(total);

        let tail = len - (len % 4);
        for i in tail..len {
            res += a[i] * b[i];
        }

        res
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::simd::scalar::dot_f32(a, b)
    }
}

/// ARM NEON elementwise addition: `out = a + b`
#[target_feature(enable = "neon")]
pub unsafe fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vo = vaddq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(i), vo);
            i += 4;
        }
        for idx in i..len {
            out[idx] = a[idx] + b[idx];
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::simd::scalar::add_f32(a, b, out);
    }
}

/// ARM NEON elementwise multiplication: `out = a * b`
#[target_feature(enable = "neon")]
pub unsafe fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vo = vmulq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(i), vo);
            i += 4;
        }
        for idx in i..len {
            out[idx] = a[idx] * b[idx];
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::simd::scalar::mul_f32(a, b, out);
    }
}

/// Blocked ARM NEON GEMM: `C = alpha * (A @ B) + beta * C`
#[target_feature(enable = "neon")]
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
    #[cfg(target_arch = "aarch64")]
    {
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            for val in c.iter_mut() {
                *val *= beta;
            }
        }

        const NR: usize = 4;
        const KC: usize = 128;

        for jc in (0..n).step_by(NR) {
            let j_len = (n - jc).min(NR);

            for pc in (0..k).step_by(KC) {
                let p_len = (k - pc).min(KC);

                for i in 0..m {
                    if j_len == NR {
                        let mut c_vec = vld1q_f32(c.as_ptr().add(i * n + jc));
                        for p in pc..(pc + p_len) {
                            let a_scalar = vdupq_n_f32(alpha * a[i * k + p]);
                            let b_vec = vld1q_f32(b.as_ptr().add(p * n + jc));
                            c_vec = vfmaq_f32(c_vec, a_scalar, b_vec);
                        }
                        vst1q_f32(c.as_mut_ptr().add(i * n + jc), c_vec);
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
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::simd::scalar::gemm_f32(m, n, k, alpha, a, b, beta, c);
    }
}
