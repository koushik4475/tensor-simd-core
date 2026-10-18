//! Scalar baseline implementations for arithmetic, dot product, and GEMM.

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    // Unroll by 4 for higher instruction-level parallelism even on scalar
    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    for (ca, cb) in chunks_a.zip(chunks_b) {
        s0 += ca[0] * cb[0];
        s1 += ca[1] * cb[1];
        s2 += ca[2] * cb[2];
        s3 += ca[3] * cb[3];
    }
    sum = (s0 + s1) + (s2 + s3);

    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        sum += x * y;
    }
    sum
}

#[inline]
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

#[inline]
pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// Cache-blocked triple-loop scalar GEMM: C = alpha * (A @ B) + beta * C
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
    if beta == 0.0 {
        c.fill(0.0);
    } else if beta != 1.0 {
        for val in c.iter_mut() {
            *val *= beta;
        }
    }

    const MC: usize = 64;
    const NC: usize = 64;
    const KC: usize = 64;

    for jc in (0..n).step_by(NC) {
        let j_end = (jc + NC).min(n);
        for pc in (0..k).step_by(KC) {
            let p_end = (pc + KC).min(k);
            for ic in (0..m).step_by(MC) {
                let i_end = (ic + MC).min(m);

                for i in ic..i_end {
                    let a_row_offset = i * k;
                    let c_row_offset = i * n;
                    for p in pc..p_end {
                        let a_val = alpha * a[a_row_offset + p];
                        if a_val == 0.0 {
                            continue;
                        }
                        let b_row_offset = p * n;
                        for j in jc..j_end {
                            c[c_row_offset + j] += a_val * b[b_row_offset + j];
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_dot() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0];
        // 2 + 6 + 12 + 20 + 30 = 70
        assert_eq!(dot_f32(&a, &b), 70.0);
    }

    #[test]
    fn test_scalar_gemm() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let mut c = vec![0.0; 4];
        gemm_f32(2, 2, 2, 1.0, &a, &b, 0.0, &mut c);
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
