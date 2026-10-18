//! NumPy / PyTorch style multidimensional broadcasting rules and coordinate mapping.

use crate::tensor::strides::Strides;

/// Determines whether two shapes can be broadcast together according to standard rules,
/// returning the resulting broadcast shape.
///
/// Rules:
/// 1. Align shapes to the right.
/// 2. For each dimension pair (d1, d2):
///    - If d1 == d2, dimension is preserved.
///    - If d1 == 1 or d2 == 1, dimension expands to max(d1, d2).
///    - Otherwise, shapes are incompatible.
pub fn broadcast_shapes(s1: &[usize], s2: &[usize]) -> Option<Vec<usize>> {
    let rank = s1.len().max(s2.len());
    let mut result = vec![0usize; rank];

    for i in 0..rank {
        let dim1 = if i < s1.len() {
            s1[s1.len() - 1 - i]
        } else {
            1
        };

        let dim2 = if i < s2.len() {
            s2[s2.len() - 1 - i]
        } else {
            1
        };

        if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
            result[rank - 1 - i] = dim1.max(dim2);
        } else {
            return None;
        }
    }

    Some(result)
}

/// Compute broadcasted strides for expanding an existing tensor into target broadcast shape.
/// Dimensions that expand from size 1 get stride 0 (zero memory overhead broadcasting).
pub fn broadcast_strides(orig_shape: &[usize], orig_strides: &Strides, target_shape: &[usize]) -> Option<Strides> {
    let orig_rank = orig_shape.len();
    let target_rank = target_shape.len();

    if orig_rank > target_rank {
        return None;
    }

    let rank_diff = target_rank - orig_rank;
    let mut new_strides = vec![0isize; target_rank];

    for i in 0..target_rank {
        if i < rank_diff {
            // Prepended broadcast dimension: stride is 0
            new_strides[i] = 0;
        } else {
            let orig_idx = i - rank_diff;
            let orig_dim = orig_shape[orig_idx];
            let target_dim = target_shape[i];

            if orig_dim == target_dim {
                new_strides[i] = orig_strides.as_slice()[orig_idx];
            } else if orig_dim == 1 {
                new_strides[i] = 0; // Broadcast dimension maps repeatedly to same element
            } else {
                return None;
            }
        }
    }

    Some(Strides::new(new_strides))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(
            broadcast_shapes(&[2, 1, 4], &[3, 1]),
            Some(vec![2, 3, 4])
        );
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 4]), None);
        assert_eq!(
            broadcast_shapes(&[5], &[2, 3, 5]),
            Some(vec![2, 3, 5])
        );
    }
}
