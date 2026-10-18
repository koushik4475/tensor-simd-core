//! # tensor-simd-core
//!
//! Cross-Platform High-Performance Matrix & Tensor Math Engine in Rust/C++ SIMD.
//! Features hardware acceleration via AVX-512, AVX2/FMA, and ARM NEON intrinsics.

pub mod error;
pub mod ops;
pub mod simd;
pub mod tensor;

pub use error::{Result, TensorError};
pub use ops::{conv2d, layernorm, leaky_relu, relu, softmax, Conv2dParams};
pub use simd::{dot_f32, gemm_f32, SimdArchitecture};
pub use tensor::slicing::{SliceIndex, SliceSpec};
pub use tensor::strides::Strides;
pub use tensor::Tensor;
