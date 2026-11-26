//! Image-model inference in Rust (candle + CUDA): int8 ConvRot weights, ai-toolkit LoRAs, sprite post-processing.
//!
//! core/    shared building blocks (weights, int8 + LoRA, attention, fused kernels, rng)
//! text/    text encoders (Qwen3-VL)
//! models/  one folder per family (qwen21, krea2, lilipixel, rdflow): network, decoder, sampler
//! model    the DiffusionModel trait and ModelKind every family implements / is picked by
//! engine   the generation loop, generic over DiffusionModel
//! locations base-model files on this machine
//! gpu      CUDA device check
//! manifest the model catalog (models.json): files, sources and defaults per family
//! catalog  trained LoRA runs on disk per family (version.json: family, scale, background)
//! sprite   pixel post-processing (keying, depixel, palette)
//! pixeloe  image -> pixel art conversion
pub mod catalog;
pub mod core;
pub mod engine;
pub mod gpu;
pub mod locations;
pub mod manifest;
pub mod model;
pub mod models;
pub mod pixeloe;
pub mod sprite;
pub mod text;
