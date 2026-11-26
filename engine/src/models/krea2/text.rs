//! Krea 2 text conditioning (krea2/src/text_encoder.py): Qwen3-VL-4B hidden states after 12 layers, a fixed
//! instruction template, the 34-token system prefix sliced off.
use anyhow::Result;
use candle_core::Tensor;

use crate::text::qwen3vl::TextEncoder;

pub const SELECT_LAYERS: [usize; 12] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];
const PREFIX: &str = "<|im_start|>system
Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>
<|im_start|>user
";
const SUFFIX: &str = "<|im_end|>
<|im_start|>assistant
";
const PREFIX_IDX: usize = 34;
const MAX_LEN: usize = 512;

/// Prompt -> [l, 12, 2560] bf16.
pub fn encode(te: &TextEncoder, prompt: &str) -> Result<Tensor> {
    te.encode_layers(PREFIX, prompt, SUFFIX, PREFIX_IDX, MAX_LEN, &SELECT_LAYERS)
}
