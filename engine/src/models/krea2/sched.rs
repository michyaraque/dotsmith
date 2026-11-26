//! Krea 2 flow-matching schedule (krea2/src/pipeline.py timesteps).

/// Krea's resolution-aware schedule (pipeline.timesteps): linspace(1, 0, steps+1) time-shifted with
/// mu = lerp over image tokens between (256 px -> 0.5) and (1280 px -> 1.15), or a fixed mu (Turbo: 1.15).
pub fn timesteps(seq_len: usize, steps: usize, mu: Option<f64>) -> Vec<f32> {
    let (x1, x2, y1, y2) = ((256f64 / 16.0).powi(2), (1280f64 / 16.0).powi(2), 0.5, 1.15);
    let mu = mu.unwrap_or_else(|| {
        let slope = (y2 - y1) / (x2 - x1);
        slope * seq_len as f64 + (y1 - slope * x1)
    });
    let e = mu.exp() as f32; // the reference does this arithmetic on an fp32 tensor
    (0..=steps)
        .map(|i| {
            let t = 1.0f32 - i as f32 / steps as f32; // torch.linspace(1, 0, steps + 1) in fp32
            if t <= 0.0 { 0.0 } else { e / (e + (1.0 / t - 1.0)) }
        })
        .collect()
}
