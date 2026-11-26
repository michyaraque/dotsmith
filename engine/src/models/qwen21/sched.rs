//! Flow-matching Euler schedules as the studio ran them (diffusers FlowMatchEulerDiscreteScheduler configs of
//! ai-toolkit's qwen_image_2): base = linspace(1, 1/steps) + exponential resolution shift (mu) + stretch to 0.02;
//! turbo = the checkpoint's fixed 8 sigmas. Both end with a terminal 0.

pub const TURBO_SIGMAS: [f32; 8] = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568];

pub fn calculate_shift(seq_len: usize) -> f64 {
    let (base_len, max_len, base_shift, max_shift) = (256.0, 8192.0, 0.5, 0.9);
    let slope = (max_shift - base_shift) / (max_len - base_len);
    seq_len as f64 * slope + base_shift - slope * base_len
}

/// Sigmas (with the terminal 0) for `steps` on a latent of `seq_len` tokens.
pub fn sigmas(steps: usize, seq_len: usize, turbo: bool) -> Vec<f32> {
    if turbo {
        let mut s = TURBO_SIGMAS.to_vec();
        s.push(0.0);
        return s;
    }
    let mu = calculate_shift(seq_len);
    let lin: Vec<f32> = (0..steps).map(|i| if steps == 1 { 1.0 } else { (1.0 + (1.0 / steps as f64 - 1.0) * i as f64 / (steps - 1) as f64) as f32 }).collect();
    let em = mu.exp() as f32;
    let shifted: Vec<f32> = lin.iter().map(|&t| em / (em + (1.0 / t - 1.0))).collect();
    let terminal = 0.02f32;
    let scale = (1.0 - shifted[steps - 1]) / (1.0 - terminal);
    let mut s: Vec<f32> = shifted.iter().map(|&t| 1.0 - (1.0 - t) / scale).collect();
    s.push(0.0);
    s
}
