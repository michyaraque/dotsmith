//! torch's CPU generator: `torch.randn(shape, generator=torch.Generator().manual_seed(seed))` bit for bit
//! (mt19937 + ATen normal_fill: 24-bit uniforms, Box-Muller in blocks of 16), so a seed gives the same noise as the
//! PyTorch studio did.

pub struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    pub fn new(seed: u64) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed as u32;
        for i in 1..624 {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Self { mt, idx: 624 }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
            let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_b0df;
            }
            self.mt[i] = v;
        }
        self.idx = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// at::uniform_real_distribution<float>(0, 1): 24 random bits * 2^-24.
    fn uniform_f32(&mut self) -> f32 {
        ((self.next_u32() & ((1 << 24) - 1)) as f64 * (2f64).powi(-24)) as f32
    }
}

fn normal_fill_16(d: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0 - d[j];
        let u2 = d[j + 8];
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        d[j] = radius * theta.cos();
        d[j + 8] = radius * theta.sin();
    }
}

/// torch.randn(n) on a fresh CPU generator seeded with `seed` (n >= 16).
pub fn randn(seed: u64, n: usize) -> Vec<f32> {
    assert!(n >= 16, "the vectorised torch path needs at least 16 values");
    let mut g = Mt19937::new(seed);
    let mut d: Vec<f32> = (0..n).map(|_| g.uniform_f32()).collect();
    let mut i = 0;
    while i + 16 <= n {
        normal_fill_16(&mut d[i..i + 16]);
        i += 16;
    }
    if !n.is_multiple_of(16) {
        let tail = n - 16;
        for v in d[tail..].iter_mut() {
            *v = g.uniform_f32();
        }
        normal_fill_16(&mut d[tail..]);
    }
    d
}
