"""Reference tensors for the LiliPixel port (engine/src/models/lilipixel). Runs no code from the model repos: the
text encoder is transformers' own Qwen3Model with bidirectional attention over the real tokens plus voyage's 1024 ->
2048 `linear` (MongoDB/voyage-4-nano), and the DiT is written out below from Pinkstack/LiliPixel-V1-190M's
pipeline.py. Everything in float32 so the port can be compared tightly.

usage: python dump_ref_lilipixel.py <voyage config.json> <voyage tokenizer.json> <voyage safetensors> <lilipixel safetensors> <out.safetensors>
"""
import math
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from transformers import Qwen3Config, Qwen3Model

cfg_json, tok_json, te_weights, dit_weights, out = sys.argv[1:6]
dev = "cuda"
torch.manual_seed(0)
PROMPT = "a red health potion in a round glass bottle"
SIZE, STEPS, GUIDANCE, TEXT_LEN = 32, 30, 3.0, 96
DIM, HEADS, LAYERS = 768, 16, 24
PAD = 151643  # <|endoftext|>

# ---- text encoder
tok = Tokenizer.from_file(tok_json)
ids = tok.encode(PROMPT, add_special_tokens=True).ids[:TEXT_LEN]
n = len(ids)
ids = ids + [PAD] * (TEXT_LEN - n)
mask = torch.tensor([[1] * n + [0] * (TEXT_LEN - n)], device=dev)
ids_t = torch.tensor([ids], device=dev)

config = Qwen3Config.from_json_file(cfg_json)
config._attn_implementation = "eager"
te = Qwen3Model(config)
sd = load_file(te_weights)
te.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")})
for layer in te.layers:
    layer.self_attn.is_causal = False
te = te.to(dev, torch.float32).eval()
linear = sd["linear.weight"].to(dev, torch.float32)
# every query sees every real token (keys of padding are masked out)
bias = torch.zeros(1, 1, TEXT_LEN, TEXT_LEN, device=dev)
bias[..., n:] = float("-inf")
with torch.no_grad():
    hidden = te(input_ids=ids_t, attention_mask=bias).last_hidden_state
    feats = hidden @ linear.T

# ---- DiT
w = {k[len("inner."):] if k.startswith("inner.") else k: v.to(dev, torch.float32) for k, v in load_file(dit_weights).items()}


def rms(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def lin(x, name, bias=True):
    y = x @ w[name + ".weight"].T
    return y + w[name + ".bias"] if bias and name + ".bias" in w else y


def rope(nh, p):
    hd = DIM // HEADS
    nf = hd // 4
    coords = torch.arange(nh, device=dev, dtype=torch.float32) * p + (p - 1) / 2.0
    ys, xs = torch.meshgrid(coords, coords, indexing="ij")
    pos_y = torch.cat([-(torch.arange(TEXT_LEN, device=dev, dtype=torch.float32) + 1.0), ys.reshape(-1)])
    pos_x = torch.cat([torch.zeros(TEXT_LEN, device=dev), xs.reshape(-1)])
    inv = 1.0 / (10000.0 ** (torch.arange(nf, device=dev, dtype=torch.float32) / nf))
    ang = torch.cat([pos_y[:, None] * inv, pos_x[:, None] * inv], dim=-1)
    return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)


def rot_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def dit(x, t, txt, m, drop):
    B, C, H, W = x.shape
    p = 2 if H >= 64 else 1
    nh = H // p
    img = x.reshape(B, C, nh, p, nh, p).permute(0, 2, 4, 1, 3, 5).reshape(B, nh * nh, C * p * p)
    img = img @ w[f"x_projs.{p}.weight"].T
    txt = rms(txt, w["txt_norm.weight"]) @ w["txt_proj.weight"].T
    mb = m.bool()
    txt = torch.where(mb.unsqueeze(-1), txt, w["pad_emb"])
    d = drop.view(B, 1, 1)
    txt = torch.where(d, w["null_txt"].unsqueeze(0), txt)
    mb = torch.where(drop.view(B, 1), torch.ones_like(mb), mb)
    seq = torch.cat([txt, img], 1)
    half = DIM // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=dev, dtype=torch.float32) / half)
    args = (t * 1000.0)[:, None] * freqs[None]
    temb = torch.cat([args.cos(), args.sin()], -1)
    temb = lin(F.silu(lin(temb, "time_mlp.0")), "time_mlp.2")
    keep = mb.unsqueeze(-1).float()
    pooled = (txt * keep).sum(1) / keep.sum(1).clamp(min=1.0)
    cond = temb + pooled + w["res_emb.weight"][(16, 32, 64).index(H)]
    mod = lin(F.silu(cond), "mod").view(B, 6, 1, DIM).unbind(1)
    cos, sin = rope(nh, p)
    h = seq
    for i in range(LAYERS):
        b = f"blocks.{i}"
        s1, sc1, g1, s2, sc2, g2 = [mm + w[f"{b}.table"][j] for j, mm in enumerate(mod)]
        a = rms(h, w[f"{b}.norm1.weight"]) * (1 + sc1) + s1
        S = a.shape[1]
        qkv = (a @ w[f"{b}.qkv.weight"].T).view(B, S, 3, HEADS, DIM // HEADS).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = q * cos + rot_half(q) * sin, k * cos + rot_half(k) * sin
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, DIM)
        h = h + (o @ w[f"{b}.proj_out.weight"].T) * g1
        a = rms(h, w[f"{b}.norm2.weight"]) * (1 + sc2) + s2
        h = h + ((F.silu(a @ w[f"{b}.fc1.weight"].T) * (a @ w[f"{b}.fc3.weight"].T)) @ w[f"{b}.fc2.weight"].T) * g2
    h = rms(h, w["norm_out.weight"])[:, TEXT_LEN:]
    o = lin(h, f"outs.{p}")
    return o.reshape(B, nh, nh, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


with torch.no_grad():
    x = torch.randn(1, 4, SIZE, SIZE, device=dev)
    t = torch.tensor([0.7], device=dev)
    v_cond = dit(x, t, feats, mask, torch.tensor([False], device=dev))
    v_null = dit(x, t, feats, mask, torch.tensor([True], device=dev))
    # diffusers FlowMatchEulerDiscreteScheduler, shift 1: sigmas = linspace(1, 1/1000, n) then 0
    sigmas = torch.cat([torch.linspace(1.0, 1.0 / 1000, STEPS), torch.zeros(1)]).to(dev)
    noise = torch.randn(1, 4, SIZE, SIZE, device=dev)
    lat = noise.clone()
    f2, m2, drop = torch.cat([feats, feats]), torch.cat([mask, mask]), torch.tensor([False, True], device=dev)
    for i in range(STEPS):
        v = dit(torch.cat([lat, lat]), sigmas[i].expand(2), f2, m2, drop)
        vc, vu = v.chunk(2)
        lat = lat + (sigmas[i + 1] - sigmas[i]) * (vu + GUIDANCE * (vc - vu))

save_file({"ids": torch.tensor([ids], dtype=torch.int64), "mask": mask.cpu().to(torch.int64).contiguous(),
           "feats": feats.cpu().contiguous(), "x": x.cpu().contiguous(), "v_cond": v_cond.cpu().contiguous(),
           "v_null": v_null.cpu().contiguous(), "noise": noise.cpu().contiguous(), "final": lat.cpu().contiguous()},
          out, metadata={"prompt": PROMPT, "size": str(SIZE), "steps": str(STEPS), "guidance": str(GUIDANCE), "t": "0.7"})
print("tokens", n, "feats", tuple(feats.shape), "final range", float(lat.min()), float(lat.max()))
img = ((lat[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
from PIL import Image
Image.fromarray(img, "RGBA").resize((256, 256), Image.NEAREST).save(out.replace(".safetensors", ".png"))
