"""Reference tensors for the RD Flow 64 port (engine/src/models/rdflow). Runs no code from the model repo: the text
encoder is transformers' own Qwen3VLTextModel cut to 27 layers, and the network and palette ramps are written out
below from Retro-Diffusion/rd-flow-64's rdflow/layers.py, model.py and ramps.py. float32.

usage: python dump_ref_rdflow.py <text_encoder config.json> <tokenizer.json> <text encoder safetensors> <rd-flow-64.safetensors> <out.safetensors>
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

cfg_json, tok_json, te_path, net_path, out = sys.argv[1:6]
dev = "cuda"
torch.manual_seed(0)
CAPTION = "A small red potion bottle with a cork stopper. The background is transparent."

# ---- text: raw tokens, 27 layers + final norm, RMS per token, bf16
import json
tc = json.load(open(cfg_json))["text_config"]
tc["num_hidden_layers"] = 27
config = Qwen3VLTextConfig(**tc)
config._attn_implementation = "eager"
te = Qwen3VLTextModel(config)
sd = load_file(te_path)
missing, unexpected = te.load_state_dict(sd, strict=False)
assert not unexpected and all("rotary" in m for m in missing), (missing[:3], unexpected[:3])
te = te.to(dev, torch.float32).eval()
ids = Tokenizer.from_file(tok_json).encode(CAPTION, add_special_tokens=False).ids
with torch.no_grad():
    h = te(input_ids=torch.tensor([ids], device=dev), use_cache=False).last_hidden_state
    text = F.rms_norm(h.float(), (h.shape[-1],)).to(torch.bfloat16).float()[0]

# ---- network
W = {k: v.to(dev, torch.float32) for k, v in load_file(net_path).items()}
HD, THETA, WIN, N0, N1, NC = 64, 100.0, 8, 64, 32, 16


def lin(x, n):
    y = x @ W[n + ".weight"].T
    return y + W[n + ".bias"] if n + ".bias" in W else y


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


def rope_tables(xy):
    q = HD // 4
    freqs = THETA ** (-torch.arange(q, device=dev, dtype=torch.float32) / q)
    has = ~torch.isnan(xy[:, 0])
    pos = torch.nan_to_num(xy, nan=0.0)
    ang = torch.cat([pos[:, :1] * freqs, pos[:, 1:] * freqs], -1) * has[:, None]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def grid_xy(n):
    ys, xs = torch.meshgrid(torch.arange(n, device=dev), torch.arange(n, device=dev), indexing="ij")
    return torch.stack([xs.flatten(), ys.flatten()], -1).float()


def attn(p, x, cos, sin, mask=None):
    B, N, D = x.shape
    h = D // HD
    q, k, v = lin(x, p + ".qkv").view(B, N, 3, h, HD).permute(2, 0, 3, 1, 4)
    q, k = apply_rope(rms(q, W[p + ".qn.w"]), cos, sin), apply_rope(rms(k, W[p + ".kn.w"]), cos, sin)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return lin(o.transpose(1, 2).reshape(B, N, D), p + ".out")


def mlp(p, x, hw=None):
    a, g = lin(x, p + ".w1").chunk(2, -1)
    if p + ".conv.weight" in W and hw is not None:
        B, N, C = a.shape
        a = F.conv2d(a.transpose(1, 2).reshape(B, C, hw, hw), W[p + ".conv.weight"], W[p + ".conv.bias"], padding=1, groups=C)
        a = a.reshape(B, C, N).transpose(1, 2)
    return lin(F.silu(g) * a, p + ".w2")


def block(p, x, mod, attn_fn, hw=None):
    sh1, sc1, g1, sh2, sc2, g2 = (mod + W[p + ".table"][None]).unbind(1)
    x = x + g1[:, None] * attn_fn(p + ".attn", rms(x, W[p + ".n1.w"]) * (1 + sc1[:, None]) + sh1[:, None])
    return x + g2[:, None] * mlp(p + ".mlp", rms(x, W[p + ".n2.w"]) * (1 + sc2[:, None]) + sh2[:, None], hw)


def windows(x, n, shift):
    B, N, D = x.shape
    g = x.view(B, n, n, D)
    if shift:
        g = torch.roll(g, (-shift, -shift), (1, 2))
    return g.view(B, n // WIN, WIN, n // WIN, WIN, D).permute(0, 1, 3, 2, 4, 5).reshape(B * (n // WIN) ** 2, WIN * WIN, D)


def unwindows(x, n, shift, B):
    D = x.shape[-1]
    g = x.view(B, n // WIN, n // WIN, WIN, WIN, D).permute(0, 1, 3, 2, 4, 5).reshape(B, n, n, D)
    if shift:
        g = torch.roll(g, (shift, shift), (1, 2))
    return g.reshape(B, n * n, D)


def level(p, x, c, n):
    B, N, D = x.shape
    mod = lin(F.silu(c), p + ".ada").view(B, 6, D)
    xy = grid_xy(n)
    i = 0
    while f"{p}.blocks.{i}.table" in W:
        shift = WIN // 2 if i % 2 else 0
        cos, sin = rope_tables(windows(xy[None], n, shift).reshape(-1, 2))
        nw = (n // WIN) ** 2
        cos, sin = cos.view(nw, WIN * WIN, -1)[:, None].repeat(B, 1, 1, 1), sin.view(nw, WIN * WIN, -1)[:, None].repeat(B, 1, 1, 1)
        x = block(f"{p}.blocks.{i}", x, mod, lambda ap, h: unwindows(attn(ap, windows(h, n, shift), cos, sin), n, shift, B), hw=n)
        i += 1
    return x


def merge(x, n):
    B, _, D = x.shape
    return x.view(B, n // 2, 2, n // 2, 2, D).permute(0, 1, 3, 2, 4, 5).reshape(B, (n // 2) ** 2, 4 * D)


def split(x, n):
    B, _, D4 = x.shape
    D = D4 // 4
    return x.view(B, n // 2, n // 2, 2, 2, D).permute(0, 1, 3, 2, 4, 5).reshape(B, n * n, D)


def oklab(rgb):
    c = rgb.float() / 255.0
    c = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = c.unbind(-1)
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).clamp_min(0) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).clamp_min(0) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).clamp_min(0) ** (1 / 3)
    return torch.stack([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], -1)


def palette_embed(rgb, ramp, lvl):
    lab = oklab(rgb)
    ang = lab[..., None] * ((2.0 ** torch.arange(8, device=dev)) * math.pi)
    feats = torch.cat([lab, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1)
    e = lin(F.silu(lin(feats, "colour.mlp.0")), "colour.mlp.2")
    return e + W["ramp_emb.weight"][ramp] + lin(lvl[..., None], "level_proj")


def timestep_embed(t, dim=256):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=dev) / half)
    a = t[:, None] * 1000 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


def forward(x_t, t, text, text_mask, pal, sc, bg):
    B = x_t.shape[0]
    dropped = ~text_mask.any(1)
    text = torch.where(dropped[:, None, None], torch.zeros_like(text), text)
    text = torch.cat([W["null_text"][None].expand(B, -1, -1), text], 1)
    tmask = torch.cat([dropped[:, None], text_mask], 1)
    pooled = (text * tmask[..., None]).sum(1) / tmask.sum(1, keepdim=True).clamp_min(1)
    c = lin(F.silu(lin(timestep_embed(t), "t_mlp.0")), "t_mlp.2") + W["style.weight"][0] + W["source.weight"][6] + lin(pooled, "text_pool")
    for j in range(4):
        c = c + W[f"attr_emb.{j}.weight"][0]
    c = c + W["bg_emb.weight"][bg]
    xin = torch.cat([x_t.reshape(B, -1, 4), sc.reshape(B, -1, 4)], -1)
    h0 = level("down0", lin(xin, "inp"), c, N0)
    h1 = level("down1", lin(merge(h0, N0), "merge0"), c, N1)
    ctx = lin(text, "text_proj")
    L0 = ctx.shape[1]
    tc, ts = rope_tables(torch.full((L0, 2), float("nan"), device=dev))
    zmod = torch.zeros(B, 6, 1024, device=dev)
    amask = tmask[:, None, None, :]
    for i in range(2):
        ctx = block(f"adapter.{i}", ctx, zmod, lambda ap, h: attn(ap, h, tc, ts, amask))
    hc = lin(merge(h1, N1), "merge1")
    R, L = 4, text.shape[1]
    rgb, ramp, lvl = pal
    Pn = rgb.shape[0]
    pt = lin(palette_embed(rgb, ramp, lvl), "pal_tok")[None].expand(B, -1, -1)
    tok = torch.cat([W["registers"][None].expand(B, -1, -1) + W["modality"][0], ctx + W["modality"][1], pt + W["modality"][3], hc + W["modality"][2]], 1)
    S = R + L + Pn
    nan = torch.full((S, 2), float("nan"), device=dev)
    cos, sin = rope_tables(torch.cat([nan, grid_xy(NC)]))
    keymask = torch.cat([torch.ones(B, R, dtype=torch.bool, device=dev), tmask, torch.ones(B, Pn, dtype=torch.bool, device=dev),
                         torch.ones(B, NC * NC, dtype=torch.bool, device=dev)], 1)[:, None, None, :]
    mod = lin(F.silu(c), "core_ada").view(B, 6, 1024)
    for i in range(16):
        tok = block(f"core.{i}", tok, mod, lambda ap, h: attn(ap, h, cos, sin, keymask))
    hc = tok[:, S:]
    u1 = level("up1", lin(torch.cat([split(lin(hc, "split1"), N1), h1], -1), "skip1"), c, N1)
    u0 = rms(level("up0", lin(torch.cat([split(lin(u1, "split0"), N0), h0], -1), "skip0"), c, N0), W["out_norm.w"])
    keys = lin(F.silu(lin(palette_embed(rgb, ramp, lvl), "k.0")), "k.2")
    logits = torch.cat([lin(u0, "transparent"), lin(u0, "q") @ keys.T / math.sqrt(128)], -1)
    return lin(u0, "x_head"), u0, logits


# ---- palette ramps (ramps.py)
def srgb_to_oklab_np(rgb):
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    l, m, s = (np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b), np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b),
               np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b))
    return np.stack([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], -1).astype(np.float32)


def ramps(lab):
    n = len(lab)
    ramp, level_ = np.zeros(n, np.int64), np.full(n, 0.5, np.float32)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma, hue = np.hypot(a, b), np.arctan2(b, a) % (2 * math.pi)
    chromatic = np.flatnonzero(chroma >= 0.035)
    groups = []
    if len(chromatic):
        order = chromatic[np.argsort(hue[chromatic])]
        h = hue[order]
        gaps = np.diff(np.concatenate([h, h[:1] + 2 * math.pi]))
        order = np.roll(order, -(int(np.argmax(gaps)) + 1))
        h = np.unwrap(hue[order])
        cur = [order[0]]
        for i in range(1, len(order)):
            if h[i] - h[i - 1] > math.radians(28):
                groups.append(cur)
                cur = []
            cur.append(order[i])
        groups.append(cur)
        split_ = []
        for g in groups:
            while len(g) > 2:
                hg = np.unwrap(hue[g])
                if hg[-1] - hg[0] <= math.radians(75):
                    break
                k = int(np.argmax(np.diff(hg))) + 1
                split_.append(g[:k])
                g = g[k:]
            split_.append(g)
        groups = split_
    neutral = np.flatnonzero(chroma < 0.035)
    for rid, g in [(0, list(neutral))] + [(min(i + 1, 15), g) for i, g in enumerate(groups)]:
        if not len(g):
            continue
        g = np.asarray(g)
        ramp[g] = rid
        if len(g) > 1:
            level_[g] = np.argsort(np.argsort(L[g])) / (len(g) - 1)
    return ramp, level_


pal_rgb = np.array([[20, 20, 24], [90, 90, 96], [230, 230, 236], [120, 20, 30], [200, 40, 50], [250, 120, 110], [30, 60, 150],
                    [70, 120, 230], [160, 200, 255], [40, 110, 40], [110, 190, 70], [220, 180, 60]], dtype=np.uint8)
ramp, lvl = ramps(srgb_to_oklab_np(pal_rgb))

with torch.no_grad():
    x_t = torch.randn(2, 64, 64, 4, device=dev)
    sc = torch.randn(2, 64, 64, 4, device=dev) * 0.5
    L = text.shape[0]
    texts = text[None].expand(2, -1, -1)
    mask = torch.tensor([[True] * L, [False] * L], device=dev)
    pal = (torch.tensor(pal_rgb, device=dev).long(), torch.tensor(ramp, device=dev), torch.tensor(lvl, device=dev))
    xh, feats, logits = forward(x_t, torch.tensor([0.7, 0.7], device=dev), texts, mask, pal, sc, torch.tensor([2, 2], device=dev))

save_file({"ids": torch.tensor(ids, dtype=torch.int64), "text": text.cpu().contiguous(), "x_t": x_t.cpu().contiguous(), "sc": sc.cpu().contiguous(),
           "pal_rgb": torch.tensor(pal_rgb.astype(np.int64)), "ramp": torch.tensor(ramp), "level": torch.tensor(lvl),
           "x_hat": xh.reshape(2, -1, 4).cpu().contiguous(), "feats": feats.cpu().contiguous(), "logits": logits.cpu().contiguous()},
          out, metadata={"caption": CAPTION, "t": "0.7"})
print("tokens", len(ids), "text", tuple(text.shape), "x_hat", tuple(xh.shape), "ramps", ramp.tolist())
