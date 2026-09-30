"""3D Net: nucleus detector for 3D+t light-sheet movies.

Input is a window of three frames (t-1, t, t+1), each (Z, Y, X) on the XY-pooled grid. Two 3D convolutional
encoders read the current frame and the change between the next and the previous frame. At the coarsest scale the
change features gate the frame features, and a few self-attention blocks mix the coarse 3D tokens. A decoder with
skip connections from both encoders returns to the input grid, where two 1x1x1 heads predict

  * a 3D unit vector field pointing from each voxel to the centre of its nucleus (Cellpose-style flow), and
  * a cell-probability logit.

Nuclei are recovered by advecting the voxels above a probability threshold along the field (see detect.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def double_conv(cin: int, cout: int) -> nn.Sequential:
    g = min(8, cout)
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.SiLU(inplace=True),
        nn.Conv3d(cout, cout, kernel_size=3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.SiLU(inplace=True))


class Encoder(nn.Module):
    """Four stages (c, 2c, 4c, 8c) separated by 2x max pooling; returns the output of every stage."""

    def __init__(self, cin: int, c: int):
        super().__init__()
        self.stages = nn.ModuleList([double_conv(cin, c), double_conv(c, 2 * c), double_conv(2 * c, 4 * c), double_conv(4 * c, 8 * c)])
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x if i == 0 else self.pool(x))
            feats.append(x)
        return feats


class ChangeGate(nn.Module):
    """Change features produce a gate in (0, 1) for the frame features; both are then merged back to C channels."""

    def __init__(self, ch: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=3, padding=1, bias=True), nn.GroupNorm(min(8, ch), ch), nn.SiLU(inplace=True),
            nn.Conv3d(ch, ch, kernel_size=1, bias=True), nn.Sigmoid())
        self.merge = double_conv(2 * ch, ch)

    def forward(self, frame, change):
        return self.merge(torch.cat([frame * self.gate_conv(change), change], dim=1))


class MixerBlock(nn.Module):
    """Depthwise-conv position term, then pre-norm multi-head self-attention and MLP over the flattened 3D tokens."""

    def __init__(self, ch: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.pos = nn.Conv3d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=True)
        self.ln1 = nn.LayerNorm(ch)
        self.mha = nn.MultiheadAttention(ch, heads, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(ch)
        hidden = int(round(ch * mlp_ratio))
        self.ffn = nn.Sequential(nn.Linear(ch, hidden), nn.GELU(), nn.Linear(hidden, ch))

    def forward(self, x):
        b, c, z, y, w = x.shape
        x = x + self.pos(x)
        tok = x.flatten(2).transpose(1, 2)
        h = self.ln1(tok)
        tok = tok + self.mha(h, h, h, need_weights=False)[0]
        tok = tok + self.ffn(self.ln2(tok))
        return tok.transpose(1, 2).reshape(b, c, z, y, w)


class Body(nn.Module):
    """Feature extractor: (B, 3, Z, Y, X) window -> (B, c, Z, Y, X) features."""

    def __init__(self, c: int = 32, heads: int = 4, blocks: int = 2):
        super().__init__()
        self.enc_frame = Encoder(1, c)
        self.enc_change = Encoder(1, c)
        self.gate = ChangeGate(8 * c)
        self.mixer = nn.ModuleList([MixerBlock(8 * c, heads) for _ in range(blocks)])
        self.ups = nn.ModuleList([nn.ConvTranspose3d(8 * c, 4 * c, kernel_size=2, stride=2),
                                  nn.ConvTranspose3d(4 * c, 2 * c, kernel_size=2, stride=2),
                                  nn.ConvTranspose3d(2 * c, c, kernel_size=2, stride=2)])
        self.decs = nn.ModuleList([double_conv(12 * c, 4 * c), double_conv(6 * c, 2 * c), double_conv(3 * c, c)])
        self.out_channels = c

    def forward(self, x):
        prev, cur, nxt = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        f = self.enc_frame(cur)
        g = self.enc_change(nxt - prev)
        h = self.gate(f[3], g[3])
        for blk in self.mixer:
            h = blk(h)
        for i, level in enumerate((2, 1, 0)):
            h = self.decs[i](torch.cat([self.ups[i](h), f[level], g[level]], dim=1))
        return h


class FlowProbHead(nn.Module):
    def __init__(self, cin: int):
        super().__init__()
        self.flow = nn.Conv3d(cin, 3, 1)
        self.prob = nn.Conv3d(cin, 1, 1)

    def forward(self, f):
        v = self.flow(f)
        return dict(flow=v / (v.norm(dim=1, keepdim=True) + 1e-6), raw_flow=v, prob=self.prob(f))


class Net3D(nn.Module):
    def __init__(self, c: int = 32, heads: int = 4, blocks: int = 2):
        super().__init__()
        self.backbone = Body(c, heads, blocks)
        self.head = FlowProbHead(c)

    def forward(self, x):
        return self.head(self.backbone(x))


def load_net3d(path, device):
    """Returns (model, config, tag); config holds the input grid (pool_xy) and the frame window (window_half)."""
    ck = torch.load(str(path), map_location=device, weights_only=False)
    cfg = dict(ck['config'])
    model = Net3D(cfg.get('c', 32), cfg.get('heads', 4), cfg.get('blocks', 2))
    model.load_state_dict(ck['model'], strict=True)
    return model.to(device).eval(), cfg, cfg.get('tag', '')
