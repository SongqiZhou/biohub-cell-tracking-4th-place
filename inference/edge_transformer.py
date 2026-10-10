"""Edge transformer (inference only): a temporal 3D U-Net encodes two consecutive frames, features are sampled at the
node positions, and a cross-attention node transformer scores every pair of nodes in the two frames.
Re-implements the linker of the competition's official baseline (tracking-cellmot, BSD 3-Clause License, Copyright (c) 2026
Thibaut Goldsborough); see LICENSE-baseline.txt.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POS_EMBED_DIM = 8                                  # per axis (t, z, y, x)


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


class TimeAttention(nn.Module):
    """Self-attention across the frames of the window, independently at every voxel."""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):                           # (B, T, C, Z, Y, X)
        B, T, C = x.shape[:3]
        spatial = x.shape[3:]
        S = math.prod(spatial)
        h = self.norm(x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C))
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.reshape(B, S, T, C).permute(0, 2, 3, 1).reshape(B, T, C, *spatial)


class TemporalUNet(nn.Module):
    """(B, T, 1, Z, Y, X) -> (B, T, C_out, Z, Y, X); time attention at every stage except the full-resolution one."""

    def __init__(self, out_channels: int = 32, layers=(32, 64, 128)):
        super().__init__()
        layers = list(layers)
        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        prev = 1
        for i, ch in enumerate(layers):
            self.encoder_blocks.append(conv_block(prev, ch))
            self.temporal_blocks.append(nn.Identity() if i == 0 else TimeAttention(ch))
            prev = ch
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(layers) - 1, 0, -1):
            self.upsamples.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
            self.decoder_blocks.append(conv_block(layers[i] + layers[i - 1], layers[i - 1]))
        self.head = nn.Conv3d(layers[0], out_channels, kernel_size=1)

    def forward(self, x):
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        skips = []
        for i, (block, temporal) in enumerate(zip(self.encoder_blocks, self.temporal_blocks)):
            if i > 0:
                x = self.pool(x)
            x = block(x)
            x = temporal(x.reshape(B, T, *x.shape[1:])).reshape(B * T, *x.shape[1:])
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)
        for up, block, skip in zip(self.upsamples, self.decoder_blocks, skips[::-1]):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        x = self.head(x)
        return x.reshape(B, T, *x.shape[1:])


class CrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, q, kv, kv_mask=None):
        a, _ = self.cross_attn(self.norm1(q), self.norm1(kv), self.norm1(kv), key_padding_mask=None if kv_mask is None else ~kv_mask)
        q = q + a
        return q + self.mlp(self.norm2(q))


class PairScorer(nn.Module):
    """Nodes of frame t and t+1 attend to each other; every (i, j) pair is then scored by an MLP on both embeddings and
    the relative position."""

    def __init__(self, feat_dim: int, hidden: int = 128, heads: int = 4, blocks: int = 4, chunk: int = 32):
        super().__init__()
        self.chunk = chunk
        self.proj = nn.Linear(feat_dim, hidden)
        self.norm_in = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList([CrossBlock(hidden, heads) for _ in range(blocks)])
        self.norm_out = nn.LayerNorm(hidden)
        self.pair_mlp = nn.Sequential(nn.Linear(hidden * 2 + 3, hidden), nn.GELU(), nn.Dropout(0.3),
                                      nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, feat_a, feat_b, pos_a, pos_b, mask_a=None, mask_b=None):
        q = self.norm_in(self.proj(feat_a))
        k = self.norm_in(self.proj(feat_b))
        for blk in self.blocks:
            q = blk(q, k, mask_b)
            k = blk(k, q, mask_a)
        q = self.norm_out(q); k = self.norm_out(k)
        out = []
        for i in range(0, q.shape[1], self.chunk):
            qc = q[:, i:i + self.chunk]; pc = pos_a[:, i:i + self.chunk]
            qe = qc.unsqueeze(2).expand(-1, -1, k.shape[1], -1)
            ke = k.unsqueeze(1).expand(-1, qc.shape[1], -1, -1)
            rel = (pc.unsqueeze(2) - pos_b.unsqueeze(1)) / 100.0
            out.append(self.pair_mlp(torch.cat([qe, ke, rel], dim=-1)).squeeze(-1))
        return torch.cat(out, dim=1)                # (B, N_a, N_b) logits


class EdgeTransformer(nn.Module):
    def __init__(self, out_channels: int = 32, layers=(32, 64, 128)):
        super().__init__()
        self.unet = TemporalUNet(out_channels, layers)
        self.detect_head = nn.Conv3d(out_channels, 1, kernel_size=1)          # unused here, kept for the checkpoint
        self.transformer = PairScorer(out_channels + 4 * POS_EMBED_DIM)

    def encode(self, imgs):                         # (B, W, Z, Y, X) -> (B, W, C, Z, Y, X)
        return self.unet(imgs.unsqueeze(2))

    @staticmethod
    def sample(feat, coords):                       # feat (C, Z, Y, X), coords (N, 3) grid -> (1, N, C)
        Z, Y, X = feat.shape[1:]
        z = coords[:, 0].long().clamp(0, Z - 1); y = coords[:, 1].long().clamp(0, Y - 1); x = coords[:, 2].long().clamp(0, X - 1)
        return feat[:, z, y, x].T.unsqueeze(0)

    def score(self, feat_a, feat_b, pos_a, pos_b, emb_a, emb_b):
        return self.transformer(torch.cat([feat_a, emb_a], -1), torch.cat([feat_b, emb_b], -1), pos_a, pos_b,
                                torch.ones(feat_a.shape[:2], dtype=torch.bool, device=feat_a.device),
                                torch.ones(feat_b.shape[:2], dtype=torch.bool, device=feat_b.device))


def position_embedding(coords: np.ndarray, shape) -> np.ndarray:
    """(N, 4) [t, z, y, x] grid coordinates -> (N, 32) sinusoidal embedding normalised by the window shape."""
    out = []
    for c, s in zip(coords.T, shape):
        v = c / max(s, 1)
        ang = v[:, None] * (2 ** np.arange(POS_EMBED_DIM // 2)) * np.pi
        out += [np.sin(ang), np.cos(ang)]
    return np.concatenate(out, axis=1).astype(np.float32)


def load_edge_transformer(weights: Path, device):
    """Returns (model, window size, grid downsampling (z, y, x)); config.json sits next to the weights."""
    cfg = json.loads((Path(weights).parent / 'config.json').read_text())
    model = EdgeTransformer(cfg['unet_out_channels'], cfg['unet_layers'])
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval(), int(cfg['window_size']), tuple(cfg['downsample'])
