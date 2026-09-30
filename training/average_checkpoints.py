#!/usr/bin/env python
"""Average epoch checkpoints into the deployed 3D Net weights.

    python training/average_checkpoints.py --out models/net3d_128.pt runs/net3d_128/ep7.pt runs/net3d_128/ep8.pt runs/net3d_128/ep9.pt runs/net3d_128/ep10.pt
    python training/average_checkpoints.py --out models/net3d_64.pt --fp16-scale-exp 7 runs/net3d_64/ep50.pt

--fp16-scale-exp k divides the weights of the first convolution of the last-but-two decoder block by 2^k. That block is
followed by GroupNorm, so the network output is unchanged, but its activations stay inside the fp16 range on GPUs
without bf16 (T4). The 64 model needed k = 7.
"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--fp16-scale-exp', type=int, default=0)
ap.add_argument('checkpoints', nargs='+')
a = ap.parse_args()
cks = [torch.load(p, map_location='cpu', weights_only=False) for p in a.checkpoints]
sds = [c['model'] for c in cks]
avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) if sds[0][k].is_floating_point() else sds[0][k]
       for k in sds[0]}
if a.fp16_scale_exp:
    key = 'backbone.decs.0.0.weight'
    avg[key] = avg[key] / float(2 ** a.fp16_scale_exp)
config = dict(cks[0]['config'])
torch.save({'model': avg, 'config': config}, a.out)
print(f'{a.out}: average of epochs {[c.get("epoch") for c in cks]}' + (f', fp16 scale 2^-{a.fp16_scale_exp}' if a.fp16_scale_exp else ''))
