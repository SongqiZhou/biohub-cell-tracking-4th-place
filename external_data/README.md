# External data

Our final models use **one external embryo**, and only in **one model** (3D Net-128-PL, the pseudo-label detector that
refines the nodes). Everything else — 3D Net-128, 3D Net-64 and every downstream model (edge transformer, cell
embedding, edge model, division models, fork verifier) — is trained on the 199 competition training movies only.

## Source

| | |
|---|---|
| dataset | the Ultrack `zebrafish_embryo` light-sheet volume, publicly available from the Royer Lab (CZ Biohub): `https://public.czbiohub.org/royerlab/ultrack/zebrafish_embryo.ome.zarr` |
| labels | **none** — no tracks or annotations are published for this volume; all labels come from our own detector |
| voxel size | (1.625, 0.40625, 0.40625) µm, identical to the competition, so no resampling is needed |

## How much we used

| item | amount |
|---|---|
| windows | **102** windows of 104 × 104 × 104 µm (64 × 256 × 256 voxels), listed in `ultrack_windows.json` |
| frames per window | 100 (the competition movie length) |
| total frames | **10,200** (the 199 competition movies have 19,900) |
| time spans | 3 start frames (t0 = 50, 200, 350), 34 windows each |
| position | a 104 µm grid in Y/X (origins 104–728 µm); z = 104–208 µm for 100 windows and 0–104 µm for 2 |
| storage | 39 GB in the competition zarr format (about 115 GB downloaded, see below) |
| pseudo-label nodes | **2,023,831** (≈ 198 per frame), from 3D Net-128; released in the Kaggle dataset `songqizhou/biohub-4th-place-pseudo-labels` |

Share in the training set of 3D Net-128-PL:

| | competition | external | share of external |
|---|---|---|---|
| movies / windows | 199 | 102 | 34% |
| frames sampled per epoch (16 per movie) | 3,184 | 1,632 | 34% |
| annotated (ground-truth) nodes | 133,318 | 0 | – |
| pseudo-label nodes | 4,776,356 (out-of-fold 3D Net-128) | 2,023,831 | 30% |

Pseudo-label nodes within 4 µm of a ground-truth node are dropped when the labels are merged (117,956 nodes), so the
model sees ground truth wherever it exists.

## Pipeline

Training uses our pseudo-labels by default: the Kaggle dataset `songqizhou/biohub-4th-place-pseudo-labels` holds them for
the competition movies (`pseudo_oof/`) and for the external windows (`pseudo_ultrack/`); `training/run_net3d.sh` checks
them against `training/pseudo_labels.sha256` and runs step 4 below with them. With `PSEUDO=regenerate` it runs all four
steps with the newly trained 3D Nets instead (see `training/README.md`, section 1, for why this is not the default).

```bash
# 0. our pseudo-labels (61 MB)
kaggle datasets download songqizhou/biohub-4th-place-pseudo-labels -p data/pseudo_labels --unzip

# 1. download the 102 windows in the competition format (~39 GB on disk)
python external_data/extract_ultrack_windows.py --out-dir data/ultrack_windows --workers 6

# 2. detect nuclei with the ground-truth-only 3D Net (the deployed 3D Net-128); inference settings, without the refinement
python inference/detect.py --data-dir data/ultrack_windows --vec inference/models/net3d_128.pt \
    --thr 0.97 --nms-adapt 4,7,10 --out work/ultrack_nodes

# 3. detections -> pseudo-labels (greedy linking, tracks of >= 3 nodes, line-fit smoothing)
python external_data/make_pseudo_labels.py work/ultrack_nodes work/pseudo_ultrack

# 4. merge with the competition movies and their out-of-fold pseudo-labels (ours, from step 0)
python external_data/make_training_mix.py --train data/train --external data/ultrack_windows \
    --pseudo-train data/pseudo_labels/pseudo_oof --pseudo-external data/pseudo_labels/pseudo_ultrack \
    --out data/mix
```

With regenerated pseudo-labels, step 4 takes `work/pseudo_oof` (made by `training/run_net3d.sh`) and `work/pseudo_ultrack`
and writes `data/mix_regenerated`. The competition pseudo-labels (`pseudo_oof/`) are the out-of-fold detections of the
five fold models of 3D Net-128 on the 199 training movies, passed through the same `make_pseudo_labels.py`. The external
windows are labelled once, by the final 3D Net-128; there is no iterative self-training.

Steps 2 and 3 reproduce our pseudo-labels exactly (checked file by file) when run with our 3D Net-128 weights.

## Notes

* **Download cost is per time span, not per window.** Level 0 of the source stores a whole YX plane per chunk
  (`[1, 1, 128, 2217, 2170]`, about 1.2 GB), so reading one window means reading full planes. Restricting the windows
  to three time spans and one z-chunk lets all windows of a span share each read; this is why the list has this
  shape and why the download is about 115 GB rather than several hundred.
* **Remote reads time out occasionally.** The extractor retries each frame block with backoff; a long run needs this.
* **Format.** The windows follow the competition layout exactly (zarr v3, one chunk per frame, zstd + bitshuffle,
  OME multiscales 0.5, the same image-statistics quantile keys), so all our readers work on them unchanged.
* **Other external data.** We also converted Zebrahub embryos that come with Ultrack's automatic tracks, but no model
  in the final pipeline uses them.
