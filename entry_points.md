# Entry points

All commands run from the repository root. The paths they read and write are listed in `SETTINGS.json`, the folder
layout in `directory_structure.txt`, hardware and run time in `README.md`.

## 1. Environment

```bash
pip install -r requirements.txt
```

## 2. Data

```bash
# competition data -> data/train (199 training movies), data/test
kaggle competitions download -c biohub-cell-tracking-during-development -p data
unzip -q data/biohub-cell-tracking-during-development.zip -d data

# our pseudo-labels (used by the pseudo-label 3D Net; PSEUDO=regenerate in step 3 rebuilds them instead)
kaggle datasets download songqizhou/biohub-4th-place-pseudo-labels -p data/pseudo_labels --unzip
```

The 102 external windows (`data/ultrack_windows`) are downloaded by step 3 itself (`external_data/README.md`).

## 3. Train all models

```bash
GPUS="0 1 2 3" bash training/train_all.sh      # -> models/ (about 15 h on 4 x H100)
cp -r models/. inference/models/
```

Each step can also be run on its own (`training/README.md`): `training/run_net3d.sh`, `training/edge_transformer/run.sh`,
`training/run_link.sh`, `training/run_division.sh`, then the two fork verifier commands at the end of
`training/train_all.sh`.

## 4. Predict

```bash
python inference/run_pipeline.py --test data/test --work work/run      # -> work/run/submission.csv
```

To predict with our released weights instead of retrained ones, skip step 3 and put the `models/` folder of the Kaggle
dataset `songqizhou/biohub-4th-place-artifacts` into `inference/models/`:

```bash
kaggle datasets download songqizhou/biohub-4th-place-artifacts -p artifacts --unzip
cp -r artifacts/models/. inference/models/
```

With these weights the pipeline reproduces our final submission. On Kaggle, the notebook `inference/kaggle_notebook.py`
runs the same code (published as `songqizhou/biohub-4th-place-solution`).
