#!/usr/bin/env python
"""Run the full inference pipeline outside Kaggle.

    python inference/run_pipeline.py --test <dir with *.zarr movies> --work <output dir>

The model weights must be in inference/models/ (the `models/` folder of the Kaggle dataset
biohub-4th-place-artifacts), and the packages in requirements.txt must be installed. The result is
<work>/submission.csv; with --keep the intermediate files (nodes, link probabilities, division scores, ...) stay in <work>.
"""
import argparse
import os
import runpy
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--test', required=True, help='directory with the *.zarr movies')
ap.add_argument('--work', required=True, help='output directory')
ap.add_argument('--keep', action='store_true', help='keep intermediate files')
a = ap.parse_args()
here = Path(__file__).resolve().parent
assert (here / 'models' / 'net3d_128.pt').exists(), 'put the model weights into inference/models/ first'
Path(a.work).mkdir(parents=True, exist_ok=True)
os.environ.update(BIOHUB_CODE=str(here), BIOHUB_TEST=str(Path(a.test).resolve()), BIOHUB_WORK=str(Path(a.work).resolve()),
                  BIOHUB_SKIP_PIP='1', BIOHUB_KEEP='1' if a.keep else '0')
runpy.run_path(str(here / 'kaggle_notebook.py'), run_name='__main__')
