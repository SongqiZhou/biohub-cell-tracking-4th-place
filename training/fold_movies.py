#!/usr/bin/env python
"""Print the held-out movies of a fold as a comma-separated list: python training/fold_movies.py 0"""
import json
import sys
from pathlib import Path

print(','.join(json.loads((Path(__file__).resolve().parent / 'folds.json').read_text())[f'fold{int(sys.argv[1])}']))
