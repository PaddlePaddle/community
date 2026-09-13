#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/swe_paddle/test_pr27247_dataloader_spawn_pickle.py -q
