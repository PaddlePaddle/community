#!/usr/bin/env bash

set -euo pipefail
python -m pytest python/paddle/fluid/tests/unittests/test_dataloader_autotune.py -q
