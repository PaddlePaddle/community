#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_dataloader_persistent_workers_structure.py -q
