#!/usr/bin/env bash
set -euo pipefail

# Run from the root of a rebuilt Paddle source checkout with both patches applied.
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pytest \
  test/legacy_test/test_api_compatibility.py::TestPixelShuffleAPI_Compatibility \
  -q

"$PYTHON_BIN" -m pytest \
  test/legacy_test/test_pixel_shuffle_op.py \
  -q
