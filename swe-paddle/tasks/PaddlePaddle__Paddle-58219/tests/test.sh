#!/usr/bin/env bash
set -euo pipefail

# Run from the root of a PaddlePaddle/Paddle source checkout.
python test/legacy_test/test_math_op_patch_pir.py -v
