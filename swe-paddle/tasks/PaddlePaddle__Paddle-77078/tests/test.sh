#!/usr/bin/env bash
set -euo pipefail

# Target test for PaddlePaddle__Paddle-77078.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
python -m pytest test/legacy_test/test_inverse_op.py -q
