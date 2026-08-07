#!/usr/bin/env bash
set -euo pipefail

# Target test for PaddlePaddle__Paddle-77078.
# Run the file directly so its __main__ block enables Paddle static mode.
python test/legacy_test/test_inverse_op.py
