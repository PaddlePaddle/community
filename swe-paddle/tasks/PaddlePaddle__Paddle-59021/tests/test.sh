#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-59021.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
python -m pytest \
  test/dygraph_to_static/test_len.py \
  -q

FLAGS_enable_pir_in_executor=true python -m pytest \
  test/legacy_test/test_fuse_elewise_add_act_pass.py \
  -q
