#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-59348.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
FLAGS_enable_pir_in_executor=true python -m pytest \
  test/sequence/test_sequence_mask.py \
  -q
