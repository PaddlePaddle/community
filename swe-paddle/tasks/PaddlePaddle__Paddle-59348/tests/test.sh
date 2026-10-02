#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-59348.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
#
# CTest runs this module through tools/test_runner.py, which calls
# paddle.enable_static() first because test_sequence_mask is in
# static_mode_white_list. pytest bypasses that runner, so static mode must be
# enabled explicitly before collection (TestSequenceMaskOpError needs it).
export PYTHONPATH="test/legacy_test:test${PYTHONPATH:+:${PYTHONPATH}}"

FLAGS_enable_pir_in_executor=true python -c '
import sys

import paddle
import pytest

paddle.enable_static()
sys.exit(pytest.main(["test/sequence/test_sequence_mask.py", "-q"]))
'
