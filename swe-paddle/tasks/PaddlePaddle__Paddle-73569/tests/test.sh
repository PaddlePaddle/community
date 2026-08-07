#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_matmul_v2_op.py::TestMatMulV2Op -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_matmul_v2_op.py::TestMatMulOp_trans_y::test_check_grad -q
