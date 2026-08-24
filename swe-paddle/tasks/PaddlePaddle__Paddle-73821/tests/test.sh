#!/bin/bash
set -e

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_pad_op.py::TestPadOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_pad_op.py::TestPadOp_ZeroSize2 -q
python -m pytest test/legacy_test/test_pad3d_op.py::TestPad3dOp_ZeroSize_Circular -q
python -m pytest test/legacy_test/test_pad3d_op.py::TestPad3dOp_ZeroSize_Replicate -q
