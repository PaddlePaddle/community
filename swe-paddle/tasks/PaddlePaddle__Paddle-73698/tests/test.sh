#!/bin/bash
set -e

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_conv2d_transpose_op.py::TestConv2DTransposeOp -q
python -m pytest test/legacy_test/test_functional_conv1d_transpose.py::TestFunctionalConv1DError -q
python -m pytest test/legacy_test/test_functional_conv3d_transpose.py::TestFunctionalConv3DTransposeError -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_conv2d_transpose_op.py::TestFunctionalConv2DTranspose_ZeroSize -q
python -m pytest test/legacy_test/test_functional_conv1d_transpose.py::TestFunctionalConv1DTranspose_ZeroSize -q
python -m pytest test/legacy_test/test_functional_conv3d_transpose.py::TestFunctionalConv3DTranspose_ZeroSize -q
