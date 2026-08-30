#!/bin/bash
set -e

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_functional_conv1d.py::TestFunctionalConv1DError -q
python -m pytest test/legacy_test/test_functional_conv2d.py::TestFunctionalConv2DError -q
python -m pytest test/legacy_test/test_functional_conv3d.py::TestFunctionalConv3DError -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_functional_conv1d.py::TestFunctionalConv1D_ZeroSize -q
python -m pytest test/legacy_test/test_functional_conv2d.py::TestFunctionalConv2D_ZeroSize -q
python -m pytest test/legacy_test/test_functional_conv3d.py::TestFunctionalConv3D_ZeroSize2 -q
