#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="test/legacy_test${PYTHONPATH:+:${PYTHONPATH}}"

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_nn_dice_loss.py::TestDiceLossOpApi -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_nn_dice_loss.py::TestDiceLossOpApi_ZeroSize -q
