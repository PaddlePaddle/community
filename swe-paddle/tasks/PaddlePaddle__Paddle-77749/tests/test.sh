#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" test/legacy_test/test_rnn_utils.py
