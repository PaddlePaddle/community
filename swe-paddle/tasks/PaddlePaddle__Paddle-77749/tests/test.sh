#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# P2P (pass-to-pass): existing RNN utility regression guard. The gold patch
# does not modify test/legacy_test/test_rnn_cell_api.py, so this node must pass
# both before and after the solution is applied.
(
    cd test/legacy_test
    "${PYTHON_BIN}" -m pytest -q \
        test_rnn_cell_api.py::TestRnnUtil::test_case
)

# F2P (fail-to-pass): all cases in the newly added API test file.
"${PYTHON_BIN}" test/legacy_test/test_rnn_utils.py
