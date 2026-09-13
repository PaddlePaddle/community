#!/usr/bin/env bash

set -euo pipefail
python -m pytest   test/legacy_test/test_launch_coverage.py::TestCoverage::test_find_free_ports   test/legacy_test/test_launch_main_kill.py   -q
