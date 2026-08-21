#!/usr/bin/env bash
set -euo pipefail

python -m pytest \
  test/legacy_test/test_api_compatibility.py::TestHsplitAPI::test_dygraph_Compatibility \
  test/legacy_test/test_api_compatibility.py::TestDsplitAPI::test_dygraph_Compatibility \
  test/legacy_test/test_api_compatibility.py::TestVsplitAPI::test_dygraph_Compatibility \
  -q
