#!/usr/bin/env bash
set -euo pipefail

python -m pytest \
  test/swe_paddle/test_pr78048_p2p.py::test_p2p_hsplit_original_parameters \
  test/swe_paddle/test_pr78048_p2p.py::test_p2p_dsplit_original_parameters \
  test/swe_paddle/test_pr78048_p2p.py::test_p2p_vsplit_original_parameters \
  test/legacy_test/test_api_compatibility.py::TestHsplitAPI::test_dygraph_Compatibility \
  test/legacy_test/test_api_compatibility.py::TestDsplitAPI::test_dygraph_Compatibility \
  test/legacy_test/test_api_compatibility.py::TestVsplitAPI::test_dygraph_Compatibility \
  -q
