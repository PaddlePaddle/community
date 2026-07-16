#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q \
  test/legacy_test/test_check_memory_usage.py::TestCheckMemoryUsage::test_existing_memory_logging \
  test/legacy_test/test_check_memory_usage.py::TestCheckMemoryUsage::test_unsupported_cpu_allocator_api
