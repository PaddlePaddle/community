#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_CONFIG="${BUILD_CONFIG:-Release}"
TEST_REGEX="${TEST_REGEX:-^inference_api_(path_p2p|utf8_path_f2p)_test$}"

if [[ -d "${BUILD_DIR}" ]]; then
  while IFS= read -r dll_dir; do
    case ":${PATH}:" in *":${dll_dir}:"*) ;; *) export PATH="${dll_dir}:${PATH}" ;; esac
  done < <(find "${BUILD_DIR}" -type f -iname '*.dll' -exec dirname {} \; 2>/dev/null | sort -u)
fi

ctest --test-dir "${BUILD_DIR}" -C "${BUILD_CONFIG}" -R "${TEST_REGEX}" --output-on-failure
