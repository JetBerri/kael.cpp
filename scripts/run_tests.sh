#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "usage: $0 [BUILD_DIR]"
    echo
    echo "Configure a Debug build if BUILD_DIR has none, build it and run the C++ test suite."
    echo "BUILD_DIR defaults to build, relative to the repository root."
    exit 0
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
build="${1:-build}"

if [[ ! -f "$build/CMakeCache.txt" ]]; then
    cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Debug
fi
cmake --build "$build" --parallel
ctest --test-dir "$build" --output-on-failure --parallel "$(getconf _NPROCESSORS_ONLN)"
