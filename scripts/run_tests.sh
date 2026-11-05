#!/bin/bash
set -euo pipefail

cmake --build build
ctest --test-dir build --output-on-failure