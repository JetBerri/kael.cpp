#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage "$0" data/mnist "Download the MNIST training and test sets in IDX format."
    exit 0
fi

directory="${1:-data/mnist}"
mirror="https://ossci-datasets.s3.amazonaws.com/mnist"

while read -r name checksum; do
    fetch_gzip "$mirror/$name.gz" "$directory/$name" "$checksum"
done <<'FILES'
train-images-idx3-ubyte ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db
train-labels-idx1-ubyte 65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5
t10k-images-idx3-ubyte 0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7
t10k-labels-idx1-ubyte ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2
FILES
