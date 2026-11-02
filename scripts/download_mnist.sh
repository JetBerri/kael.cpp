#!/usr/bin/env bash
set -euo pipefail

mirror="https://ossci-datasets.s3.amazonaws.com/mnist"
target="${1:-data/mnist}"
mkdir -p "$target"

checksum_of() {
    if command -v md5sum >/dev/null; then
        md5sum "$1" | cut -d' ' -f1
    else
        md5 -q "$1"
    fi
}

while read -r name expected; do
    archive="$target/$name.gz"
    if [[ ! -f "$target/$name" ]]; then
        curl --fail --silent --show-error --location --output "$archive" "$mirror/$name.gz"
        actual="$(checksum_of "$archive")"
        if [[ "$actual" != "$expected" ]]; then
            rm -f "$archive"
            echo "checksum mismatch for $name: got $actual, expected $expected" >&2
            exit 1
        fi
        gunzip --force "$archive"
    fi
    echo "$name ready"
done <<'FILES'
train-images-idx3-ubyte f68b3c2dcbeaaa9fbdd348bbdeb94873
train-labels-idx1-ubyte d53e105ee54ea40749a09fcbcd1e9432
t10k-images-idx3-ubyte 9fb629c4189551a2d022fa330f9573f3
t10k-labels-idx1-ubyte ec29112dd5afa0611ce80d1b7f02629c
FILES
