#!/bin/bash
set -euo pipefail

directory="${1:-data/gpt2}"
base="https://huggingface.co/gpt2/resolve/main"
expected="248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"

mkdir -p "$directory"
curl --fail --location --continue-at - --output "$directory/model.safetensors" "$base/model.safetensors"
curl --fail --location --output "$directory/config.json" "$base/config.json"
curl --fail --location --output "$directory/vocab.json" "$base/vocab.json"
curl --fail --location --output "$directory/merges.txt" "$base/merges.txt"
echo "$expected  $directory/model.safetensors" | shasum -a 256 --check
