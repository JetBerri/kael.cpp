#!/bin/bash
set -euo pipefail

directory="${1:-data/shakespeare}"
url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
expected="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"

mkdir -p "$directory"
curl --fail --location --output "$directory/input.txt" "$url"
echo "$expected  $directory/input.txt" | shasum -a 256 --check
