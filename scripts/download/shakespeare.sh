#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage "$0" data/shakespeare "Download Tiny Shakespeare, the corpus of the character-level GPT example."
    exit 0
fi

directory="${1:-data/shakespeare}"

fetch "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
    "$directory/input.txt" 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
