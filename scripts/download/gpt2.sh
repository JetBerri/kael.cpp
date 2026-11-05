#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage "$0" data/gpt2 "Download the GPT-2 small weights and BPE tokenizer files from Hugging Face."
    exit 0
fi

directory="${1:-data/gpt2}"
base="https://huggingface.co/gpt2/resolve/main"

fetch "$base/model.safetensors" "$directory/model.safetensors" \
    248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707
fetch "$base/vocab.json" "$directory/vocab.json" \
    196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783
fetch "$base/merges.txt" "$directory/merges.txt" \
    1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5
