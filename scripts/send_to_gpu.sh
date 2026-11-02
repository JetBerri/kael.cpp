#!/bin/bash
set -euo pipefail

usage() {
    echo "usage: $0 user@host [--arch native] [--without-weights] [--no-run]" >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
host="$1"
shift

architecture="native"
weights=true
run=true
while [[ $# -gt 0 ]]; do
    case "$1" in
    --arch)
        [[ $# -ge 2 ]] || usage
        architecture="$2"
        shift
        ;;
    --without-weights) weights=false ;;
    --no-run) run=false ;;
    *) usage ;;
    esac
    shift
done

cd "$(git rev-parse --show-toplevel)"
commit="$(git rev-parse --short HEAD)"
remote_directory="kael.cpp-$commit"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

if ! git diff --quiet HEAD; then
    echo "note: uncommitted changes are not sent, only commit $commit" >&2
fi

git archive --format=tar.gz --output="$staging/source.tar.gz" HEAD
data_directories=()
for directory in data/mnist data/shakespeare; do
    [[ -d "$directory" ]] && data_directories+=("$directory")
done
if [[ "$weights" == true && -d data/gpt2 ]]; then
    data_directories+=(data/gpt2)
fi
if [[ ${#data_directories[@]} -gt 0 ]]; then
    COPYFILE_DISABLE=1 tar --no-xattrs -czf "$staging/data.tar.gz" "${data_directories[@]}"
else
    tar -czf "$staging/data.tar.gz" --files-from /dev/null
fi

ssh "$host" "mkdir -p $remote_directory"
scp "$staging/source.tar.gz" "$staging/data.tar.gz" "$host:$remote_directory/"
ssh "$host" "cd $remote_directory && tar -xzf source.tar.gz && tar -xzf data.tar.gz && rm source.tar.gz data.tar.gz"
echo "sent commit $commit to $host:$remote_directory"

if [[ "$run" == false ]]; then
    exit 0
fi

status=0
ssh -o ServerAliveInterval=30 "$host" \
    "cd $remote_directory && bash scripts/test_cuda.sh $(printf '%q' "$architecture")" || status=$?
if scp -r "$host:$remote_directory/cuda-report" "./cuda-report-$commit"; then
    echo "report copied to cuda-report-$commit"
else
    echo "could not copy the report from $host:$remote_directory/cuda-report" >&2
fi
exit "$status"
