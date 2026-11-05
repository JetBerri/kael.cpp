# Shared by the download scripts. Source it; do not run it.

sha256_of() {
    if command -v sha256sum >/dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# Downloads URL to DESTINATION unless the file already has the expected SHA-256.
# An interrupted download resumes from DESTINATION.partial.
fetch() {
    local url="$1" destination="$2" expected="$3"
    if [[ -f "$destination" && "$(sha256_of "$destination")" == "$expected" ]]; then
        echo "present  $destination"
        return 0
    fi
    mkdir -p "$(dirname "$destination")"
    curl --fail --silent --show-error --location --retry 3 --continue-at - \
        --output "$destination.partial" "$url"
    verify_and_place "$destination.partial" "$destination" "$expected"
}

# Like fetch, for a gzip file: EXPECTED is the checksum of the decompressed content.
fetch_gzip() {
    local url="$1" destination="$2" expected="$3"
    if [[ -f "$destination" && "$(sha256_of "$destination")" == "$expected" ]]; then
        echo "present  $destination"
        return 0
    fi
    mkdir -p "$(dirname "$destination")"
    curl --fail --silent --show-error --location --retry 3 --output "$destination.partial.gz" "$url"
    gunzip --force "$destination.partial.gz"
    verify_and_place "$destination.partial" "$destination" "$expected"
}

verify_and_place() {
    local partial="$1" destination="$2" expected="$3"
    local actual
    actual="$(sha256_of "$partial")"
    if [[ "$actual" != "$expected" ]]; then
        rm -f "$partial"
        echo "checksum mismatch for $destination: expected $expected, got $actual" >&2
        return 1
    fi
    mv "$partial" "$destination"
    echo "fetched  $destination"
}

print_usage() {
    local script="$1" default="$2" description="$3"
    echo "usage: $script [DIRECTORY]"
    echo
    echo "$description"
    echo "DIRECTORY defaults to $default. Files that already match their checksum are kept."
}
