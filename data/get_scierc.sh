#!/usr/bin/env bash
# Download and extract the processed SciERC dataset (Luan et al., 2018).
# Source: http://nlp.cs.washington.edu/sciIE/
# Usage (from anywhere in the repo): ./data/get_scierc.sh

set -euo pipefail

URL="http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"
ARCHIVE="sciERC_processed.tar.gz"

# Resolve paths relative to this script, so it works from any directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/raw/scierc"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

# Download (skip if already present); use wget, or curl as a fallback
if [ -f "${ARCHIVE}" ]; then
    echo "Archive already downloaded, skipping."
elif command -v wget > /dev/null; then
    wget "${URL}" -O "${ARCHIVE}"
elif command -v curl > /dev/null; then
    curl -L "${URL}" -o "${ARCHIVE}"
else
    echo "Error: neither wget nor curl is installed." >&2
    exit 1
fi

# Extract
tar -xzf "${ARCHIVE}"

# Verify the expected split files exist and report where they are
echo "SciERC extracted to ${TARGET_DIR}"
missing=0
for split in train dev test; do
    path="$(find . -name "${split}.json" -print -quit)"
    if [ -n "${path}" ]; then
        echo "  ${split}: ${TARGET_DIR}/${path#./} ($(wc -l < "${path}") documents)"
    else
        echo "  ${split}: NOT FOUND" >&2
        missing=1
    fi
done

if [ "${missing}" -ne 0 ]; then
    echo "Error: some split files are missing. Check the archive contents." >&2
    exit 1
fi