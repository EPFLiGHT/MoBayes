#!/usr/bin/env bash
# Clone zhiyuanhubj/UoT into baselines/uot/repo/ (gitignored).
# Override target with a positional arg.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-${SCRIPT_DIR}/repo}"
REPO_URL="https://github.com/zhiyuanhubj/UoT.git"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "UoT repository already present at ${TARGET_DIR}"
  exit 0
fi

mkdir -p "$(dirname "${TARGET_DIR}")"
git clone "${REPO_URL}" "${TARGET_DIR}"
echo "Fetched UoT into ${TARGET_DIR}"
