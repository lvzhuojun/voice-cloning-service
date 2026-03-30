#!/usr/bin/env bash
# Clone GPT-SoVITS into ./GPT-SoVITS/
# Usage: bash setup/clone_gptsovits.sh

set -e

REPO_URL="https://github.com/RVC-Boss/GPT-SoVITS.git"
TARGET_DIR="GPT-SoVITS"

# Run from project root regardless of where the script is called from
cd "$(dirname "$0")/.."

echo ""
echo "============================================================"
echo " Clone GPT-SoVITS"
echo " Target: ./${TARGET_DIR}/"
echo "============================================================"
echo ""

if [ -d "${TARGET_DIR}/.git" ]; then
    echo "[OK] GPT-SoVITS already cloned. Pulling latest changes..."
    git -C "${TARGET_DIR}" pull origin main
    echo "[OK] Done."
else
    echo "[INFO] Cloning from ${REPO_URL} ..."
    git clone "${REPO_URL}" "${TARGET_DIR}"
    echo ""
    echo "[OK] GPT-SoVITS cloned successfully to ./${TARGET_DIR}/"
fi

echo ""
echo "============================================================"
echo " Next step: download pretrained models"
echo "   conda run -n voice-cloning python setup/download_models.py"
echo "============================================================"
echo ""
