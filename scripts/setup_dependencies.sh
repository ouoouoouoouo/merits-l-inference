#!/usr/bin/env bash
# setup_dependencies.sh — clone and pip-install the three model repositories.
#
# After running this, you can:
#     from merits_l_llama.src.models.stage3_fusion import Stage3Fusion
#     from merits_l_llama.src.models.text_stage2   import TextStage2
#     from merits_l_text.src.models.audio_classifier import AudioStage2
#     import care_training  # for CARE-WavLM loader
#
# Usage: bash scripts/setup_dependencies.sh [deps_dir]
#   deps_dir defaults to ./deps

set -euo pipefail

DEPS_DIR="${1:-deps}"
mkdir -p "$DEPS_DIR"
cd "$DEPS_DIR"

REPOS=(
    "https://github.com/ouoouoouoouo/care-training.git"
    "https://github.com/ouoouoouoouo/merits-l-text.git"
    "https://github.com/ouoouoouoouo/merits-l-llama.git"
)

for url in "${REPOS[@]}"; do
    name="$(basename "$url" .git)"
    echo ""
    echo "===================================================="
    echo "  $name"
    echo "===================================================="
    if [ -d "$name" ]; then
        echo "  → already cloned, pulling latest…"
        (cd "$name" && git pull)
    else
        git clone --depth 1 "$url"
    fi

    if [ -f "$name/setup.py" ] || [ -f "$name/pyproject.toml" ]; then
        pip install -e "./$name"
    else
        # No packaging metadata — add to PYTHONPATH via a .pth file
        SITE_PACKAGES="$(python -c 'import site, sys; print(site.getsitepackages()[0])')"
        echo "$(realpath "$name")" > "$SITE_PACKAGES/${name}.pth"
        echo "  → added $(realpath "$name") to sys.path via ${name}.pth"
    fi
done

echo ""
echo "===================================================="
echo "  Verifying imports"
echo "===================================================="
python - <<'EOF'
import importlib, sys
targets = [
    "merits_l_llama",
    "merits_l_text",
    "care_training",
]
ok = True
for name in targets:
    try:
        m = importlib.import_module(name)
        print(f"  ✓ {name}  ({m.__file__})")
    except Exception as e:
        print(f"  ✗ {name}  ({e})")
        ok = False
sys.exit(0 if ok else 1)
EOF

echo ""
echo "Done. Next step:  bash scripts/download_checkpoints.sh"
