#!/usr/bin/env bash
# enable_mode2.sh — one-shot setup for new-audio inference (Mode 2).
#
# This script:
#   1. Downloads the CARE-WavLM base checkpoint (~974 MB)
#   2. Extracts 13-layer + pooled features on IEMOCAP train/val/test
#   3. Retrains CARE downstream Stage I head (~5 min)  → stage1 model + 256-d hidden cache
#   4. Retrains Audio Stage II Bi-GRU on the new 256-d hidden features (~10 min)
#   5. Retrains Stage III fusion head (5 seeds × ~5 min = 25 min)
#
# Total wall-clock: ~ 30 min on RTX 4090 or similar.
#
# Prerequisites:
#   - bash scripts/setup_dependencies.sh    (3 code repos installed)
#   - bash scripts/download_checkpoints.sh  (Mode 1 checkpoints present)
#   - Llama-3.1-8B license accepted on HuggingFace + huggingface-cli login
#
# Result: checkpoints/mode2/ populated with:
#   care_wavlm/best.pth                 (from HF Hub)
#   care_downstream_stage1/model.pt     (newly trained)
#   audio_stage2_mode2.pt               (newly trained)
#   stage3_fusion_mode2/seed{1..5}.pt   (newly trained)

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-checkpoints}"
MODE2_DIR="$CKPT_DIR/mode2"
DEPS_DIR="${DEPS_DIR:-deps}"
mkdir -p "$MODE2_DIR/care_wavlm" "$MODE2_DIR/care_downstream_stage1" "$MODE2_DIR/stage3_fusion_mode2"

# -----------------------------------------------------------------------------
# 1. Download CARE-WavLM base
# -----------------------------------------------------------------------------
if [ ! -f "$MODE2_DIR/care_wavlm/best.pth" ]; then
    echo "===================================================="
    echo " Step 1/5: downloading CARE-WavLM base (~974 MB) …"
    echo "===================================================="
    python - <<EOF
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="ouoouoouoouo/merits-l-llama-care",
    filename="care_wavlm/best.pth",
    local_dir="$MODE2_DIR",
    local_dir_use_symlinks=False,
)
print(f"Downloaded to {p}")
EOF
fi

# -----------------------------------------------------------------------------
# 2. Extract 13-layer + pooled features on IEMOCAP (~3 min)
# -----------------------------------------------------------------------------
FEATURES_PT="$MODE2_DIR/care_features_iemocap.pt"
if [ ! -f "$FEATURES_PT" ]; then
    echo ""
    echo "===================================================="
    echo " Step 2/5: extracting CARE features on IEMOCAP …"
    echo "===================================================="
    python -m care_training.scripts.extract_iemocap_care_downstream_style \
        --care-ckpt   "$MODE2_DIR/care_wavlm/best.pth" \
        --manifest-dir "$DEPS_DIR/merits-l-text/data/manifests/iemocap" \
        --output-pt    "$FEATURES_PT"
fi

# -----------------------------------------------------------------------------
# 3. Retrain CARE downstream Stage I head (~5 min)
# -----------------------------------------------------------------------------
HEAD_PT="$MODE2_DIR/care_downstream_stage1/model.pt"
HIDDEN_PT="$MODE2_DIR/care_downstream_stage1/hidden.pt"
echo ""
echo "===================================================="
echo " Step 3/5: retraining CARE downstream Stage I head …"
echo "===================================================="
python -m care_training.care-training.scripts.train_care_downstream_iemocap \
    --features-pt   "$FEATURES_PT" \
    --manifest-dir  "$DEPS_DIR/merits-l-text/data/manifests/iemocap" \
    --seed          42 \
    --save-model-pt "$HEAD_PT" \
    --save-hidden-pt "$HIDDEN_PT"

# -----------------------------------------------------------------------------
# 4. Retrain Audio Stage II (~10 min)
# -----------------------------------------------------------------------------
AUDIO_STAGE2_PT="$MODE2_DIR/audio_stage2_mode2.pt"
echo ""
echo "===================================================="
echo " Step 4/5: retraining Audio Stage II Bi-GRU …"
echo "===================================================="
python -m merits_l_text.src.train_stage2 \
    --config "$DEPS_DIR/merits-l-text/configs/iemocap_audio_care_downstream_stage2.yaml" \
    --features-pt "$HIDDEN_PT" \
    --output-pt   "$AUDIO_STAGE2_PT"

# -----------------------------------------------------------------------------
# 5. Retrain Stage III fusion (5 seeds × ~5 min)
# -----------------------------------------------------------------------------
echo ""
echo "===================================================="
echo " Step 5/5: retraining Stage III fusion (5 seeds) …"
echo "===================================================="
for s in 1 2 3 4 5; do
    python -m merits_l_llama.src.train_stage3 \
        --config "$DEPS_DIR/merits-l-llama/configs/iemocap_stage3_llama.yaml" \
        --audio-features-pt "$AUDIO_STAGE2_PT" \
        --seed $s \
        --output-pt "$MODE2_DIR/stage3_fusion_mode2/seed$s.pt"
done

echo ""
echo "===================================================="
echo " Mode 2 setup complete!"
echo "===================================================="
echo ""
echo "You can now run inference on arbitrary audio:"
echo "  python inference_example.py --audio your_file.wav --mode 2"
echo ""
echo "Reported accuracy for this retrained pipeline is stored in"
echo "  $MODE2_DIR/stage3_fusion_mode2/  (per-seed test_report.txt)"
