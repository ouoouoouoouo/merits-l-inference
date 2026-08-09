#!/usr/bin/env bash
# upload_to_hf_hub.sh — (AUTHOR ONLY) upload local checkpoints to HuggingFace Hub.
#
# Prerequisites on the author's machine:
#   1. huggingface-cli login  (with a write token)
#   2. Create the target repo once:
#          huggingface-cli repo create merits-l-llama-care --type model
#   3. Assemble the local checkpoint tree under $CKPT_DIR:
#          checkpoints/
#          ├── care_wavlm/best.pth                (from care_training/ckpts/)
#          ├── care_downstream_stage1/*.pt        (see MODEL_CARD.md)
#          ├── audio_stage2.pt                    (from merits-l-text/outputs/iemocap_audio_care_ds_stage2/best/stage2.pt)
#          ├── llama_lora/                        (from merits-l-llama/outputs/iemocap_text_llama_stage1/best/)
#          ├── text_stage2.pt                     (from merits-l-llama/outputs/iemocap_text_llama_stage2/best/stage2.pt)
#          └── stage3_fusion.pt                   (from merits-l-llama/outputs/iemocap_stage3_llama_seed1/best/stage3.pt)
#
# Usage: bash scripts/upload_to_hf_hub.sh
#   or:  HF_REPO=your-user/your-repo bash scripts/upload_to_hf_hub.sh

set -euo pipefail

HF_REPO="${HF_REPO:-ouoouoouoouo/merits-l-llama-care}"
CKPT_DIR="${1:-checkpoints}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: $CKPT_DIR does not exist. See header of this script for expected layout."
    exit 1
fi

echo "About to upload contents of '$CKPT_DIR' to https://huggingface.co/$HF_REPO"
echo ""
find "$CKPT_DIR" -type f | sort | while read -r f; do
    printf "  %s  (%s)\n" "$f" "$(du -h "$f" | cut -f1)"
done
echo ""
echo "Total: $(du -sh "$CKPT_DIR" | cut -f1)"
echo ""
read -r -p "Proceed? [y/N] " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

huggingface-cli upload "$HF_REPO" "$CKPT_DIR" . \
    --repo-type=model \
    --commit-message="Upload MERITS-L (Llama-3.1-8B + CARE) checkpoints"

# Also upload the model card (README) if present
if [ -f "MODEL_CARD.md" ]; then
    cp MODEL_CARD.md "$CKPT_DIR/README.md"
    huggingface-cli upload "$HF_REPO" "$CKPT_DIR/README.md" README.md \
        --repo-type=model \
        --commit-message="Update model card"
    rm "$CKPT_DIR/README.md"
fi

echo ""
echo "Upload complete."
echo "→ https://huggingface.co/$HF_REPO"
