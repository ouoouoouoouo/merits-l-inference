#!/usr/bin/env bash
# download_checkpoints.sh — pull the 5 fine-tuned checkpoints from HuggingFace Hub.
#
# Requires: huggingface_hub (bundled with `transformers`) and a HF login.
#     huggingface-cli login
#
# The Llama-3.1-8B base model is NOT downloaded here — transformers pulls it
# on first use from meta-llama/Meta-Llama-3.1-8B, which requires you to have
# accepted the Llama 3.1 Community License on your HF account.

set -euo pipefail

HF_REPO="${HF_REPO:-ouoouoouoouo/merits-l-llama-care}"   # placeholder — update after upload
CKPT_DIR="${1:-checkpoints}"

mkdir -p "$CKPT_DIR"

echo "Pulling $HF_REPO → $CKPT_DIR/"

python - <<EOF
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$HF_REPO",
    local_dir="$CKPT_DIR",
    local_dir_use_symlinks=False,   # copy files, don't symlink
    allow_patterns=[
        # ---------- Mode 1: reproducibility (default, ~290 MB) ----------
        "care_downstream_features_cache.pt",   # 7.5 MB — 256-d hidden per IEMOCAP utt
        "audio_stage2.pt",                     # 3.3 MB — Audio Stage II Bi-GRU
        "llama_lora/adapter_config.json",
        "llama_lora/adapter_model.safetensors",# 27 MB  — LoRA r=16
        "llama_lora/head.pt",                  # 66 KB — Text Stage I head
        "text_stage2.pt",                      # 257 MB — Text Stage II Bi-GRU
        "stage3_fusion.pt",                    # 5.3 MB — Stage III fusion
        "README.md",
        # ---------- Mode 2: new-audio inference (add --with-care-wavlm) ----------
        # Add "care_wavlm/best.pth" (~ 974 MB) below to enable feature extraction
        # from arbitrary audio. See USAGE.md § Mode 2.
    ],
)
print("Download complete.")
EOF

echo ""
echo "===================================================="
echo "  Downloaded files"
echo "===================================================="
find "$CKPT_DIR" -type f | sort | while read -r f; do
    printf "  %s  (%s)\n" "$f" "$(du -h "$f" | cut -f1)"
done

echo ""
echo "Total size: $(du -sh "$CKPT_DIR" | cut -f1)"
echo ""
echo "Done. Try it now:  python inference_example.py --audio samples/happy.wav"
