# Usage guide

Step-by-step: install → download → run inference.

## Table of contents

1. [System requirements](#1-system-requirements)
2. [Install this repo](#2-install-this-repo)
3. [Install the three dependency repos](#3-install-the-three-dependency-repos)
4. [Get the Llama-3.1 base model](#4-get-the-llama-31-base-model)
5. [Download the fine-tuned checkpoints](#5-download-the-fine-tuned-checkpoints)
6. [Run inference — CLI](#6-run-inference--cli)
7. [Run inference — Python API](#7-run-inference--python-api)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. System requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| OS | Linux (Ubuntu 22.04) or Windows 11 (WSL2 recommended) | Ubuntu 22.04 |
| GPU VRAM | 12 GB (4-bit Llama) | 24 GB (RTX 4090 / A5000 / A6000) |
| System RAM | 16 GB | 32 GB |
| Disk space | 25 GB (base models + checkpoints) | 40 GB |
| CUDA | 12.1 | 12.4+ |
| Python | 3.10 | 3.11 |

Tested on Ubuntu 22.04 + RTX 4090 + PyTorch 2.4.

## 2. Install this repo

```bash
git clone https://github.com/ouoouoouoouo/merits-l-inference.git
cd merits-l-inference

# Create a fresh virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Install the three dependency repos

The model class definitions live in three sibling GitHub repositories.
Install all three in editable mode so `inference_example.py` can import
`Stage3Fusion`, `TextStage2`, `AudioStage2`, etc.:

```bash
bash scripts/setup_dependencies.sh
```

This clones and `pip install -e` the following into a `deps/` directory:

- [`ouoouoouoouo/care-training`](https://github.com/ouoouoouoouo/care-training)
  — audio encoder (CARE-WavLM) + CARE downstream Stage I
- [`ouoouoouoouo/merits-l-text`](https://github.com/ouoouoouoouo/merits-l-text)
  — Audio Stage II model, feature extraction scripts
- [`ouoouoouoouo/merits-l-llama`](https://github.com/ouoouoouoouo/merits-l-llama)
  — Text Stage I/II, Stage III co-attention fusion

Verify with:

```python
from merits_l_llama.src.models.stage3_fusion import Stage3Fusion   # noqa
from merits_l_llama.src.models.text_stage2 import TextStage2       # noqa
from merits_l_text.src.models.audio_classifier import AudioStage2  # noqa
```

## 4. Get the Llama-3.1 base model

Llama-3.1-8B is not bundled — Transformers pulls it on first run from
HuggingFace. You must:

1. **Accept the [Llama-3.1 license](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)**
   on HuggingFace. Meta usually approves within a few hours.
2. **Log in** on your machine:

   ```bash
   huggingface-cli login
   # paste your HF read token
   ```

Once accepted, the base model is downloaded automatically on first inference
(~16 GB, cached in `~/.cache/huggingface/hub/`).

## 5. Download the fine-tuned checkpoints

The 6 fine-tuned checkpoints (~1.3 GB) are hosted at:

> https://huggingface.co/ouoouoouo/merits-l-llama-care

```bash
bash scripts/download_checkpoints.sh
```

This populates `checkpoints/`:

```
checkpoints/
├── care_wavlm/
│   └── best.pth                          974 MB — CARE-WavLM (200K-step, paper-faithful)
├── care_downstream_stage1/
│   └── model.pt                          ~ 8 MB — CARE downstream Stage I head
├── audio_stage2.pt                       3.3 MB — Audio Stage II Bi-GRU
├── llama_lora/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors         27 MB  — LoRA r=16, α=32
│   └── head.pt                           66 KB  — Text Stage I linear head
├── text_stage2.pt                        257 MB — Text Stage II Bi-GRU
└── stage3_fusion.pt                      5.3 MB — Stage III co-attention head
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for what each checkpoint contains.

## 6. Run inference — CLI

```bash
# Whisper auto-transcribes
python inference_example.py --audio samples/happy.wav

# Faster + more accurate — supply your own transcript
python inference_example.py --audio samples/happy.wav --text "I finally passed the exam!"

# Batch mode
python inference_example.py --audio-dir samples/ --output-json predictions.json
```

Example output:

```json
{
  "audio": "samples/happy.wav",
  "transcript": "I finally passed the exam!",
  "prediction": "happy",
  "confidence": 0.87,
  "probabilities": {
    "angry":   0.02,
    "happy":   0.87,
    "sad":     0.03,
    "neutral": 0.08
  }
}
```

## 7. Run inference — Python API

```python
from inference_example import MERITSLInference

model = MERITSLInference.from_pretrained(
    checkpoint_dir="checkpoints/",
    device="cuda",
    load_in_4bit=True,          # set False if you have >24 GB VRAM
    whisper_model="large-v3",   # "medium" saves ~1.5 GB VRAM
)

# Single utterance
result = model.predict_audio(audio_path="samples/happy.wav")
print(result["prediction"], result["confidence"])

# With provided transcript (skips Whisper)
result = model.predict_audio(
    audio_path="samples/happy.wav",
    text="I finally passed the exam!",
)
```

## 8. Troubleshooting

<details>
<summary><b>CUDA out of memory</b></summary>

- Make sure `load_in_4bit=True` (default). This brings Llama-3.1-8B from
  ~16 GB down to ~5 GB.
- Try Whisper `medium` instead of `large-v3` (saves ~1.5 GB).
- Reduce batch size to 1.
- Close other GPU processes: `nvidia-smi`.
</details>

<details>
<summary><b>ImportError: cannot import name 'Stage3Fusion' from 'merits_l_llama'</b></summary>

You skipped step 3. Run `bash scripts/setup_dependencies.sh`.
</details>

<details>
<summary><b>OSError: You are trying to access a gated repo (meta-llama/Meta-Llama-3.1-8B)</b></summary>

You need to:
1. Accept the [Llama-3.1 license](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B).
2. Run `huggingface-cli login` with a read token that has the accepted license.
</details>

<details>
<summary><b>Whisper transcription is wrong / garbled</b></summary>

- Whisper large-v3 is generally accurate but can fail on very short (<1 s) or
  very noisy audio.
- Try supplying `--text` manually to skip Whisper.
- Ensure audio is 16 kHz mono.
</details>

<details>
<summary><b>Confidence is always ~0.25 across all classes</b></summary>

The pipeline is not loading a checkpoint correctly. Enable verbose logging:

```bash
python inference_example.py --audio a.wav --verbose
```

Check for `[warning] missing keys` messages during checkpoint loading.
</details>

<details>
<summary><b>My audio is not in English</b></summary>

The model is trained on English (IEMOCAP) only. Whisper can transcribe other
languages, but the emotion classifier has not been evaluated on non-English
speech and results will be unreliable.
</details>

<details>
<summary><b>Why is the accuracy 0.8541 and not 0.8746 like the thesis says?</b></summary>

The thesis reports the author's original training pipeline
(RoBERTa-then-swap-to-Llama), which peaked at 0.8746 on seed 1. That pipeline
had a gap — the CARE downstream Stage I head was trained but its weights were
not saved to disk (only the extracted features were).

To ship a model that can accept new audio (not just IEMOCAP test utterances),
we re-trained the three downstream stages (CARE ds Stage I → Audio Stage II
→ Stage III fusion, 5 seeds) with the seed-fixed pipeline. The re-trained
end-to-end model — which is what's in this repository — scores 0.8541 on the
same test split (best seed). This is 2 pp lower than the thesis number and
1 pp lower than the paper baseline (0.8648), which is a fair reflection of
what a downloaded and used model actually delivers.
</details>

---

If your issue is not listed, open a GitHub issue with:

- OS + Python + CUDA version
- Full traceback
- `pip list | grep -E "torch|transformers|peft|bitsandbytes"` output
