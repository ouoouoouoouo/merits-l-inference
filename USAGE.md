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
Install all three so `inference_example.py` can import
`Stage3Fusion`, `TextStage2`, `CAREEmotionClassifier`, etc.:

```bash
bash scripts/setup_dependencies.sh
```

The script clones the following into `deps/`, creates underscored symlinks
(so Python's import machinery — which forbids dashes — can find them),
and adds `deps/` to `sys.path` via a `.pth` file:

- [`ouoouoouoouo/care-training`](https://github.com/ouoouoouoouo/care-training)
  — audio encoder (CARE-WavLM) + CARE downstream Stage I
- [`ouoouoouoouo/merits-l-text`](https://github.com/ouoouoouoouo/merits-l-text)
  — Audio Stage II feature-extraction scripts (RoBERTa baseline pipeline)
- [`ouoouoouoouo/merits-l-llama`](https://github.com/ouoouoouoouo/merits-l-llama)
  — Text Stage I/II (Llama LoRA), Stage III co-attention fusion

Verify with:

```python
from merits_l_llama.src.models.stage3_fusion import Stage3Fusion            # noqa
from merits_l_llama.src.models.text_stage2 import TextStage2                # noqa
from care_training.scripts.train_care_downstream_iemocap import CAREEmotionClassifier  # noqa
from care_training.scripts.extract_iemocap_care_downstream_style import CAREDownstreamExtractor  # noqa
```

Note: `audio_stage2.pt` was trained with the generic **TextStage2** (Bi-GRU +
self-attention) architecture, not the SUPERB-13-layer `AudioClassifier` in
`merits_l_text/src/models/audio_classifier.py`. The loader in
`inference_example.py` handles this correctly — you don't need to worry
about it, but if you `git blame` the code that's why.

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
<summary><b>ImportError: No module named 'merits_l_text' (even after setup)</b></summary>

Python cannot import a directory whose name contains a dash
(`merits-l-text` → attempted `import merits-l-text` is a syntax error). Our
`setup_dependencies.sh` creates underscored symlinks (`merits_l_text` →
`merits-l-text`) inside `deps/` and adds `deps/` to `sys.path` via a `.pth`
file. If it didn't run cleanly, do it manually:

```bash
cd deps
ln -sf merits-l-text merits_l_text
ln -sf merits-l-llama merits_l_llama
ln -sf care-training care_training           # if not already underscored
export PYTHONPATH="$PWD:$PYTHONPATH"
```
</details>

<details>
<summary><b>ModuleNotFoundError: No module named 'model_pase'</b></summary>

Pass `--care-repo` pointing to the directory containing `model_pase.py`
(inside the CARE-training clone):

```bash
python inference_example.py --audio a.wav --care-repo deps/care-training/CARE/pretraining
```
</details>

<details>
<summary><b>PackageNotFoundError: No package metadata was found for bitsandbytes</b></summary>

```bash
pip install bitsandbytes
```
If you don't want 4-bit Llama, use `--no-4bit` (requires ≥16 GB VRAM).
</details>

<details>
<summary><b>ValueError: RNN input dtype (torch.float16) does not match weight dtype (torch.float32)</b></summary>

Bug in an earlier version — the Llama 4-bit hidden output (bf16) needs to be
cast to fp32 before entering the Text Stage II Bi-GRU. This is already fixed
in the current `inference_example.py`; if you see this, `git pull` the repo.
</details>

<details>
<summary><b>AttributeError: 'AudioClassifier' object has no attribute 'encode'</b></summary>

Bug in an earlier version. `audio_stage2.pt` was trained with the generic
`TextStage2` architecture (Bi-GRU + self-attention), NOT `AudioClassifier`
(which is the Stage I SUPERB-13-layer weighted classifier). The current
`inference_example.py` uses `TextStage2` for both text and audio Stage II —
`git pull` if you see this.
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
