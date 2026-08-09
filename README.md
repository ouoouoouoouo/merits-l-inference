# MERITS-L (Llama-3.1-8B + CARE)

**Multimodal speech emotion recognition** on IEMOCAP 4-class (angry / happy / sad / neutral).
This repository releases the trained checkpoints and documentation for the model reported in my thesis, which extends [MERITS-L (Dutta & Ganapathy, ICASSP 2025)](https://arxiv.org/abs/2409.14547) by replacing the RoBERTa text encoder with **Meta Llama-3.1-8B (LoRA)**.

| System                                        | Test wF1 (5 seeds)   | Best     | Δ vs paper |
|-----------------------------------------------|---------------------|----------|-----------|
| MERITS-L paper (RoBERTa + CARE)               | 0.8648 (1 seed)     | —        | baseline  |
| Our original training (Llama LoRA + CARE)     | 0.8567 ± 0.0139     | 0.8746   | +0.98%    |
| **Released Mode 2 model (this checkpoint)**   | **0.8474 ± 0.0110** | **0.8541** | −1.07%  |

> **Note on the two numbers**: our original training pipeline (0.8746 best) had a
> gap — the CARE downstream Stage I head was trained but its weights were not
> saved to disk, only the extracted features. To ship an end-to-end model that
> accepts arbitrary new audio, we re-trained the downstream 3 stages (CARE ds
> Stage I → Audio Stage II → Stage III fusion, 5 seeds) with the seed-fixed
> pipeline. The retrained model in this repo scores 0.8541 (best seed) —
> slightly lower than the original 0.8746, but reproducible from raw wav.

📄 **[MODEL_CARD.md](MODEL_CARD.md)** — model card (intended use, evals, limitations, bias)
🏗️ **[ARCHITECTURE.md](ARCHITECTURE.md)** — full architecture + pipeline diagram
🧑‍💻 **[USAGE.md](USAGE.md)** — step-by-step setup & inference

---

## What this repo ships

**End-to-end audio → emotion pipeline (~1.3 GB total).**
6 checkpoints, all pre-computed on the author's hardware — you don't need to
retrain anything. Predict emotion for arbitrary English audio in one command:

```bash
python inference_example.py --audio your_file.wav
# → Prediction: happy (confidence 0.87)
```

Measured accuracy on IEMOCAP session 5: **wF1 = 0.8541** (best seed of 5).

See [USAGE.md](USAGE.md) for the full setup + inference guide.

---

## Model overview

```
Audio wav (16kHz)
    │
    ▼
┌─────────────────────────────────────┐
│  CARE-WavLM  (pretrained on MSP-Podcast, 200K steps)      │  974 MB
│    ↓ SUPERB-13 per-frame features (1536-d)                │
│  CARE Downstream Stage I head                             │  small
│    ↓ per-frame 256-d                                      │
│  Audio Stage II Bi-GRU + self-attention                   │  3.3 MB
│    ↓ utt hidden (256-d)                                   │
└─────────────────────────────────────┘
                                       ╲
                                        ╲
                                         ▶ Stage III Co-Attention Fusion ─► emotion (4-class)
                                        ╱                                    (5.3 MB)
                                       ╱
┌─────────────────────────────────────┐
│  Text (whisper or user-supplied)                          │
│    ↓ tokens                                               │
│  Meta Llama-3.1-8B (4-bit NF4)  +  LoRA r=16 α=32         │  16 GB + 27 MB
│    ↓ per-token 4096-d                                     │
│  Text Stage I linear head (mean-pool → 4)                 │  66 KB
│  Text Stage II Bi-GRU + self-attention                    │  257 MB
│    ↓ utt hidden (2048-d)                                  │
└─────────────────────────────────────┘
```

## Repository layout

```
merits-l-inference/
├── README.md                       ← you are here
├── MODEL_CARD.md                   ← HuggingFace-style model card
├── ARCHITECTURE.md                 ← per-component arch + shapes
├── USAGE.md                        ← full setup guide
├── LICENSE                         ← Apache 2.0 code / Llama 3.1 CL weights
├── requirements.txt
├── inference_example.py            ← minimal end-to-end example (uses the 3 GitHub repos below)
├── configs/
│   └── best_model.yaml             ← the exact config that produced wF1 = 0.8746
├── scripts/
│   ├── setup_dependencies.sh       ← clones and pip-installs the 3 code repos
│   ├── download_checkpoints.sh     ← pulls 5 checkpoints from HuggingFace Hub
│   └── upload_to_hf_hub.sh         ← (author-only) publish new checkpoints
├── checkpoints/                    ← populated by download_checkpoints.sh (git-ignored)
└── tests/
    └── test_pipeline.py            ← sanity check
```

## Source-code repositories

The **model code** lives in three sibling GitHub repositories:

| Repo | Role | Contents |
|------|------|----------|
| [ouoouoouoouo/care-training](https://github.com/ouoouoouoouo/care-training) | Audio encoder | CARE-WavLM SSL pretraining (fork of iiscleap/care), MSP-Podcast pretraining pipeline |
| [ouoouoouoouo/merits-l-text](https://github.com/ouoouoouoouo/merits-l-text) | Text branch (RoBERTa baseline) + audio Stage II | RoBERTa Text Stage I/II, CARE downstream Stage I/II, Stage III fusion, extraction scripts |
| [ouoouoouoouo/merits-l-llama](https://github.com/ouoouoouoouo/merits-l-llama) | Text branch (Llama extension) | Llama Text Stage I/II (LoRA + Full FT variants), Stage III fusion with Llama features |

`inference_example.py` in this repo imports directly from these three so no re-implementation is duplicated.

## Quick start (60 seconds)

```bash
# 1. Clone this repo and install its own deps
git clone https://github.com/ouoouoouoouo/merits-l-inference.git
cd merits-l-inference
pip install -r requirements.txt

# 2. Install the three model repos (one-shot script)
bash scripts/setup_dependencies.sh

# 3. Download the 5 fine-tuned checkpoints (~1.3 GB from HuggingFace Hub)
#    Requires you to first accept the Llama-3.1 license on HF
huggingface-cli login
bash scripts/download_checkpoints.sh

# 4. Run inference on a sample wav
python inference_example.py --audio samples/happy.wav
# → Prediction: happy (confidence 0.87)
```

Full details in [USAGE.md](USAGE.md).

## Results

Test weighted-F1 on IEMOCAP session-independent split (session 5 as test), 5 seeds
of the retrained end-to-end pipeline (what's in this repo):

| Seed | Weighted F1 |
|:----:|:-----------:|
| 1 ⭐ | **0.8541**  |
| 2    | 0.8279      |
| 3    | 0.8522      |
| 4    | 0.8494      |
| 5    | 0.8532      |
| **Mean ± std** | **0.8474 ± 0.0110** |

The seed-1 Stage III fusion checkpoint is what's released (`stage3_fusion.pt`).

**For reference**, the author's original training run (before the downstream
retraining needed for end-to-end release) achieved 0.8567 ± 0.0139 (best 0.8746);
that number appears in the accompanying thesis. See [MODEL_CARD.md § Evaluation]
for the full explanation.

## Requirements

- Python ≥ 3.10
- CUDA ≥ 12.1
- PyTorch ≥ 2.4
- **≥ 12 GB VRAM** with 4-bit Llama (fits RTX 3060 12 GB, RTX 4090 24 GB, etc.)
- ~24 GB disk for base models + checkpoints

Tested on Ubuntu 22.04 + RTX 4090.

## Citation

If you use this model, please cite:

```bibtex
@mastersthesis{yourthesis2026,
  title  = {Extending MERITS-L with Llama-3.1-8B: A LLM-Augmented Multimodal Speech Emotion Recognition Pipeline},
  author = {Your Name},
  school = {National Taiwan University of Science and Technology},
  year   = {2026}
}

@inproceedings{dutta2025merits,
  title     = {MERITS-L: Multimodal Emotion Recognition via Interactive Speech and Language},
  author    = {Dutta, Soumya and Ganapathy, Sriram},
  booktitle = {ICASSP},
  year      = {2025}
}

@article{grattafiori2024llama,
  title   = {The Llama 3 Herd of Models},
  author  = {Grattafiori, Aaron and others},
  journal = {arXiv preprint arXiv:2407.21783},
  year    = {2024}
}
```

## License & credits

- **Code** — Apache 2.0 (see [LICENSE](LICENSE))
- **Model weights** — [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/)
  (LoRA adapter is a derivative work of Llama-3.1)
- **CARE encoder** — adapted from [iiscleap/care](https://github.com/iiscleap/care) (MIT, per their repo)
- **Whisper** — MIT © OpenAI

**Built with Llama** — this project uses Meta Llama 3.1 under the Llama 3.1 Community License Agreement, section 5(a).

## Contact

Questions or issues → open a GitHub issue on this repo, or email wanwan0424@gmail.com.
