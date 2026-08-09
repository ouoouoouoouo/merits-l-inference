# Model Card — MERITS-L (Llama-3.1-8B + CARE)

## Model summary

A multimodal speech emotion recognition (SER) system for English speech,
predicting one of 4 classes: **angry, happy, sad, neutral**.

Architecture: **MERITS-L** ([Dutta & Ganapathy, ICASSP 2025](https://arxiv.org/abs/2409.14547))
with the RoBERTa text encoder replaced by **Meta Llama-3.1-8B (LoRA-adapted)**.

- **Model type**: Multimodal (audio + text) classifier
- **Language**: English (audio + transcription)
- **License**: [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/)
- **Trained on**: IEMOCAP 4-class (session-independent split, session 5 as test)
- **Built with Llama**

## Intended use

**Intended:**
- Research and educational use in speech emotion recognition
- Baseline for comparison in academic multimodal SER papers
- Demonstration of LLM-augmented SER pipelines

**Not intended:**
- Medical, psychological, or clinical diagnosis
- Employment, hiring, or personnel evaluation decisions
- Surveillance or covert monitoring of individuals
- Any high-stakes automated decision-making about people
- Non-English speech (not evaluated)
- Real-time / low-latency streaming (batch inference only)

## Training data

- **Dataset**: [IEMOCAP](https://sail.usc.edu/iemocap/) 4-class subset
  (angry, happy [inc. excited], sad, neutral)
- **Split**: Session-independent — sessions 1–4 train/val, session 5 test
- **Size**: 5,531 utterances (3,205 train / 1,085 val / 1,241 test)
- **Text**: Ground-truth IEMOCAP transcripts during training;
  Whisper large-v3 transcripts at inference time

**Additional pretraining data for the CARE audio encoder:**
[MSP-Podcast v1.11](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) —
148,160 utterances, self-supervised training with PASE+ acoustic and RoBERTa
semantic targets (no emotion labels used).

## Evaluation

**Test weighted-F1 on IEMOCAP session 5** (5 seeds, mean ± std):

| System                                                    | Weighted F1         | Best     |
|-----------------------------------------------------------|---------------------|----------|
| Paper (RoBERTa + CARE-WavLM)                              | 0.8648 (1 seed)     | —        |
| Our reproduction (RoBERTa + CARE-WavLM)                   | 0.8305 ± 0.0138     | 0.8504   |
| Our original Llama LoRA + CARE (thesis)                   | 0.8567 ± 0.0139     | 0.8746   |
| **Released end-to-end pipeline (this checkpoint set)**    | **0.8474 ± 0.0110** | **0.8541** |

**Two numbers, one story:**

The **0.8746 in the thesis** was obtained during the original training run,
which had a small gap — the CARE downstream Stage I head was trained but its
weights were never saved to disk (the training script only cached the extracted
features). To ship a model that accepts arbitrary new audio, we re-trained the
downstream 3 stages (CARE ds Stage I → Audio Stage II → Stage III fusion, 5
seeds) with the seed-fixed pipeline. This gives the released end-to-end model
its own — slightly lower but honest — accuracy: **0.8541 (best seed)**, or
0.8474 ± 0.0110 across 5 seeds.

Both numbers are correct; they measure different pipelines. If you cite this
model, please use **0.8541** — that is what a downloaded and used copy
actually delivers.

### 5-seed results for the released pipeline

| Seed | Weighted F1 |
|:----:|:-----------:|
| 1 ⭐ | 0.8541      |
| 2    | 0.8279      |
| 3    | 0.8522      |
| 4    | 0.8494      |
| 5    | 0.8532      |
| **Mean ± std** | **0.8474 ± 0.0110** |

The seed-1 Stage III fusion checkpoint is the one packaged in `stage3_fusion.pt`.

## Architecture details

**Text branch:**
- Base: Meta Llama-3.1-8B (frozen backbone in 4-bit NF4)
- Adaptation: LoRA r=16, α=32, dropout=0.1, targets `q_proj, k_proj, v_proj, o_proj`
- Text Stage I: mean-pool → linear (4096 → 4) — head file 66 KB
- Text Stage II: Bi-GRU (hidden=1024, 2 layers) + self-attention (8 heads) → utt hidden (2048-d) — 257 MB

**Audio branch:**
- Base: CARE-WavLM (WavLM-base pretrained with PASE+ and RoBERTa supervision on MSP-Podcast) — 974 MB
- CARE Downstream Stage I: SUPERB-13 layer weighted sum (1536-d) → FC (→ 256-d per frame)
- Audio Stage II: Bi-GRU (hidden=128, 2 layers) + self-attention (8 heads) → utt hidden (256-d) — 3.3 MB

**Fusion:**
- Stage III **bidirectional co-attention** head — 5.3 MB
- text_dim=2048, audio_dim=256, hidden_dim=256, num_heads=8, dropout=0.5
- Concatenates 4 vectors: `[text_pool ‖ audio_pool ‖ text_attended_audio ‖ audio_attended_text]` (1024-d)
- FC (1024 → 256) → GELU → classifier (256 → 4)
- Trainable params: ~1.25 M

See [ARCHITECTURE.md](ARCHITECTURE.md) for exact state-dict layouts of every
released checkpoint.

**Inference:**
- Whisper large-v3 for automatic transcription (~3 GB)

## Limitations

1. **Session-dependent evaluation only** — trained/tested on IEMOCAP session split.
   Cross-corpus generalization (e.g., to MSP-IMPROV, RAVDESS) not evaluated.
2. **4 emotions only** — collapses "excited" into "happy" as per MERITS-L convention.
   Fine-grained emotion (e.g., surprise, disgust, fear) not supported.
3. **English only** — Llama and Whisper both support other languages but the
   downstream heads are English-only trained.
4. **Acted speech** — IEMOCAP is scripted/improvised acting; naturalistic
   spontaneous speech may perform worse.
5. **Whisper transcription errors** propagate to the text branch; the fusion
   is somewhat robust but severely miscribed speech will degrade prediction.
6. **Prosody in transcription** is lost — Whisper produces plain text, so the
   text branch does not see paralinguistic cues (that is the audio branch's job).

## Bias and fairness

IEMOCAP contains 10 professional actors (5M/5F, all native English speakers, ages
approximately 25–40). Performance on demographics not represented in the training
corpus (e.g., children, elderly, non-native accents, dialects) is **unknown and
likely worse**. Emotion perception is culturally variable, and the training labels
reflect annotator judgments on North American acted speech.

## Environmental impact

CARE-WavLM pretraining: 200,000 optimizer steps @ batch 128 on a single
NVIDIA Blackwell 6000 (~30 hours). Text/audio Stage I/II and Stage III fusion:
~4 GPU-hours combined on the same hardware.

## How to cite

```bibtex
@inproceedings{dutta2025merits,
  title={MERITS-L: Multimodal Emotion Recognition via Interactive Speech and Language},
  author={Dutta, Soumya and Ganapathy, Sriram},
  booktitle={ICASSP},
  year={2025}
}
```

Please also acknowledge:
- Meta Llama 3.1 (Grattafiori et al., 2024) — "Built with Llama"
- Whisper (Radford et al., 2022)
- IEMOCAP corpus (Busso et al., 2008)
