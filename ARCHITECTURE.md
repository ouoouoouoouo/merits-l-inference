# Architecture

This document describes the exact architecture of the released MERITS-L (Llama-3.1-8B + CARE)
model, per-component parameter counts, and how the 5 released checkpoints connect.

## High-level pipeline

```
                            ┌────────────────────────────────────────┐
                            │              INFERENCE PATH             │
                            └────────────────────────────────────────┘

  Audio wav (16 kHz mono)
    │
    │   [1] Feature extraction  ─────────────────  CARE-WavLM
    │        SUPERB-13 layers × T frames × 1536-d           checkpoint 1 (974 MB)
    │
    ▼
  Frame features (T, 1536)
    │
    │   [2] CARE Downstream Stage I head  ─────  linear + non-linear
    │        (T, 1536) → (T, 256)                          checkpoint 2 (part of CARE ds Stage I)
    │
    ▼
  Frame features (T, 256)
    │
    │   [3] Audio Stage II  ────────────────────  Bi-GRU (hidden=128, 2 layers) + self-attn 8h
    │        pool over time                                checkpoint 3 (3.3 MB)
    │
    ▼
  Utterance hidden (256)  ═══════════════════════════════════════════════════╗
                                                                              ║
                                                                              ║
  Text (English string, from Whisper or user)                                 ║
    │                                                                         ║
    │   [4] Tokenization  ────────────────────  Llama-3.1 tokenizer           ║
    │                                                                         ║
    ▼                                                                         ║
  Token IDs (L,)                                                              ║
    │                                                                         ║
    │   [5] Llama-3.1-8B forward  ─────────────  base model (4-bit NF4)       ║
    │        + LoRA r=16 α=32  on q/k/v/o proj             checkpoints 4a+4b ║
    │                                                       (16 GB + 27 MB)   ║
    ▼                                                                         ║
  Token hiddens (L, 4096)                                                     ║
    │                                                                         ║
    │   [6] Text Stage II Bi-GRU + self-attn  ─  hidden=1024, 2 layers, 8h    ║
    │        pool over tokens                              checkpoint 5 (257 MB)
    ▼                                                                         ║
  Utterance hidden (2048)  ══════════════════════════════════════════════════╣
                                                                              ║
                                                                              ▼
                                                          ┌────────────────────────────┐
                                                          │  Stage III Co-Attention    │
                                                          │      Fusion (5.3 MB)       │
                                                          │        checkpoint 6         │
                                                          └────────────────────────────┘
                                                                              │
                                                                              ▼
                                                          Emotion logits (4)
                                                          {angry, happy, sad, neutral}
```

## Per-checkpoint specification

Below are the exact tensor shapes recovered from the released checkpoints, so
consumers can reconstruct any component if needed.

### 1. `care_wavlm/best.pth` — CARE-WavLM (paper-faithful, 200 K steps)

- **Backbone**: `microsoft/wavlm-base` (~ 95 M params)
- **Additional CARE modules**: PASE+ acoustic head, RoBERTa semantic head,
  downblock/upblock semantic-injection adapters
- **File size**: 974 MB (state_dict + config)
- **Loader**: `care_training/CARE/pretraining/model_pase.load_pretrained(...)`
- **Pretraining**: **200 K** optimizer updates on MSP-Podcast v1.11 (~149 K utterances),
  effective batch 128, AdamW lr = 1e-5, PASE+ acoustic loss + RoBERTa semantic loss —
  this is the **paper-faithful** protocol (matches CARE paper Sec IV-D-1).
- **Source on training cluster**: `care_training/ckpts_faithful/best.pth`
  (MD5 `67ab39526f14b1129a8a230120a80d76`).
  Note: the same tree also contains `care_training/ckpts/best.pth`
  (MD5 `7e542bd0b04ee96b59e1a5254e9da818`) which is a longer 600 K-step continuation —
  it was used for some exploratory runs but is **not** the checkpoint behind the
  released best model (0.8746). Do not confuse the two.
- **Output when called**: 13-layer stack (1 conv + 12 transformer layers, 1536-d per
  frame) plus a pooled 768-d WavLM representation — both fed into the downstream
  Stage I head below.

### 2. `care_downstream_stage1/model.pt` — CARE Downstream Stage I head

The 256-d per-utterance features fed into Audio Stage II come from a small
classifier head trained on CARE-WavLM features. Architecture (from
[`train_care_downstream_iemocap.py::CAREEmotionClassifier`](https://github.com/ouoouoouoouo/care-training/blob/main/care-training/scripts/train_care_downstream_iemocap.py)):

```python
weights: Parameter(13, 1)                    # learnable softmax over layers
fc:      Linear(1536 + 768, 768)             # fuse weighted layers + pooled audio
fc_1:    Linear(768, 256)                    # ← Stage II input (256-d hidden)
out:     Linear(256, 4)                      # classifier (unused at inference)
dropout: Dropout(p=0.2)
```

**Total: ~1.97 M parameters. File size: ~8 MB.**

Trained on IEMOCAP with Adam (lr = 1e-4, batch = 32, 50 epochs, seed = 42) —
same protocol as the paper's downstream Stage I.

Loader:
```python
from care_training.scripts.train_care_downstream_iemocap import CAREEmotionClassifier
ckpt = torch.load("checkpoints/care_downstream_stage1/model.pt", map_location=device)
model = CAREEmotionClassifier(**ckpt["model_cfg"])
model.load_state_dict(ckpt["model_state_dict"])
```

Inference call (returns 256-d hidden for Stage II):
```python
logits, hidden_256 = model(feat_13x1536, pooled_768, return_hidden=True)
```

### 3. `audio_stage2.pt` — Audio Stage II Bi-GRU + self-attention

Config (embedded in checkpoint under `model_cfg`):

```python
{
  "input_dim": 256,       # from CARE downstream Stage I
  "gru_hidden": 128,      # 2 * 128 = 256 output
  "gru_layers": 2,
  "num_heads": 8,         # 256 / 8 = 32 dim per head
  "dropout": 0.5,
  "num_labels": 4,        # (unused at Stage II — hidden state is what's exported)
}
```

State-dict layout (from live inspection):

```
bigru.weight_ih_l0                : (384, 256)
bigru.weight_hh_l0                : (384, 128)
bigru.bias_ih_l0                  : (384,)
bigru.bias_hh_l0                  : (384,)
bigru.weight_ih_l0_reverse        : (384, 256)
bigru.weight_hh_l0_reverse        : (384, 128)
bigru.bias_ih_l0_reverse          : (384,)
bigru.bias_hh_l0_reverse          : (384,)
bigru.weight_ih_l1                : (384, 256)
bigru.weight_hh_l1                : (384, 128)
bigru.bias_ih_l1                  : (384,)
bigru.bias_hh_l1                  : (384,)
bigru.weight_ih_l1_reverse        : (384, 256)
bigru.weight_hh_l1_reverse        : (384, 128)
bigru.bias_ih_l1_reverse          : (384,)
bigru.bias_hh_l1_reverse          : (384,)
attn.in_proj_weight               : (768, 256)     # 3 × 256 for q,k,v
attn.in_proj_bias                 : (768,)
attn.out_proj.weight              : (256, 256)
attn.out_proj.bias                : (256,)
```

Class: `merits_l_text.src.models.audio_classifier.AudioStage2`
(see `merits-l-text/src/models/audio_classifier.py`).

- **File size**: 3.3 MB
- **Parameters**: ~ 0.85 M

### 4. Llama-3.1-8B + LoRA

**4a. Base**: [`meta-llama/Meta-Llama-3.1-8B`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
downloaded from HuggingFace (~ 16 GB fp16, ~ 5 GB in 4-bit NF4 at load time).
Users must accept the Llama 3.1 Community License on HF first.

**4b. LoRA adapter** (this release):

```
adapter_config.json   :  PEFT configuration
adapter_model.safetensors : 27 MB
best_metadata.json    :  {seed: 42, best_val_wF1: ...}
head.pt               :  66 KB — Text Stage I linear head (4096 → 4)
```

LoRA hyperparameters (from `merits-l-llama/configs/iemocap_text_llama.yaml`):

```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  bias: none
  target_modules: [q_proj, k_proj, v_proj, o_proj]
```

**How to load**:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    quantization_config=bnb, device_map="auto",
)
model = PeftModel.from_pretrained(base, "checkpoints/llama_lora")
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
```

### 5. `text_stage2.pt` — Text Stage II Bi-GRU + self-attention

Config (from `model_cfg`):

```python
{
  "input_dim": 4096,      # Llama-3.1-8B hidden size
  "gru_hidden": 1024,     # 2 * 1024 = 2048 output
  "gru_layers": 2,
  "num_heads": 8,
  "dropout": 0.5,
  "num_labels": 4,
}
```

State-dict layout:

```
bigru.weight_ih_l0                : (3072, 4096)
bigru.weight_hh_l0                : (3072, 1024)
bigru.bias_ih_l0                  : (3072,)
bigru.bias_hh_l0                  : (3072,)
bigru.weight_ih_l0_reverse        : (3072, 4096)
bigru.weight_hh_l0_reverse        : (3072, 1024)
bigru.bias_ih_l0_reverse          : (3072,)
bigru.bias_hh_l0_reverse          : (3072,)
bigru.weight_ih_l1                : (3072, 2048)
bigru.weight_hh_l1                : (3072, 1024)
bigru.bias_ih_l1                  : (3072,)
bigru.bias_hh_l1                  : (3072,)
bigru.weight_ih_l1_reverse        : (3072, 2048)
bigru.weight_hh_l1_reverse        : (3072, 1024)
bigru.bias_ih_l1_reverse          : (3072,)
bigru.bias_hh_l1_reverse          : (3072,)
attn.in_proj_weight               : (6144, 2048)   # 3 × 2048
attn.in_proj_bias                 : (6144,)
attn.out_proj.weight              : (2048, 2048)
attn.out_proj.bias                : (2048,)
```

Class: `merits_l_llama.src.models.text_stage2.TextStage2`
(see `merits-l-llama/src/models/text_stage2.py`).

- **File size**: 257 MB
- **Parameters**: ~ 64 M

### 6. `stage3.pt` — Stage III Co-Attention Fusion

Config:

```python
{
  "text_dim": 2048,
  "audio_dim": 256,
  "hidden_dim": 256,
  "num_heads": 8,
  "dropout": 0.5,
  "num_labels": 4,
}
```

State-dict layout (bidirectional co-attention):

```
proj_text.weight                      : (256, 2048)    # project text 2048 → 256
proj_text.bias                        : (256,)
proj_audio.weight                     : (256, 256)     # project audio 256 → 256
proj_audio.bias                       : (256,)
text_attends_audio.in_proj_weight     : (768, 256)     # MHA(text ← audio)
text_attends_audio.in_proj_bias       : (768,)
text_attends_audio.out_proj.weight    : (256, 256)
text_attends_audio.out_proj.bias      : (256,)
audio_attends_text.in_proj_weight     : (768, 256)     # MHA(audio ← text)
audio_attends_text.in_proj_bias       : (768,)
audio_attends_text.out_proj.weight    : (256, 256)
audio_attends_text.out_proj.bias      : (256,)
ln_text.weight                        : (256,)
ln_text.bias                          : (256,)
ln_audio.weight                       : (256,)
ln_audio.bias                         : (256,)
fuse_fc.weight                        : (256, 1024)    # 1024 = 4 × 256 concat
fuse_fc.bias                          : (256,)
classifier.weight                     : (4, 256)
classifier.bias                       : (4,)
```

Class: `merits_l_llama.src.models.stage3_fusion.Stage3Fusion`
(see `merits-l-llama/src/models/stage3_fusion.py`).

Note: `fuse_fc` input is 1024 = 4 × hidden_dim (256). This corresponds to the
concatenation `[text_pool ‖ audio_pool ‖ text_attended_audio ‖ audio_attended_text]`
after co-attention, before the final 4-class classifier.

- **File size**: 5.3 MB
- **Parameters**: ~ 1.25 M

## Parameter budget summary

| Component | Params | Frozen? | Trainable @ this stage |
|-----------|-------:|:-------:|:----------------------:|
| Llama-3.1-8B base           | ~8.0 B    | ✓ frozen (4-bit) | — |
| Llama LoRA adapter          | ~9 M      | trainable         | Text Stage I & II |
| Text Stage I head           | ~16 K     | trainable         | Text Stage I only |
| Text Stage II (Bi-GRU+attn) | ~64 M     | trainable         | Text Stage II only |
| CARE-WavLM base             | ~95 M     | ✓ frozen at Stage III | — (pretrained separately) |
| CARE downstream Stage I     | ~0.5 M    | ✓ frozen at Stage III | — |
| Audio Stage II              | ~0.85 M   | ✓ frozen at Stage III | — |
| **Stage III fusion head**   | **~1.25 M** | **trainable**   | Stage III (this checkpoint) |

## Training stages

1. **CARE-WavLM self-supervised pretraining** (200 K steps on MSP-Podcast) — one-time
2. **CARE downstream Stage I fine-tuning** on IEMOCAP (SUPERB-13 + FC head)
3. **Audio Stage II** — Bi-GRU + self-attn over CARE ds Stage I 256-d features
4. **Text Stage I** — LoRA fine-tuning of Llama-3.1-8B with linear head
5. **Text Stage II** — Bi-GRU + self-attn over Llama-frozen Stage I hiddens
6. **Stage III fusion** — co-attention over Text Stage II ⨯ Audio Stage II hiddens

Only the outputs of stages 1, 3, 4, 5, 6 are released (5 checkpoints).

## Data conventions

- **Sampling rate**: 16 kHz mono
- **Max clip duration**: 15 s at inference (longer clips truncated)
- **Text**: normalized English, up to 128 Llama tokens
- **Labels**: `[angry, happy, sad, neutral]` — the "happy" class includes IEMOCAP's
  "excited" tag, following the standard MERITS-L 4-class convention
