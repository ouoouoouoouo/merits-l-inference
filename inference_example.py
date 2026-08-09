"""End-to-end inference for MERITS-L (Llama-3.1-8B + CARE).

Predicts one of {angry, happy, sad, neutral} from a raw English audio clip.
Transcription is done automatically with Whisper (or user-supplied).

Requires:
  - `bash scripts/setup_dependencies.sh` — installs 3 code repos to importable paths
  - `bash scripts/download_checkpoints.sh` — downloads 6 checkpoints (~1.3 GB)
  - Llama-3.1-8B license accepted on HuggingFace + `huggingface-cli login`

Usage:
  python inference_example.py --audio samples/happy.wav
  python inference_example.py --audio samples/happy.wav --text "I passed the exam!"
  python inference_example.py --audio-dir samples/ --output-json out.json

Pipeline (matches ARCHITECTURE.md):

  wav ──► CARE-WavLM (SpeechTextModel, wrapped in EmotionClassifier)
             ↓ extract_audio_features
          (fusion_out, pooled_audio, fusion_out_aud)
             ↓ mean-pool over T, concat semantic ‖ acoustic
          feat (13, 1536), pooled (768,)
             ↓ CAREEmotionClassifier(feat, pooled, return_hidden=True)
          256-d hidden
             ↓ AudioClassifier.encode(seq_len=1)
          256-d utt hidden ─────────────────────────────┐
                                                         │
                                                         ▼
                                                     Stage3Fusion → 4-class logits
                                                         ▲
                                                         │
  text ──► Llama-3.1-8B (4-bit NF4) + LoRA              │
             ↓ mean-pool tokens                         │
          4096-d per utt                                │
             ↓ TextStage2.encode                       │
          2048-d utt hidden ────────────────────────────┘
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ---------- dependency repo imports (verify with scripts/setup_dependencies.sh) ----------
try:
    from merits_l_text.src.models.audio_classifier import AudioClassifier, build_audio_classifier
except ImportError:
    AudioClassifier = None
    build_audio_classifier = None

try:
    from merits_l_llama.src.models.text_stage2 import TextStage2, build_text_stage2
except ImportError:
    TextStage2 = None
    build_text_stage2 = None

try:
    from merits_l_llama.src.models.stage3_fusion import Stage3Fusion, build_stage3_fusion
except ImportError:
    Stage3Fusion = None
    build_stage3_fusion = None

try:
    from care_training.scripts.extract_iemocap_care_downstream_style import CAREDownstreamExtractor
except ImportError:
    CAREDownstreamExtractor = None

try:
    from care_training.scripts.train_care_downstream_iemocap import CAREEmotionClassifier
except ImportError:
    CAREEmotionClassifier = None
# --------------------------------------------------------------------------------

log = logging.getLogger("merits-l-inference")

LABEL_NAMES = ["angry", "happy", "sad", "neutral"]


def _make_cfg_namespace(d: dict):
    """Convert dict → object with attribute access, so model_cfg-dicts work with
    build_*(cfg) helpers that expect cfg.foo instead of cfg["foo"]."""
    from types import SimpleNamespace
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _make_cfg_namespace(v) for k, v in d.items()})
    return d


def _load_stage_ckpt(path: Path, builder, cls, device):
    """Common loader for Text Stage II / Audio Stage II / Stage III fusion.

    Each checkpoint stores {"model_state_dict": ..., "model_cfg": {...}, ...}.
    We first try `builder(cfg)` (build_*); if unavailable, fall back to `cls(**cfg)`.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("model_cfg", {})

    model = None
    if builder is not None:
        try:
            model = builder(_make_cfg_namespace(cfg_dict))
        except TypeError:
            model = builder(cfg_dict)          # some builders take dict directly
    if model is None:
        model = cls(**cfg_dict)

    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"{path.name} missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        log.warning(f"{path.name} unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    return model.to(device).eval()


class MERITSLInference:
    """End-to-end inference wrapper."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        care_repo: str = "/home/ouo/care_training/CARE/pretraining",
        device: str = "cuda",
        load_in_4bit: bool = True,
        whisper_model: str = "large-v3",
        max_audio_seconds: float = 15.0,
        sample_rate: int = 16000,
    ):
        self.ckpt_dir = Path(checkpoint_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.load_in_4bit = load_in_4bit
        self.max_audio_seconds = max_audio_seconds
        self.sample_rate = sample_rate

        self._check_dependencies()

        # ---- Audio branch ----
        log.info("Loading CARE-WavLM extractor …")
        self.care_extractor = CAREDownstreamExtractor(
            care_ckpt=str(self.ckpt_dir / "care_wavlm" / "best.pth"),
            care_repo=care_repo,
            device=str(self.device),
        )

        log.info("Loading CARE downstream Stage I head …")
        ds1_ckpt = torch.load(
            self.ckpt_dir / "care_downstream_stage1" / "model.pt",
            map_location=self.device, weights_only=False,
        )
        self.care_ds_stage1 = CAREEmotionClassifier(**ds1_ckpt["model_cfg"])
        self.care_ds_stage1.load_state_dict(ds1_ckpt["model_state_dict"])
        self.care_ds_stage1.to(self.device).eval()

        log.info("Loading Audio Stage II Bi-GRU …")
        self.audio_stage2 = _load_stage_ckpt(
            self.ckpt_dir / "audio_stage2.pt",
            build_audio_classifier, AudioClassifier, self.device,
        )

        # ---- Text branch ----
        self._load_text_branch()

        # ---- Fusion ----
        log.info("Loading Stage III co-attention fusion …")
        self.fusion = _load_stage_ckpt(
            self.ckpt_dir / "stage3_fusion.pt",
            build_stage3_fusion, Stage3Fusion, self.device,
        )

        # ---- Whisper (lazy — loaded on first predict without --text) ----
        self._whisper_model_name = whisper_model
        self._whisper = None

    # ---------------------------------------------------------------- deps

    def _check_dependencies(self):
        missing = []
        for name, obj in [
            ("merits_l_text.src.models.audio_classifier.AudioClassifier", AudioClassifier),
            ("merits_l_llama.src.models.text_stage2.TextStage2", TextStage2),
            ("merits_l_llama.src.models.stage3_fusion.Stage3Fusion", Stage3Fusion),
            ("care_training.scripts.extract_iemocap_care_downstream_style.CAREDownstreamExtractor",
             CAREDownstreamExtractor),
            ("care_training.scripts.train_care_downstream_iemocap.CAREEmotionClassifier",
             CAREEmotionClassifier),
        ]:
            if obj is None:
                missing.append(name)
        if missing:
            raise ImportError(
                "Missing dependency imports:\n  - "
                + "\n  - ".join(missing)
                + "\n\nRun:  bash scripts/setup_dependencies.sh"
                "\nand make sure PYTHONPATH contains the deps root."
            )

    # ---------------------------------------------------------------- text

    def _load_text_branch(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        log.info("Loading Llama-3.1-8B (base + LoRA) …")
        base_model_id = "meta-llama/Meta-Llama-3.1-8B"

        if self.load_in_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id, quantization_config=bnb, device_map="auto",
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=torch.bfloat16, device_map="auto",
            )
        self.llama = PeftModel.from_pretrained(base, str(self.ckpt_dir / "llama_lora")).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info("Loading Text Stage II Bi-GRU …")
        self.text_stage2 = _load_stage_ckpt(
            self.ckpt_dir / "text_stage2.pt",
            build_text_stage2, TextStage2, self.device,
        )

    # ---------------------------------------------------------------- whisper

    def _get_whisper(self):
        if self._whisper is None:
            import whisper
            log.info(f"Loading Whisper {self._whisper_model_name} …")
            self._whisper = whisper.load_model(self._whisper_model_name, device=str(self.device))
        return self._whisper

    # ---------------------------------------------------------------- audio io

    def _load_audio(self, path: str | Path) -> np.ndarray:
        import soundfile as sf
        arr, sr = sf.read(str(path), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != self.sample_rate:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sample_rate)
        arr = arr[: int(self.max_audio_seconds * self.sample_rate)]
        return arr.astype(np.float32)

    def _transcribe(self, audio_path: str | Path) -> str:
        w = self._get_whisper()
        return w.transcribe(str(audio_path), language="en")["text"].strip()

    # ============================================================ encoders

    @torch.no_grad()
    def _encode_audio(self, audio_np: np.ndarray) -> torch.Tensor:
        """Full audio pipeline: wav (np) → 256-d utt hidden (torch on device)."""
        # 1. CARE-WavLM → (13, 1536) features + (768,) pooled
        feat, pooled = self.care_extractor.extract(audio_np)      # float32 cpu tensors
        feat = feat.unsqueeze(0).to(self.device)                  # (1, 13, 1536)
        pooled = pooled.unsqueeze(0).to(self.device)              # (1, 768)

        # 2. CARE downstream Stage I → 256-d hidden per utt
        _, h256 = self.care_ds_stage1(feat, pooled, return_hidden=True)
        # h256: (1, 256)

        # 3. Audio Stage II Bi-GRU + self-attn — treat as seq_len=1 dialogue
        seq = h256.unsqueeze(1)                                    # (1, 1, 256)
        mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        encoded = self.audio_stage2.encode(seq, mask)              # (1, 1, 256)
        return encoded.squeeze(0).squeeze(0)                       # (256,)

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        """text → 2048-d utt hidden."""
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128, padding=True,
        ).to(self.device)

        outputs = self.llama(**tokens, output_hidden_states=True)
        h = outputs.hidden_states[-1]                              # (1, L, 4096)

        # Mean-pool tokens (matches how Text Stage I / II training extracted utt-level features)
        attn = tokens.attention_mask.unsqueeze(-1).to(h.dtype)     # (1, L, 1)
        utt = (h * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)  # (1, 4096)

        # Text Stage II — treat as seq_len=1 dialogue
        seq = utt.unsqueeze(1)                                     # (1, 1, 4096)
        mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        encoded = self.text_stage2.encode(seq, mask)               # (1, 1, 2048)
        return encoded.squeeze(0).squeeze(0)                       # (2048,)

    # ============================================================ predict

    @torch.no_grad()
    def predict(
        self,
        audio_path: str | Path,
        text: str | None = None,
    ) -> dict[str, Any]:
        audio_np = self._load_audio(audio_path)
        if text is None:
            text = self._transcribe(audio_path)
            log.info(f"Whisper transcript: {text}")

        audio_h = self._encode_audio(audio_np)                     # (256,)
        text_h = self._encode_text(text)                           # (2048,)

        logits = self.fusion(text_h.unsqueeze(0), audio_h.unsqueeze(0))  # (1, 4)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        pred_idx = int(np.argmax(probs))

        return {
            "audio": str(audio_path),
            "transcript": text,
            "prediction": LABEL_NAMES[pred_idx],
            "confidence": probs[pred_idx],
            "probabilities": dict(zip(LABEL_NAMES, probs)),
        }

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path = "checkpoints", **kwargs):
        return cls(checkpoint_dir=checkpoint_dir, **kwargs)


# ============================================================================
#   CLI
# ============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MERITS-L (Llama-3.1-8B + CARE) inference")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", type=str, help="Path to a single audio file")
    src.add_argument("--audio-dir", type=str, help="Directory of .wav files (batch)")

    p.add_argument("--text", type=str, default=None,
                   help="Transcript. If omitted, Whisper transcribes.")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--care-repo", type=str, default="/home/ouo/care_training/CARE/pretraining",
                   help="Path to CARE/pretraining/ (contains model_pase.py). Only used at load time.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-4bit", action="store_true", help="Load Llama in bf16 instead of NF4")
    p.add_argument("--whisper-model", type=str, default="large-v3",
                   help="whisper model: tiny/base/small/medium/large-v3")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    model = MERITSLInference.from_pretrained(
        checkpoint_dir=args.checkpoint_dir,
        care_repo=args.care_repo,
        device=args.device,
        load_in_4bit=not args.no_4bit,
        whisper_model=args.whisper_model,
    )

    if args.audio:
        r = model.predict(args.audio, text=args.text)
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        wavs = sorted(Path(args.audio_dir).glob("*.wav"))
        if not wavs:
            print(f"No .wav in {args.audio_dir}", file=sys.stderr)
            sys.exit(1)
        results = [model.predict(w) for w in wavs]
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Wrote {len(results)} predictions to {args.output_json}")
        else:
            for r in results:
                print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
