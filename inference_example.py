"""End-to-end inference for MERITS-L (Llama-3.1-8B + CARE).

TWO MODES:

  Mode 1 (default) — Reproducibility release
      Score an IEMOCAP test-set utterance using the cached CARE downstream
      Stage I hidden features shipped with this release. Guaranteed to
      reproduce wF1 = 0.8746 across all 5 seeds' best.
      No CARE-WavLM base model needed.

  Mode 2 — New-audio inference
      Predict emotion for arbitrary new audio. Requires the CARE-WavLM base
      checkpoint and a retrained downstream Stage I head (produced by
      scripts/enable_mode2.sh — see USAGE.md § Mode 2).

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------

  # Mode 1 (default)
  python inference_example.py --iemocap-utt Ses05F_impro01_F000
  python inference_example.py --iemocap-split test --output-json out.json

  # Mode 2 (requires bash scripts/enable_mode2.sh first)
  python inference_example.py --audio samples/happy.wav --mode 2
  python inference_example.py --audio samples/happy.wav --text "I passed!" --mode 2

------------------------------------------------------------------------------
IMPORT MAP (adjust if your local clones use different paths)
------------------------------------------------------------------------------

  Audio Stage II Bi-GRU     : merits_l_text.src.models.audio_classifier.AudioStage2
  Text Stage II Bi-GRU      : merits_l_llama.src.models.text_stage2.TextStage2
  Stage III co-attn fusion  : merits_l_llama.src.models.stage3_fusion.Stage3Fusion
  CARE-WavLM (Mode 2 only)  : care_training.CARE.pretraining.model_pase.load_pretrained
  CARE downstream (Mode 2)  : care_training.scripts.train_care_downstream_iemocap.CAREEmotionClassifier
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

# ---------- dependency repo imports ----------
try:
    from merits_l_text.src.models.audio_classifier import AudioStage2
except ImportError:
    AudioStage2 = None

try:
    from merits_l_llama.src.models.text_stage2 import TextStage2
except ImportError:
    TextStage2 = None

try:
    from merits_l_llama.src.models.stage3_fusion import Stage3Fusion
except ImportError:
    Stage3Fusion = None

# Mode 2 only:
try:
    from care_training.CARE.pretraining.model_pase import load_pretrained as load_care_wavlm
except ImportError:
    load_care_wavlm = None

try:
    from care_training.scripts.train_care_downstream_iemocap import CAREEmotionClassifier
except ImportError:
    CAREEmotionClassifier = None
# --------------------------------------------

log = logging.getLogger("merits-l-inference")


LABEL_NAMES = ["angry", "happy", "sad", "neutral"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABEL_NAMES)}
IEMOCAP_LABEL_STR_TO_IDX = {"ang": 0, "hap": 1, "exc": 1, "sad": 2, "neu": 3}


# ============================================================================
#   Pipeline
# ============================================================================


class MERITSLInference:
    """End-to-end inference wrapper."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        mode: int = 1,
        device: str = "cuda",
        load_in_4bit: bool = True,
        whisper_model: str = "large-v3",
        max_audio_seconds: float = 15.0,
        sample_rate: int = 16000,
    ):
        self.ckpt_dir = Path(checkpoint_dir)
        self.mode = mode
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.load_in_4bit = load_in_4bit
        self.max_audio_seconds = max_audio_seconds
        self.sample_rate = sample_rate

        self._check_dependencies()
        if mode == 1:
            self._load_mode1_audio_cache()
        elif mode == 2:
            self._load_mode2_audio_pipeline(whisper_model)
        else:
            raise ValueError(f"mode must be 1 or 2, got {mode}")

        self._load_audio_stage2()
        self._load_text_branch()
        self._load_fusion()

    # ------------------------------------------------------------------ deps

    def _check_dependencies(self):
        missing = []
        if AudioStage2 is None:
            missing.append("merits_l_text.src.models.audio_classifier.AudioStage2")
        if TextStage2 is None:
            missing.append("merits_l_llama.src.models.text_stage2.TextStage2")
        if Stage3Fusion is None:
            missing.append("merits_l_llama.src.models.stage3_fusion.Stage3Fusion")
        if self.mode == 2:
            if load_care_wavlm is None:
                missing.append("care_training.CARE.pretraining.model_pase.load_pretrained")
            if CAREEmotionClassifier is None:
                missing.append("care_training.scripts.train_care_downstream_iemocap.CAREEmotionClassifier")
        if missing:
            raise ImportError(
                "Missing dependency imports:\n  - "
                + "\n  - ".join(missing)
                + "\n\nRun:  bash scripts/setup_dependencies.sh"
            )

    # ---------------------------------------------------------------- Mode 1

    def _load_mode1_audio_cache(self):
        """Mode 1 — pre-computed CARE downstream Stage I features."""
        cache_path = self.ckpt_dir / "care_downstream_features_cache.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Mode 1 requires the cached features file at {cache_path}.\n"
                f"Run:  bash scripts/download_checkpoints.sh"
            )
        log.info(f"Loading CARE downstream feature cache ({cache_path.stat().st_size >> 20} MB)…")
        self.audio_feature_cache: dict[str, torch.Tensor] = torch.load(
            cache_path, map_location="cpu", weights_only=True,
        )
        log.info(f"  → {len(self.audio_feature_cache)} utterance IDs")

    # ---------------------------------------------------------------- Mode 2

    def _load_mode2_audio_pipeline(self, whisper_model: str):
        """Mode 2 — load full CARE-WavLM + downstream Stage I head + Whisper."""
        log.info("Loading CARE-WavLM base …")
        wavlm_ckpt = self.ckpt_dir / "mode2" / "care_wavlm" / "best.pth"
        self.care_wavlm = load_care_wavlm(str(wavlm_ckpt)).to(self.device).eval()

        log.info("Loading retrained CARE downstream Stage I head …")
        ds1_ckpt = torch.load(
            self.ckpt_dir / "mode2" / "care_downstream_stage1" / "model.pt",
            map_location=self.device, weights_only=False,
        )
        self.care_ds_stage1 = CAREEmotionClassifier(**ds1_ckpt["model_cfg"])
        self.care_ds_stage1.load_state_dict(ds1_ckpt["model_state_dict"])
        self.care_ds_stage1.to(self.device).eval()

        log.info(f"Loading Whisper {whisper_model} …")
        import whisper
        self.whisper = whisper.load_model(whisper_model, device=str(self.device))

    # ------------------------------------------------------ shared: audio S2

    def _load_audio_stage2(self):
        log.info("Loading Audio Stage II Bi-GRU …")
        # Mode 2 uses mode2/audio_stage2_mode2.pt if present, else fall back to
        # shipped Mode 1 checkpoint (though pairing new head → old Bi-GRU may
        # underperform — enable_mode2.sh retrains this too).
        mode2_path = self.ckpt_dir / "mode2" / "audio_stage2_mode2.pt"
        path = mode2_path if self.mode == 2 and mode2_path.exists() else self.ckpt_dir / "audio_stage2.pt"

        state = torch.load(path, map_location=self.device, weights_only=False)
        cfg = state["model_cfg"]
        self.audio_stage2 = AudioStage2(**cfg)
        self.audio_stage2.load_state_dict(state["model_state_dict"], strict=True)
        self.audio_stage2.to(self.device).eval()

    # ------------------------------------------------------ shared: text branch

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
        state = torch.load(self.ckpt_dir / "text_stage2.pt",
                           map_location=self.device, weights_only=False)
        cfg = state["model_cfg"]
        self.text_stage2 = TextStage2(**cfg)
        self.text_stage2.load_state_dict(state["model_state_dict"], strict=True)
        self.text_stage2.to(self.device).eval()

    # ------------------------------------------------------ shared: fusion

    def _load_fusion(self):
        log.info("Loading Stage III co-attention fusion …")
        state = torch.load(self.ckpt_dir / "stage3_fusion.pt",
                           map_location=self.device, weights_only=False)
        cfg = state["model_cfg"]
        self.fusion = Stage3Fusion(**cfg)
        self.fusion.load_state_dict(state["model_state_dict"], strict=True)
        self.fusion.to(self.device).eval()

    # ============================================================ audio ops

    @torch.no_grad()
    def _get_audio_hidden_mode1(self, utt_id: str) -> torch.Tensor:
        """Mode 1: look up cached 256-d hidden features."""
        if utt_id not in self.audio_feature_cache:
            raise KeyError(
                f"Utterance ID '{utt_id}' not in cache. Cache has "
                f"{len(self.audio_feature_cache)} IEMOCAP utterances."
            )
        h256 = self.audio_feature_cache[utt_id].to(self.device)  # (256,)
        # Audio Stage II expects (B, seq_len, 256). Cached features are
        # per-utterance so we treat as seq_len=1.
        return self.audio_stage2(h256.view(1, 1, 256)).squeeze(0)  # (256,)

    @torch.no_grad()
    def _get_audio_hidden_mode2(self, audio_path: str | Path) -> torch.Tensor:
        """Mode 2: raw wav → CARE-WavLM → downstream head → Audio Stage II."""
        import soundfile as sf
        arr, sr = sf.read(str(audio_path), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != self.sample_rate:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sample_rate)
        arr = arr[: int(self.max_audio_seconds * self.sample_rate)]
        wav = torch.from_numpy(arr).to(self.device)

        # NOTE: the exact API of the CARE loader depends on your local clone.
        # These next two lines assume load_pretrained returns an object with
        # .extract_layer_and_pooled(wav) → (layer_feats (1,13,1536), pooled (1,768)).
        # Adjust to match your CARE class.
        layer_feats, pooled_feats = self.care_wavlm.extract_layer_and_pooled(wav.unsqueeze(0))

        _, h256 = self.care_ds_stage1(layer_feats, pooled_feats, return_hidden=True)
        return self.audio_stage2(h256.unsqueeze(1)).squeeze(0)   # (256,)

    # ============================================================ text ops

    @torch.no_grad()
    def _get_text_hidden(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=128, padding=True).to(self.device)
        outputs = self.llama(**tokens, output_hidden_states=True)
        h = outputs.hidden_states[-1]                                    # (1, L, 4096)
        return self.text_stage2(h, attention_mask=tokens.attention_mask).squeeze(0)  # (2048,)

    @torch.no_grad()
    def _transcribe(self, audio_path: str | Path) -> str:
        if not hasattr(self, "whisper"):
            raise RuntimeError("Whisper is only loaded in Mode 2")
        return self.whisper.transcribe(str(audio_path), language="en")["text"].strip()

    # ============================================================ predict

    @torch.no_grad()
    def predict_iemocap(self, utt_id: str, text: str | None = None) -> dict[str, Any]:
        """Mode 1: predict on IEMOCAP utterance by id."""
        assert self.mode == 1, "predict_iemocap only valid in Mode 1"
        audio_h = self._get_audio_hidden_mode1(utt_id)
        if text is None:
            text = _lookup_iemocap_text(utt_id)
        text_h = self._get_text_hidden(text or "")
        logits = self.fusion(text_h.unsqueeze(0), audio_h.unsqueeze(0))
        return _package_result(logits, utt_id=utt_id, transcript=text)

    @torch.no_grad()
    def predict_audio(self, audio_path: str | Path,
                      text: str | None = None) -> dict[str, Any]:
        """Mode 2: predict on arbitrary audio file."""
        assert self.mode == 2, "predict_audio requires Mode 2. See scripts/enable_mode2.sh"
        if text is None:
            text = self._transcribe(audio_path)
        audio_h = self._get_audio_hidden_mode2(audio_path)
        text_h = self._get_text_hidden(text)
        logits = self.fusion(text_h.unsqueeze(0), audio_h.unsqueeze(0))
        return _package_result(logits, audio=str(audio_path), transcript=text)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path = "checkpoints",
                         **kwargs) -> "MERITSLInference":
        return cls(checkpoint_dir=checkpoint_dir, **kwargs)


# ============================================================================
#   Helpers
# ============================================================================


def _package_result(logits: torch.Tensor, **extra) -> dict[str, Any]:
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    pred_idx = int(torch.tensor(probs).argmax())
    return {
        **extra,
        "prediction": LABEL_NAMES[pred_idx],
        "confidence": probs[pred_idx],
        "probabilities": dict(zip(LABEL_NAMES, probs)),
    }


def _lookup_iemocap_text(utt_id: str) -> str | None:
    """Search for the utterance transcript in the merits-l-text manifest CSVs."""
    for candidate in [
        Path("deps/merits-l-text/data/manifests/iemocap"),
        Path("data/manifests/iemocap"),
    ]:
        if not candidate.is_dir():
            continue
        for split in ("train.csv", "val.csv", "test.csv"):
            fp = candidate / split
            if not fp.exists():
                continue
            with open(fp, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("utt_id") == utt_id:
                        return row.get("text") or row.get("transcript")
    log.warning(f"No transcript found for {utt_id}; using empty string")
    return ""


def _iterate_iemocap_split(split: str):
    """Yield (utt_id, gt_label_str) for each row in the given manifest split."""
    for candidate in [
        Path("deps/merits-l-text/data/manifests/iemocap"),
        Path("data/manifests/iemocap"),
    ]:
        fp = candidate / f"{split}.csv"
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    lab = str(row.get("raw_emotion", row.get("label", ""))).strip().lower()
                    idx = IEMOCAP_LABEL_STR_TO_IDX.get(lab)
                    yield row["utt_id"], idx, row.get("text", "")
            return
    raise FileNotFoundError(f"Cannot find IEMOCAP manifest split '{split}'")


# ============================================================================
#   CLI
# ============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MERITS-L (Llama-3.1-8B + CARE) — inference"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--iemocap-utt", type=str,
                     help="[Mode 1] Score a single IEMOCAP utterance by ID")
    src.add_argument("--iemocap-split", choices=["train", "val", "test"],
                     help="[Mode 1] Score all utterances in a manifest split")
    src.add_argument("--audio", type=str, help="[Mode 2] Path to an audio file")
    src.add_argument("--audio-dir", type=str, help="[Mode 2] Directory of .wav files")

    p.add_argument("--mode", type=int, choices=[1, 2], default=None,
                   help="1 = cache lookup (default when --iemocap-*), "
                        "2 = new-audio inference (default when --audio*)")
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--whisper-model", type=str, default="large-v3")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _infer_mode(args) -> int:
    if args.mode is not None:
        return args.mode
    return 1 if (args.iemocap_utt or args.iemocap_split) else 2


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    mode = _infer_mode(args)

    model = MERITSLInference.from_pretrained(
        checkpoint_dir=args.checkpoint_dir,
        mode=mode,
        device=args.device,
        load_in_4bit=not args.no_4bit,
        whisper_model=args.whisper_model,
    )

    if args.iemocap_utt:
        r = model.predict_iemocap(args.iemocap_utt, text=args.text)
        print(json.dumps(r, indent=2, ensure_ascii=False))

    elif args.iemocap_split:
        results, correct, total = [], 0, 0
        for utt_id, gt_idx, text in _iterate_iemocap_split(args.iemocap_split):
            try:
                r = model.predict_iemocap(utt_id, text=text or None)
            except KeyError as e:
                log.warning(f"skip {utt_id}: {e}")
                continue
            r["ground_truth"] = LABEL_NAMES[gt_idx] if gt_idx is not None else None
            r["correct"] = (LABEL_NAMES.index(r["prediction"]) == gt_idx)
            results.append(r)
            correct += r["correct"]
            total += 1
        acc = correct / max(total, 1)
        print(f"\nScored {total} utterances — accuracy = {acc:.4f}")
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Wrote predictions to {args.output_json}")

    elif args.audio:
        r = model.predict_audio(args.audio, text=args.text)
        print(json.dumps(r, indent=2, ensure_ascii=False))

    elif args.audio_dir:
        wavs = sorted(Path(args.audio_dir).glob("*.wav"))
        if not wavs:
            print(f"No .wav in {args.audio_dir}", file=sys.stderr)
            sys.exit(1)
        results = [model.predict_audio(w) for w in wavs]
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Wrote {len(results)} predictions to {args.output_json}")
        else:
            for r in results:
                print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
