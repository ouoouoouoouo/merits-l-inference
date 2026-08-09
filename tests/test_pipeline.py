"""Smoke test — verify all 5 checkpoints load and the pipeline produces a prediction.

Run with:  pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


CKPT_DIR = Path(os.environ.get("MERITS_CKPT_DIR", "checkpoints"))


@pytest.fixture(scope="module")
def sample_audio(tmp_path_factory):
    """Generate a 3-second 16 kHz silent wav for smoke test."""
    import soundfile as sf
    import numpy as np
    tmp = tmp_path_factory.mktemp("samples")
    path = tmp / "silent_3s.wav"
    sf.write(path, np.zeros(16000 * 3, dtype="float32"), 16000)
    return str(path)


def test_checkpoints_exist():
    """The 5 required checkpoints are present."""
    required = [
        CKPT_DIR / "care_wavlm" / "best.pth",
        CKPT_DIR / "audio_stage2.pt",
        CKPT_DIR / "llama_lora" / "adapter_model.safetensors",
        CKPT_DIR / "text_stage2.pt",
        CKPT_DIR / "stage3_fusion.pt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f"Missing checkpoints:\n  " + "\n  ".join(missing)


def test_state_dict_shapes():
    """Load each state_dict and verify tensor shapes match the released arch."""
    expected_shapes = {
        "audio_stage2.pt": {
            "bigru.weight_ih_l0":    (384, 256),
            "attn.in_proj_weight":   (768, 256),
        },
        "text_stage2.pt": {
            "bigru.weight_ih_l0":    (3072, 4096),
            "attn.in_proj_weight":   (6144, 2048),
        },
        "stage3_fusion.pt": {
            "proj_text.weight":      (256, 2048),
            "proj_audio.weight":     (256, 256),
            "fuse_fc.weight":        (256, 1024),
            "classifier.weight":     (4, 256),
        },
    }

    for fname, expected in expected_shapes.items():
        path = CKPT_DIR / fname
        if not path.exists():
            pytest.skip(f"{path} not present")
        state = torch.load(path, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state.get("model", state))
        for key, shape in expected.items():
            assert key in sd, f"{fname}: missing key {key}"
            actual = tuple(sd[key].shape)
            assert actual == shape, f"{fname}: {key} shape {actual} != {shape}"


@pytest.mark.slow
@pytest.mark.gpu
def test_end_to_end_inference(sample_audio):
    """Full pipeline runs and returns a valid emotion prediction."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from inference_example import MERITSLInference, LABEL_NAMES

    model = MERITSLInference.from_pretrained(
        checkpoint_dir=CKPT_DIR,
        device="cuda",
        load_in_4bit=True,
        whisper_model="tiny",   # fastest, for smoke test only
    )
    result = model.predict(audio_path=sample_audio, text="hello world")

    assert result["prediction"] in LABEL_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"].keys()) == set(LABEL_NAMES)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
