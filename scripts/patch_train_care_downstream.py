"""One-off patch: add `--save-model-pt` to train_care_downstream_iemocap.py.

The original script only saves the 256-d hidden features cache (via
--save-hidden-pt). For releasing an end-to-end pipeline we also need to
save the trained CAREEmotionClassifier weights so users can extract 256-d
features from *new* audio.

Apply once, then re-run training with `--seed 42 --save-model-pt <path>`.

Usage:
    python scripts/patch_train_care_downstream.py \
        /home/ouo/care_training/care-training/scripts/train_care_downstream_iemocap.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PATCH_ARG = '''    parser.add_argument("--save-model-pt", default=None, type=str,
                        help="If set, save the trained CAREEmotionClassifier state_dict + "
                             "config to this .pt file (for releasing an inference model).")
'''

PATCH_SAVE = '''
    # Save trained CAREEmotionClassifier weights (for end-to-end inference release).
    if args.save_model_pt:
        Path(args.save_model_pt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_cfg": {
                "num_layers": num_layers,
                "layer_dim": layer_dim,
                "pooled_dim": pooled_dim,
                "hidden_dim": args.hidden,
                "output_dim": args.num_classes,
            },
            "best_val_weighted_f1": best_val_f1,
            "seed": args.seed,
            "arch": "CAREEmotionClassifier",
        }, args.save_model_pt)
        print(f"Saved {args.save_model_pt}  ({sum(p.numel() for p in model.parameters()) / 1e3:.1f}K params)")
'''


def apply_patch(script_path: str | Path) -> None:
    p = Path(script_path)
    src = p.read_text(encoding="utf-8")

    if "--save-model-pt" in src:
        print(f"Already patched: {p}")
        return

    # 1. Insert the CLI argument next to --save-hidden-pt
    anchor_arg = '    parser.add_argument("--save-hidden-pt"'
    if anchor_arg not in src:
        raise RuntimeError("Cannot locate --save-hidden-pt anchor in script")
    # Find the end of that argparse block (next line starting with 'parser' or blank line)
    idx = src.index(anchor_arg)
    # Find end of the multi-line argparse call
    end = src.index(")\n", idx) + 2
    src = src[:end] + PATCH_ARG + src[end:]

    # 2. Append save-model-pt logic right before `if args.save_hidden_pt:`
    anchor_save = "    if args.save_hidden_pt:"
    if anchor_save not in src:
        raise RuntimeError("Cannot locate save_hidden_pt block in script")
    src = src.replace(anchor_save, PATCH_SAVE.rstrip() + "\n\n" + anchor_save, 1)

    # 3. Ensure Path is imported (already is in this script — pathlib.Path)
    if "from pathlib import Path" not in src:
        src = src.replace("from __future__ import annotations",
                          "from __future__ import annotations\n\nfrom pathlib import Path", 1)

    p.write_text(src, encoding="utf-8")
    print(f"Patched: {p}")
    print("  + --save-model-pt argument")
    print("  + save block writing model_state_dict + model_cfg")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: patch_train_care_downstream.py <path-to-train_care_downstream_iemocap.py>")
        sys.exit(1)
    apply_patch(sys.argv[1])
