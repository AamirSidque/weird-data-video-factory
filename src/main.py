from __future__ import annotations
import argparse
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from .generators import choose_generator
from .renderer import render_video
from .telegram import send_video


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"


def load_config():
    with open(ROOT / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generator", default=None,
                   help="Specific generator, e.g. H06 or D01")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31 - 1)

    import numpy as np
    rng = np.random.default_rng(seed)

    fixed = args.generator or cfg.get("fixed_generator")
    spec = choose_generator(rng, fixed)

    if cfg.get("fixed_data_type") and spec.data_type != cfg["fixed_data_type"]:
        # Pick a generator from the requested data type.
        candidates = [g for g in __import__(
            "src.generators", fromlist=["all_generators"]).all_generators()
                      if g.data_type == cfg["fixed_data_type"]]
        spec = candidates[int(rng.integers(0, len(candidates)))]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    OUTPUT.mkdir(exist_ok=True)
    video_path = OUTPUT / f"{stamp}_{spec.generator_id}_{seed}.mp4"

    print(f"Generator: {spec.generator_id}")
    print(f"Data type: {spec.data_type}")
    print(f"Method: {spec.method}")
    print(f"Seed: {seed}")
    print(f"Title: {spec.title}")

    render_video(
        spec, seed, video_path,
        int(cfg["width"]), int(cfg["height"]),
        int(cfg["fps"]), int(cfg["duration_seconds"]),
        int(cfg["crf"]), cfg["preset"]
    )

    metadata = {
        "title": spec.title,
        "generator_id": spec.generator_id,
        "data_type": spec.data_type,
        "method": spec.method,
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "video": str(video_path.name),
    }
    metadata_path = video_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if cfg.get("send_to_telegram", True):
        if not token or not chat_id:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required."
            )
        send_video(token, chat_id, str(video_path), spec.title,
                   spec.generator_id, spec.data_type, seed)

    print("DONE:", video_path)


if __name__ == "__main__":
    main()
