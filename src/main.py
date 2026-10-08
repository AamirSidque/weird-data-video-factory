import argparse
import json
import os
import secrets
import subprocess
import wave
from pathlib import Path

import numpy as np

from .generators import (
    all_generators,
    find_generator,
    make_data,
    values,
    title,
    frame,
)
from .telegram import send_video


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
TMP = ROOT / "tmp"

OUT.mkdir(exist_ok=True)
TMP.mkdir(exist_ok=True)


def render_video(g, cfg, seed, data_vals, duration, out):
    fps = int(cfg["fps"])
    width = int(cfg["width"])
    height = int(cfg["height"])

    rng = np.random.default_rng(seed + 77)

    params = {
        "density": float(rng.uniform(0.35, 1.0)),
        "chaos": float(rng.uniform(0.2, 1.0)),
        "spin": float(rng.uniform(0.2, 1.8)),
    }

    total_frames = max(1, int(duration * fps))

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        cfg.get("preset", "veryfast"),
        "-crf",
        str(cfg.get("crf", 23)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
    )

    try:
        for i in range(total_frames):
            progress = i / max(1, total_frames - 1)

            image = frame(
                width,
                height,
                progress,
                data_vals,
                g["data_type"],
                g["method"],
                seed,
                params,
            )

            process.stdin.write(
                np.asarray(image, dtype=np.uint8).tobytes()
            )

    finally:
        if process.stdin:
            process.stdin.close()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError("FFmpeg video render failed")


def audio(g, vals, duration, rate, seed, out):
    total_samples = int(duration * rate)

    values_array = np.resize(
        np.asarray(vals, dtype=float),
        256,
    )

    base_frequencies = {
        "frequency": 220,
        "morse": 120,
        "dna": 180,
        "binary": 90,
        "hex": 130,
        "hashes": 100,
    }

    base_frequency = base_frequencies.get(
        g["data_type"],
        145,
    )

    chunk_size = max(1, rate * 2)

    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)

        phase0 = 0.0

        for start in range(0, total_samples, chunk_size):
            count = min(
                chunk_size,
                total_samples - start,
            )

            positions = np.arange(
                start,
                start + count,
            )

            if total_samples > 1:
                normalized_positions = np.linspace(
                    0,
                    total_samples - 1,
                    256,
                )

                control = np.interp(
                    positions,
                    normalized_positions,
                    values_array,
                )
            else:
                control = np.full(
                    count,
                    values_array[0],
                )

            time = positions / rate

            frequency = base_frequency * (
                0.55 + 1.5 * control
            )

            phase = (
                phase0
                + 2
                * np.pi
                * np.cumsum(frequency)
                / rate
            )

            phase0 = float(phase[-1])

            signal = (
                0.18 * np.sin(phase)
                + 0.08 * np.sin(
                    phase * 2.01 + control * 4
                )
                + 0.04 * np.sin(
                    phase * 3.01
                )
            )

            gate = (
                control > 0.62
            ).astype(float)

            kernel_size = max(
                1,
                rate // 120,
            )

            kernel_size = min(
                kernel_size,
                count,
            )

            if kernel_size > 1:
                gate = np.convolve(
                    gate,
                    np.ones(kernel_size) / kernel_size,
                    mode="same",
                )

            signal += (
                0.1
                * gate
                * np.sin(
                    2
                    * np.pi
                    * base_frequency
                    * 2
                    * time
                )
            )

            # Fade in
            if start == 0:
                fade_samples = min(
                    count,
                    rate * 2,
                )

                envelope = np.ones(count)

                if fade_samples > 1:
                    envelope[:fade_samples] = np.linspace(
                        0,
                        1,
                        fade_samples,
                    )

            # Fade out
            elif start + count >= total_samples - rate * 2:
                fade_start = max(
                    0,
                    total_samples - rate * 2 - start,
                )

                envelope = np.ones(count)

                fade_length = count - fade_start

                if fade_length > 1:
                    envelope[fade_start:] = np.linspace(
                        1,
                        0,
                        fade_length,
                    )

            else:
                envelope = np.ones(count)

            signal = np.clip(
                signal * envelope * 0.9,
                -0.95,
                0.95,
            )

            wav.writeframes(
                (
                    signal * 32767
                ).astype(np.int16).tobytes()
            )


def choose_duration(cfg, rng):
    minimum = int(
        cfg["min_duration_seconds"]
    )

    maximum = int(
        cfg["max_duration_seconds"]
    )

    if maximum <= minimum:
        return minimum

    duration = (
        minimum
        + (
            maximum - minimum
        )
        * rng.beta(2, 2.8)
    )

    return int(round(duration))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generator",
        default="",
        help="Generator ID. Blank = random.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    config_path = ROOT / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    cfg = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    # ---------------------------------------------------------
    # Generate unique seed
    # ---------------------------------------------------------

    seed = secrets.randbelow(
        2**31 - 1
    )

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # Select generator
    # ---------------------------------------------------------

    if args.generator:
        g = find_generator(
            args.generator
        )

        if g is None:
            raise ValueError(
                f"Unknown generator: {args.generator}"
            )
    else:
        generators = all_generators()

        if not generators:
            raise RuntimeError(
                "No generators are available."
            )

        g = rng.choice(generators)

    # ---------------------------------------------------------
    # Generate duration
    # ---------------------------------------------------------

    duration = choose_duration(
        cfg,
        rng,
    )

    # ---------------------------------------------------------
    # Generate raw data
    # ---------------------------------------------------------

    data = make_data(
        g["data_type"],
        rng,
    )

    data_vals = values(
        g["data_type"],
        data,
    )

    # ---------------------------------------------------------
    # Generate title
    # ---------------------------------------------------------

    ttl = title(
        g,
        rng,
    )

    # ---------------------------------------------------------
    # File paths
    # ---------------------------------------------------------

    stem = (
        f'{g["id"]}_{seed}'
    )

    raw_video = (
        TMP / f"{stem}_raw.mp4"
    )

    audio_file = (
        TMP / f"{stem}.wav"
    )

    final_video = (
        OUT / f"{stem}.mp4"
    )

    metadata_file = (
        OUT / f"{stem}.json"
    )

    # ---------------------------------------------------------
    # Render video
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("RawSignal Generation")
    print("=" * 60)
    print(f"Generator : {g['id']}")
    print(f"Data      : {g['data_label']}")
    print(f"Seed      : {seed}")
    print(f"Duration  : {duration}s")
    print(
        f"Resolution: "
        f"{cfg['width']}x{cfg['height']}"
    )
    print(f"FPS       : {cfg['fps']}")
    print("=" * 60)
    print()

    render_video(
        g,
        cfg,
        seed,
        data_vals,
        duration,
        raw_video,
    )

    # ---------------------------------------------------------
    # Add data-derived audio
    # ---------------------------------------------------------

    audio_enabled = bool(
        cfg.get(
            "audio_enabled",
            True,
        )
    )

    if audio_enabled:
        audio(
            g,
            data_vals,
            duration,
            int(
                cfg.get(
                    "audio_sample_rate",
                    44100,
                )
            ),
            seed,
            audio_file,
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_video),
                "-i",
                str(audio_file),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(final_video),
            ],
            check=True,
        )

        raw_video.unlink(
            missing_ok=True
        )

        audio_file.unlink(
            missing_ok=True
        )

    else:
        raw_video.replace(
            final_video
        )

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------

    metadata = {
        "title": ttl,
        "generator": g["id"],
        "data_type": g["data_type"],
        "data_label": g["data_label"],
        "visual_method": g["method"],
        "visual_method_label": g["method_label"],
        "seed": seed,
        "duration_seconds": duration,
        "fps": cfg["fps"],
        "resolution": (
            f'{cfg["width"]}x{cfg["height"]}'
        ),
        "audio": audio_enabled,
    }

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # Send to Telegram
    # ---------------------------------------------------------

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    send_to_telegram = bool(
        cfg.get(
            "send_to_telegram",
            True,
        )
    )

    if (
        send_to_telegram
        and token
        and chat
    ):
        print()
        print(
            "Sending video to Telegram..."
        )

        send_video(
            token,
            chat,
            final_video,
            ttl,
            g["id"],
            g["data_label"],
            seed,
        )

        print(
            "Telegram upload successful."
        )

        # Delete video after successful upload.
        # The video is NOT stored as a GitHub artifact.
        final_video.unlink(
            missing_ok=True
        )

    else:
        print()
        print(
            "Telegram upload skipped."
        )

    # ---------------------------------------------------------
    # Cleanup temporary files
    # ---------------------------------------------------------

    raw_video.unlink(
        missing_ok=True
    )

    audio_file.unlink(
        missing_ok=True
    )

    print()
    print(
        "Generation completed successfully."
    )


if __name__ == "__main__":
    main()
