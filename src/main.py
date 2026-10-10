import argparse
import json
import os
import secrets
import subprocess
import wave
from pathlib import Path

import numpy as np

from .youtube_uploader import (
    upload_and_schedule,
    telegram_notify,
)

from .generators import (
    all_generators,
    find_generator,
    make_data,
    values,
    title,
    frame,
)
from .telegram import send_video


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "output"
TMP = ROOT / "tmp"
HISTORY_FILE = ROOT / "data" / "generation_history.json"

OUT.mkdir(exist_ok=True)
TMP.mkdir(exist_ok=True)
HISTORY_FILE.parent.mkdir(exist_ok=True)


# =========================================================
# GENERATION HISTORY
# =========================================================

def load_generation_history():
    """
    Load previously generated signatures.

    The history file is intentionally small and contains
    only generation signatures, not videos.
    """

    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text(
            json.dumps(
                {"signatures": []},
                indent=2,
            ),
            encoding="utf-8",
        )

        return []

    try:
        data = json.loads(
            HISTORY_FILE.read_text(
                encoding="utf-8"
            )
        )

        signatures = data.get(
            "signatures",
            [],
        )

        if not isinstance(signatures, list):
            return []

        return signatures

    except (json.JSONDecodeError, OSError):
        print(
            "Warning: generation history could not "
            "be read. Starting with empty history."
        )

        return []


def save_generation_signature(signature):
    """
    Add a successful generation signature to history.

    History is capped at 10,000 entries so the JSON file
    remains small.
    """

    signatures = load_generation_history()

    if signature not in signatures:
        signatures.append(signature)

    signatures = signatures[-10000:]

    HISTORY_FILE.write_text(
        json.dumps(
            {
                "signatures": signatures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def create_generation_signature(
    generator_id,
    seed,
    duration,
    params,
):
    """
    Create a deterministic signature for the complete
    generation configuration.

    Same generator + same seed + same duration +
    same visual parameters = same generation.
    """

    return (
        f"{generator_id}|"
        f"{seed}|"
        f"{duration}|"
        f"{params['density']:.8f}|"
        f"{params['chaos']:.8f}|"
        f"{params['spin']:.8f}"
    )


# =========================================================
# DURATION
# =========================================================

def choose_duration(cfg, rng):
    """
    Select a variable duration between the configured
    minimum and maximum.

    Beta distribution makes medium-length videos more
    common while still allowing short and long videos.
    """

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


# =========================================================
# VISUAL PARAMETERS
# =========================================================

def generate_visual_parameters(seed):
    """
    Generate deterministic visual parameters from the seed.

    The same seed always produces the same parameters.
    """

    rng = np.random.default_rng(
        seed + 77
    )

    return {
        "density": float(
            rng.uniform(
                0.35,
                1.0,
            )
        ),
        "chaos": float(
            rng.uniform(
                0.2,
                1.0,
            )
        ),
        "spin": float(
            rng.uniform(
                0.2,
                1.8,
            )
        ),
    }


# =========================================================
# VIDEO RENDERING
# =========================================================

def render_video(
    g,
    cfg,
    seed,
    data_vals,
    duration,
    out,
    params,
    raw_data=None,
):
    fps = int(
        cfg["fps"]
    )

    width = int(
        cfg["width"]
    )

    height = int(
        cfg["height"]
    )

    total_frames = max(
        1,
        int(duration * fps),
    )

    command = [
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
        cfg.get(
            "preset",
            "veryfast",
        ),
        "-crf",
        str(
            cfg.get(
                "crf",
                23,
            )
        ),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
    )

    try:
        for i in range(total_frames):

            progress = (
                i
                / max(
                    1,
                    total_frames - 1,
                )
            )

            image = frame(
                width,
                height,
                progress,
                data_vals,
                g["data_type"],
                g["method"],
                seed,
                params,
                raw_data=raw_data,
            )

            process.stdin.write(
                np.asarray(
                    image,
                    dtype=np.uint8,
                ).tobytes()
            )

    finally:

        if process.stdin:
            process.stdin.close()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            "FFmpeg video render failed."
        )


# =========================================================
# DATA-DERIVED AUDIO
# =========================================================

def audio(
    g,
    vals,
    duration,
    rate,
    seed,
    out,
):
    """Render a gentle, seed-driven ambient/electronic soundtrack.

    The data influences the musical key and pacing, but does not directly
    bend a continuous oscillator. A simple diatonic chord progression,
    soft sustained pads, rounded bass, and a quiet plucked arpeggio create
    a more musical bed without external APIs or audio assets.
    """
    total_samples = int(duration * rate)
    if total_samples <= 0 or rate < 8000:
        raise ValueError("Audio duration or sample rate is invalid.")

    rng = np.random.default_rng(int(seed) % (2**32))
    values_array = np.resize(np.asarray(vals, dtype=float), 128)
    values_array = np.nan_to_num(values_array, nan=0.5, posinf=1.0, neginf=0.0)
    values_array = np.clip(values_array, 0.0, 1.0)

    # A stable musical key and tempo for each generated seed.
    root_midi = int(rng.choice([48, 50, 52, 53, 55, 57]))
    bpm = int(rng.choice([84, 88, 92, 96, 100]))
    beat_seconds = 60.0 / bpm
    chord_seconds = beat_seconds * 4.0
    beat_samples = max(1, int(beat_seconds * rate))
    chord_samples = max(1, int(chord_seconds * rate))

    # I - V - vi - IV, voiced for a calm, modern ambient feel.
    chord_intervals = ((0, 4, 7), (7, 11, 14), (9, 12, 16), (5, 9, 12))
    progression = [
        tuple(root_midi + interval for interval in intervals)
        for intervals in chord_intervals
    ]

    def midi_hz(note):
        return 440.0 * (2.0 ** ((note - 69) / 12.0))

    # Very slow control movement keeps the track related to the visual data,
    # while avoiding the unpleasant, continuous pitch wobble of the old sound.
    control = float(np.mean(values_array))
    pad_level = 0.075 + 0.012 * control
    bass_level = 0.050 + 0.008 * (1.0 - control)
    pluck_level = 0.030 + 0.006 * control
    fade_samples = min(total_samples // 2, int(rate * 2.5))
    chunk_size = max(1, int(rate * 2))

    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(rate)

        for start in range(0, total_samples, chunk_size):
            count = min(chunk_size, total_samples - start)
            positions = np.arange(start, start + count, dtype=np.int64)
            t = positions.astype(np.float64) / rate
            chord_idx = np.floor(t / chord_seconds).astype(int) % len(progression)
            local_chord_t = np.mod(t, chord_seconds)

            left = np.zeros(count, dtype=np.float64)
            right = np.zeros(count, dtype=np.float64)

            # Sustained chord pad: rounded sine fundamentals with a quiet octave.
            for chord_i, chord in enumerate(progression):
                mask = chord_idx == chord_i
                if not np.any(mask):
                    continue
                local_t = local_chord_t[mask]
                # Gentle swell within each chord; no hard gating.
                swell = 0.82 + 0.18 * np.sin(np.pi * np.clip(local_t / chord_seconds, 0, 1))
                for voice_i, note in enumerate(chord):
                    freq = midi_hz(note)
                    phase = 2.0 * np.pi * freq * t[mask]
                    voice = np.sin(phase) + 0.16 * np.sin(2.0 * phase)
                    voice *= (pad_level / 3.0) * swell
                    pan = (voice_i - 1) * 0.16
                    left[mask] += voice * (0.72 - pan)
                    right[mask] += voice * (0.72 + pan)

            # Soft bass note on each bar, smoothed with a short attack/release.
            beat_idx = positions // beat_samples
            beat_pos = (positions % beat_samples) / float(beat_samples)
            bar_start = (beat_idx % 4) == 0
            bass_env = np.zeros(count, dtype=np.float64)
            bass_env[bar_start] = np.exp(-beat_pos[bar_start] * 3.8)
            bass_notes = np.array([root_midi, root_midi + 7, root_midi + 9, root_midi + 5])
            bass_midi = bass_notes[chord_idx]
            bass = np.sin(2.0 * np.pi * np.array([midi_hz(int(n)) for n in bass_midi]) * t)
            bass *= bass_level * bass_env
            left += bass * 0.72
            right += bass * 0.72

            # A low-volume, rounded arpeggio. A short raised-cosine envelope
            # avoids clicks and keeps transients softer than a sharp pluck.
            beat_pos_samples = positions % beat_samples
            beat_number = positions // beat_samples
            arp_note_index = (beat_number % 4).astype(int)
            arp_notes = np.empty(count, dtype=np.int32)
            for chord_i, chord in enumerate(progression):
                mask = chord_idx == chord_i
                if np.any(mask):
                    arp_notes[mask] = np.asarray(chord, dtype=np.int32)[arp_note_index[mask] % 3] + 12
            attack = np.clip(beat_pos_samples / max(1.0, rate * 0.025), 0.0, 1.0)
            release = np.exp(-beat_pos_samples / (rate * 0.22))
            arp_env = (0.5 - 0.5 * np.cos(np.pi * attack)) * release
            arp_freq = np.array([midi_hz(int(n)) for n in arp_notes])
            arp_phase = 2.0 * np.pi * arp_freq * t
            arp = (np.sin(arp_phase) + 0.12 * np.sin(2.0 * arp_phase)) * pluck_level * arp_env
            # Alternating stereo placement gives subtle width without extreme panning.
            arp_pan = np.where((beat_number % 2) == 0, 0.16, -0.16)
            left += arp * (0.68 - arp_pan)
            right += arp * (0.68 + arp_pan)

            # Gentle, slow stereo movement and clean fade edges.
            movement = 0.97 + 0.03 * np.sin(2.0 * np.pi * t / 11.0)
            left *= movement
            right *= movement
            envelope = np.ones(count, dtype=np.float64)
            if fade_samples > 1:
                fade_in = positions < fade_samples
                envelope[fade_in] = positions[fade_in] / (fade_samples - 1)
                fade_out_start = total_samples - fade_samples
                fade_out = positions >= fade_out_start
                envelope[fade_out] = np.minimum(
                    envelope[fade_out],
                    (total_samples - 1 - positions[fade_out]) / (fade_samples - 1),
                )
            stereo = np.column_stack((left, right)) * envelope[:, None]
            # Conservative ceiling leaves headroom for platform encoding.
            stereo = np.clip(stereo, -0.45, 0.45)
            wav.writeframes((stereo * 32767).astype(np.int16).tobytes())


# =========================================================
# UNIQUE GENERATION SELECTION
# =========================================================

def create_unique_generation(
    cfg,
    requested_generator,
    history,
    max_attempts=50,
):
    """
    Generate a new configuration that does not already
    exist in generation_history.json.
    """

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        # -------------------------------------------------
        # New cryptographically strong random seed
        # -------------------------------------------------

        seed = secrets.randbelow(
            2**31 - 1
        )

        rng = np.random.default_rng(
            seed
        )

        # -------------------------------------------------
        # Generator
        # -------------------------------------------------

        if requested_generator:

            g = find_generator(
                requested_generator
            )

            if g is None:
                raise ValueError(
                    "Unknown generator: "
                    f"{requested_generator}"
                )

        else:

            generators = all_generators()

            if not generators:
                raise RuntimeError(
                    "No generators are available."
                )

            g = rng.choice(
                generators
            )

        # -------------------------------------------------
        # Duration
        # -------------------------------------------------

        duration = choose_duration(
            cfg,
            rng,
        )

        # -------------------------------------------------
        # Visual parameters
        # -------------------------------------------------

        visual_params = (
            generate_visual_parameters(
                seed
            )
        )

        # -------------------------------------------------
        # Signature
        # -------------------------------------------------

        signature = (
            create_generation_signature(
                g["id"],
                seed,
                duration,
                visual_params,
            )
        )

        # -------------------------------------------------
        # Duplicate check
        # -------------------------------------------------

        if signature not in history:

            return (
                g,
                seed,
                duration,
                visual_params,
                signature,
            )

        print(
            f"Duplicate detected. "
            f"Retry {attempt}/{max_attempts}..."
        )

    raise RuntimeError(
        "Unable to create a unique generation "
        f"after {max_attempts} attempts."
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # Arguments
    # -----------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generator",
        default="",
        help=(
            "Generator ID. "
            "Blank = random."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    config_path = (
        ROOT / "config.json"
    )

    if not config_path.exists():

        raise FileNotFoundError(
            "Configuration file not found: "
            f"{config_path}"
        )

    cfg = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    # -----------------------------------------------------
    # Load duplicate history
    # -----------------------------------------------------

    history = (
        load_generation_history()
    )

    print(
        f"Generation history: "
        f"{len(history)} entries"
    )

    # -----------------------------------------------------
    # Select unique generation
    # -----------------------------------------------------

    (
        g,
        seed,
        duration,
        visual_params,
        signature,
    ) = create_unique_generation(
        cfg,
        args.generator,
        history,
    )

    # -----------------------------------------------------
    # Generate data
    # -----------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    data = make_data(
        g["data_type"],
        rng,
    )

    data_vals = values(
        g["data_type"],
        data,
    )

    # -----------------------------------------------------
    # Generate title
    # -----------------------------------------------------

    ttl = title(
        g,
        rng,
    )

    # -----------------------------------------------------
    # File names
    # -----------------------------------------------------

    stem = (
        f'{g["id"]}_{seed}'
    )

    raw_video = (
        TMP
        / f"{stem}_raw.mp4"
    )

    audio_file = (
        TMP
        / f"{stem}.wav"
    )

    final_video = (
        OUT
        / f"{stem}.mp4"
    )

    metadata_file = (
        OUT
        / f"{stem}.json"
    )

    # -----------------------------------------------------
    # Generation information
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("RawSignal Generation")
    print("=" * 60)

    print(
        f"Generator : {g['id']}"
    )

    print(
        f"Data      : {g['data_label']}"
    )

    print(
        f"Seed      : {seed}"
    )

    print(
        f"Duration  : {duration}s"
    )

    print(
        f"Resolution: "
        f"{cfg['width']}x{cfg['height']}"
    )

    print(
        f"FPS       : {cfg['fps']}"
    )

    print(
        f"Signature : {signature}"
    )

    print("=" * 60)
    print()

    # -----------------------------------------------------
    # Render video
    # -----------------------------------------------------

    render_video(
        g,
        cfg,
        seed,
        data_vals,
        duration,
        raw_video,
        visual_params,
        raw_data=data,
    )

    # -----------------------------------------------------
    # Audio
    # -----------------------------------------------------

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

        # -------------------------------------------------
        # Combine video + audio
        # -------------------------------------------------

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

    # -----------------------------------------------------
    # Save metadata
    # -----------------------------------------------------

    metadata = {
        "title": ttl,
        "generator": g["id"],
        "data_type": g["data_type"],
        "data_label": g["data_label"],
        "visual_method": g["method"],
        "visual_method_label": g[
            "method_label"
        ],
        "seed": seed,
        "duration_seconds": duration,
        "fps": cfg["fps"],
        "resolution": (
            f'{cfg["width"]}x'
            f'{cfg["height"]}'
        ),
        "audio": audio_enabled,
        "signature": signature,
        "visual_parameters": (
            visual_params
        ),
        "experiment_id": "prime-gap-visual-pilot" if g["id"].upper() == "P09" else None,
        "learning_objective": (
            "Show that prime numbers are greater than 1, have exactly two positive divisors, "
            "and that gaps between consecutive primes vary."
            if g["id"].upper() == "P09" else None
        ),
    }

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # Save signature only AFTER successful rendering.
    # -----------------------------------------------------

    save_generation_signature(
        signature
    )

    print()
    print(
        "Generation signature saved."
    )

    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # YouTube upload and scheduling
    # -----------------------------------------------------
    # Keep final_video on disk until both YouTube upload and
    # Telegram delivery have had a chance to use it.
    telegram_notify(
        "✅ RawSignal video generated\n"
        f"Title: {ttl}\n"
        f"Generator: {g['id']}\n"
        f"Data type: {g['data_label']}\n"
        f"Seed: {seed}\n"
        f"Duration: {duration} seconds\n"
        f"Resolution: {cfg['width']}x{cfg['height']}"
    )

    try:
        upload_result = upload_and_schedule(
            final_video,
            metadata,
        )
        print(
            "YouTube upload/scheduling completed: "
            f"{upload_result.get('youtube_video_id', 'already scheduled')}"
        )
    except Exception as exc:
        # Do not prevent the existing Telegram video delivery if
        # YouTube configuration, quota, or upload fails.
        error_message = str(exc)
        print(f"YouTube upload/scheduling failed: {error_message}")
        telegram_notify(
            "❌ RawSignal YouTube upload/scheduling failed\n"
            f"Title: {ttl}\n"
            f"Generator: {g['id']}\n"
            f"Error: {error_message[:1200]}"
        )

    # -----------------------------------------------------
    # Telegram
    # -----------------------------------------------------

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

        # -------------------------------------------------
        # Delete only after Telegram delivery succeeds.
        # YouTube upload/scheduling was attempted above.
        # -------------------------------------------------

        final_video.unlink(
            missing_ok=True
        )

        print(
            "Video deleted from runner."
        )

    else:

        print()
        print(
            "Telegram upload skipped."
        )

    # -----------------------------------------------------
    # Final cleanup
    # -----------------------------------------------------

    raw_video.unlink(
        missing_ok=True
    )

    audio_file.unlink(
        missing_ok=True
    )

    print()
    print("=" * 60)
    print(
        "Generation completed successfully."
    )
    print("=" * 60)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
















# import argparse
# import json
# import os
# import secrets
# import subprocess
# import wave
# from pathlib import Path

# import numpy as np

# from .youtube_uploader import (
#     upload_and_schedule,
#     telegram_notify,
# )

# from .generators import (
#     all_generators,
#     find_generator,
#     make_data,
#     values,
#     title,
#     frame,
# )
# from .telegram import send_video


# # =========================================================
# # PATHS
# # =========================================================

# ROOT = Path(__file__).resolve().parents[1]

# OUT = ROOT / "output"
# TMP = ROOT / "tmp"
# HISTORY_FILE = ROOT / "data" / "generation_history.json"

# OUT.mkdir(exist_ok=True)
# TMP.mkdir(exist_ok=True)
# HISTORY_FILE.parent.mkdir(exist_ok=True)


# # =========================================================
# # GENERATION HISTORY
# # =========================================================

# def load_generation_history():
#     """
#     Load previously generated signatures.

#     The history file is intentionally small and contains
#     only generation signatures, not videos.
#     """

#     if not HISTORY_FILE.exists():
#         HISTORY_FILE.write_text(
#             json.dumps(
#                 {"signatures": []},
#                 indent=2,
#             ),
#             encoding="utf-8",
#         )

#         return []

#     try:
#         data = json.loads(
#             HISTORY_FILE.read_text(
#                 encoding="utf-8"
#             )
#         )

#         signatures = data.get(
#             "signatures",
#             [],
#         )

#         if not isinstance(signatures, list):
#             return []

#         return signatures

#     except (json.JSONDecodeError, OSError):
#         print(
#             "Warning: generation history could not "
#             "be read. Starting with empty history."
#         )

#         return []


# def save_generation_signature(signature):
#     """
#     Add a successful generation signature to history.

#     History is capped at 10,000 entries so the JSON file
#     remains small.
#     """

#     signatures = load_generation_history()

#     if signature not in signatures:
#         signatures.append(signature)

#     signatures = signatures[-10000:]

#     HISTORY_FILE.write_text(
#         json.dumps(
#             {
#                 "signatures": signatures,
#             },
#             indent=2,
#         ),
#         encoding="utf-8",
#     )


# def create_generation_signature(
#     generator_id,
#     seed,
#     duration,
#     params,
# ):
#     """
#     Create a deterministic signature for the complete
#     generation configuration.

#     Same generator + same seed + same duration +
#     same visual parameters = same generation.
#     """

#     return (
#         f"{generator_id}|"
#         f"{seed}|"
#         f"{duration}|"
#         f"{params['density']:.8f}|"
#         f"{params['chaos']:.8f}|"
#         f"{params['spin']:.8f}"
#     )


# # =========================================================
# # DURATION
# # =========================================================

# def choose_duration(cfg, rng):
#     """
#     Select a variable duration between the configured
#     minimum and maximum.

#     Beta distribution makes medium-length videos more
#     common while still allowing short and long videos.
#     """

#     minimum = int(
#         cfg["min_duration_seconds"]
#     )

#     maximum = int(
#         cfg["max_duration_seconds"]
#     )

#     if maximum <= minimum:
#         return minimum

#     duration = (
#         minimum
#         + (
#             maximum - minimum
#         )
#         * rng.beta(2, 2.8)
#     )

#     return int(round(duration))


# # =========================================================
# # VISUAL PARAMETERS
# # =========================================================

# def generate_visual_parameters(seed):
#     """
#     Generate deterministic visual parameters from the seed.

#     The same seed always produces the same parameters.
#     """

#     rng = np.random.default_rng(
#         seed + 77
#     )

#     return {
#         "density": float(
#             rng.uniform(
#                 0.35,
#                 1.0,
#             )
#         ),
#         "chaos": float(
#             rng.uniform(
#                 0.2,
#                 1.0,
#             )
#         ),
#         "spin": float(
#             rng.uniform(
#                 0.2,
#                 1.8,
#             )
#         ),
#     }


# # =========================================================
# # VIDEO RENDERING
# # =========================================================

# def render_video(
#     g,
#     cfg,
#     seed,
#     data_vals,
#     duration,
#     out,
#     params,
#     raw_data=None,
# ):
#     fps = int(
#         cfg["fps"]
#     )

#     width = int(
#         cfg["width"]
#     )

#     height = int(
#         cfg["height"]
#     )

#     total_frames = max(
#         1,
#         int(duration * fps),
#     )

#     command = [
#         "ffmpeg",
#         "-y",
#         "-loglevel",
#         "error",
#         "-f",
#         "rawvideo",
#         "-pix_fmt",
#         "rgb24",
#         "-s",
#         f"{width}x{height}",
#         "-r",
#         str(fps),
#         "-i",
#         "-",
#         "-an",
#         "-c:v",
#         "libx264",
#         "-preset",
#         cfg.get(
#             "preset",
#             "veryfast",
#         ),
#         "-crf",
#         str(
#             cfg.get(
#                 "crf",
#                 23,
#             )
#         ),
#         "-pix_fmt",
#         "yuv420p",
#         "-movflags",
#         "+faststart",
#         str(out),
#     ]

#     process = subprocess.Popen(
#         command,
#         stdin=subprocess.PIPE,
#     )

#     try:
#         for i in range(total_frames):

#             progress = (
#                 i
#                 / max(
#                     1,
#                     total_frames - 1,
#                 )
#             )

#             image = frame(
#                 width,
#                 height,
#                 progress,
#                 data_vals,
#                 g["data_type"],
#                 g["method"],
#                 seed,
#                 params,
#                 raw_data=raw_data,
#             )

#             process.stdin.write(
#                 np.asarray(
#                     image,
#                     dtype=np.uint8,
#                 ).tobytes()
#             )

#     finally:

#         if process.stdin:
#             process.stdin.close()

#         return_code = process.wait()

#     if return_code != 0:
#         raise RuntimeError(
#             "FFmpeg video render failed."
#         )


# # =========================================================
# # DATA-DERIVED AUDIO
# # =========================================================

# def audio(
#     g,
#     vals,
#     duration,
#     rate,
#     seed,
#     out,
# ):
#     total_samples = int(
#         duration * rate
#     )

#     values_array = np.resize(
#         np.asarray(
#             vals,
#             dtype=float,
#         ),
#         256,
#     )

#     base_frequencies = {
#         "frequency": 220,
#         "morse": 120,
#         "dna": 180,
#         "binary": 90,
#         "hex": 130,
#         "hashes": 100,
#     }

#     base_frequency = base_frequencies.get(
#         g["data_type"],
#         145,
#     )

#     chunk_size = max(
#         1,
#         rate * 2,
#     )

#     phase0 = 0.0

#     with wave.open(
#         str(out),
#         "wb",
#     ) as wav:

#         wav.setnchannels(1)
#         wav.setsampwidth(2)
#         wav.setframerate(rate)

#         for start in range(
#             0,
#             total_samples,
#             chunk_size,
#         ):

#             count = min(
#                 chunk_size,
#                 total_samples - start,
#             )

#             positions = np.arange(
#                 start,
#                 start + count,
#             )

#             if total_samples > 1:

#                 normalized_positions = (
#                     np.linspace(
#                         0,
#                         total_samples - 1,
#                         256,
#                     )
#                 )

#                 control = np.interp(
#                     positions,
#                     normalized_positions,
#                     values_array,
#                 )

#             else:

#                 control = np.full(
#                     count,
#                     values_array[0],
#                 )

#             time = positions / rate

#             frequency = (
#                 base_frequency
#                 * (
#                     0.55
#                     + 1.5 * control
#                 )
#             )

#             phase = (
#                 phase0
#                 + 2
#                 * np.pi
#                 * np.cumsum(
#                     frequency
#                 )
#                 / rate
#             )

#             phase0 = float(
#                 phase[-1]
#             )

#             signal = (
#                 0.18
#                 * np.sin(phase)
#             )

#             signal += (
#                 0.08
#                 * np.sin(
#                     phase * 2.01
#                     + control * 4
#                 )
#             )

#             signal += (
#                 0.04
#                 * np.sin(
#                     phase * 3.01
#                 )
#             )

#             # Data-controlled gate
#             gate = (
#                 control > 0.62
#             ).astype(float)

#             kernel_size = max(
#                 1,
#                 rate // 120,
#             )

#             kernel_size = min(
#                 kernel_size,
#                 count,
#             )

#             if kernel_size > 1:

#                 gate = np.convolve(
#                     gate,
#                     np.ones(
#                         kernel_size
#                     )
#                     / kernel_size,
#                     mode="same",
#                 )

#             signal += (
#                 0.1
#                 * gate
#                 * np.sin(
#                     2
#                     * np.pi
#                     * base_frequency
#                     * 2
#                     * time
#                 )
#             )

#             # -------------------------------------------------
#             # Fade in
#             # -------------------------------------------------

#             if start == 0:

#                 fade_samples = min(
#                     count,
#                     rate * 2,
#                 )

#                 envelope = np.ones(
#                     count
#                 )

#                 if fade_samples > 1:

#                     envelope[
#                         :fade_samples
#                     ] = np.linspace(
#                         0,
#                         1,
#                         fade_samples,
#                     )

#             # -------------------------------------------------
#             # Fade out
#             # -------------------------------------------------

#             elif (
#                 start + count
#                 >= total_samples
#                 - rate * 2
#             ):

#                 fade_start = max(
#                     0,
#                     total_samples
#                     - rate * 2
#                     - start,
#                 )

#                 envelope = np.ones(
#                     count
#                 )

#                 fade_length = (
#                     count
#                     - fade_start
#                 )

#                 if fade_length > 1:

#                     envelope[
#                         fade_start:
#                     ] = np.linspace(
#                         1,
#                         0,
#                         fade_length,
#                     )

#             else:

#                 envelope = np.ones(
#                     count
#                 )

#             signal = np.clip(
#                 signal
#                 * envelope
#                 * 0.9,
#                 -0.95,
#                 0.95,
#             )

#             wav.writeframes(
#                 (
#                     signal * 32767
#                 )
#                 .astype(
#                     np.int16
#                 )
#                 .tobytes()
#             )


# # =========================================================
# # UNIQUE GENERATION SELECTION
# # =========================================================

# def create_unique_generation(
#     cfg,
#     requested_generator,
#     history,
#     max_attempts=50,
# ):
#     """
#     Generate a new configuration that does not already
#     exist in generation_history.json.
#     """

#     for attempt in range(
#         1,
#         max_attempts + 1,
#     ):

#         # -------------------------------------------------
#         # New cryptographically strong random seed
#         # -------------------------------------------------

#         seed = secrets.randbelow(
#             2**31 - 1
#         )

#         rng = np.random.default_rng(
#             seed
#         )

#         # -------------------------------------------------
#         # Generator
#         # -------------------------------------------------

#         if requested_generator:

#             g = find_generator(
#                 requested_generator
#             )

#             if g is None:
#                 raise ValueError(
#                     "Unknown generator: "
#                     f"{requested_generator}"
#                 )

#         else:

#             generators = all_generators()

#             if not generators:
#                 raise RuntimeError(
#                     "No generators are available."
#                 )

#             g = rng.choice(
#                 generators
#             )

#         # -------------------------------------------------
#         # Duration
#         # -------------------------------------------------

#         duration = choose_duration(
#             cfg,
#             rng,
#         )

#         # -------------------------------------------------
#         # Visual parameters
#         # -------------------------------------------------

#         visual_params = (
#             generate_visual_parameters(
#                 seed
#             )
#         )

#         # -------------------------------------------------
#         # Signature
#         # -------------------------------------------------

#         signature = (
#             create_generation_signature(
#                 g["id"],
#                 seed,
#                 duration,
#                 visual_params,
#             )
#         )

#         # -------------------------------------------------
#         # Duplicate check
#         # -------------------------------------------------

#         if signature not in history:

#             return (
#                 g,
#                 seed,
#                 duration,
#                 visual_params,
#                 signature,
#             )

#         print(
#             f"Duplicate detected. "
#             f"Retry {attempt}/{max_attempts}..."
#         )

#     raise RuntimeError(
#         "Unable to create a unique generation "
#         f"after {max_attempts} attempts."
#     )


# # =========================================================
# # MAIN
# # =========================================================

# def main():

#     # -----------------------------------------------------
#     # Arguments
#     # -----------------------------------------------------

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--generator",
#         default="",
#         help=(
#             "Generator ID. "
#             "Blank = random."
#         ),
#     )

#     args = parser.parse_args()

#     # -----------------------------------------------------
#     # Configuration
#     # -----------------------------------------------------

#     config_path = (
#         ROOT / "config.json"
#     )

#     if not config_path.exists():

#         raise FileNotFoundError(
#             "Configuration file not found: "
#             f"{config_path}"
#         )

#     cfg = json.loads(
#         config_path.read_text(
#             encoding="utf-8"
#         )
#     )

#     # -----------------------------------------------------
#     # Load duplicate history
#     # -----------------------------------------------------

#     history = (
#         load_generation_history()
#     )

#     print(
#         f"Generation history: "
#         f"{len(history)} entries"
#     )

#     # -----------------------------------------------------
#     # Select unique generation
#     # -----------------------------------------------------

#     (
#         g,
#         seed,
#         duration,
#         visual_params,
#         signature,
#     ) = create_unique_generation(
#         cfg,
#         args.generator,
#         history,
#     )

#     # -----------------------------------------------------
#     # Generate data
#     # -----------------------------------------------------

#     rng = np.random.default_rng(
#         seed
#     )

#     data = make_data(
#         g["data_type"],
#         rng,
#     )

#     data_vals = values(
#         g["data_type"],
#         data,
#     )

#     # -----------------------------------------------------
#     # Generate title
#     # -----------------------------------------------------

#     ttl = title(
#         g,
#         rng,
#     )

#     # -----------------------------------------------------
#     # File names
#     # -----------------------------------------------------

#     stem = (
#         f'{g["id"]}_{seed}'
#     )

#     raw_video = (
#         TMP
#         / f"{stem}_raw.mp4"
#     )

#     audio_file = (
#         TMP
#         / f"{stem}.wav"
#     )

#     final_video = (
#         OUT
#         / f"{stem}.mp4"
#     )

#     metadata_file = (
#         OUT
#         / f"{stem}.json"
#     )

#     # -----------------------------------------------------
#     # Generation information
#     # -----------------------------------------------------

#     print()
#     print("=" * 60)
#     print("RawSignal Generation")
#     print("=" * 60)

#     print(
#         f"Generator : {g['id']}"
#     )

#     print(
#         f"Data      : {g['data_label']}"
#     )

#     print(
#         f"Seed      : {seed}"
#     )

#     print(
#         f"Duration  : {duration}s"
#     )

#     print(
#         f"Resolution: "
#         f"{cfg['width']}x{cfg['height']}"
#     )

#     print(
#         f"FPS       : {cfg['fps']}"
#     )

#     print(
#         f"Signature : {signature}"
#     )

#     print("=" * 60)
#     print()

#     # -----------------------------------------------------
#     # Render video
#     # -----------------------------------------------------

#     render_video(
#         g,
#         cfg,
#         seed,
#         data_vals,
#         duration,
#         raw_video,
#         visual_params,
#         raw_data=data,
#     )

#     # -----------------------------------------------------
#     # Audio
#     # -----------------------------------------------------

#     audio_enabled = bool(
#         cfg.get(
#             "audio_enabled",
#             True,
#         )
#     )

#     if audio_enabled:

#         audio(
#             g,
#             data_vals,
#             duration,
#             int(
#                 cfg.get(
#                     "audio_sample_rate",
#                     44100,
#                 )
#             ),
#             seed,
#             audio_file,
#         )

#         # -------------------------------------------------
#         # Combine video + audio
#         # -------------------------------------------------

#         subprocess.run(
#             [
#                 "ffmpeg",
#                 "-y",
#                 "-loglevel",
#                 "error",
#                 "-i",
#                 str(raw_video),
#                 "-i",
#                 str(audio_file),
#                 "-map",
#                 "0:v:0",
#                 "-map",
#                 "1:a:0",
#                 "-c:v",
#                 "copy",
#                 "-c:a",
#                 "aac",
#                 "-b:a",
#                 "160k",
#                 "-shortest",
#                 "-movflags",
#                 "+faststart",
#                 str(final_video),
#             ],
#             check=True,
#         )

#         raw_video.unlink(
#             missing_ok=True
#         )

#         audio_file.unlink(
#             missing_ok=True
#         )

#     else:

#         raw_video.replace(
#             final_video
#         )

#     # -----------------------------------------------------
#     # Save metadata
#     # -----------------------------------------------------

#     metadata = {
#         "title": ttl,
#         "generator": g["id"],
#         "data_type": g["data_type"],
#         "data_label": g["data_label"],
#         "visual_method": g["method"],
#         "visual_method_label": g[
#             "method_label"
#         ],
#         "seed": seed,
#         "duration_seconds": duration,
#         "fps": cfg["fps"],
#         "resolution": (
#             f'{cfg["width"]}x'
#             f'{cfg["height"]}'
#         ),
#         "audio": audio_enabled,
#         "signature": signature,
#         "visual_parameters": (
#             visual_params
#         ),
#         "experiment_id": "prime-gap-visual-pilot" if g["id"].upper() == "P09" else None,
#         "learning_objective": (
#             "Show that prime numbers are greater than 1, have exactly two positive divisors, "
#             "and that gaps between consecutive primes vary."
#             if g["id"].upper() == "P09" else None
#         ),
#     }

#     metadata_file.write_text(
#         json.dumps(
#             metadata,
#             indent=2,
#         ),
#         encoding="utf-8",
#     )

#     # -----------------------------------------------------
#     # IMPORTANT:
#     # Save signature only AFTER successful rendering.
#     # -----------------------------------------------------

#     save_generation_signature(
#         signature
#     )

#     print()
#     print(
#         "Generation signature saved."
#     )

#     print(
#         json.dumps(
#             metadata,
#             indent=2,
#         )
#     )

#     # -----------------------------------------------------
#     # YouTube upload and scheduling
#     # -----------------------------------------------------
#     # Keep final_video on disk until both YouTube upload and
#     # Telegram delivery have had a chance to use it.
#     telegram_notify(
#         "✅ RawSignal video generated\n"
#         f"Title: {ttl}\n"
#         f"Generator: {g['id']}\n"
#         f"Data type: {g['data_label']}\n"
#         f"Seed: {seed}\n"
#         f"Duration: {duration} seconds\n"
#         f"Resolution: {cfg['width']}x{cfg['height']}"
#     )

#     try:
#         upload_result = upload_and_schedule(
#             final_video,
#             metadata,
#         )
#         print(
#             "YouTube upload/scheduling completed: "
#             f"{upload_result.get('youtube_video_id', 'already scheduled')}"
#         )
#     except Exception as exc:
#         # Do not prevent the existing Telegram video delivery if
#         # YouTube configuration, quota, or upload fails.
#         error_message = str(exc)
#         print(f"YouTube upload/scheduling failed: {error_message}")
#         telegram_notify(
#             "❌ RawSignal YouTube upload/scheduling failed\n"
#             f"Title: {ttl}\n"
#             f"Generator: {g['id']}\n"
#             f"Error: {error_message[:1200]}"
#         )

#     # -----------------------------------------------------
#     # Telegram
#     # -----------------------------------------------------

#     token = os.getenv(
#         "TELEGRAM_BOT_TOKEN"
#     )

#     chat = os.getenv(
#         "TELEGRAM_CHAT_ID"
#     )

#     send_to_telegram = bool(
#         cfg.get(
#             "send_to_telegram",
#             True,
#         )
#     )

#     if (
#         send_to_telegram
#         and token
#         and chat
#     ):

#         print()
#         print(
#             "Sending video to Telegram..."
#         )

#         send_video(
#             token,
#             chat,
#             final_video,
#             ttl,
#             g["id"],
#             g["data_label"],
#             seed,
#         )

#         print(
#             "Telegram upload successful."
#         )

#         # -------------------------------------------------
#         # Delete only after Telegram delivery succeeds.
#         # YouTube upload/scheduling was attempted above.
#         # -------------------------------------------------

#         final_video.unlink(
#             missing_ok=True
#         )

#         print(
#             "Video deleted from runner."
#         )

#     else:

#         print()
#         print(
#             "Telegram upload skipped."
#         )

#     # -----------------------------------------------------
#     # Final cleanup
#     # -----------------------------------------------------

#     raw_video.unlink(
#         missing_ok=True
#     )

#     audio_file.unlink(
#         missing_ok=True
#     )

#     print()
#     print("=" * 60)
#     print(
#         "Generation completed successfully."
#     )
#     print("=" * 60)


# # =========================================================
# # ENTRY POINT
# # =========================================================

# if __name__ == "__main__":
#     main()




















# import argparse
# import json
# import os
# import secrets
# import subprocess
# import wave
# from pathlib import Path

# import numpy as np

# from .youtube_uploader import (
#     upload_and_schedule,
#     telegram_notify,
# )

# from .generators import (
#     all_generators,
#     find_generator,
#     make_data,
#     values,
#     title,
#     frame,
# )
# from .telegram import send_video


# # =========================================================
# # PATHS
# # =========================================================

# ROOT = Path(__file__).resolve().parents[1]

# OUT = ROOT / "output"
# TMP = ROOT / "tmp"
# HISTORY_FILE = ROOT / "data" / "generation_history.json"

# OUT.mkdir(exist_ok=True)
# TMP.mkdir(exist_ok=True)
# HISTORY_FILE.parent.mkdir(exist_ok=True)


# # =========================================================
# # GENERATION HISTORY
# # =========================================================

# def load_generation_history():
#     """
#     Load previously generated signatures.

#     The history file is intentionally small and contains
#     only generation signatures, not videos.
#     """

#     if not HISTORY_FILE.exists():
#         HISTORY_FILE.write_text(
#             json.dumps(
#                 {"signatures": []},
#                 indent=2,
#             ),
#             encoding="utf-8",
#         )

#         return []

#     try:
#         data = json.loads(
#             HISTORY_FILE.read_text(
#                 encoding="utf-8"
#             )
#         )

#         signatures = data.get(
#             "signatures",
#             [],
#         )

#         if not isinstance(signatures, list):
#             return []

#         return signatures

#     except (json.JSONDecodeError, OSError):
#         print(
#             "Warning: generation history could not "
#             "be read. Starting with empty history."
#         )

#         return []


# def save_generation_signature(signature):
#     """
#     Add a successful generation signature to history.

#     History is capped at 10,000 entries so the JSON file
#     remains small.
#     """

#     signatures = load_generation_history()

#     if signature not in signatures:
#         signatures.append(signature)

#     signatures = signatures[-10000:]

#     HISTORY_FILE.write_text(
#         json.dumps(
#             {
#                 "signatures": signatures,
#             },
#             indent=2,
#         ),
#         encoding="utf-8",
#     )


# def create_generation_signature(
#     generator_id,
#     seed,
#     duration,
#     params,
# ):
#     """
#     Create a deterministic signature for the complete
#     generation configuration.

#     Same generator + same seed + same duration +
#     same visual parameters = same generation.
#     """

#     return (
#         f"{generator_id}|"
#         f"{seed}|"
#         f"{duration}|"
#         f"{params['density']:.8f}|"
#         f"{params['chaos']:.8f}|"
#         f"{params['spin']:.8f}"
#     )


# # =========================================================
# # DURATION
# # =========================================================

# def choose_duration(cfg, rng):
#     """
#     Select a variable duration between the configured
#     minimum and maximum.

#     Beta distribution makes medium-length videos more
#     common while still allowing short and long videos.
#     """

#     minimum = int(
#         cfg["min_duration_seconds"]
#     )

#     maximum = int(
#         cfg["max_duration_seconds"]
#     )

#     if maximum <= minimum:
#         return minimum

#     duration = (
#         minimum
#         + (
#             maximum - minimum
#         )
#         * rng.beta(2, 2.8)
#     )

#     return int(round(duration))


# # =========================================================
# # VISUAL PARAMETERS
# # =========================================================

# def generate_visual_parameters(seed):
#     """
#     Generate deterministic visual parameters from the seed.

#     The same seed always produces the same parameters.
#     """

#     rng = np.random.default_rng(
#         seed + 77
#     )

#     return {
#         "density": float(
#             rng.uniform(
#                 0.35,
#                 1.0,
#             )
#         ),
#         "chaos": float(
#             rng.uniform(
#                 0.2,
#                 1.0,
#             )
#         ),
#         "spin": float(
#             rng.uniform(
#                 0.2,
#                 1.8,
#             )
#         ),
#     }


# # =========================================================
# # VIDEO RENDERING
# # =========================================================

# def render_video(
#     g,
#     cfg,
#     seed,
#     data_vals,
#     duration,
#     out,
#     params,
# ):
#     fps = int(
#         cfg["fps"]
#     )

#     width = int(
#         cfg["width"]
#     )

#     height = int(
#         cfg["height"]
#     )

#     total_frames = max(
#         1,
#         int(duration * fps),
#     )

#     command = [
#         "ffmpeg",
#         "-y",
#         "-loglevel",
#         "error",
#         "-f",
#         "rawvideo",
#         "-pix_fmt",
#         "rgb24",
#         "-s",
#         f"{width}x{height}",
#         "-r",
#         str(fps),
#         "-i",
#         "-",
#         "-an",
#         "-c:v",
#         "libx264",
#         "-preset",
#         cfg.get(
#             "preset",
#             "veryfast",
#         ),
#         "-crf",
#         str(
#             cfg.get(
#                 "crf",
#                 23,
#             )
#         ),
#         "-pix_fmt",
#         "yuv420p",
#         "-movflags",
#         "+faststart",
#         str(out),
#     ]

#     process = subprocess.Popen(
#         command,
#         stdin=subprocess.PIPE,
#     )

#     try:
#         for i in range(total_frames):

#             progress = (
#                 i
#                 / max(
#                     1,
#                     total_frames - 1,
#                 )
#             )

#             image = frame(
#                 width,
#                 height,
#                 progress,
#                 data_vals,
#                 g["data_type"],
#                 g["method"],
#                 seed,
#                 params,
#             )

#             process.stdin.write(
#                 np.asarray(
#                     image,
#                     dtype=np.uint8,
#                 ).tobytes()
#             )

#     finally:

#         if process.stdin:
#             process.stdin.close()

#         return_code = process.wait()

#     if return_code != 0:
#         raise RuntimeError(
#             "FFmpeg video render failed."
#         )


# # =========================================================
# # DATA-DERIVED AUDIO
# # =========================================================

# def audio(
#     g,
#     vals,
#     duration,
#     rate,
#     seed,
#     out,
# ):
#     total_samples = int(
#         duration * rate
#     )

#     values_array = np.resize(
#         np.asarray(
#             vals,
#             dtype=float,
#         ),
#         256,
#     )

#     base_frequencies = {
#         "frequency": 220,
#         "morse": 120,
#         "dna": 180,
#         "binary": 90,
#         "hex": 130,
#         "hashes": 100,
#     }

#     base_frequency = base_frequencies.get(
#         g["data_type"],
#         145,
#     )

#     chunk_size = max(
#         1,
#         rate * 2,
#     )

#     phase0 = 0.0

#     with wave.open(
#         str(out),
#         "wb",
#     ) as wav:

#         wav.setnchannels(1)
#         wav.setsampwidth(2)
#         wav.setframerate(rate)

#         for start in range(
#             0,
#             total_samples,
#             chunk_size,
#         ):

#             count = min(
#                 chunk_size,
#                 total_samples - start,
#             )

#             positions = np.arange(
#                 start,
#                 start + count,
#             )

#             if total_samples > 1:

#                 normalized_positions = (
#                     np.linspace(
#                         0,
#                         total_samples - 1,
#                         256,
#                     )
#                 )

#                 control = np.interp(
#                     positions,
#                     normalized_positions,
#                     values_array,
#                 )

#             else:

#                 control = np.full(
#                     count,
#                     values_array[0],
#                 )

#             time = positions / rate

#             frequency = (
#                 base_frequency
#                 * (
#                     0.55
#                     + 1.5 * control
#                 )
#             )

#             phase = (
#                 phase0
#                 + 2
#                 * np.pi
#                 * np.cumsum(
#                     frequency
#                 )
#                 / rate
#             )

#             phase0 = float(
#                 phase[-1]
#             )

#             signal = (
#                 0.18
#                 * np.sin(phase)
#             )

#             signal += (
#                 0.08
#                 * np.sin(
#                     phase * 2.01
#                     + control * 4
#                 )
#             )

#             signal += (
#                 0.04
#                 * np.sin(
#                     phase * 3.01
#                 )
#             )

#             # Data-controlled gate
#             gate = (
#                 control > 0.62
#             ).astype(float)

#             kernel_size = max(
#                 1,
#                 rate // 120,
#             )

#             kernel_size = min(
#                 kernel_size,
#                 count,
#             )

#             if kernel_size > 1:

#                 gate = np.convolve(
#                     gate,
#                     np.ones(
#                         kernel_size
#                     )
#                     / kernel_size,
#                     mode="same",
#                 )

#             signal += (
#                 0.1
#                 * gate
#                 * np.sin(
#                     2
#                     * np.pi
#                     * base_frequency
#                     * 2
#                     * time
#                 )
#             )

#             # -------------------------------------------------
#             # Fade in
#             # -------------------------------------------------

#             if start == 0:

#                 fade_samples = min(
#                     count,
#                     rate * 2,
#                 )

#                 envelope = np.ones(
#                     count
#                 )

#                 if fade_samples > 1:

#                     envelope[
#                         :fade_samples
#                     ] = np.linspace(
#                         0,
#                         1,
#                         fade_samples,
#                     )

#             # -------------------------------------------------
#             # Fade out
#             # -------------------------------------------------

#             elif (
#                 start + count
#                 >= total_samples
#                 - rate * 2
#             ):

#                 fade_start = max(
#                     0,
#                     total_samples
#                     - rate * 2
#                     - start,
#                 )

#                 envelope = np.ones(
#                     count
#                 )

#                 fade_length = (
#                     count
#                     - fade_start
#                 )

#                 if fade_length > 1:

#                     envelope[
#                         fade_start:
#                     ] = np.linspace(
#                         1,
#                         0,
#                         fade_length,
#                     )

#             else:

#                 envelope = np.ones(
#                     count
#                 )

#             signal = np.clip(
#                 signal
#                 * envelope
#                 * 0.9,
#                 -0.95,
#                 0.95,
#             )

#             wav.writeframes(
#                 (
#                     signal * 32767
#                 )
#                 .astype(
#                     np.int16
#                 )
#                 .tobytes()
#             )


# # =========================================================
# # UNIQUE GENERATION SELECTION
# # =========================================================

# def create_unique_generation(
#     cfg,
#     requested_generator,
#     history,
#     max_attempts=50,
# ):
#     """
#     Generate a new configuration that does not already
#     exist in generation_history.json.
#     """

#     for attempt in range(
#         1,
#         max_attempts + 1,
#     ):

#         # -------------------------------------------------
#         # New cryptographically strong random seed
#         # -------------------------------------------------

#         seed = secrets.randbelow(
#             2**31 - 1
#         )

#         rng = np.random.default_rng(
#             seed
#         )

#         # -------------------------------------------------
#         # Generator
#         # -------------------------------------------------

#         if requested_generator:

#             g = find_generator(
#                 requested_generator
#             )

#             if g is None:
#                 raise ValueError(
#                     "Unknown generator: "
#                     f"{requested_generator}"
#                 )

#         else:

#             generators = all_generators()

#             if not generators:
#                 raise RuntimeError(
#                     "No generators are available."
#                 )

#             g = rng.choice(
#                 generators
#             )

#         # -------------------------------------------------
#         # Duration
#         # -------------------------------------------------

#         duration = choose_duration(
#             cfg,
#             rng,
#         )

#         # -------------------------------------------------
#         # Visual parameters
#         # -------------------------------------------------

#         visual_params = (
#             generate_visual_parameters(
#                 seed
#             )
#         )

#         # -------------------------------------------------
#         # Signature
#         # -------------------------------------------------

#         signature = (
#             create_generation_signature(
#                 g["id"],
#                 seed,
#                 duration,
#                 visual_params,
#             )
#         )

#         # -------------------------------------------------
#         # Duplicate check
#         # -------------------------------------------------

#         if signature not in history:

#             return (
#                 g,
#                 seed,
#                 duration,
#                 visual_params,
#                 signature,
#             )

#         print(
#             f"Duplicate detected. "
#             f"Retry {attempt}/{max_attempts}..."
#         )

#     raise RuntimeError(
#         "Unable to create a unique generation "
#         f"after {max_attempts} attempts."
#     )


# # =========================================================
# # MAIN
# # =========================================================

# def main():

#     # -----------------------------------------------------
#     # Arguments
#     # -----------------------------------------------------

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--generator",
#         default="",
#         help=(
#             "Generator ID. "
#             "Blank = random."
#         ),
#     )

#     args = parser.parse_args()

#     # -----------------------------------------------------
#     # Configuration
#     # -----------------------------------------------------

#     config_path = (
#         ROOT / "config.json"
#     )

#     if not config_path.exists():

#         raise FileNotFoundError(
#             "Configuration file not found: "
#             f"{config_path}"
#         )

#     cfg = json.loads(
#         config_path.read_text(
#             encoding="utf-8"
#         )
#     )

#     # -----------------------------------------------------
#     # Load duplicate history
#     # -----------------------------------------------------

#     history = (
#         load_generation_history()
#     )

#     print(
#         f"Generation history: "
#         f"{len(history)} entries"
#     )

#     # -----------------------------------------------------
#     # Select unique generation
#     # -----------------------------------------------------

#     (
#         g,
#         seed,
#         duration,
#         visual_params,
#         signature,
#     ) = create_unique_generation(
#         cfg,
#         args.generator,
#         history,
#     )

#     # -----------------------------------------------------
#     # Generate data
#     # -----------------------------------------------------

#     rng = np.random.default_rng(
#         seed
#     )

#     data = make_data(
#         g["data_type"],
#         rng,
#     )

#     data_vals = values(
#         g["data_type"],
#         data,
#     )

#     # -----------------------------------------------------
#     # Generate title
#     # -----------------------------------------------------

#     ttl = title(
#         g,
#         rng,
#     )

#     # -----------------------------------------------------
#     # File names
#     # -----------------------------------------------------

#     stem = (
#         f'{g["id"]}_{seed}'
#     )

#     raw_video = (
#         TMP
#         / f"{stem}_raw.mp4"
#     )

#     audio_file = (
#         TMP
#         / f"{stem}.wav"
#     )

#     final_video = (
#         OUT
#         / f"{stem}.mp4"
#     )

#     metadata_file = (
#         OUT
#         / f"{stem}.json"
#     )

#     # -----------------------------------------------------
#     # Generation information
#     # -----------------------------------------------------

#     print()
#     print("=" * 60)
#     print("RawSignal Generation")
#     print("=" * 60)

#     print(
#         f"Generator : {g['id']}"
#     )

#     print(
#         f"Data      : {g['data_label']}"
#     )

#     print(
#         f"Seed      : {seed}"
#     )

#     print(
#         f"Duration  : {duration}s"
#     )

#     print(
#         f"Resolution: "
#         f"{cfg['width']}x{cfg['height']}"
#     )

#     print(
#         f"FPS       : {cfg['fps']}"
#     )

#     print(
#         f"Signature : {signature}"
#     )

#     print("=" * 60)
#     print()

#     # -----------------------------------------------------
#     # Render video
#     # -----------------------------------------------------

#     render_video(
#         g,
#         cfg,
#         seed,
#         data_vals,
#         duration,
#         raw_video,
#         visual_params,
#     )

#     # -----------------------------------------------------
#     # Audio
#     # -----------------------------------------------------

#     audio_enabled = bool(
#         cfg.get(
#             "audio_enabled",
#             True,
#         )
#     )

#     if audio_enabled:

#         audio(
#             g,
#             data_vals,
#             duration,
#             int(
#                 cfg.get(
#                     "audio_sample_rate",
#                     44100,
#                 )
#             ),
#             seed,
#             audio_file,
#         )

#         # -------------------------------------------------
#         # Combine video + audio
#         # -------------------------------------------------

#         subprocess.run(
#             [
#                 "ffmpeg",
#                 "-y",
#                 "-loglevel",
#                 "error",
#                 "-i",
#                 str(raw_video),
#                 "-i",
#                 str(audio_file),
#                 "-map",
#                 "0:v:0",
#                 "-map",
#                 "1:a:0",
#                 "-c:v",
#                 "copy",
#                 "-c:a",
#                 "aac",
#                 "-b:a",
#                 "160k",
#                 "-shortest",
#                 "-movflags",
#                 "+faststart",
#                 str(final_video),
#             ],
#             check=True,
#         )

#         raw_video.unlink(
#             missing_ok=True
#         )

#         audio_file.unlink(
#             missing_ok=True
#         )

#     else:

#         raw_video.replace(
#             final_video
#         )

#     # -----------------------------------------------------
#     # Save metadata
#     # -----------------------------------------------------

#     metadata = {
#         "title": ttl,
#         "generator": g["id"],
#         "data_type": g["data_type"],
#         "data_label": g["data_label"],
#         "visual_method": g["method"],
#         "visual_method_label": g[
#             "method_label"
#         ],
#         "seed": seed,
#         "duration_seconds": duration,
#         "fps": cfg["fps"],
#         "resolution": (
#             f'{cfg["width"]}x'
#             f'{cfg["height"]}'
#         ),
#         "audio": audio_enabled,
#         "signature": signature,
#         "visual_parameters": (
#             visual_params
#         ),
#     }

#     metadata_file.write_text(
#         json.dumps(
#             metadata,
#             indent=2,
#         ),
#         encoding="utf-8",
#     )

#     # -----------------------------------------------------
#     # IMPORTANT:
#     # Save signature only AFTER successful rendering.
#     # -----------------------------------------------------

#     save_generation_signature(
#         signature
#     )

#     print()
#     print(
#         "Generation signature saved."
#     )

#     print(
#         json.dumps(
#             metadata,
#             indent=2,
#         )
#     )

#     # -----------------------------------------------------
#     # YouTube upload and scheduling
#     # -----------------------------------------------------
#     # Keep final_video on disk until both YouTube upload and
#     # Telegram delivery have had a chance to use it.
#     telegram_notify(
#         "✅ RawSignal video generated\\n"
#         f"Title: {ttl}\\n"
#         f"Generator: {g['id']}\\n"
#         f"Data type: {g['data_label']}\\n"
#         f"Seed: {seed}\\n"
#         f"Duration: {duration} seconds\\n"
#         f"Resolution: {cfg['width']}x{cfg['height']}"
#     )

#     youtube_upload_ok = False

#     try:
#         upload_result = upload_and_schedule(
#             final_video,
#             metadata,
#         )
#         youtube_upload_ok = True
#         print(
#             "YouTube upload/scheduling completed: "
#             f"{upload_result.get('youtube_video_id', 'already scheduled')}"
#         )
#     except Exception as exc:
#         # Do not prevent the existing Telegram video delivery if
#         # YouTube configuration, quota, or upload fails.
#         error_message = str(exc)
#         print(f"YouTube upload/scheduling failed: {error_message}")
#         telegram_notify(
#             "❌ RawSignal YouTube upload/scheduling failed\\n"
#             f"Title: {ttl}\\n"
#             f"Generator: {g['id']}\\n"
#             f"Error: {error_message[:1200]}"
#         )

#     # -----------------------------------------------------
#     # Telegram
#     # -----------------------------------------------------

#     token = os.getenv(
#         "TELEGRAM_BOT_TOKEN"
#     )

#     chat = os.getenv(
#         "TELEGRAM_CHAT_ID"
#     )

#     send_to_telegram = bool(
#         cfg.get(
#             "send_to_telegram",
#             True,
#         )
#     )

#     if (
#         send_to_telegram
#         and token
#         and chat
#     ):

#         print()
#         print(
#             "Sending video to Telegram..."
#         )

#         send_video(
#             token,
#             chat,
#             final_video,
#             ttl,
#             g["id"],
#             g["data_label"],
#             seed,
#         )

#         print(
#             "Telegram upload successful."
#         )

#         # -------------------------------------------------
#         # Delete only after Telegram delivery succeeds.
#         # YouTube upload/scheduling was attempted above.
#         # -------------------------------------------------

#         final_video.unlink(
#             missing_ok=True
#         )

#         print(
#             "Video deleted from runner."
#         )

#     else:

#         print()
#         print(
#             "Telegram upload skipped."
#         )

#     # -----------------------------------------------------
#     # Final cleanup
#     # -----------------------------------------------------

#     raw_video.unlink(
#         missing_ok=True
#     )

#     audio_file.unlink(
#         missing_ok=True
#     )

#     print()
#     print("=" * 60)
#     print(
#         "Generation completed successfully."
#     )
#     print("=" * 60)


# # =========================================================
# # ENTRY POINT
# # =========================================================

# if __name__ == "__main__":
#     main()
