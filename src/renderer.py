from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
from .generators import render_frame, make_data, GeneratorSpec


def render_video(spec: GeneratorSpec, seed: int, output: Path,
                 width: int, height: int, fps: int, duration: int,
                 crf: int, preset: str):
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = make_data(spec.data_type, rng, 8192)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)

    total = fps * duration
    try:
        for frame_no in range(total):
            img = render_frame(spec, data, frame_no, fps, (width,height), seed)
            proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        proc.stdin.close()
        rc = proc.wait()
    except Exception:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.kill()
        proc.wait()
        raise

    if rc != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg failed:\n{err[-4000:]}")

    return output
