from __future__ import annotations
import requests


def send_video(bot_token: str, chat_id: str, video_path: str, title: str,
               generator_id: str, data_type: str, seed: int):
    url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
    caption = (
        f"🎬 New video ready\n\n"
        f"Title: {title}\n"
        f"Generator: {generator_id}\n"
        f"Data: {data_type}\n"
        f"Seed: {seed}"
    )
    with open(video_path, "rb") as f:
        r = requests.post(
            url,
            data={"chat_id": chat_id, "caption": caption},
            files={"video": (video_path.name, f, "video/mp4")},
            timeout=180,
        )
    r.raise_for_status()
    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(payload)
    return payload
