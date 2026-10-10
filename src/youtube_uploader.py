from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "youtube_state.json"
IST = ZoneInfo("Asia/Kolkata")
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]
MIN_GAP_HOURS = int(os.getenv("YT_MIN_GAP_HOURS", "6"))
# Initial test slots, not guaranteed optimal times.
PREFERRED_SLOTS = [(18, 0), (21, 0)]
CATEGORY_ID = os.getenv("YT_CATEGORY_ID", "28")
DESCRIPTION = os.getenv(
    "YT_DEFAULT_DESCRIPTION",
    "RawSignal — where raw data becomes visual reality.\n"
    "Procedurally generated data-driven visuals and sound.\n\n"
    "#RawSignal #DataVisualization #GenerativeArt",
)
TAGS = ["RawSignal", "data visualization", "generative art",
        "procedural animation", "data art", "abstract visuals"]


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_state():
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        return {"scheduled_videos": [], "last_scheduled_at": None, "notified_public": []}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("JSON root is not an object")
        state.setdefault("scheduled_videos", [])
        state.setdefault("last_scheduled_at", None)
        state.setdefault("notified_public", [])
        return state
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        # Fail closed: don't erase state and risk duplicates.
        raise RuntimeError(f"Cannot safely read {STATE_PATH}: {exc}") from exc


def _write_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(STATE_PATH)


def telegram_notify(message):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram notification skipped: token/chat ID not configured.")
        return
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": message, "disable_web_page_preview": True},
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"Telegram notification failed: {exc}")


def _youtube():
    required = ["YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError("Missing GitHub Actions secrets: " + ", ".join(missing))
    credentials = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        scopes=SCOPES,
    )
    return build("youtube", "v3", credentials=credentials, cache_discovery=False)


def _title(value):
    value = re.sub(r"\s+", " ", str(value or "").strip())[:100].strip()
    return value or "RawSignal — Data-Driven Visual Experience"


def _next_publish_at(state):
    now = _now()
    earliest = now + timedelta(minutes=30)
    last = state.get("last_scheduled_at")
    if last:
        try:
            last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
            earliest = max(earliest, last_dt + timedelta(hours=MIN_GAP_HOURS))
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"Invalid last_scheduled_at in state: {last!r}") from exc

    local_earliest = earliest.astimezone(IST)
    for offset in range(9):
        day = (local_earliest + timedelta(days=offset)).date()
        for hour, minute in sorted(PREFERRED_SLOTS):
            candidate = datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)
            if candidate >= local_earliest:
                return candidate.astimezone(timezone.utc)
    return earliest


def upload_and_schedule(video_path, metadata):
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    state = _read_state()
    generator = str(metadata.get("generator") or metadata.get("generator_id") or "unknown")
    seed = str(metadata.get("seed", ""))
    signature = str(metadata.get("signature") or f"{generator}|{seed}|{path.name}")

    for item in state["scheduled_videos"]:
        if item.get("signature") == signature:
            telegram_notify(
                "⚠️ RawSignal duplicate upload blocked\n"
                f"Title: {item.get('title', '')}\n"
                f"Existing: https://youtu.be/{item.get('youtube_video_id', '')}"
            )
            return item

    title = _title(metadata.get("title"))
    publish_at = _next_publish_at(state)
    telegram_notify(
        "⬆️ RawSignal YouTube upload started\n"
        f"Title: {title}\nGenerator: {generator}\n"
        f"Duration: {metadata.get('duration_seconds', 'unknown')} seconds\n"
        f"Target: {publish_at.astimezone(IST).strftime('%d %b %Y, %I:%M %p IST')}"
    )

    youtube = _youtube()
    body = {
        "snippet": {
            "title": title,
            "description": DESCRIPTION,
            "tags": TAGS,
            "categoryId": CATEGORY_ID,
            "defaultLanguage": "en",
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": _iso(publish_at),
            "selfDeclaredMadeForKids": False,
        },
    }

    try:
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=MediaFileUpload(
                str(path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True
            ),
        )
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"YouTube upload progress: {int(status.progress() * 100)}%")
    except HttpError as exc:
        telegram_notify(f"❌ RawSignal YouTube upload failed\nTitle: {title}\nError: {str(exc)[:1000]}")
        raise

    video_id = response.get("id")
    if not video_id:
        raise RuntimeError(f"YouTube API returned no video ID: {response}")

    item = {
        "signature": signature,
        "youtube_video_id": video_id,
        "title": title,
        "generator": generator,
        "seed": seed,
        "scheduled_publish_at": _iso(publish_at),
        "scheduled_publish_at_ist": publish_at.astimezone(IST).isoformat(),
        "upload_status": "scheduled",
        "created_at": _iso(_now()),
    }
    state["scheduled_videos"].append(item)
    state["scheduled_videos"] = state["scheduled_videos"][-1000:]
    state["last_scheduled_at"] = _iso(publish_at)
    _write_state(state)

    url = f"https://www.youtube.com/watch?v={video_id}"
    telegram_notify(
        "🗓️ RawSignal video uploaded and scheduled\n"
        f"Title: {title}\nGenerator: {generator}\nVideo ID: {video_id}\n"
        f"Publish time: {publish_at.astimezone(IST).strftime('%d %b %Y, %I:%M %p IST')}\n"
        f"Status: Private until scheduled publication\n{url}"
    )
    return item


def check_publication_status():
    state = _read_state()
    pending = [
        item for item in state["scheduled_videos"]
        if item.get("youtube_video_id")
        and item["youtube_video_id"] not in state["notified_public"]
    ]
    if not pending:
        print("No pending publication notifications.")
        return

    youtube = _youtube()
    changed = False
    for start in range(0, len(pending), 50):
        batch = pending[start:start + 50]
        ids = ",".join(item["youtube_video_id"] for item in batch)
        result = youtube.videos().list(part="snippet,status", id=ids, maxResults=50).execute()
        found = {video["id"]: video for video in result.get("items", [])}
        for item in batch:
            video_id = item["youtube_video_id"]
            video = found.get(video_id)
            if not video:
                continue
            status = video.get("status", {})
            if status.get("privacyStatus") == "public":
                state["notified_public"].append(video_id)
                item["upload_status"] = "public"
                changed = True
                telegram_notify(
                    "🟢 RawSignal video is now public\n"
                    f"Title: {video.get('snippet', {}).get('title', item.get('title', ''))}\n"
                    f"https://www.youtube.com/watch?v={video_id}"
                )
            elif status.get("uploadStatus") in {"failed", "rejected"}:
                reason = status.get("failureReason") or status.get("rejectionReason") or "unknown"
                item["upload_status"] = "failed"
                item["failure_reason"] = reason
                state["notified_public"].append(video_id)
                changed = True
                telegram_notify(
                    "❌ RawSignal video failed/rejected\n"
                    f"Title: {item.get('title', '')}\nVideo ID: {video_id}\nReason: {reason}"
                )
    if changed:
        _write_state(state)
