"""Run locally once to obtain a YouTube refresh token for GitHub Actions.

Download your Google OAuth Desktop-app client JSON and save it as
client_secret.json in the repository root, then run:
    python -m pip install google-auth-oauthlib
    python scripts/authorize_youtube.py

The script prints a refresh token. Add it to GitHub Actions secrets as
YT_REFRESH_TOKEN, then delete client_secret.json and never commit it.
"""
import json
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

ROOT = Path(__file__).resolve().parents[1]
CLIENT_FILE = ROOT / "client_secret.json"
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

if not CLIENT_FILE.exists():
    raise SystemExit(
        "Missing client_secret.json. Download the OAuth Desktop-app JSON "
        "from Google Cloud Console and place it in the repository root."
    )

flow = InstalledAppFlow.from_client_secrets_file(
    str(CLIENT_FILE),
    scopes=SCOPES,
)
credentials = flow.run_local_server(
    port=0,
    access_type="offline",
    prompt="consent",
)

if not credentials.refresh_token:
    raise SystemExit(
        "Google did not return a refresh token. Revoke this app's access in "
        "your Google Account, then rerun the script with prompt=consent."
    )

print("\nCopy this refresh token into GitHub Actions secret YT_REFRESH_TOKEN:\n")
print(credentials.refresh_token)
print("\nDo not commit the token or client_secret.json.")
