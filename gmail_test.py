"""
gmail_test.py — checks that Gmail access works. Read-only: it can't send, delete, or change anything.

First run opens your browser to sign in; after that, token.json remembers you.
Run:  python gmail_test.py
"""
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
ROOT = Path(__file__).resolve().parent
TOKEN, CREDS = ROOT / "token.json", ROOT / "credentials.json"


def gmail():
    creds = Credentials.from_authorized_user_file(TOKEN, SCOPES) if TOKEN.exists() else None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            creds = None  # in Testing mode Google expires sign-ins after 7 days; just sign in again
    if not creds or not creds.valid:
        if not CREDS.exists():
            raise SystemExit("credentials.json not found in ~/style-matcher. See the setup steps.")
        creds = InstalledAppFlow.from_client_secrets_file(CREDS, SCOPES).run_local_server(port=0)
        TOKEN.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


if __name__ == "__main__":
    service = gmail()
    me = service.users().getProfile(userId="me").execute()
    query = 'subject:(order OR confirmation OR shipped OR return OR refund) newer_than:1y'
    found = service.users().messages().list(userId="me", q=query, maxResults=50).execute()
    print(f"Connected as {me['emailAddress']}.")
    print(f"Found {len(found.get('messages', []))} order-related emails from the past year (showing up to 50).")
