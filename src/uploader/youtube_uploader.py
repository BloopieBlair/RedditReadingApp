"""YouTube OAuth2 uploader with resumable upload, token management, and dry-run mode."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import CLIENT_SECRETS_PATH, TEMP_DIR
from src.models import UploadMetadata

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


class YouTubeUploader:
    """OAuth2 YouTube video uploader supporting authentication, status checks, and uploads."""

    def __init__(
        self,
        client_secrets_file: Optional[str] = None,
        token_file: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.client_secrets_file = str(client_secrets_file or CLIENT_SECRETS_PATH)
        self.token_file = str(token_file or (TEMP_DIR / "youtube_token.json"))
        self.dry_run = dry_run

    def get_credentials(self) -> Optional[Any]:
        """
        Loads saved OAuth2 credentials from token file.
        Auto-refreshes expired credentials if refresh_token is present.
        Returns Credentials object or None if unauthenticated/invalid.
        """
        if not os.path.exists(self.token_file) or os.path.getsize(self.token_file) == 0:
            return None

        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request

            with open(self.token_file, "r", encoding="utf-8") as f:
                token_data = json.load(f)

            creds = Credentials.from_authorized_user_info(token_data, SCOPES)

            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                self.save_credentials(creds)

            return creds if creds and creds.valid else None
        except Exception as e:
            logger.warning(f"Error loading YouTube OAuth credentials from {self.token_file}: {e}")
            return None

    def save_credentials(self, creds: Any) -> None:
        """Saves authorized user credentials to JSON token file."""
        try:
            token_info = json.loads(creds.to_json())
            token_path = Path(self.token_file)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(token_path, "w", encoding="utf-8") as f:
                json.dump(token_info, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving YouTube credentials to {self.token_file}: {e}")
            raise

    def authenticate_user(self, port: int = 8080) -> Any:
        """
        Launches browser-based Google OAuth2 login flow.
        Validates client_secrets_file existence and non-placeholder credentials.
        """
        secret_path = Path(self.client_secrets_file)
        if not secret_path.exists():
            sample_secret = {
                "installed": {
                    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
                    "project_id": "reddit-reading-app",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_secret": "YOUR_CLIENT_SECRET",
                    "redirect_uris": ["http://localhost:8080/", "http://127.0.0.1:8080/"],
                }
            }
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            with open(secret_path, "w", encoding="utf-8") as f:
                json.dump(sample_secret, f, indent=2)
            raise FileNotFoundError(
                f"Missing client_secrets_file! Template created at: {secret_path}. "
                "Please configure real Google Cloud OAuth credentials."
            )

        try:
            with open(secret_path, "r", encoding="utf-8") as f:
                secret_data = json.load(f)
            client_info = secret_data.get("installed", {}) or secret_data.get("web", {})
            client_id = client_info.get("client_id", "")
            if "YOUR_CLIENT_ID" in client_id or not client_id:
                raise ValueError(
                    "Google Cloud OAuth Credentials not configured! "
                    "Please replace placeholder 'YOUR_CLIENT_ID' in client_secrets_file."
                )
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error reading client secrets file {secret_path}: {e}")
            raise ValueError(f"Invalid client secrets file: {e}") from e

        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), SCOPES)
        creds = flow.run_local_server(port=port, prompt="select_account consent")
        self.save_credentials(creds)
        return creds

    def has_valid_client_secrets(self) -> bool:
        """Checks if client_secrets_file exists and contains valid, non-placeholder credentials."""
        secret_path = Path(self.client_secrets_file)
        if not secret_path.exists():
            return False
        try:
            with open(secret_path, "r", encoding="utf-8") as f:
                secret_data = json.load(f)
            client_info = secret_data.get("installed", {}) or secret_data.get("web", {})
            client_id = client_info.get("client_id", "")
            client_secret = client_info.get("client_secret", "")
            if not client_id or not client_secret:
                return False
            if "YOUR_CLIENT_ID" in client_id or "YOUR_CLIENT_SECRET" in client_secret:
                return False
            return True
        except Exception:
            return False

    def revoke_credentials(self) -> bool:
        """Deletes cached token file to allow account swapping."""
        try:
            token_path = Path(self.token_file)
            if token_path.exists():
                token_path.unlink()
                logger.info(f"Deleted token file: {token_path}")
                return True
        except Exception as e:
            logger.warning(f"Failed to delete token file {self.token_file}: {e}")
        return False

    def get_channel_status(self) -> Dict[str, Any]:
        """
        Returns channel profile dict (authenticated, channel_id, channel_name, avatar_url, has_client_secrets).
        Handles dry_run mode gracefully.
        """
        has_secrets = self.has_valid_client_secrets()
        if self.dry_run:
            return {
                "authenticated": True,
                "channel_id": "mock_channel_id",
                "channel_name": "Mock Channel (Dry Run)",
                "avatar_url": None,
                "has_client_secrets": has_secrets,
            }

        creds = self.get_credentials()
        if not creds:
            return {
                "authenticated": False,
                "channel_id": None,
                "channel_name": None,
                "avatar_url": None,
                "has_client_secrets": has_secrets,
            }

        try:
            from googleapiclient.discovery import build

            youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
            request = youtube.channels().list(mine=True, part="snippet,statistics")
            response = request.execute()

            if response.get("items"):
                channel = response["items"][0]
                snippet = channel.get("snippet", {})
                return {
                    "authenticated": True,
                    "channel_id": channel.get("id"),
                    "channel_name": snippet.get("title", "YouTube Channel"),
                    "avatar_url": snippet.get("thumbnails", {}).get("default", {}).get("url"),
                }
            else:
                return {
                    "authenticated": True,
                    "channel_id": None,
                    "channel_name": "Account Connected (No Channel Created)",
                    "avatar_url": None,
                    "has_no_channel": True,
                }
        except Exception as e:
            logger.error(f"Error fetching YouTube channel status: {e}")

        return {
            "authenticated": False,
            "channel_id": None,
            "channel_name": None,
            "avatar_url": None,
        }

    def format_metadata(self, metadata: UploadMetadata) -> Dict[str, Any]:
        """
        Ensures #Shorts tag in title, truncates title <= 100 chars, description <= 500 chars,
        and applies default tags if empty.
        Returns YouTube API insert payload dictionary.
        """
        title = metadata.title.strip()
        if "#Shorts" not in title and "#shorts" not in title:
            if len(title) > 92:
                title = f"{title[:92].strip()} #Shorts"
            else:
                title = f"{title} #Shorts"

        title = title[:100]
        description = (metadata.description or "")[:500]
        tags = metadata.tags if metadata.tags else ["Shorts", "Reddit", "Minecraft", "RedditStories"]
        category_id = metadata.category_id or "24"

        return {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": metadata.privacy_status,
                "selfDeclaredMadeForKids": False,
            },
        }

    def upload(self, video_path: str, metadata: UploadMetadata) -> Dict[str, Any]:
        """
        Uploads a vertical video to YouTube Shorts using YouTube Data API v3.
        Supports dry_run=True and real OAuth resumable chunk upload.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found for upload: {video_path}")

        body = self.format_metadata(metadata)

        if self.dry_run:
            mock_id = "mock_youtube_video_id_12345"
            return {
                "status": "success",
                "video_id": mock_id,
                "video_url": f"https://youtube.com/shorts/{mock_id}",
                "title": body["snippet"]["title"],
                "snippet": body["snippet"],
                "upload_status": "processed",
                "dry_run": True,
            }

        creds = self.get_credentials()
        if not creds:
            raise PermissionError("Not authenticated with YouTube. Please connect your channel first or specify dry_run=True.")

        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
        media = MediaFileUpload(
            video_path, chunksize=1024 * 1024, resumable=True, mimetype="video/mp4"
        )
        request = youtube.videos().insert(
            part="snippet,status", body=body, media_body=media
        )

        response = None
        try:
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress_pct = int(status.progress() * 100)
                    logger.info(f"YouTube Upload Progress: {progress_pct}%")
        except Exception as e:
            err_str = str(e)
            if "uploadLimitExceeded" in err_str or "exceeded the number of videos" in err_str:
                raise RuntimeError(
                    "YouTube Channel Daily Upload Limit Exceeded! "
                    "YouTube caps unverified channels to ~6-10 uploads per 24 hours. "
                    "To fix: 1) Wait 24 hours for the rolling limit to reset, or "
                    "2) Enable 'Advanced Features' in YouTube Studio (https://studio.youtube.com -> Settings -> Channel -> Feature Eligibility) to unlock up to 100 uploads per day."
                ) from e
            raise

        video_id = response.get("id") if response else None
        video_url = f"https://youtube.com/shorts/{video_id}" if video_id else ""
        upload_status = (
            response.get("status", {}).get("uploadStatus", "processed")
            if response
            else "processed"
        )

        return {
            "status": "success",
            "video_id": video_id,
            "video_url": video_url,
            "title": body["snippet"]["title"],
            "snippet": body["snippet"],
            "upload_status": upload_status,
        }


def upload_short(
    video_path: str,
    title: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    client_secrets_file: Optional[str] = None,
    token_file: Optional[str] = None,
    privacy_status: str = "unlisted",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Standalone helper function to upload a YouTube Short video."""
    uploader = YouTubeUploader(
        client_secrets_file=client_secrets_file,
        token_file=token_file,
        dry_run=dry_run,
    )
    metadata = UploadMetadata(
        title=title,
        description=description,
        tags=tags or [],
        privacy_status=privacy_status,
    )
    return uploader.upload(video_path=video_path, metadata=metadata)


def get_channel_status(
    client_secrets_file: Optional[str] = None,
    token_file: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Standalone helper function to get channel authentication status."""
    uploader = YouTubeUploader(
        client_secrets_file=client_secrets_file,
        token_file=token_file,
        dry_run=dry_run,
    )
    return uploader.get_channel_status()


def authenticate_user(
    client_secrets_file: Optional[str] = None,
    token_file: Optional[str] = None,
    port: int = 8080,
) -> Any:
    """Standalone helper function to trigger interactive OAuth user login flow."""
    uploader = YouTubeUploader(
        client_secrets_file=client_secrets_file,
        token_file=token_file,
    )
    return uploader.authenticate_user(port=port)
