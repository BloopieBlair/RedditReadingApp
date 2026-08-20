"""YouTube Uploader package for OAuth authentication and Shorts upload."""

from src.uploader.youtube_uploader import (
    YouTubeUploader,
    upload_short,
    get_channel_status,
    authenticate_user,
)

__all__ = [
    "YouTubeUploader",
    "upload_short",
    "get_channel_status",
    "authenticate_user",
]
