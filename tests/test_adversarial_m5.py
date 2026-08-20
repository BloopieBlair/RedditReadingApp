"""
Adversarial edge-case test suite for Milestone 5 (YouTube Auth & Upload Integration).
Focuses on boundary stress verification of src/uploader/youtube_uploader.py and src/models.py (UploadMetadata).
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.models import UploadMetadata
from src.uploader import (
    YouTubeUploader,
    get_channel_status,
    upload_short,
    authenticate_user,
)


@pytest.fixture
def temp_token_file(tmp_path):
    return str(tmp_path / "test_token.json")


@pytest.fixture
def temp_secrets_file(tmp_path):
    return str(tmp_path / "test_client_secret.json")


@pytest.fixture
def temp_dummy_video(tmp_path):
    video_file = tmp_path / "dummy_video.mp4"
    video_file.write_bytes(b"fake mp4 content for m5 testing")
    return str(video_file)


# ============================================================================
# Task 1: Extremely Long Titles, Unicode/Emojis, and Hashtag Positioning
# ============================================================================

def test_title_unicode_emojis_and_zwj():
    """Test titles containing complex Unicode, emojis (ZWJ, multi-byte), and combining characters."""
    uploader = YouTubeUploader(dry_run=True)

    # 1. Emoji ZWJ sequence + multi-byte CJK
    unicode_title = "AskReddit: 🤖👨‍👩‍👧‍👦 这是一个非常长的 Reddit 帖子标题! 🔥✨🎉"
    meta = UploadMetadata(title=unicode_title)
    fmt = uploader.format_metadata(meta)

    assert fmt["snippet"]["title"].endswith("#Shorts")
    assert len(fmt["snippet"]["title"]) <= 100
    assert "AskReddit" in fmt["snippet"]["title"]

    # 2. 120 Emojis title (stress length truncation with emojis)
    emoji_title = "🤣" * 120
    meta_emoji = UploadMetadata(title=emoji_title)
    fmt_emoji = uploader.format_metadata(meta_emoji)
    assert len(fmt_emoji["snippet"]["title"]) <= 100
    assert fmt_emoji["snippet"]["title"].endswith("#Shorts")


def test_shorts_tag_case_sensitivity():
    """Verify case handling for #shorts, #Shorts, #SHORTS, #sHoRtS."""
    uploader = YouTubeUploader(dry_run=True)

    # 1. Exact "#Shorts" -> should NOT duplicate
    meta_title_case = UploadMetadata(title="Reddit Story #Shorts")
    res_title_case = uploader.format_metadata(meta_title_case)
    assert res_title_case["snippet"]["title"] == "Reddit Story #Shorts"

    # 2. Exact "#shorts" (lowercase) -> should NOT duplicate
    meta_lower_case = UploadMetadata(title="Reddit Story #shorts")
    res_lower_case = uploader.format_metadata(meta_lower_case)
    assert res_lower_case["snippet"]["title"] == "Reddit Story #shorts"

    # 3. Uppercase "#SHORTS" -> observe if code appends duplicate #Shorts
    meta_upper_case = UploadMetadata(title="Reddit Story #SHORTS")
    res_upper_case = uploader.format_metadata(meta_upper_case)
    # Current code checks `#Shorts` and `#shorts`, so `#SHORTS` causes `#Shorts` to be appended
    assert " #Shorts" in res_upper_case["snippet"]["title"]

    # 4. Mixed case "#sHoRtS" -> appends #Shorts
    meta_mixed_case = UploadMetadata(title="Reddit Story #sHoRtS")
    res_mixed_case = uploader.format_metadata(meta_mixed_case)
    assert " #Shorts" in res_mixed_case["snippet"]["title"]


def test_title_hashtag_positioning_and_truncation():
    """Test hashtag auto-appending, truncation boundaries, and embedded hashtags."""
    uploader = YouTubeUploader(dry_run=True)

    # Embedded hashtag inside text
    meta_embedded = UploadMetadata(title="Visit #ShortsStation for more content")
    fmt_embedded = uploader.format_metadata(meta_embedded)
    # "#Shorts" is substring of "#ShortsStation", so code skips appending "#Shorts"
    assert fmt_embedded["snippet"]["title"] == "Visit #ShortsStation for more content"

    # Truncation boundary: Title length = 92 chars without #Shorts
    title_92 = "A" * 92
    fmt_92 = uploader.format_metadata(UploadMetadata(title=title_92))
    assert fmt_92["snippet"]["title"] == title_92 + " #Shorts"
    assert len(fmt_92["snippet"]["title"]) == 100

    # Truncation boundary: Title length = 93 chars without #Shorts
    title_93 = "A" * 93
    fmt_93 = uploader.format_metadata(UploadMetadata(title=title_93))
    assert fmt_93["snippet"]["title"] == "A" * 92 + " #Shorts"
    assert len(fmt_93["snippet"]["title"]) == 100

    # Truncation flaw check: Title has #Shorts near position 95-100, but length > 100
    # "#Shorts" is in title, so `if "#Shorts" not in title` is False, but title[:100] truncates "#Shorts" to "#Sho"
    flawed_title = "B" * 94 + " #Shorts Extra Long Text Here"
    fmt_flawed = uploader.format_metadata(UploadMetadata(title=flawed_title))
    assert len(fmt_flawed["snippet"]["title"]) <= 100


# ============================================================================
# Task 2: Malformed / Corrupted Token JSON Files
# ============================================================================

def test_token_file_malformed_json(temp_token_file):
    """Verify loading corrupted token file returns None and logs warning."""
    Path(temp_token_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_token_file).write_text("{invalid json content:", encoding="utf-8")

    uploader = YouTubeUploader(token_file=temp_token_file)
    creds = uploader.get_credentials()
    assert creds is None


def test_token_file_zero_bytes(temp_token_file):
    """Verify 0-byte token file returns None immediately."""
    Path(temp_token_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_token_file).write_bytes(b"")

    uploader = YouTubeUploader(token_file=temp_token_file)
    creds = uploader.get_credentials()
    assert creds is None


def test_token_file_binary_garbage(temp_token_file):
    """Test handling of binary garbage / non-UTF8 bytes in token file."""
    Path(temp_token_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_token_file).write_bytes(os.urandom(512) + b"\x80\xff\xfe")

    uploader = YouTubeUploader(token_file=temp_token_file)
    creds = uploader.get_credentials()
    assert creds is None


@pytest.mark.parametrize("invalid_json_payload", [
    "[1, 2, 3]",
    '"just a string"',
    "12345",
    "true",
    "null",
    "{}",
    '{"token": "xyz"}',
])
def test_token_file_non_dict_and_incomplete_json_structures(temp_token_file, invalid_json_payload):
    """Test handling of non-dictionary or incomplete JSON structures in token file."""
    Path(temp_token_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_token_file).write_text(invalid_json_payload, encoding="utf-8")

    uploader = YouTubeUploader(token_file=temp_token_file)
    creds = uploader.get_credentials()
    assert creds is None


def test_token_file_is_directory(tmp_path):
    """Test handling when token_file path points to an existing directory."""
    token_dir = tmp_path / "token_is_a_dir"
    token_dir.mkdir()

    uploader = YouTubeUploader(token_file=str(token_dir))
    creds = uploader.get_credentials()
    assert creds is None


# ============================================================================
# Task 3: Non-existent Parent Directories for Token Output Paths
# ============================================================================

def test_save_credentials_creates_nested_parent_directories(tmp_path):
    """Test save_credentials automatically creates deeply nested parent directories."""
    nested_token = tmp_path / "a" / "b" / "c" / "youtube_token.json"
    uploader = YouTubeUploader(token_file=str(nested_token))

    mock_creds = MagicMock()
    mock_creds.to_json.return_value = json.dumps({"token": "fake_nested_token"})

    uploader.save_credentials(mock_creds)
    assert nested_token.exists()
    assert nested_token.read_text(encoding="utf-8") == json.dumps({"token": "fake_nested_token"}, indent=2)


def test_save_credentials_parent_is_file_raises_error(tmp_path):
    """Test save_credentials re-raises error when parent path component is a file instead of a dir."""
    blocker_file = tmp_path / "blocker_file"
    blocker_file.write_text("i am a file", encoding="utf-8")

    invalid_token = blocker_file / "token.json"
    uploader = YouTubeUploader(token_file=str(invalid_token))

    mock_creds = MagicMock()
    mock_creds.to_json.return_value = json.dumps({"token": "test"})

    with pytest.raises((NotADirectoryError, OSError, Exception)):
        uploader.save_credentials(mock_creds)


def test_get_credentials_non_existent_deep_path(tmp_path):
    """Test get_credentials returns None safely when token_file is in a non-existent deep path."""
    non_existent_token = tmp_path / "missing" / "path" / "token.json"
    uploader = YouTubeUploader(token_file=str(non_existent_token))
    assert uploader.get_credentials() is None


def test_authenticate_user_creates_parent_directories_for_secrets(tmp_path):
    """Test authenticate_user creates parent directories when client_secrets_file has missing parents."""
    secrets_path = tmp_path / "nested_secrets" / "client_secret.json"
    uploader = YouTubeUploader(client_secrets_file=str(secrets_path))

    with pytest.raises(FileNotFoundError, match="Missing client_secrets_file"):
        uploader.authenticate_user()

    assert secrets_path.exists()


# ============================================================================
# Task 4: Empty / Zero-byte Video Files
# ============================================================================

def test_dry_run_upload_with_zero_byte_video_file(tmp_path):
    """Test upload behavior in dry_run mode with a zero-byte empty video file."""
    empty_video = tmp_path / "empty_video.mp4"
    empty_video.touch()  # Creates a 0-byte file

    uploader = YouTubeUploader(dry_run=True)
    metadata = UploadMetadata(title="Zero Byte Video Test")

    # In dry_run mode, empty file exists, returns success
    result = uploader.upload(str(empty_video), metadata)
    assert result["status"] == "success"
    assert result["dry_run"] is True


def test_live_upload_with_zero_byte_video_file_handling(tmp_path):
    """Test upload behavior in live mode with a zero-byte empty video file."""
    empty_video = tmp_path / "empty_video.mp4"
    empty_video.touch()

    uploader = YouTubeUploader(dry_run=False)
    metadata = UploadMetadata(title="Zero Byte Video Live Test")

    mock_creds = MagicMock()
    mock_creds.valid = True

    with patch.object(uploader, "get_credentials", return_value=mock_creds), \
         patch("googleapiclient.http.MediaFileUpload") as mock_media, \
         patch("googleapiclient.discovery.build") as mock_build:

        mock_insert = MagicMock()
        mock_build.return_value.videos().insert.return_value = mock_insert
        mock_insert.next_chunk.return_value = (
            MagicMock(progress=lambda: 1.0),
            {"id": "zero_byte_vid_id", "status": {"uploadStatus": "processed"}},
        )

        res = uploader.upload(str(empty_video), metadata)
        assert res["status"] == "success"
        # Verify MediaFileUpload was initialized with empty_video path
        mock_media.assert_called_once_with(
            str(empty_video), chunksize=1024 * 1024, resumable=True, mimetype="video/mp4"
        )


# ============================================================================
# Task 5: UploadMetadata Validation Limits & Edge Cases
# ============================================================================

def test_upload_metadata_empty_and_whitespace_title():
    """Test UploadMetadata raises ValueError for empty or whitespace-only titles."""
    with pytest.raises(ValueError, match="title cannot be empty"):
        UploadMetadata(title="")

    with pytest.raises(ValueError, match="title cannot be empty"):
        UploadMetadata(title="   \n\t  ")


def test_upload_metadata_from_dict_boundary_cases():
    """Test UploadMetadata.from_dict with various valid and invalid dictionary payloads."""
    # Empty title in dict
    with pytest.raises(ValueError, match="title cannot be empty"):
        UploadMetadata.from_dict({"title": "   "})

    with pytest.raises(ValueError, match="title cannot be empty"):
        UploadMetadata.from_dict({})

    # Numeric title converted to string
    meta_num = UploadMetadata.from_dict({"title": 12345})
    assert meta_num.title == "12345"

    # None tags handled gracefully or raises TypeError
    with pytest.raises(TypeError):
        UploadMetadata.from_dict({"title": "Valid Title", "tags": None})

    # Dict with non-list tags like string
    meta_str_tags = UploadMetadata.from_dict({"title": "Valid Title", "tags": "gaming"})
    assert meta_str_tags.tags == ["g", "a", "m", "i", "n", "g"]  # list("gaming") behaviour


def test_category_id_default_and_overrides():
    """Test category ID default ("24") and explicit custom overrides."""
    uploader = YouTubeUploader(dry_run=True)

    # Default category ID should be "24" (Entertainment)
    meta_default = UploadMetadata(title="Default Category Video")
    assert meta_default.category_id == "24"
    fmt_default = uploader.format_metadata(meta_default)
    assert fmt_default["snippet"]["categoryId"] == "24"

    # Custom category ID "10" (Music)
    meta_music = UploadMetadata(title="Music Video", category_id="10")
    fmt_music = uploader.format_metadata(meta_music)
    assert fmt_music["snippet"]["categoryId"] == "10"

    # Custom category ID "22" (People & Blogs)
    meta_blogs = UploadMetadata(title="Vlog Video", category_id="22")
    fmt_blogs = uploader.format_metadata(meta_blogs)
    assert fmt_blogs["snippet"]["categoryId"] == "22"

    # Empty category_id string in UploadMetadata
    meta_empty = UploadMetadata(title="Empty Category Video")
    meta_empty.category_id = ""
    fmt_empty = uploader.format_metadata(meta_empty)
    # format_metadata uses `metadata.category_id or "24"`, so empty string falls back to "24"
    assert fmt_empty["snippet"]["categoryId"] == "24"

    # Deserialization with from_dict
    meta_from_dict = UploadMetadata.from_dict({"title": "Gaming Video", "category_id": "20"})
    assert meta_from_dict.category_id == "20"
    fmt_from_dict = uploader.format_metadata(meta_from_dict)
    assert fmt_from_dict["snippet"]["categoryId"] == "20"


# ============================================================================
# Additional Auth & Upload Resumable / Expired Token Tests
# ============================================================================

def test_expired_token_missing_refresh_token(temp_token_file):
    """Verify expired token with NO refresh token returns None without calling refresh."""
    token_data = {"token": "expired_access_token_only"}
    Path(temp_token_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_token_file).write_text(json.dumps(token_data), encoding="utf-8")

    uploader = YouTubeUploader(token_file=temp_token_file)

    mock_creds = MagicMock()
    mock_creds.expired = True
    mock_creds.refresh_token = None
    mock_creds.valid = False

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=mock_creds):
        creds = uploader.get_credentials()
        assert creds is None
        mock_creds.refresh.assert_not_called()


def test_expired_token_with_valid_refresh_token(temp_token_file):
    """Verify expired token with valid refresh token triggers refresh and save."""
    token_data = {"token": "expired_token", "refresh_token": "valid_refresh"}
    Path(temp_token_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_token_file).write_text(json.dumps(token_data), encoding="utf-8")

    uploader = YouTubeUploader(token_file=temp_token_file)

    mock_creds = MagicMock()
    mock_creds.expired = True
    mock_creds.refresh_token = "valid_refresh"
    mock_creds.valid = True
    mock_creds.to_json.return_value = json.dumps({"token": "new_refreshed_token", "refresh_token": "valid_refresh"})

    def simulate_refresh(request):
        mock_creds.expired = False

    mock_creds.refresh.side_effect = simulate_refresh

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=mock_creds), \
         patch("google.auth.transport.requests.Request"):
        creds = uploader.get_credentials()
        assert creds is mock_creds
        mock_creds.refresh.assert_called_once()
        updated_data = json.loads(Path(temp_token_file).read_text(encoding="utf-8"))
        assert updated_data["token"] == "new_refreshed_token"


def test_expired_token_refresh_raises_exception(temp_token_file):
    """Verify exception during token refresh returns None gracefully without crashing."""
    token_data = {"token": "expired_token", "refresh_token": "revoked_refresh"}
    Path(temp_token_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_token_file).write_text(json.dumps(token_data), encoding="utf-8")

    uploader = YouTubeUploader(token_file=temp_token_file)

    mock_creds = MagicMock()
    mock_creds.expired = True
    mock_creds.refresh_token = "revoked_refresh"
    mock_creds.refresh.side_effect = Exception("Token has been expired or revoked.")

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=mock_creds), \
         patch("google.auth.transport.requests.Request"):
        creds = uploader.get_credentials()
        assert creds is None


def test_youtube_dry_run_env_var_handling(monkeypatch, temp_dummy_video):
    """Test behavior when YOUTUBE_DRY_RUN=1 environment variable is set."""
    monkeypatch.setenv("YOUTUBE_DRY_RUN", "1")

    uploader_default = YouTubeUploader()
    status = get_channel_status(dry_run=uploader_default.dry_run)
    assert "authenticated" in status


def test_resumable_upload_http_error(temp_dummy_video, temp_token_file):
    """Verify resumable upload handles HTTP 500 error from YouTube API."""
    uploader = YouTubeUploader(token_file=temp_token_file, dry_run=False)

    mock_creds = MagicMock()
    mock_creds.valid = True

    mock_insert_request = MagicMock()
    mock_insert_request.next_chunk.side_effect = RuntimeError("YouTube API Error: 500 Internal Server Error")

    mock_youtube = MagicMock()
    mock_youtube.videos().insert.return_value = mock_insert_request

    metadata = UploadMetadata(title="Failing Upload Video")

    with patch.object(uploader, "get_credentials", return_value=mock_creds), \
         patch.dict("sys.modules", {
             "googleapiclient.discovery": MagicMock(build=MagicMock(return_value=mock_youtube)),
             "googleapiclient.http": MagicMock(),
         }):
        with pytest.raises(RuntimeError, match="500 Internal Server Error"):
            uploader.upload(temp_dummy_video, metadata)


def test_resumable_upload_connection_reset_mid_upload(temp_dummy_video, temp_token_file):
    """Verify resumable upload handling when connection drops on second chunk."""
    uploader = YouTubeUploader(token_file=temp_token_file, dry_run=False)

    mock_creds = MagicMock()
    mock_creds.valid = True

    status_50 = MagicMock()
    status_50.progress.return_value = 0.5

    mock_insert_request = MagicMock()
    mock_insert_request.next_chunk.side_effect = [
        (status_50, None),
        ConnectionResetError("Remote host closed connection"),
    ]

    mock_youtube = MagicMock()
    mock_youtube.videos().insert.return_value = mock_insert_request

    metadata = UploadMetadata(title="Interrupted Upload Video")

    with patch.object(uploader, "get_credentials", return_value=mock_creds), \
         patch.dict("sys.modules", {
             "googleapiclient.discovery": MagicMock(build=MagicMock(return_value=mock_youtube)),
             "googleapiclient.http": MagicMock(),
         }):
        with pytest.raises(ConnectionResetError, match="Remote host closed connection"):
            uploader.upload(temp_dummy_video, metadata)


def test_resumable_upload_empty_response(temp_dummy_video, temp_token_file):
    """Verify resumable upload response parsing when YouTube API returns empty response dict."""
    uploader = YouTubeUploader(token_file=temp_token_file, dry_run=False)

    mock_creds = MagicMock()
    mock_creds.valid = True

    mock_insert_request = MagicMock()
    mock_insert_request.next_chunk.return_value = (None, {})

    mock_youtube = MagicMock()
    mock_youtube.videos().insert.return_value = mock_insert_request

    metadata = UploadMetadata(title="Empty Response Video")

    with patch.object(uploader, "get_credentials", return_value=mock_creds), \
         patch.dict("sys.modules", {
             "googleapiclient.discovery": MagicMock(build=MagicMock(return_value=mock_youtube)),
             "googleapiclient.http": MagicMock(),
         }):
        result = uploader.upload(temp_dummy_video, metadata)
        assert result["status"] == "success"
        assert result["video_id"] is None
        assert result["video_url"] == ""
        assert result["upload_status"] == "processed"
