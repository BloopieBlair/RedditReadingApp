"""Comprehensive unit tests for YouTube Auth & Upload Integration."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models import UploadMetadata
from src.uploader import (
    YouTubeUploader,
    upload_short,
    get_channel_status,
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
    video_file.write_bytes(b"fake mp4 content")
    return str(video_file)


def test_uploader_package_exports():
    """Verify src.uploader package exports all required symbols."""
    import src.uploader as uploader_module

    assert hasattr(uploader_module, "YouTubeUploader")
    assert hasattr(uploader_module, "upload_short")
    assert hasattr(uploader_module, "get_channel_status")
    assert hasattr(uploader_module, "authenticate_user")


def test_upload_short_signature_and_defaults(temp_dummy_video):
    """Test upload_short standalone function with default arguments and dry_run=True."""
    result = upload_short(
        video_path=temp_dummy_video,
        title="My Amazing Short",
        dry_run=True,
    )
    assert result["status"] == "success"
    assert "mock_youtube_video_id" in result["video_id"]
    assert "https://youtube.com/shorts/" in result["video_url"]
    assert "#Shorts" in result["title"]
    assert result["dry_run"] is True
    assert result["snippet"]["tags"] == ["Shorts", "Reddit", "Minecraft", "RedditStories"]


def test_dry_run_upload(temp_dummy_video, temp_token_file, temp_secrets_file):
    """Test YouTubeUploader class in dry_run mode."""
    uploader = YouTubeUploader(
        client_secrets_file=temp_secrets_file,
        token_file=temp_token_file,
        dry_run=True,
    )
    metadata = UploadMetadata(
        title="Minecraft Parkour Story",
        description="Great story",
        tags=["fun", "gaming"],
        privacy_status="public",
    )
    res = uploader.upload(temp_dummy_video, metadata)
    assert res["status"] == "success"
    assert res["dry_run"] is True
    assert res["snippet"]["title"] == "Minecraft Parkour Story #Shorts"
    assert res["snippet"]["tags"] == ["fun", "gaming"]
    assert res["snippet"]["categoryId"] == "24"


def test_title_shorts_hashtag_auto_append_and_truncation():
    """Test hashtag appending, title length bounds, description bounds, and default tags."""
    uploader = YouTubeUploader(dry_run=True)

    # Title without #Shorts tag
    meta1 = UploadMetadata(title="Cool Story")
    fmt1 = uploader.format_metadata(meta1)
    assert fmt1["snippet"]["title"] == "Cool Story #Shorts"
    assert fmt1["status"]["privacyStatus"] == "unlisted"

    # Title already having #shorts tag
    meta2 = UploadMetadata(title="Cool Story #shorts")
    fmt2 = uploader.format_metadata(meta2)
    assert fmt2["snippet"]["title"] == "Cool Story #shorts"

    # Extremely long title without #Shorts
    long_title = "A" * 150
    meta3 = UploadMetadata(title=long_title)
    fmt3 = uploader.format_metadata(meta3)
    assert len(fmt3["snippet"]["title"]) <= 100
    assert fmt3["snippet"]["title"].endswith("#Shorts")

    # Extremely long description
    long_desc = "D" * 800
    meta4 = UploadMetadata(title="Short Title", description=long_desc)
    fmt4 = uploader.format_metadata(meta4)
    assert len(fmt4["snippet"]["description"]) == 500

    # Default tags when empty
    meta5 = UploadMetadata(title="Short Title", tags=[])
    fmt5 = uploader.format_metadata(meta5)
    assert fmt5["snippet"]["tags"] == ["Shorts", "Reddit", "Minecraft", "RedditStories"]


def test_missing_video_file_raises_error():
    """Test that upload raises FileNotFoundError when video file does not exist."""
    uploader = YouTubeUploader(dry_run=True)
    metadata = UploadMetadata(title="Missing File Video")
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        uploader.upload("non_existent_video_path_12345.mp4", metadata)


def test_unauthenticated_upload_raises_permission_error(temp_dummy_video, temp_token_file):
    """Test that upload with dry_run=False raises PermissionError when unauthenticated."""
    uploader = YouTubeUploader(token_file=temp_token_file, dry_run=False)
    metadata = UploadMetadata(title="Unauthenticated Video")
    with pytest.raises(PermissionError, match="Not authenticated with YouTube"):
        uploader.upload(temp_dummy_video, metadata)


def test_get_credentials_valid_and_save(temp_token_file):
    """Test loading valid credentials from file and saving credentials."""
    uploader = YouTubeUploader(token_file=temp_token_file)

    mock_creds = MagicMock()
    mock_creds.expired = False
    mock_creds.valid = True
    mock_creds.to_json.return_value = json.dumps({"token": "fake_access_token"})

    # Test save_credentials
    uploader.save_credentials(mock_creds)
    assert Path(temp_token_file).exists()

    # Test get_credentials with mocked google.oauth2.credentials.Credentials
    with patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=mock_creds):
        loaded_creds = uploader.get_credentials()
        assert loaded_creds is mock_creds


def test_get_credentials_expired_token_refresh(temp_token_file):
    """Test auto-refreshing expired credentials during get_credentials."""
    token_data = {"token": "expired_token", "refresh_token": "valid_refresh"}
    Path(temp_token_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_token_file).write_text(json.dumps(token_data), encoding="utf-8")

    uploader = YouTubeUploader(token_file=temp_token_file)

    mock_creds = MagicMock()
    mock_creds.expired = True
    mock_creds.refresh_token = "valid_refresh"
    mock_creds.valid = True
    mock_creds.to_json.return_value = json.dumps({"token": "refreshed_token"})

    def mock_refresh(request):
        mock_creds.expired = False

    mock_creds.refresh.side_effect = mock_refresh

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=mock_creds), \
         patch("google.auth.transport.requests.Request") as mock_request:
        creds = uploader.get_credentials()
        assert creds is mock_creds
        mock_creds.refresh.assert_called_once()


def test_authenticate_user_missing_secrets(temp_secrets_file, temp_token_file):
    """Test authenticate_user creates template and raises FileNotFoundError if secrets file missing."""
    uploader = YouTubeUploader(client_secrets_file=temp_secrets_file, token_file=temp_token_file)
    with pytest.raises(FileNotFoundError, match="Missing client_secrets_file"):
        uploader.authenticate_user()
    assert Path(temp_secrets_file).exists()


def test_authenticate_user_placeholder_secret(temp_secrets_file, temp_token_file):
    """Test authenticate_user raises ValueError if client secret contains YOUR_CLIENT_ID placeholder."""
    placeholder_data = {
        "installed": {
            "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
            "client_secret": "YOUR_CLIENT_SECRET",
        }
    }
    Path(temp_secrets_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_secrets_file).write_text(json.dumps(placeholder_data), encoding="utf-8")

    uploader = YouTubeUploader(client_secrets_file=temp_secrets_file, token_file=temp_token_file)
    with pytest.raises(ValueError, match="Credentials not configured"):
        uploader.authenticate_user()


def test_authenticate_user_success_mock(temp_secrets_file, temp_token_file):
    """Test successful user authentication flow with mocked OAuth flow."""
    valid_secret = {
        "installed": {
            "client_id": "12345-realid.apps.googleusercontent.com",
            "client_secret": "real_secret_key",
        }
    }
    Path(temp_secrets_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_secrets_file).write_text(json.dumps(valid_secret), encoding="utf-8")

    uploader = YouTubeUploader(client_secrets_file=temp_secrets_file, token_file=temp_token_file)

    mock_flow_class = MagicMock()
    mock_flow = MagicMock()
    mock_creds = MagicMock()
    mock_creds.to_json.return_value = json.dumps({"token": "authed_token"})
    mock_flow.run_local_server.return_value = mock_creds
    mock_flow_class.from_client_secrets_file.return_value = mock_flow

    mock_oauthlib_flow = MagicMock()
    mock_oauthlib_flow.InstalledAppFlow = mock_flow_class

    with patch.dict("sys.modules", {
        "google_auth_oauthlib": MagicMock(),
        "google_auth_oauthlib.flow": mock_oauthlib_flow,
    }):
        creds = uploader.authenticate_user(port=8080)
        assert creds is mock_creds
        mock_flow_class.from_client_secrets_file.assert_called_once_with(temp_secrets_file, ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"])
        mock_flow.run_local_server.assert_called_once_with(port=8080, prompt="select_account consent")
        assert Path(temp_token_file).exists()


def test_get_channel_status_dry_run_and_unauthenticated(temp_token_file):
    """Test get_channel_status for dry_run and unauthenticated states."""
    # Dry run status
    status_dry = get_channel_status(dry_run=True)
    assert status_dry["authenticated"] is True
    assert status_dry["channel_id"] == "mock_channel_id"

    # Unauthenticated status
    status_unauth = get_channel_status(token_file=temp_token_file, dry_run=False)
    assert status_unauth["authenticated"] is False
    assert status_unauth["channel_name"] is None


def test_get_channel_status_authenticated_mock(temp_token_file):
    """Test get_channel_status with mocked YouTube Data API response."""
    uploader = YouTubeUploader(token_file=temp_token_file)

    mock_creds = MagicMock()
    mock_creds.valid = True

    mock_response = {
        "items": [
            {
                "id": "UC_TEST_CHANNEL_123",
                "snippet": {
                    "title": "My Test YouTube Channel",
                    "thumbnails": {"default": {"url": "https://example.com/avatar.jpg"}},
                },
            }
        ]
    }

    mock_request = MagicMock()
    mock_request.execute.return_value = mock_response

    mock_youtube = MagicMock()
    mock_youtube.channels().list.return_value = mock_request

    with patch.object(uploader, "get_credentials", return_value=mock_creds), \
         patch.dict("sys.modules", {"googleapiclient.discovery": MagicMock(build=MagicMock(return_value=mock_youtube))}):
        status = uploader.get_channel_status()
        assert status["authenticated"] is True
        assert status["channel_id"] == "UC_TEST_CHANNEL_123"
        assert status["channel_name"] == "My Test YouTube Channel"
        assert status["avatar_url"] == "https://example.com/avatar.jpg"


def test_resumable_upload_chunk_loop_mock(temp_dummy_video, temp_token_file):
    """Test YouTube video upload progress chunk loop with mocked MediaFileUpload and discovery build."""
    uploader = YouTubeUploader(token_file=temp_token_file, dry_run=False)

    mock_creds = MagicMock()
    mock_creds.valid = True

    # Progress status mock
    status_50 = MagicMock()
    status_50.progress.return_value = 0.5

    # Media insert request mock
    mock_insert_request = MagicMock()
    mock_insert_request.next_chunk.side_effect = [
        (status_50, None),
        (
            MagicMock(progress=lambda: 1.0),
            {
                "id": "real_yt_video_999",
                "snippet": {"title": "Test Title #Shorts"},
                "status": {"uploadStatus": "processed"},
            },
        ),
    ]

    mock_youtube = MagicMock()
    mock_youtube.videos().insert.return_value = mock_insert_request

    metadata = UploadMetadata(title="Test Real Upload", description="Real upload test")

    mock_discovery = MagicMock(build=MagicMock(return_value=mock_youtube))
    mock_http = MagicMock()

    with patch.object(uploader, "get_credentials", return_value=mock_creds), \
         patch.dict("sys.modules", {
             "googleapiclient.discovery": mock_discovery,
             "googleapiclient.http": mock_http,
         }):
        result = uploader.upload(temp_dummy_video, metadata)

        assert result["status"] == "success"
        assert result["video_id"] == "real_yt_video_999"
        assert result["video_url"] == "https://youtube.com/shorts/real_yt_video_999"
        assert result["upload_status"] == "processed"
        assert mock_insert_request.next_chunk.call_count == 2


def test_has_valid_client_secrets(tmp_path):
    """Test has_valid_client_secrets for missing, placeholder, and valid secrets."""
    secret_file = tmp_path / "client_secret.json"
    uploader = YouTubeUploader(client_secrets_file=str(secret_file))

    # 1. Missing file
    assert uploader.has_valid_client_secrets() is False

    # 2. Placeholder file
    secret_file.write_text(json.dumps({"installed": {"client_id": "YOUR_CLIENT_ID", "client_secret": "YOUR_CLIENT_SECRET"}}), encoding="utf-8")
    assert uploader.has_valid_client_secrets() is False

    # 3. Valid credentials file
    secret_file.write_text(json.dumps({"installed": {"client_id": "123-abc.apps.googleusercontent.com", "client_secret": "real_secret_123"}}), encoding="utf-8")
    assert uploader.has_valid_client_secrets() is True


def test_api_save_and_upload_credentials(tmp_path, monkeypatch):
    """Test web API endpoints for saving and uploading client secrets."""
    from fastapi.testclient import TestClient
    from src.web.app import app
    import src.config as config

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "ASSETS_DIR", assets_dir)

    client = TestClient(app)

    # 1. Save manual credentials
    resp = client.post("/api/auth/save_credentials", json={
        "client_id": "999-xyz.apps.googleusercontent.com",
        "client_secret": "my_super_secret"
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    saved_path = assets_dir / "client_secret.json"
    assert saved_path.exists()
    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_data["installed"]["client_id"] == "999-xyz.apps.googleusercontent.com"
    assert saved_data["installed"]["client_secret"] == "my_super_secret"

    # 2. Save raw json
    raw_payload = json.dumps({
        "installed": {
            "client_id": "raw-id.apps.googleusercontent.com",
            "client_secret": "raw_secret_val"
        }
    })
    resp_json = client.post("/api/auth/save_credentials", json={"json_content": raw_payload})
    assert resp_json.status_code == 200
    saved_data2 = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_data2["installed"]["client_id"] == "raw-id.apps.googleusercontent.com"

    # 3. Upload client_secret.json file
    upload_payload = json.dumps({
        "installed": {
            "client_id": "uploaded-id.apps.googleusercontent.com",
            "client_secret": "uploaded_secret"
        }
    })
    resp_upload = client.post(
        "/api/auth/upload_client_secrets",
        files={"file": ("client_secret.json", upload_payload.encode("utf-8"), "application/json")}
    )
    assert resp_upload.status_code == 200
    assert resp_upload.json()["status"] == "success"
    saved_data3 = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_data3["installed"]["client_id"] == "uploaded-id.apps.googleusercontent.com"

