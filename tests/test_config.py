"""Unit tests for src/config.py configuration management and asset generators."""

from pathlib import Path
import pytest
import cv2
import src.config as config


def test_config_constants():
    """Verify video resolution and frame rate constants."""
    assert config.VIDEO_WIDTH == 1080
    assert config.VIDEO_HEIGHT == 1920
    assert config.VIDEO_FPS == 30
    assert config.ASSETS_DIR.name == "assets"
    assert config.TEMPLATES_DIR.name == "templates"
    assert config.OUTPUT_DIR.name == "output"
    assert config.TEMP_DIR.name == "temp"


def test_ensure_directories(tmp_path, monkeypatch):
    """Test directory creation function."""
    assets = tmp_path / "assets"
    templates = assets / "templates"
    output = tmp_path / "output"
    temp = tmp_path / "temp"

    monkeypatch.setattr(config, "ASSETS_DIR", assets)
    monkeypatch.setattr(config, "TEMPLATES_DIR", templates)
    monkeypatch.setattr(config, "OUTPUT_DIR", output)
    monkeypatch.setattr(config, "TEMP_DIR", temp)

    config.ensure_directories()

    assert assets.is_dir()
    assert templates.is_dir()
    assert output.is_dir()
    assert temp.is_dir()


def test_client_secrets_path_fallback(tmp_path, monkeypatch):
    """Test resolution logic for YouTube client secret JSON path."""
    primary = tmp_path / "assets" / "client_secret.json"
    custom_env_file = tmp_path / "custom" / "my_secret.json"

    monkeypatch.setattr(config, "PRIMARY_CLIENT_SECRET", primary)
    monkeypatch.delenv("YOUTUBE_CLIENT_SECRETS_FILE", raising=False)

    # Scenario 1: No env var set -> returns primary default path
    assert config.get_client_secrets_path() == primary

    # Scenario 2: Env var set and file exists -> returns env path
    custom_env_file.parent.mkdir(parents=True, exist_ok=True)
    custom_env_file.touch()
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRETS_FILE", str(custom_env_file))
    assert config.get_client_secrets_path() == custom_env_file

    # Scenario 3: Env var set to non-existent file -> returns primary default path
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRETS_FILE", str(tmp_path / "non_existent.json"))
    assert config.get_client_secrets_path() == primary


def test_generate_fallback_background(tmp_path):
    """Test synthetic background video generation and video parameters."""
    target_video = tmp_path / "test_fallback.mp4"
    generated_path = config.generate_fallback_background(
        output_path=target_video, duration_sec=1.0, width=1080, height=1920, fps=30
    )

    assert generated_path.exists()
    assert generated_path == target_video
    assert target_video.stat().st_size > 0

    # Verify video properties using OpenCV
    cap = cv2.VideoCapture(str(generated_path))
    assert cap.isOpened()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert width == 1080
    assert height == 1920
    assert fps == 30
    assert frame_count == 30


def test_get_background_video_path(tmp_path):
    """Test fallback generation when minecraft_parkour.mp4 is missing."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. When missing minecraft_parkour.mp4, should generate fallback_background.mp4
    bg_path = config.get_background_video_path(assets_dir=assets_dir)
    assert bg_path.exists()
    assert bg_path.name == "fallback_background.mp4"

    # 2. When minecraft_parkour.mp4 exists, should return minecraft_parkour.mp4
    parkour_file = assets_dir / "minecraft_parkour.mp4"
    parkour_file.touch()
    bg_path_primary = config.get_background_video_path(assets_dir=assets_dir)
    assert bg_path_primary == parkour_file
