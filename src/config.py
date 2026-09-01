"""Configuration management, constants, directory management, and asset resolution."""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Configurable directory paths
ASSETS_DIR = BASE_DIR / "assets"
BACKGROUNDS_DIR = ASSETS_DIR / "backgrounds"
TEMPLATES_DIR = ASSETS_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# Video specifications (YouTube Shorts 9:16 vertical format)
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30

# YouTube OAuth client secret location
PRIMARY_CLIENT_SECRET = ASSETS_DIR / "client_secret.json"

# Reddit Sock Puppet Session & Bot credentials file location
REDDIT_SESSION_FILE = ASSETS_DIR / "reddit_user_session.json"
REDDIT_CREDENTIALS_FILE = ASSETS_DIR / "reddit_credentials.json"
AI_OP_POSTS_FILE = OUTPUT_DIR / "ai_op_posts.json"

# AI OP Default Configuration
AI_OP_DEFAULT_MIN_COMMENTS = int(os.environ.get("AI_OP_MIN_COMMENTS", "2"))
AI_OP_DEFAULT_POLL_INTERVAL = int(os.environ.get("AI_OP_POLL_INTERVAL", "60"))
AI_OP_DEFAULT_MAX_WAIT_HOURS = int(os.environ.get("AI_OP_MAX_WAIT_HOURS", "24"))


def ensure_directories() -> None:
    """Auto-create all required application directories."""
    for directory in [ASSETS_DIR, BACKGROUNDS_DIR, TEMPLATES_DIR, OUTPUT_DIR, TEMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_client_secrets_path() -> Path:
    """
    Resolve the path for YouTube OAuth client secrets.
    Checks environment variable YOUTUBE_CLIENT_SECRETS_FILE first,
    then defaults to assets/client_secret.json.
    """
    env_path = os.environ.get("YOUTUBE_CLIENT_SECRETS_FILE")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    return PRIMARY_CLIENT_SECRET


CLIENT_SECRETS_PATH = get_client_secrets_path()


def get_reddit_credentials() -> Dict[str, str]:
    """
    Resolve Reddit bot account credentials from environment variables or saved json file.
    Returns dictionary with client_id, client_secret, username, password, user_agent.
    """
    creds = {
        "client_id": os.environ.get("REDDIT_CLIENT_ID", "").strip(),
        "client_secret": os.environ.get("REDDIT_CLIENT_SECRET", "").strip(),
        "username": os.environ.get("REDDIT_USERNAME", "").strip(),
        "password": os.environ.get("REDDIT_PASSWORD", "").strip(),
        "user_agent": os.environ.get("REDDIT_USER_AGENT", "python:ai-op-shorts-generator:v1.0 (by /u/RedditBot)").strip(),
    }

    # If any required credential is missing in env, check saved credentials file
    if not (creds["client_id"] and creds["client_secret"] and creds["username"] and creds["password"]):
        if REDDIT_CREDENTIALS_FILE.exists():
            try:
                with open(REDDIT_CREDENTIALS_FILE, "r", encoding="utf-8") as f:
                    file_creds = json.load(f)
                    if isinstance(file_creds, dict):
                        for k in ["client_id", "client_secret", "username", "password", "user_agent"]:
                            if not creds.get(k) and file_creds.get(k):
                                creds[k] = str(file_creds[k]).strip()
            except Exception as e:
                logger.warning(f"Failed to read Reddit credentials file ({e})")

    return creds


def save_reddit_credentials(
    client_id: str,
    client_secret: str,
    username: str,
    password: str,
    user_agent: Optional[str] = None,
) -> bool:
    """Save Reddit credentials to assets/reddit_credentials.json."""
    ensure_directories()
    ua = user_agent.strip() if user_agent and user_agent.strip() else f"python:ai-op-shorts-generator:v1.0 (by /u/{username.strip()})"
    data = {
        "client_id": client_id.strip(),
        "client_secret": client_secret.strip(),
        "username": username.strip(),
        "password": password.strip(),
        "user_agent": ua,
    }
    try:
        with open(REDDIT_CREDENTIALS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # Also update os.environ for current runtime session
        os.environ["REDDIT_CLIENT_ID"] = data["client_id"]
        os.environ["REDDIT_CLIENT_SECRET"] = data["client_secret"]
        os.environ["REDDIT_USERNAME"] = data["username"]
        os.environ["REDDIT_PASSWORD"] = data["password"]
        os.environ["REDDIT_USER_AGENT"] = data["user_agent"]
        return True
    except Exception as e:
        logger.error(f"Failed to save Reddit credentials: {e}")
        return False


def generate_fallback_background(
    output_path: Optional[Path] = None,
    duration_sec: float = 10.0,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    fps: int = VIDEO_FPS,
) -> Path:
    """
    Generate and cache a synthetic 9:16 vertical fallback video clip.
    Creates a dynamic background animation when minecraft_parkour.mp4 is missing.
    """
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height, and fps must be positive integers")

    target_path = output_path or (ASSETS_DIR / "fallback_background.mp4")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target_path), fourcc, float(fps), (width, height))
    total_frames = int(duration_sec * fps)

    for i in range(total_frames):
        t = i / float(total_frames) if total_frames > 0 else 0.0
        # Generate dynamic gradient frame
        b = np.linspace(30 + 20 * np.sin(t * np.pi * 2), 60, height, dtype=np.uint8)
        g = np.linspace(15, 40 + 20 * np.cos(t * np.pi * 2), height, dtype=np.uint8)
        r = np.linspace(40 + 30 * np.sin(t * np.pi * 4), 80, height, dtype=np.uint8)

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = b[:, None]
        frame[:, :, 1] = g[:, None]
        frame[:, :, 2] = r[:, None]
        writer.write(frame)

    writer.release()
    return target_path


def get_background_video_path(assets_dir: Optional[Path] = None) -> Path:
    """
    Returns assets/minecraft_parkour.mp4 if present, or generates and caches
    a synthetic 9:16 vertical fallback video (assets/fallback_background.mp4) if missing.
    """
    base_assets = assets_dir or ASSETS_DIR
    primary_bg = base_assets / "minecraft_parkour.mp4"
    if primary_bg.exists():
        return primary_bg

    fallback_bg = base_assets / "fallback_background.mp4"
    if not fallback_bg.exists():
        generate_fallback_background(fallback_bg)
    return fallback_bg
