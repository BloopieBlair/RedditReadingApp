"""
Unit and integration tests for the video production module.
Tests BackgroundManager, ShortsVideoComposer, drop shadow/rounded corners, and synthetic fallback rendering.
"""

import os
from pathlib import Path
from unittest.mock import patch

import cv2
import pytest
from PIL import Image

from src.models import AudioClip, RedditComment, RedditPost, ScrapedContent
from src.tts.voice_engine import generate_fallback_audio
from src.video import (
    BackgroundManager,
    ShortsVideoComposer,
    apply_card_shadow_and_corners,
    compose_shorts_video,
    get_background_clip,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def sample_scraped_content(temp_dir):
    """Creates a sample ScrapedContent object with rendered card PNG images."""
    post = RedditPost(
        post_id="post123",
        title="What is the most interesting fact you know?",
        author="fact_lover",
        subreddit="AskReddit",
        body="Share your favorite mind-blowing facts below!",
    )
    comments = [
        RedditComment(
            comment_id="comm1",
            author="space_geek",
            body="Neutron stars can spin at a rate of 600 rotations per second.",
        ),
        RedditComment(
            comment_id="comm2",
            author="ocean_fan",
            body="We have mapped more of the surface of Mars than the ocean floor.",
        ),
    ]

    # Create dummy PNG cards
    op_card_path = temp_dir / "op_card.png"
    c1_card_path = temp_dir / "c1_card.png"
    c2_card_path = temp_dir / "c2_card.png"

    img_op = Image.new("RGBA", (800, 400), (255, 255, 255, 255))
    img_op.save(op_card_path)

    img_c1 = Image.new("RGBA", (800, 300), (250, 250, 250, 255))
    img_c1.save(c1_card_path)

    img_c2 = Image.new("RGBA", (800, 300), (245, 245, 245, 255))
    img_c2.save(c2_card_path)

    return ScrapedContent(
        post=post,
        comments=comments,
        op_card_image_path=str(op_card_path),
        comment_card_image_paths=[str(c1_card_path), str(c2_card_path)],
    )


@pytest.fixture
def sample_audio_clips(temp_dir):
    """Creates sample WAV audio clips and returns AudioClip dictionary."""
    op_audio_path = temp_dir / "op.wav"
    c1_audio_path = temp_dir / "c1.wav"
    c2_audio_path = temp_dir / "c2.wav"

    dur_op = generate_fallback_audio(op_audio_path, 2.0)
    dur_c1 = generate_fallback_audio(c1_audio_path, 1.5)
    dur_c2 = generate_fallback_audio(c2_audio_path, 1.5)

    return {
        "op": AudioClip(
            clip_id="op", file_path=str(op_audio_path), duration_seconds=dur_op
        ),
        "comment_1": AudioClip(
            clip_id="c1", file_path=str(c1_audio_path), duration_seconds=dur_c1
        ),
        "comment_2": AudioClip(
            clip_id="c2", file_path=str(c2_audio_path), duration_seconds=dur_c2
        ),
    }


def test_video_package_exports():
    """Verify src.video package exposes required interfaces."""
    from src.video import (
        BackgroundManager,
        ShortsVideoComposer,
        apply_card_shadow_and_corners,
        compose_shorts_video,
        get_background_clip,
    )

    assert BackgroundManager is not None
    assert ShortsVideoComposer is not None
    assert callable(get_background_clip)
    assert callable(compose_shorts_video)
    assert callable(apply_card_shadow_and_corners)


def test_background_manager_fallback():
    """Verify BackgroundManager resolves video source and creates fallback background if missing."""
    manager = BackgroundManager(background_video_path="non_existent_file.mp4")
    video_source = manager.get_video_source()
    assert os.path.exists(video_source)
    assert video_source.endswith(".mp4")


def test_background_manager_crop_and_subclip():
    """Verify get_background_clip crops to 9:16 (1080x1920) @ 30fps and trims duration."""
    target_dur = 3.0
    clip = get_background_clip(target_duration=target_dur)
    try:
        assert clip.size == (1080, 1920)
        assert abs(clip.duration - target_dur) < 0.1
    finally:
        clip.close()


def test_apply_card_shadow_and_corners(temp_dir):
    """Verify apply_card_shadow_and_corners resizes, adds rounded corners and drop shadow."""
    test_img_path = temp_dir / "test_card.png"
    img = Image.new("RGBA", (1200, 600), (255, 255, 255, 255))
    img.save(test_img_path)

    shadow_blur = 20
    processed_img = apply_card_shadow_and_corners(
        image_path=str(test_img_path),
        max_width=960,
        corner_radius=24,
        shadow_blur=shadow_blur,
    )

    assert isinstance(processed_img, Image.Image)
    assert processed_img.mode == "RGBA"
    # Max width 960 + padding (shadow_blur * 2 on each side = +80) -> width 1040
    assert processed_img.width == 960 + shadow_blur * 4
    # Height original 600 scaled to 480 + padding 80 -> height 560
    assert processed_img.height == 480 + shadow_blur * 4


def test_compose_shorts_video_success(sample_scraped_content, sample_audio_clips, temp_dir):
    """Verify compose_shorts_video renders a valid 1080x1920 30fps MP4 video file."""
    output_path = str(temp_dir / "output_short.mp4")
    result_path = compose_shorts_video(
        scraped_content=sample_scraped_content,
        audio_clips=sample_audio_clips,
        output_path=output_path,
    )

    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 0

    cap = cv2.VideoCapture(result_path)
    try:
        assert cap.isOpened()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert width == 1080
        assert height == 1920
        assert fps == 30
        assert frame_count > 0
    finally:
        cap.release()


def test_compose_shorts_video_moviepy_failure_fallback(
    sample_scraped_content, sample_audio_clips, temp_dir
):
    """Verify synthetic fallback renderer creates valid MP4 when MoviePy raises an exception."""
    output_path = str(temp_dir / "fallback_short.mp4")

    composer = ShortsVideoComposer()

    with patch.object(
        composer,
        "_compose_moviepy",
        side_effect=RuntimeError("Simulated MoviePy Rendering Error"),
    ):
        result_path = composer.compose(
            scraped_content=sample_scraped_content,
            audio_clips=sample_audio_clips,
            output_path=output_path,
        )

    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 0

    cap = cv2.VideoCapture(result_path)
    try:
        assert cap.isOpened()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

        assert width == 1080
        assert height == 1920
        assert fps == 30
    finally:
        cap.release()


def test_compose_shorts_video_missing_comments(sample_scraped_content, temp_dir):
    """Verify video composition with single OP clip and no comment clips."""
    op_audio_path = temp_dir / "op_only.wav"
    dur_op = generate_fallback_audio(op_audio_path, 2.5)

    single_audio = {
        "op": AudioClip(
            clip_id="op", file_path=str(op_audio_path), duration_seconds=dur_op
        )
    }

    output_path = str(temp_dir / "op_only_short.mp4")
    result_path = compose_shorts_video(
        scraped_content=sample_scraped_content,
        audio_clips=single_audio,
        output_path=output_path,
    )

    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 0


def test_apply_card_shadow_and_corners_max_height_scaling(temp_dir):
    """Verify apply_card_shadow_and_corners scales down super tall OP card image to prevent vertical overflow."""
    tall_card_path = temp_dir / "tall_op_card.png"
    img = Image.new("RGBA", (960, 2600), (255, 255, 255, 255))
    img.save(tall_card_path)

    shadow_blur = 20
    processed_img = apply_card_shadow_and_corners(
        image_path=str(tall_card_path),
        max_width=960,
        max_height=1600,
        shadow_blur=shadow_blur,
    )

    # Processed card height (excluding 80px shadow canvas padding) must be <= max_height (1600)
    assert processed_img.height <= 1600 + shadow_blur * 4
    # Bottom position when placed at y=140 should fit inside 1920px canvas with padding
    bottom_y = 140 + processed_img.height
    assert bottom_y < 1900


def test_get_gpu_status():
    """Verify get_gpu_status detects GPU hardware acceleration or returns CPU fallback."""
    from src.video.composer import get_gpu_status
    gpu_info = get_gpu_status()
    assert isinstance(gpu_info, dict)
    assert "has_gpu" in gpu_info
    assert "encoder" in gpu_info
    assert "vendor" in gpu_info
    assert "label" in gpu_info
    assert isinstance(gpu_info["has_gpu"], bool)


def test_api_get_system_gpu():
    """Verify FastAPI /api/system/gpu endpoint returns GPU detection payload."""
    from fastapi.testclient import TestClient
    from src.web.app import app
    client = TestClient(app)
    resp = client.get("/api/system/gpu")
    assert resp.status_code == 200
    data = resp.json()
    assert "has_gpu" in data
    assert "label" in data
    assert "encoder" in data

