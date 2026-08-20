"""
Adversarial edge-case test suite for Milestone 4 (Card Processing & Synthetic Fallback).
"""

import os
from pathlib import Path
from unittest.mock import patch

import cv2
import pytest
from PIL import Image

from src.models import AudioClip, RedditComment, RedditPost, ScrapedContent
from src.video.composer import (
    ShortsVideoComposer,
    apply_card_shadow_and_corners,
    create_fallback_card,
)


# ============================================================================
# Task 1: Adversarial Tests for apply_card_shadow_and_corners
# ============================================================================

def test_apply_card_shadow_small_images():
    """Test 1x1, 2x2, 10x10 images with apply_card_shadow_and_corners."""
    for size in [(1, 1), (2, 2), (10, 10)]:
        img = Image.new("RGBA", size, (255, 0, 0, 255))
        result = apply_card_shadow_and_corners(img, max_width=960, corner_radius=5, shadow_blur=5)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        # Canvas dimensions should be image size + shadow_blur * 4
        assert result.width == size[0] + 5 * 4
        assert result.height == size[1] + 5 * 4


def test_apply_card_shadow_large_images():
    """Test very large image (5000x5000) scaling down to max_width=960."""
    img = Image.new("RGBA", (5000, 2500), (0, 255, 0, 255))
    shadow_blur = 20
    result = apply_card_shadow_and_corners(img, max_width=960, corner_radius=24, shadow_blur=shadow_blur)
    
    assert isinstance(result, Image.Image)
    assert result.mode == "RGBA"
    # Original width 5000 scaled to max_width 960 -> height scaled to 480.
    # Canvas width = 960 + 80 = 1040, height = 480 + 80 = 560
    assert result.width == 960 + shadow_blur * 4
    assert result.height == 480 + shadow_blur * 4


def test_apply_card_shadow_extreme_aspect_ratios():
    """Test extreme aspect ratios (wide 5000x100, tall 100x5000)."""
    wide_img = Image.new("RGBA", (5000, 100), (0, 0, 255, 255))
    result_wide = apply_card_shadow_and_corners(wide_img, max_width=960, corner_radius=10, shadow_blur=10)
    assert result_wide.width == 960 + 40
    assert result_wide.height == int(100 * (960 / 5000)) + 40

    tall_img = Image.new("RGBA", (100, 5000), (255, 255, 0, 255))
    result_tall = apply_card_shadow_and_corners(tall_img, max_width=960, corner_radius=10, shadow_blur=10)
    assert result_tall.width == 100 + 40
    assert result_tall.height == 5000 + 40


def test_apply_card_shadow_non_rgba_modes(tmp_path):
    """Test non-RGBA image modes: RGB, L, 1, CMYK, P."""
    modes = ["RGB", "L", "1", "CMYK", "P"]
    for mode in modes:
        if mode == "1":
            img = Image.new(mode, (400, 200), 1)
        elif mode == "L":
            img = Image.new(mode, (400, 200), 128)
        elif mode == "CMYK":
            img = Image.new(mode, (400, 200), (0, 255, 255, 0))
        elif mode == "P":
            img = Image.new("RGB", (400, 200), (100, 150, 200)).convert("P")
        else:
            img = Image.new(mode, (400, 200), (200, 100, 50))
        
        result = apply_card_shadow_and_corners(img)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"

        # Also test loading from file path for non-RGBA
        ext = "jpg" if mode == "CMYK" else "png"
        file_path = tmp_path / f"test_{mode}.{ext}"
        img.save(file_path)
        result_file = apply_card_shadow_and_corners(file_path)
        assert isinstance(result_file, Image.Image)
        assert result_file.mode == "RGBA"



def test_apply_card_shadow_missing_files():
    """Test non-existent file paths raise FileNotFoundError or OSError."""
    with pytest.raises((FileNotFoundError, OSError)):
        apply_card_shadow_and_corners("non_existent_file_xyz_12345.png")

    with pytest.raises((FileNotFoundError, OSError)):
        apply_card_shadow_and_corners(Path("invalid/path/to/card.png"))


def test_apply_card_shadow_invalid_input_types():
    """Test passing invalid types raises TypeError."""
    invalid_inputs = [123, 45.67, ["path.png"], {"file": "path.png"}, None]
    for invalid in invalid_inputs:
        with pytest.raises(TypeError):
            apply_card_shadow_and_corners(invalid)


def test_apply_card_shadow_extreme_parameters():
    """Test zero and large values for corner_radius and shadow_blur."""
    img = Image.new("RGBA", (500, 300), (100, 100, 100, 255))
    
    # Corner radius 0, shadow blur 0
    res1 = apply_card_shadow_and_corners(img, corner_radius=0, shadow_blur=0)
    assert res1.width == 500
    assert res1.height == 300

    # Large values
    res2 = apply_card_shadow_and_corners(img, corner_radius=200, shadow_blur=50)
    assert res2.width == 500 + 50 * 4
    assert res2.height == 300 + 50 * 4


# ============================================================================
# Task 2: Stress Tests for Synthetic Fallback Rendering
# ============================================================================

@pytest.fixture
def synthetic_test_content():
    """Minimal ScrapedContent with missing image files to force fallback card generation."""
    post = RedditPost(
        post_id="adv123",
        title="Adversarial Test Question",
        author="adv_tester",
        subreddit="test",
        body="This is an adversarial test body.",
    )
    comments = [
        RedditComment(comment_id="c1", author="comm_tester1", body="Comment 1 text"),
        RedditComment(comment_id="c2", author="comm_tester2", body="Comment 2 text"),
    ]
    return ScrapedContent(
        post=post,
        comments=comments,
        op_card_image_path="non_existent_op_card.png",
        comment_card_image_paths=["non_existent_c1.png", "non_existent_c2.png"],
    )


@pytest.mark.parametrize(
    "exception_cls",
    [
        RuntimeError,
        ValueError,
        MemoryError,
        OSError,
        AttributeError,
        ZeroDivisionError,
    ],
)
def test_synthetic_fallback_moviepy_exceptions(tmp_path, synthetic_test_content, exception_cls):
    """Stress test synthetic fallback by forcing various MoviePy exceptions in _compose_moviepy."""
    output_mp4 = tmp_path / f"fallback_{exception_cls.__name__}.mp4"
    composer = ShortsVideoComposer()

    with patch.object(
        composer,
        "_compose_moviepy",
        side_effect=exception_cls(f"Forced {exception_cls.__name__}"),
    ):
        res_path = composer.compose(
            scraped_content=synthetic_test_content,
            audio_clips={},
            output_path=str(output_mp4),
        )

    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 0

    # Verify MP4 integrity with OpenCV
    cap = cv2.VideoCapture(res_path)
    try:
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert w == 1080
        assert h == 1920
        assert fps == 30
        assert frame_count > 0
    finally:
        cap.release()


def test_synthetic_fallback_with_missing_bg_and_audio(tmp_path, synthetic_test_content):
    """Verify synthetic fallback works when background video and audio clips are completely missing."""
    output_mp4 = tmp_path / "fallback_no_bg_no_audio.mp4"
    composer = ShortsVideoComposer()

    with patch.object(
        composer,
        "_compose_moviepy",
        side_effect=RuntimeError("MoviePy failed"),
    ):
        res_path = composer.compose(
            scraped_content=synthetic_test_content,
            audio_clips={},
            output_path=str(output_mp4),
            background_video_path="non_existent_bg_video.mp4",
        )

    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 0

    cap = cv2.VideoCapture(res_path)
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1080
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 1920
    finally:
        cap.release()
