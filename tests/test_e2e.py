"""End-to-end integration tests for the Reddit Reading Shorts pipeline.

These tests mock external dependencies (Reddit API, edge-tts, YouTube API)
and verify the full pipeline orchestration with synthetic/mock assets.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.config import VIDEO_HEIGHT, VIDEO_WIDTH, ensure_directories
from src.models import (
    AudioClip,
    RedditComment,
    RedditPost,
    ScrapedContent,
    UploadMetadata,
)
from src.pipeline import RedditReadingPipeline, run_pipeline


def _create_mock_scraped_content(tmp_dir: str) -> ScrapedContent:
    """Create a ScrapedContent with synthetic card images."""
    post = RedditPost(
        post_id="test_e2e_001",
        title="What's the funniest thing that happened to you today?",
        author="test_user",
        subreddit="AskReddit",
        body="I'll start: I accidentally waved back at someone who wasn't waving at me.",
        ups=15200,
        num_comments=3500,
    )
    comments = [
        RedditComment(
            comment_id="c1",
            author="funny_redditor",
            body="I tried to push a pull door in front of my crush. Multiple times.",
            ups=8700,
        ),
        RedditComment(
            comment_id="c2",
            author="another_user",
            body="My cat knocked over my coffee right onto my keyboard during a work meeting.",
            ups=5400,
        ),
    ]

    # Create synthetic card images
    op_card_path = os.path.join(tmp_dir, "op_card_test.png")
    _create_test_card_image(op_card_path, "OP Card")

    comment_card_paths = []
    for i in range(2):
        c_path = os.path.join(tmp_dir, f"comment_{i+1}_card_test.png")
        _create_test_card_image(c_path, f"Comment {i+1}")
        comment_card_paths.append(c_path)

    return ScrapedContent(
        post=post,
        comments=comments,
        op_card_image_path=op_card_path,
        comment_card_image_paths=comment_card_paths,
    )


def _create_test_card_image(path: str, label: str = "Test") -> None:
    """Generate a simple test card PNG image."""
    img = Image.new("RGBA", (960, 400), (26, 26, 27, 255))
    img.save(path)


def _create_mock_audio_clips(tmp_dir: str) -> dict:
    """Create mock AudioClip objects with minimal silent MP3 files."""
    clips = {}
    for key, text in [
        ("op", "What's the funniest thing today?"),
        ("comment_1", "I tried to push a pull door."),
        ("comment_2", "My cat knocked over my coffee."),
    ]:
        audio_path = os.path.join(tmp_dir, f"{key}.mp3")
        # Write minimal valid-ish MP3 data (silent frame)
        with open(audio_path, "wb") as f:
            f.write(b"\xff\xfb\x90\xc4" + b"\x00" * 413)
        clips[key] = AudioClip(
            clip_id=key,
            file_path=audio_path,
            duration_seconds=2.0,
            text=text,
        )
    return clips


class TestPipelineUnit:
    """Unit tests for RedditReadingPipeline class construction."""

    def test_pipeline_init_defaults(self):
        pipeline = RedditReadingPipeline()
        assert pipeline.voice == "en-US-ChristopherNeural"
        assert pipeline.dry_run is False
        assert pipeline.background_video is None

    def test_pipeline_init_custom(self):
        pipeline = RedditReadingPipeline(
            output_dir="/custom/out",
            voice="en-US-AnaNeural",
            background_video="/bg.mp4",
            dry_run=True,
        )
        assert pipeline.output_dir == "/custom/out"
        assert pipeline.voice == "en-US-AnaNeural"
        assert pipeline.background_video == "/bg.mp4"
        assert pipeline.dry_run is True


class TestEndToEndPipeline:
    """Integration tests running the full pipeline with mocked externals."""

    @pytest.mark.asyncio
    async def test_pipeline_dry_run_produces_manifest(self, tmp_path):
        """Full pipeline with mocked scraper/TTS/video should produce a manifest JSON."""
        output_dir = str(tmp_path / "output")
        video_output = str(tmp_path / "output" / "test_short.mp4")

        mock_scraped = _create_mock_scraped_content(str(tmp_path))
        mock_audio = _create_mock_audio_clips(str(tmp_path))

        # Create a tiny dummy video file for the composer output
        dummy_video_path = video_output
        os.makedirs(os.path.dirname(dummy_video_path), exist_ok=True)
        with open(dummy_video_path, "wb") as f:
            f.write(b"\x00" * 1024)

        with patch("src.pipeline.fetch_reddit_post", new_callable=AsyncMock) as mock_scrape, \
             patch("src.pipeline.generate_voiceover", new_callable=AsyncMock) as mock_tts, \
             patch("src.pipeline.compose_shorts_video") as mock_compose, \
             patch("src.pipeline.upload_short") as mock_upload:

            mock_scrape.return_value = mock_scraped
            mock_tts.return_value = mock_audio
            mock_compose.return_value = video_output
            mock_upload.return_value = {
                "status": "success",
                "video_id": "mock_id",
                "video_url": "https://youtube.com/shorts/mock_id",
                "dry_run": True,
            }

            result = await run_pipeline(
                subreddit="AskReddit",
                output_video_path=video_output,
                output_dir=output_dir,
                upload=True,
                dry_run=True,
            )

        # Verify pipeline result structure
        assert result["status"] == "success"
        assert result["video_path"] == video_output
        assert result["post_title"] == mock_scraped.post.title
        assert result["subreddit"] == "AskReddit"
        assert result["dry_run"] is True
        assert result["upload_result"] is not None
        assert result["upload_result"]["dry_run"] is True

        # Verify manifest was saved
        manifest_path = Path(output_dir) / f"manifest_{mock_scraped.post.post_id}.json"
        assert manifest_path.exists()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["status"] == "success"
        assert manifest["post_id"] == "test_e2e_001"
        assert "timestamp" in manifest
        assert "audio_clips" in manifest

    @pytest.mark.asyncio
    async def test_pipeline_no_upload(self, tmp_path):
        """Pipeline with upload=False should skip upload and return None upload_result."""
        output_dir = str(tmp_path / "output")
        video_output = str(tmp_path / "output" / "test_short.mp4")

        mock_scraped = _create_mock_scraped_content(str(tmp_path))
        mock_audio = _create_mock_audio_clips(str(tmp_path))

        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        with open(video_output, "wb") as f:
            f.write(b"\x00" * 1024)

        with patch("src.pipeline.fetch_reddit_post", new_callable=AsyncMock) as mock_scrape, \
             patch("src.pipeline.generate_voiceover", new_callable=AsyncMock) as mock_tts, \
             patch("src.pipeline.compose_shorts_video") as mock_compose:

            mock_scrape.return_value = mock_scraped
            mock_tts.return_value = mock_audio
            mock_compose.return_value = video_output

            result = await run_pipeline(
                subreddit="funny",
                output_video_path=video_output,
                output_dir=output_dir,
                upload=False,
            )

        assert result["status"] == "success"
        assert result["upload_result"] is None
        assert result["subreddit"] == "funny"

    @pytest.mark.asyncio
    async def test_pipeline_calls_all_stages(self, tmp_path):
        """Verify the pipeline invokes scraper, TTS, composer, and uploader in correct order."""
        output_dir = str(tmp_path / "output")
        video_output = str(tmp_path / "output" / "test_short.mp4")

        mock_scraped = _create_mock_scraped_content(str(tmp_path))
        mock_audio = _create_mock_audio_clips(str(tmp_path))

        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        with open(video_output, "wb") as f:
            f.write(b"\x00" * 1024)

        call_order = []

        async def track_scrape(*a, **kw):
            call_order.append("scrape")
            return mock_scraped

        async def track_tts(*a, **kw):
            call_order.append("tts")
            return mock_audio

        def track_compose(*a, **kw):
            call_order.append("compose")
            return video_output

        def track_upload(*a, **kw):
            call_order.append("upload")
            return {"status": "success", "dry_run": True}

        with patch("src.pipeline.fetch_reddit_post", side_effect=track_scrape), \
             patch("src.pipeline.generate_voiceover", side_effect=track_tts), \
             patch("src.pipeline.compose_shorts_video", side_effect=track_compose), \
             patch("src.pipeline.upload_short", side_effect=track_upload):

            await run_pipeline(
                subreddit="AskReddit",
                output_video_path=video_output,
                output_dir=output_dir,
                upload=True,
                dry_run=True,
            )

        assert call_order == ["scrape", "tts", "compose", "upload"]

    @pytest.mark.asyncio
    async def test_pipeline_voice_propagation(self, tmp_path):
        """Pipeline should pass the voice parameter to the TTS engine."""
        output_dir = str(tmp_path / "output")
        video_output = str(tmp_path / "output" / "test_short.mp4")

        mock_scraped = _create_mock_scraped_content(str(tmp_path))
        mock_audio = _create_mock_audio_clips(str(tmp_path))

        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        with open(video_output, "wb") as f:
            f.write(b"\x00" * 1024)

        with patch("src.pipeline.fetch_reddit_post", new_callable=AsyncMock) as mock_scrape, \
             patch("src.pipeline.generate_voiceover", new_callable=AsyncMock) as mock_tts, \
             patch("src.pipeline.compose_shorts_video") as mock_compose:

            mock_scrape.return_value = mock_scraped
            mock_tts.return_value = mock_audio
            mock_compose.return_value = video_output

            await run_pipeline(
                subreddit="AskReddit",
                output_video_path=video_output,
                output_dir=output_dir,
                voice="en-US-AnaNeural",
            )

        # Verify voice was passed through
        mock_tts.assert_called_once()
        call_kwargs = mock_tts.call_args
        assert call_kwargs.kwargs.get("voice") == "en-US-AnaNeural" or \
               (len(call_kwargs.args) > 2 and call_kwargs.args[2] == "en-US-AnaNeural")

    @pytest.mark.asyncio
    async def test_pipeline_manifest_contains_audio_metadata(self, tmp_path):
        """Manifest should contain audio clip metadata (durations, file paths)."""
        output_dir = str(tmp_path / "output")
        video_output = str(tmp_path / "output" / "test_short.mp4")

        mock_scraped = _create_mock_scraped_content(str(tmp_path))
        mock_audio = _create_mock_audio_clips(str(tmp_path))

        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        with open(video_output, "wb") as f:
            f.write(b"\x00" * 1024)

        with patch("src.pipeline.fetch_reddit_post", new_callable=AsyncMock) as mock_scrape, \
             patch("src.pipeline.generate_voiceover", new_callable=AsyncMock) as mock_tts, \
             patch("src.pipeline.compose_shorts_video") as mock_compose:

            mock_scrape.return_value = mock_scraped
            mock_tts.return_value = mock_audio
            mock_compose.return_value = video_output

            result = await run_pipeline(
                subreddit="AskReddit",
                output_video_path=video_output,
                output_dir=output_dir,
            )

        assert "audio_clips" in result
        audio_meta = result["audio_clips"]
        assert "op" in audio_meta
        assert audio_meta["op"]["duration_seconds"] == 2.0
        assert "comment_1" in audio_meta
        assert "comment_2" in audio_meta


class TestRunPipelineFunction:
    """Tests for the run_pipeline() functional wrapper."""

    @pytest.mark.asyncio
    async def test_run_pipeline_creates_pipeline_instance(self, tmp_path):
        """run_pipeline should create and execute a RedditReadingPipeline."""
        output_dir = str(tmp_path / "output")
        video_output = str(tmp_path / "output" / "short.mp4")

        mock_scraped = _create_mock_scraped_content(str(tmp_path))
        mock_audio = _create_mock_audio_clips(str(tmp_path))

        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        with open(video_output, "wb") as f:
            f.write(b"\x00" * 1024)

        with patch("src.pipeline.fetch_reddit_post", new_callable=AsyncMock) as mock_scrape, \
             patch("src.pipeline.generate_voiceover", new_callable=AsyncMock) as mock_tts, \
             patch("src.pipeline.compose_shorts_video") as mock_compose:

            mock_scrape.return_value = mock_scraped
            mock_tts.return_value = mock_audio
            mock_compose.return_value = video_output

            result = await run_pipeline(
                subreddit="funny",
                output_video_path=video_output,
                output_dir=output_dir,
                dry_run=True,
            )

        assert result["status"] == "success"
        assert result["subreddit"] == "funny"
