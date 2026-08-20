"""Unit and integration tests for Batch Mode pipeline execution."""

import json
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.pipeline import run_batch_pipeline
from src.models import ScrapedContent, RedditPost, RedditComment


@pytest.mark.asyncio
async def test_run_batch_pipeline_dry_run(tmp_path):
    """Test batch pipeline generation with multiple subreddits in dry run mode."""
    mock_post1 = RedditPost(post_id="p1", title="Title 1", body="Body 1", author="user1", ups=100, subreddit="AskReddit")
    mock_post2 = RedditPost(post_id="p2", title="Title 2", body="Body 2", author="user2", ups=200, subreddit="funny")

    content1 = ScrapedContent(post=mock_post1, comments=[RedditComment(comment_id="c1", body="Comment 1", author="u3", ups=50)])
    content2 = ScrapedContent(post=mock_post2, comments=[RedditComment(comment_id="c2", body="Comment 2", author="u4", ups=60)])

    mock_audio = {"op": MagicMock(file_path=str(tmp_path / "op.mp3"), duration=5.0)}

    with patch("src.pipeline.fetch_reddit_post", side_effect=[content1, content2]), \
         patch("src.pipeline.generate_voiceover", AsyncMock(return_value=mock_audio)), \
         patch("src.pipeline.compose_shorts_video", return_value=str(tmp_path / "out.mp4")), \
         patch("src.pipeline.OUTPUT_DIR", tmp_path):

        status_calls = []
        async def mock_cb(stage, message, current, total):
            status_calls.append((current, total, stage))

        res = await run_batch_pipeline(
            subreddits=["AskReddit", "funny"],
            batch_count=2,
            dry_run=True,
            upload=False,
            status_callback=mock_cb,
        )

        assert res["is_batch"] is True
        assert res["total_requested"] == 2
        assert res["successful_count"] == 2
        assert len(status_calls) == 2

        batch_folder = Path(res["output_folder"])
        assert batch_folder.exists()

        summary_file = batch_folder / "batch_summary.json"
        assert summary_file.exists()

        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        assert summary_data["total_requested"] == 2
        assert summary_data["successful_count"] == 2
        assert summary_data["subreddits"] == ["AskReddit", "funny"]


@pytest.mark.asyncio
async def test_max_comments_configuration(tmp_path):
    """Verify max_comments parameter is passed down from run_batch_pipeline to fetch_reddit_post."""
    mock_post = RedditPost(post_id="p1", title="Title 1", body="Body 1", author="user1", ups=100, subreddit="AskReddit")
    content = ScrapedContent(post=mock_post, comments=[RedditComment(comment_id="c1", body="Comment 1", author="u3", ups=50)])
    mock_audio = {"op": MagicMock(file_path=str(tmp_path / "op.mp3"), duration=5.0)}

    with patch("src.pipeline.fetch_reddit_post", AsyncMock(return_value=content)) as mock_fetch, \
         patch("src.pipeline.generate_voiceover", AsyncMock(return_value=mock_audio)), \
         patch("src.pipeline.compose_shorts_video", return_value=str(tmp_path / "out.mp4")), \
         patch("src.pipeline.OUTPUT_DIR", tmp_path):

        await run_batch_pipeline(
            subreddits=["AskReddit"],
            batch_count=1,
            max_comments=4,
            dry_run=True,
        )

        mock_fetch.assert_called_once_with(subreddit="AskReddit", post_id=None, max_comments=4)
