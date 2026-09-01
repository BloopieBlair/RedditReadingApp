"""Unit tests for AIPostWatcher and autonomous workflow orchestrator."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from src.poster.watcher import AIPostWatcher
from src.poster.tracker import AIPostTracker
from src.models import AIPostRecord


class TestAIPostWatcher:
    """Test AIPostWatcher comment checking and automated video triggering."""

    @pytest.mark.asyncio
    async def test_check_post_comments_simulated(self, tmp_path):
        tracker = AIPostTracker(file_path=tmp_path / "sim_posts.json")
        record = AIPostRecord(
            post_id="sim_123456",
            subreddit="AskReddit",
            title="Simulated question?",
            min_comments_target=2,
            is_simulated=True,
        )
        tracker.add_post(record)

        watcher = AIPostWatcher(tracker=tracker)
        res = await watcher.check_post_comments("sim_123456")

        assert res["status"] == "success"
        assert res["current_comments_count"] >= 2
        assert res["is_ready"] is True

    @pytest.mark.asyncio
    async def test_check_post_comments_live_mock(self, tmp_path):
        tracker = AIPostTracker(file_path=tmp_path / "live_posts.json")
        record = AIPostRecord(
            post_id="live999",
            subreddit="AskReddit",
            title="Live question?",
            min_comments_target=2,
            is_simulated=False,
        )
        tracker.add_post(record)

        watcher = AIPostWatcher(tracker=tracker)

        # Mock Reddit JSON structure (post + comments listing)
        mock_reddit_json = [
            {"data": {"children": [{"data": {"id": "live999", "title": "Live question?"}}]}},
            {
                "data": {
                    "children": [
                        {"data": {"author": "AutoModerator", "body": "Rule reminder", "stickied": True}},
                        {"data": {"author": "FunnyUser1", "body": "Hilarious story here", "ups": 42}},
                        {"data": {"author": "WittyUser2", "body": "Another great reply", "ups": 15}},
                    ]
                }
            }
        ]

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_reddit_json

            res = await watcher.check_post_comments("live999")
            assert res["status"] == "success"
            # AutoModerator filtered out -> 2 valid meatbag comments
            assert res["current_comments_count"] == 2
            assert res["is_ready"] is True
            assert len(res["previews"]) == 2

    @pytest.mark.asyncio
    async def test_render_post_video_dispatches_pipeline(self, tmp_path):
        tracker = AIPostTracker(file_path=tmp_path / "render_posts.json")
        record = AIPostRecord(
            post_id="render123",
            subreddit="AskReddit",
            title="Render Question?",
            min_comments_target=2,
        )
        tracker.add_post(record)

        watcher = AIPostWatcher(tracker=tracker)

        mock_pipeline_result = {
            "status": "success",
            "video_path": str(tmp_path / "video.mp4"),
            "output_folder": str(tmp_path),
        }

        with patch("src.poster.watcher.RedditReadingPipeline.execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_pipeline_result

            res = await watcher.render_post_video(post_id="render123")
            assert res["status"] == "success"
            assert res["video_path"] == str(tmp_path / "video.mp4")

            # Check tracker status updated
            updated_record = tracker.get_post("render123")
            assert updated_record.status == "rendered"
            assert updated_record.rendered_video_path == str(tmp_path / "video.mp4")

    @pytest.mark.asyncio
    async def test_run_ai_op_workflow_dry_run(self, tmp_path):
        tracker = AIPostTracker(file_path=tmp_path / "workflow_posts.json")
        watcher = AIPostWatcher(tracker=tracker)

        mock_pipeline_result = {
            "status": "success",
            "video_path": str(tmp_path / "workflow_video.mp4"),
            "output_folder": str(tmp_path),
        }

        with patch("src.poster.watcher.RedditReadingPipeline.execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_pipeline_result

            res = await watcher.run_ai_op_workflow(
                subreddit="AskReddit",
                theme="testing",
                dry_run=True,
                poll_interval=1,
            )

            assert res["status"] == "success"
            assert res["video_path"] == str(tmp_path / "workflow_video.mp4")
            assert tracker.get_post(res["post_id"]) is not None
