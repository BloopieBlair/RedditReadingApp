"""Unit tests for AIPostTracker."""

import pytest
import json
from pathlib import Path
from src.poster.tracker import AIPostTracker
from src.models import AIPostRecord


class TestAIPostTracker:
    """Test AIPostTracker JSON persistence and CRUD operations."""

    def test_tracker_add_and_get_post(self, tmp_path):
        manifest_file = tmp_path / "ai_op_test_posts.json"
        tracker = AIPostTracker(file_path=manifest_file)

        record = AIPostRecord(
            post_id="test123",
            subreddit="AskReddit",
            title="What is your favourite snack?",
            url="https://reddit.com/r/AskReddit/comments/test123/",
            min_comments_target=3,
        )

        tracker.add_post(record)
        fetched = tracker.get_post("test123")
        assert fetched is not None
        assert fetched.post_id == "test123"
        assert fetched.min_comments_target == 3
        assert fetched.status == "submitted"

        # Verify disk persistence
        assert manifest_file.exists()
        tracker2 = AIPostTracker(file_path=manifest_file)
        assert tracker2.get_post("test123") is not None

    def test_tracker_list_posts_with_filter(self, tmp_path):
        manifest_file = tmp_path / "ai_op_test_posts.json"
        tracker = AIPostTracker(file_path=manifest_file)

        p1 = AIPostRecord(post_id="p1", subreddit="AskReddit", title="Title 1", created_utc=100.0, status="submitted")
        p2 = AIPostRecord(post_id="p2", subreddit="funny", title="Title 2", created_utc=200.0, status="rendered")
        p3 = AIPostRecord(post_id="p3", subreddit="AskReddit", title="Title 3", created_utc=300.0, status="waiting_for_comments")

        tracker.add_post(p1)
        tracker.add_post(p2)
        tracker.add_post(p3)

        all_posts = tracker.list_posts()
        assert len(all_posts) == 3
        # Sorted newest first (created_utc descending)
        assert all_posts[0].post_id == "p3"

        ask_posts = tracker.list_posts(subreddit="AskReddit")
        assert len(ask_posts) == 2

        rendered_posts = tracker.list_posts(status="rendered")
        assert len(rendered_posts) == 1
        assert rendered_posts[0].post_id == "p2"

    def test_tracker_update_comments_status_transition(self, tmp_path):
        manifest_file = tmp_path / "ai_op_test_posts.json"
        tracker = AIPostTracker(file_path=manifest_file)

        record = AIPostRecord(
            post_id="p_test",
            subreddit="AskReddit",
            title="Question?",
            min_comments_target=2,
            current_comments_count=0,
            status="submitted",
        )
        tracker.add_post(record)

        # 1 comment -> waiting_for_comments
        tracker.update_comments("p_test", comments_count=1, previews=["Meatbag 1"])
        updated = tracker.get_post("p_test")
        assert updated.current_comments_count == 1
        assert updated.status == "waiting_for_comments"

        # 2 comments -> ready_to_render
        tracker.update_comments("p_test", comments_count=2, previews=["Meatbag 1", "Meatbag 2"])
        updated2 = tracker.get_post("p_test")
        assert updated2.current_comments_count == 2
        assert updated2.status == "ready_to_render"

    def test_tracker_mark_rendered_and_delete(self, tmp_path):
        manifest_file = tmp_path / "ai_op_test_posts.json"
        tracker = AIPostTracker(file_path=manifest_file)

        record = AIPostRecord(post_id="p_del", subreddit="AskReddit", title="Title")
        tracker.add_post(record)

        tracker.mark_rendered("p_del", video_path="output/video.mp4", output_folder="output/folder")
        p = tracker.get_post("p_del")
        assert p.status == "rendered"
        assert p.rendered_video_path == "output/video.mp4"

        assert tracker.delete_post("p_del") is True
        assert tracker.get_post("p_del") is None
