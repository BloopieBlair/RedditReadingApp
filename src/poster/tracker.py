"""Tracker for storing, querying, and updating AI OP posts across their lifecycle."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.config import AI_OP_POSTS_FILE, ensure_directories
from src.models import AIPostRecord

logger = logging.getLogger(__name__)


class AIPostTracker:
    """Manages persistent tracking of AI OP submissions in a JSON manifest file."""

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = Path(file_path) if file_path is not None else Path(AI_OP_POSTS_FILE)
        ensure_directories()
        self._posts: Dict[str, AIPostRecord] = {}
        self._load()

    def _load(self) -> None:
        """Load tracked posts from JSON file."""
        self._posts = {}
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and item.get("post_id"):
                                record = AIPostRecord.from_dict(item)
                                self._posts[record.post_id] = record
            except Exception as e:
                logger.warning(f"Failed to load AI OP posts from {self.file_path}: {e}")

    def _save(self) -> None:
        """Save tracked posts to JSON file."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            records_list = [p.to_dict() for p in self._posts.values()]
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(records_list, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save AI OP posts to {self.file_path}: {e}")

    def add_post(self, record: AIPostRecord) -> AIPostRecord:
        """Add or update an AI post record."""
        self._posts[record.post_id] = record
        self._save()
        logger.info(f"Tracked new AI OP post: [{record.post_id}] in r/{record.subreddit}")
        return record

    def get_post(self, post_id: str) -> Optional[AIPostRecord]:
        """Retrieve a tracked post record by post ID."""
        clean_id = str(post_id).replace("t3_", "").strip()
        return self._posts.get(clean_id)

    def list_posts(
        self,
        status: Optional[str] = None,
        subreddit: Optional[str] = None,
    ) -> List[AIPostRecord]:
        """
        List tracked post records with optional filtering by status or subreddit.
        Sorted by created_utc descending (newest first).
        """
        results = list(self._posts.values())

        if status:
            results = [p for p in results if p.status.lower() == status.lower()]

        if subreddit:
            clean_sub = subreddit.replace("r/", "").strip().lower()
            results = [p for p in results if p.subreddit.lower() == clean_sub]

        results.sort(key=lambda p: p.created_utc, reverse=True)
        return results

    def update_post(self, post_id: str, **kwargs) -> Optional[AIPostRecord]:
        """Update fields on a tracked post record."""
        record = self.get_post(post_id)
        if not record:
            return None

        for k, v in kwargs.items():
            if hasattr(record, k):
                setattr(record, k, v)

        self._save()
        return record

    def update_comments(
        self,
        post_id: str,
        comments_count: int,
        previews: Optional[List[str]] = None,
        last_checked_at: Optional[str] = None,
    ) -> Optional[AIPostRecord]:
        """Update comment counts and preview snippets."""
        record = self.get_post(post_id)
        if not record:
            return None

        record.current_comments_count = comments_count
        if previews is not None:
            record.top_comments_preview = previews
        if last_checked_at:
            record.last_checked_at = last_checked_at

        # Transition status based on comment count
        if record.status in ("submitted", "waiting_for_comments"):
            if record.current_comments_count >= record.min_comments_target:
                record.status = "ready_to_render"
            else:
                record.status = "waiting_for_comments"

        self._save()
        return record

    def mark_rendered(
        self,
        post_id: str,
        video_path: str,
        output_folder: Optional[str] = None,
    ) -> Optional[AIPostRecord]:
        """Mark post as successfully rendered into a video."""
        record = self.get_post(post_id)
        if not record:
            return None

        record.status = "rendered"
        record.rendered_video_path = video_path
        if output_folder:
            record.rendered_folder = output_folder

        self._save()
        logger.info(f"Marked AI OP post [{post_id}] as rendered: {video_path}")
        return record

    def delete_post(self, post_id: str) -> bool:
        """Remove a tracked post from history."""
        clean_id = str(post_id).replace("t3_", "").strip()
        if clean_id in self._posts:
            del self._posts[clean_id]
            self._save()
            return True
        return False
