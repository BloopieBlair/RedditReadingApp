"""Post History Manager for tracking processed post IDs across runs."""

import json
import logging
from pathlib import Path
from typing import Set, List
from src.config import OUTPUT_DIR, ensure_directories

logger = logging.getLogger(__name__)

HISTORY_FILE = Path(OUTPUT_DIR) / "seen_posts.json"


class PostHistoryManager:
    """Manages seen Reddit post IDs to ensure unique content on every run."""

    def __init__(self, history_file: Path = HISTORY_FILE):
        self.history_file = history_file
        ensure_directories()
        self._seen_ids: Set[str] = self._load()

    def _load(self) -> Set[str]:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return set(data)
            except Exception as e:
                logger.warning(f"Failed to load post history from {self.history_file}: {e}")
        return set()

    def is_seen(self, post_id: str) -> bool:
        return str(post_id).strip() in self._seen_ids

    def mark_seen(self, post_id: str) -> None:
        clean_id = str(post_id).strip()
        if clean_id:
            self._seen_ids.add(clean_id)
            self._save()

    def _save(self) -> None:
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(list(self._seen_ids), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save post history to {self.history_file}: {e}")
