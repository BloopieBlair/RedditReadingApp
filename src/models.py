"""Data models and serialization helpers for Reddit Reading YouTube Shorts app."""

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


@dataclass
class RedditComment:
    """Dataclass representing a Reddit comment."""
    comment_id: str
    author: str
    body: str
    ups: int = 0
    created_utc: float = 0.0
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RedditComment":
        if not d.get("comment_id") or not isinstance(d.get("comment_id"), str):
            raise ValueError("comment_id must be a non-empty string")
        return cls(
            comment_id=str(d["comment_id"]),
            author=str(d.get("author", "anonymous")),
            body=str(d.get("body", "")),
            ups=int(d.get("ups", 0)),
            created_utc=float(d.get("created_utc", 0.0)),
            parent_id=d.get("parent_id"),
        )


@dataclass
class RedditPost:
    """Dataclass representing a Reddit submission/post."""
    post_id: str
    title: str
    author: str
    subreddit: str
    body: str = ""
    ups: int = 0
    created_utc: float = 0.0
    num_comments: int = 0
    url: str = ""
    image_url: Optional[str] = None

    def __post_init__(self):
        if not self.title or not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RedditPost":
        if not d.get("post_id") or not isinstance(d.get("post_id"), str):
            raise ValueError("post_id must be a non-empty string")
        title = d.get("title")
        if not title or not isinstance(title, str) or not str(title).strip():
            raise ValueError("title must be a non-empty string")
        return cls(
            post_id=str(d["post_id"]),
            title=str(title),
            author=str(d.get("author", "anonymous")),
            subreddit=str(d.get("subreddit", "AskReddit")),
            body=str(d.get("body", "")),
            ups=int(d.get("ups", 0)),
            created_utc=float(d.get("created_utc", 0.0)),
            num_comments=int(d.get("num_comments", 0)),
            url=str(d.get("url", "")),
            image_url=d.get("image_url"),
        )


@dataclass
class ScrapedContent:
    """Dataclass encapsulating scraped post, comments, and rendered card images."""
    post: RedditPost
    comments: List[RedditComment] = field(default_factory=list)
    op_card_image_path: Optional[str] = None
    comment_card_image_paths: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.comments is None:
            self.comments = []
        if self.comment_card_image_paths is None:
            self.comment_card_image_paths = []

    @property
    def post_title(self) -> str:
        return self.post.title

    @property
    def post_body(self) -> str:
        return self.post.body

    @property
    def author(self) -> str:
        return self.post.author

    @property
    def subreddit(self) -> str:
        return self.post.subreddit

    @property
    def comment1_card_image_path(self) -> Optional[str]:
        return self.comment_card_image_paths[0] if len(self.comment_card_image_paths) > 0 else None

    @property
    def comment2_card_image_path(self) -> Optional[str]:
        return self.comment_card_image_paths[1] if len(self.comment_card_image_paths) > 1 else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "post": self.post.to_dict(),
            "comments": [c.to_dict() for c in self.comments if c is not None],
            "op_card_image_path": self.op_card_image_path,
            "comment_card_image_paths": list(self.comment_card_image_paths),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScrapedContent":
        post_data = d.get("post")
        if not isinstance(post_data, dict):
            raise ValueError("ScrapedContent from_dict requires a valid 'post' dictionary")
        post = RedditPost.from_dict(post_data)
        comments_data = d.get("comments")
        if comments_data is None:
            comments_data = []
        comments = [RedditComment.from_dict(c) for c in comments_data if isinstance(c, dict)]
        return cls(
            post=post,
            comments=comments,
            op_card_image_path=d.get("op_card_image_path"),
            comment_card_image_paths=d.get("comment_card_image_paths") or [],
        )


@dataclass
class AudioClip:
    """Dataclass representing a synthesized audio segment."""
    clip_id: str
    file_path: str
    duration_seconds: float
    text: str = ""

    def __post_init__(self):
        if math.isnan(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("duration_seconds must be a non-negative, non-NaN number")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AudioClip":
        if not d.get("clip_id"):
            raise ValueError("clip_id must be provided")
        if not d.get("file_path"):
            raise ValueError("file_path must be provided")
        duration = float(d.get("duration_seconds", 0.0))
        if math.isnan(duration):
            raise ValueError("duration_seconds must be a non-negative, non-NaN number")
        return cls(
            clip_id=str(d["clip_id"]),
            file_path=str(d["file_path"]),
            duration_seconds=duration,
            text=str(d.get("text", "")),
        )


@dataclass
class VideoRenderConfig:
    """Dataclass specifying video rendering parameters."""
    width: int = 1080
    height: int = 1920
    fps: int = 30
    max_duration_seconds: float = 60.0
    card_display_margin: float = 0.2
    background_video_path: Optional[str] = None
    enable_subtitles: bool = True
    subtitle_style: str = "yellow_pill"

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("width, height, and fps must be positive integers")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoRenderConfig":
        w = int(d.get("width", 1080))
        h = int(d.get("height", 1920))
        fps = int(d.get("fps", 30))
        if w <= 0 or h <= 0 or fps <= 0:
            raise ValueError("width, height, and fps must be positive integers")
        return cls(
            width=w,
            height=h,
            fps=fps,
            max_duration_seconds=float(d.get("max_duration_seconds", 60.0)),
            card_display_margin=float(d.get("card_display_margin", 0.2)),
            background_video_path=d.get("background_video_path"),
            enable_subtitles=bool(d.get("enable_subtitles", True)),
            subtitle_style=str(d.get("subtitle_style", "yellow_pill")),
        )


@dataclass
class UploadMetadata:
    """Dataclass representing YouTube video upload metadata."""
    title: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    privacy_status: str = "unlisted"
    category_id: str = "24"  # Default 24 = Entertainment

    def __post_init__(self):
        if not self.title or not self.title.strip():
            raise ValueError("UploadMetadata title cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UploadMetadata":
        title = str(d.get("title", "")).strip()
        if not title:
            raise ValueError("UploadMetadata title cannot be empty")
        return cls(
            title=title,
            description=str(d.get("description", "")),
            tags=list(d.get("tags", [])),
            privacy_status=str(d.get("privacy_status", "unlisted")),
            category_id=str(d.get("category_id", "24")),
        )


def calculate_total_audio_duration(
    audio_clips: Union[List[AudioClip], Dict[str, AudioClip]]
) -> float:
    """
    Calculate the cumulative duration of audio clips in seconds.
    Accepts either a list of AudioClip objects or a dict mapping clip names to AudioClip.
    """
    if isinstance(audio_clips, dict):
        clips = audio_clips.values()
    else:
        clips = audio_clips
    return sum(clip.duration_seconds for clip in clips)


def calculate_clip_timeline(
    audio_clips: Dict[str, AudioClip]
) -> List[Dict[str, Any]]:
    """
    Computes start and end timestamps for audio clips in sequence.
    Returns list of dicts with clip_id, start_time, end_time, duration.
    """
    timeline = []
    current_time = 0.0
    for clip_key, clip in audio_clips.items():
        start_time = current_time
        end_time = start_time + clip.duration_seconds
        timeline.append({
            "key": clip_key,
            "clip_id": clip.clip_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": clip.duration_seconds,
            "file_path": clip.file_path,
        })
        current_time = end_time
    return timeline
