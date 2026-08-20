"""Video production package for background asset management and vertical YouTube Shorts composition."""

from src.video.background_manager import BackgroundManager, get_background_clip
from src.video.composer import (
    ShortsVideoComposer,
    apply_card_shadow_and_corners,
    compose_shorts_video,
)

__all__ = [
    "get_background_clip",
    "BackgroundManager",
    "compose_shorts_video",
    "ShortsVideoComposer",
    "apply_card_shadow_and_corners",
]
