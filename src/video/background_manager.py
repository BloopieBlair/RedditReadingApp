"""Background video asset manager and random crop selector using MoviePy 2.x."""

import logging
import random
from pathlib import Path
from typing import Optional

from moviepy import VideoFileClip, vfx
from src.config import (
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    generate_fallback_background,
    get_background_video_path,
)

logger = logging.getLogger(__name__)


class BackgroundManager:
    """Manager for resolving, cropping, and trimming background video footage."""

    def __init__(self, background_video_path: Optional[str] = None):
        self.background_video_path = background_video_path

    def get_video_source(self) -> str:
        """Returns valid background video path."""
        if self.background_video_path and Path(self.background_video_path).exists():
            return str(self.background_video_path)
        return str(get_background_video_path())

    def get_background_clip(
        self, target_duration: float, video_source_path: Optional[str] = None
    ) -> VideoFileClip:
        """
        Loads background video, crops to 9:16 aspect ratio, resizes to 1080x1920 @ 30fps,
        and trims to target_duration using a random start timestamp or looping.
        """
        if target_duration <= 0:
            target_duration = 5.0

        source_path = video_source_path or self.get_video_source()

        # Handle missing or unreadable source file gracefully
        try:
            clip = VideoFileClip(source_path)
        except Exception as e:
            logger.warning(
                f"Failed to load video source '{source_path}': {e}. Generating fallback background."
            )
            fallback_path = generate_fallback_background(
                duration_sec=max(10.0, target_duration)
            )
            clip = VideoFileClip(str(fallback_path))

        try:
            w, h = clip.size
            target_ar = 9.0 / 16.0
            current_ar = w / float(h) if h > 0 else 1.0

            if current_ar > target_ar:
                # Wider horizontal clip -> crop width centered horizontally
                crop_w = int(h * target_ar)
                x1 = (w - crop_w) // 2
                y1 = 0
                crop_h = h
            elif current_ar < target_ar:
                # Taller vertical clip -> crop height centered vertically
                crop_h = int(w / target_ar)
                y1 = (h - crop_h) // 2
                x1 = 0
                crop_w = w
            else:
                x1 = 0
                y1 = 0
                crop_w = w
                crop_h = h

            clip = clip.with_effects(
                [
                    vfx.Crop(x1=x1, y1=y1, width=crop_w, height=crop_h),
                    vfx.Resize((VIDEO_WIDTH, VIDEO_HEIGHT)),
                ]
            )

            d_bg = float(clip.duration) if clip.duration is not None else 0.0

            if d_bg > target_duration:
                max_start = d_bg - target_duration
                t_start = random.uniform(0.0, max_start)
                clip = clip.subclipped(t_start, t_start + target_duration)
            else:
                # Loop to cover target_duration
                clip = clip.with_effects(
                    [vfx.Loop(duration=target_duration)]
                ).subclipped(0, target_duration)

            # Strip background audio to avoid interference with voiceover
            clip = clip.without_audio()
            return clip

        except Exception as e:
            logger.warning(
                f"Error processing background clip: {e}. Falling back to dynamic generator."
            )
            fallback_path = generate_fallback_background(
                duration_sec=max(10.0, target_duration)
            )
            return (
                VideoFileClip(str(fallback_path))
                .with_effects([vfx.Resize((VIDEO_WIDTH, VIDEO_HEIGHT))])
                .subclipped(0, target_duration)
                .without_audio()
            )


def get_background_clip(
    target_duration: float,
    background_video_path: Optional[str] = None,
    video_source_path: Optional[str] = None,
) -> VideoFileClip:
    """Functional interface for obtaining background video clip."""
    path = video_source_path or background_video_path
    manager = BackgroundManager(background_video_path=path)
    return manager.get_background_clip(
        target_duration=target_duration, video_source_path=path
    )
