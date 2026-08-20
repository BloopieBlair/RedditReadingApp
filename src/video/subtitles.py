"""
Animated On-Screen Subtitle & Caption Generator for Reddit Reading Shorts.
Renders phrase-synchronized caption overlays with zero-overlap positioning logic.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def split_text_into_phrases(text: str, max_words: int = 4) -> List[str]:
    """Splits text body into natural 3-5 word phrase chunks for readable captions."""
    clean = re.sub(r"\s+", " ", text).strip()
    words = clean.split()
    if not words:
        return []

    phrases = []
    curr = []
    for w in words:
        curr.append(w)
        if len(curr) >= max_words or w.endswith((".", "!", "?", ";", ":")):
            phrases.append(" ".join(curr))
            curr = []
    if curr:
        phrases.append(" ".join(curr))
    return phrases


def generate_caption_timeline(
    text: str, duration: float, start_time: float = 0.0, max_words: int = 4
) -> List[Dict[str, Any]]:
    """
    Splits text into phrases and distributes phrase durations proportionally across clip duration.
    Returns list of dicts: [{'phrase': '...', 'start': 0.0, 'end': 1.5}, ...]
    """
    phrases = split_text_into_phrases(text, max_words=max_words)
    if not phrases:
        return []

    total_chars = sum(len(p) for p in phrases)
    if total_chars <= 0 or duration <= 0:
        return []

    timeline = []
    curr_t = start_time
    for p in phrases:
        frac = len(p) / float(total_chars)
        p_dur = max(0.3, duration * frac)
        timeline.append({"phrase": p, "start": curr_t, "end": curr_t + p_dur})
        curr_t += p_dur

    return timeline


def render_caption_image(
    phrase: str,
    canvas_w: int = 1080,
    font_size: int = 44,
    style: str = "yellow_pill",
) -> Image.Image:
    """
    Renders a crisp RGBA caption overlay image with rounded background pill and 4px stroke.
    Canvas height is 200px (to be pasted onto 1080x1920 video at target Y position).
    """
    canvas_h = 200
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Calculate text bounds
    left, top, right, bottom = draw.textbbox((0, 0), phrase, font=font)
    text_w = right - left
    text_h = bottom - top

    padding_x = 28
    padding_y = 14
    box_w = min(canvas_w - 60, text_w + padding_x * 2)
    box_h = text_h + padding_y * 2

    box_x = (canvas_w - box_w) // 2
    box_y = (canvas_h - box_h) // 2

    # Style colors
    if style == "cyan_glow":
        text_fill = (34, 211, 238, 255)
        bg_color = (15, 23, 42, 235)
        stroke_color = (0, 0, 0, 255)
    elif style == "minimal_white":
        text_fill = (255, 255, 255, 255)
        bg_color = (0, 0, 0, 180)
        stroke_color = (0, 0, 0, 255)
    else:  # yellow_pill (default)
        text_fill = (250, 204, 21, 255)  # Bright Yellow
        bg_color = (15, 23, 42, 230)     # Dark Slate
        stroke_color = (0, 0, 0, 255)

    # 1. Pill mask & container
    pill_mask = Image.new("L", (box_w, box_h), 0)
    pill_draw = ImageDraw.Draw(pill_mask)
    pill_draw.rounded_rectangle((0, 0, box_w, box_h), radius=18, fill=255)

    pill_img = Image.new("RGBA", (box_w, box_h), bg_color)
    img.paste(pill_img, (box_x, box_y), pill_mask)

    # 2. Render text with stroke
    text_x = (canvas_w - text_w) // 2
    text_y = box_y + padding_y - 2
    draw.text(
        (text_x, text_y),
        phrase,
        font=font,
        fill=text_fill,
        stroke_width=4,
        stroke_fill=stroke_color,
    )

    return img
