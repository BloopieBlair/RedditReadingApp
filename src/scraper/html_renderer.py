"""HTML Card Renderer for post and comment overlay image generation using Jinja2, Playwright, and Pillow fallback."""

import os
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from jinja2 import Template
from PIL import Image, ImageDraw, ImageFont

from src.models import RedditPost, RedditComment
from src.config import TEMPLATES_DIR, TEMP_DIR

logger = logging.getLogger(__name__)


def format_ups(ups: Union[int, float, None]) -> str:
    """Format upvote count (e.g. 15200 -> '15.2k', 1200000 -> '1.2m', 950 -> '950')."""
    if ups is None:
        return "0"
    try:
        val = int(ups)
    except (ValueError, TypeError):
        return "0"

    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1_000_000:
        formatted = f"{abs_val / 1_000_000:.1f}m"
        return f"{sign}{formatted}"
    elif abs_val >= 1_000:
        formatted = f"{abs_val / 1_000:.1f}k"
        return f"{sign}{formatted}"
    return str(val)


def format_time_ago(created_utc: Union[int, float, str, None]) -> str:
    """Format UTC creation timestamp to relative time string (e.g. '2h ago', '15m ago', '3d ago')."""
    if created_utc is None:
        return "recently"
    try:
        ts = float(created_utc)
    except (ValueError, TypeError):
        return "recently"

    if ts <= 0:
        return "recently"

    now = time.time()
    diff = max(0.0, now - ts)

    if diff < 60:
        return "just now"
    elif diff < 3600:
        mins = int(diff // 60)
        return f"{mins}m ago"
    elif diff < 86400:
        hours = int(diff // 3600)
        return f"{hours}h ago"
    elif diff < 2592000:  # 30 days
        days = int(diff // 86400)
        return f"{days}d ago"
    elif diff < 31536000:  # 365 days
        months = int(diff // 2592000)
        return f"{months}mo ago"
    else:
        years = int(diff // 31536000)
        return f"{years}y ago"


def _render_with_playwright(html_content: str, output_path: str) -> str:
    """Render HTML string to PNG using Playwright sync browser at 2x device scale factor."""
    from playwright.sync_api import sync_playwright

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(device_scale_factor=2.0)
        page = context.new_page()
        page.set_content(html_content, wait_until="domcontentloaded", timeout=5000)

        card = page.query_selector("#card")
        if card:
            card.screenshot(path=str(out_file), omit_background=True)
        else:
            page.screenshot(path=str(out_file), full_page=True, omit_background=True)

        browser.close()
    return str(out_file)


async def _render_with_playwright_async(html_content: str, output_path: str) -> str:
    """Render HTML string to PNG using Playwright async browser at 2x device scale factor."""
    from playwright.async_api import async_playwright

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(device_scale_factor=2.0)
        page = await context.new_page()
        await page.set_content(html_content, wait_until="domcontentloaded", timeout=5000)

        card = await page.query_selector("#card")
        if card:
            await card.screenshot(path=str(out_file), omit_background=True)
        else:
            await page.screenshot(path=str(out_file), full_page=True, omit_background=True)

        await browser.close()
    return str(out_file)


def _render_with_pillow(
    item: Union[RedditPost, RedditComment],
    output_path: str,
    subreddit: str = "AskReddit",
) -> str:
    """Fallback pure-Python Pillow card image generator when Playwright is unavailable or fails."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    scale = 2
    width = 960 * scale
    padding = 24 * scale

    is_post = isinstance(item, RedditPost)
    sub_name = item.subreddit if is_post else subreddit
    if sub_name.startswith("r/"):
        sub_name = sub_name[2:]
    author = item.author or "anonymous"
    created = format_time_ago(item.created_utc)
    title = item.title if is_post else ""
    body = item.body or ""
    ups_str = format_ups(item.ups)

    try:
        font_title = ImageFont.truetype("arial.ttf", 26 * scale)
        font_body = ImageFont.truetype("arial.ttf", 20 * scale)
        font_header = ImageFont.truetype("arial.ttf", 18 * scale)
        font_meta = ImageFont.truetype("arial.ttf", 14 * scale)
    except IOError:
        font_title = font_body = font_header = font_meta = ImageFont.load_default()

    def wrap_text(text: str, font: ImageFont.ImageFont, max_w: int) -> list:
        lines = []
        for paragraph in text.split("\n"):
            words = paragraph.split(" ")
            current_line = ""
            for word in words:
                test_line = f"{current_line} {word}".strip()
                bbox = font.getbbox(test_line)
                w = bbox[2] - bbox[0]
                if w <= max_w:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
        return lines

    max_text_width = width - (padding * 2)
    title_lines = wrap_text(title, font_title, max_text_width) if title else []
    body_lines = wrap_text(body, font_body, max_text_width) if body else []

    line_height_title = int(35 * scale)
    line_height_body = int(28 * scale)

    header_h = 50 * scale
    title_h = len(title_lines) * line_height_title + (16 * scale if title_lines else 0)
    body_h = len(body_lines) * line_height_body + (16 * scale if body_lines else 0)
    footer_h = 50 * scale
    card_h = padding * 2 + header_h + title_h + body_h + footer_h

    img = Image.new("RGBA", (width, card_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg_color = (26, 26, 27, 255)
    border_color = (52, 53, 54, 255)
    corner_radius = 16 * scale

    draw.rounded_rectangle(
        [0, 0, width - 1, card_h - 1],
        radius=corner_radius,
        fill=bg_color,
        outline=border_color,
        width=scale,
    )

    curr_y = padding

    icon_size = 40 * scale
    draw.ellipse(
        [padding, curr_y, padding + icon_size, curr_y + icon_size],
        fill=(255, 69, 0, 255),
    )
    draw.text(
        (padding + 10 * scale, curr_y + 8 * scale),
        "r/",
        fill=(255, 255, 255, 255),
        font=font_header,
    )

    meta_x = padding + icon_size + 12 * scale
    draw.text(
        (meta_x, curr_y),
        f"r/{sub_name}",
        fill=(242, 244, 245, 255),
        font=font_header,
    )
    draw.text(
        (meta_x, curr_y + 22 * scale),
        f"Posted by u/{author} • {created}",
        fill=(129, 131, 132, 255),
        font=font_meta,
    )
    curr_y += icon_size + 16 * scale

    if title_lines:
        for line in title_lines:
            draw.text((padding, curr_y), line, fill=(255, 255, 255, 255), font=font_title)
            curr_y += line_height_title
        curr_y += 12 * scale

    if body_lines:
        for line in body_lines:
            draw.text((padding, curr_y), line, fill=(215, 218, 220, 255), font=font_body)
            curr_y += line_height_body
        curr_y += 12 * scale

    draw.line(
        [(padding, curr_y), (width - padding, curr_y)],
        fill=(39, 39, 41, 255),
        width=scale,
    )
    curr_y += 12 * scale

    badge_x = padding
    badge_w = 120 * scale
    badge_h = 32 * scale
    draw.rounded_rectangle(
        [badge_x, curr_y, badge_x + badge_w, curr_y + badge_h],
        radius=16 * scale,
        fill=(39, 39, 41, 255),
    )
    draw.text(
        (badge_x + 12 * scale, curr_y + 4 * scale),
        f"▲ {ups_str}",
        fill=(255, 69, 0, 255),
        font=font_meta,
    )

    img.save(str(out_file))
    return str(out_file)


class HTMLCardRenderer:
    """Renderer for converting Reddit post/comment cards into PNG images."""

    def __init__(self, template_path: Optional[str] = None):
        self.template_path = Path(template_path) if template_path else (TEMPLATES_DIR / "card_template.html")

    def render_html(
        self,
        item: Union[RedditPost, RedditComment],
        subreddit: str = "AskReddit",
    ) -> str:
        """Render card item to HTML string using Jinja2 template."""
        is_post = isinstance(item, RedditPost)
        sub_name = item.subreddit if is_post else subreddit
        if sub_name.startswith("r/"):
            sub_name = sub_name[2:]

        author = item.author or "anonymous"
        created = format_time_ago(item.created_utc)
        title = item.title if is_post else None
        body = item.body if item.body else ""
        ups = format_ups(item.ups)
        image_url = getattr(item, "image_url", None) if is_post else None

        if self.template_path.exists():
            template_str = self.template_path.read_text(encoding="utf-8")
        else:
            template_str = """<!DOCTYPE html><html><body><div id="card">r/{{ SUBREDDIT }} u/{{ AUTHOR }} {{ TITLE }} {{ BODY }} {{ UPS }}</div></body></html>"""

        template = Template(template_str)
        return template.render(
            SUBREDDIT=sub_name,
            AUTHOR=author,
            CREATED=created,
            TITLE=title,
            BODY=body,
            UPS=ups,
            IMAGE_URL=image_url,
        )

    def render(
        self,
        item: Union[RedditPost, RedditComment],
        output_path: str,
        subreddit: str = "AskReddit",
    ) -> str:
        """Render Reddit card to PNG image. Tries Playwright, falls back to Pillow."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        html_content = self.render_html(item, subreddit=subreddit)

        try:
            # Check if asyncio event loop is currently running
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and loop.is_running():
                # If inside asyncio event loop, run async_playwright or Pillow
                # We can run async_render coroutine or fallback to Pillow
                try:
                    fut = _render_with_playwright_async(html_content, str(out_file))
                    # If running inside a task in an event loop
                    return loop.run_until_complete(fut) if not loop.is_running() else _render_with_pillow(item, str(out_file), subreddit=subreddit)
                except Exception:
                    return _render_with_pillow(item, str(out_file), subreddit=subreddit)

            return _render_with_playwright(html_content, str(out_file))
        except Exception as e:
            logger.warning(f"Playwright card rendering failed: {e}. Falling back to Pillow renderer.")
            return _render_with_pillow(item, str(out_file), subreddit=subreddit)

    async def async_render(
        self,
        item: Union[RedditPost, RedditComment],
        output_path: str,
        subreddit: str = "AskReddit",
    ) -> str:
        """Async render method using async Playwright with Pillow fallback."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        html_content = self.render_html(item, subreddit=subreddit)

        try:
            return await _render_with_playwright_async(html_content, str(out_file))
        except Exception as e:
            logger.warning(f"Async Playwright card rendering failed: {e}. Falling back to Pillow renderer.")
            return _render_with_pillow(item, str(out_file), subreddit=subreddit)


def render_card_image(
    title: str,
    body: str,
    author: str,
    subreddit: str,
    ups: int,
    output_path: str,
    card_type: str = "post",
) -> str:
    """Functional interface for rendering card images."""
    renderer = HTMLCardRenderer()
    item = (
        RedditPost(
            post_id="card",
            title=title,
            author=author,
            subreddit=subreddit,
            body=body,
            ups=ups,
        )
        if card_type == "post"
        else RedditComment(
            comment_id="card", author=author, body=body, ups=ups
        )
    )
    return renderer.render(item, output_path=output_path, subreddit=subreddit)
