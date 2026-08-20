"""Unit and integration tests for src/scraper/html_renderer.py and src/scraper/reddit_scraper.py."""

import os
import time
import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from src.models import RedditPost, RedditComment, ScrapedContent
from src.scraper.html_renderer import (
    HTMLCardRenderer,
    format_ups,
    format_time_ago,
    _render_with_pillow,
    render_card_image,
)
from src.scraper.reddit_scraper import (
    RedditScraper,
    fetch_reddit_post,
    is_bot_or_deleted_author,
    is_valid_post_data,
    is_valid_comment_data,
)


def test_format_ups():
    """Test format_ups upvote number formatting."""
    assert format_ups(0) == "0"
    assert format_ups(950) == "950"
    assert format_ups(-50) == "-50"
    assert format_ups(1000) == "1.0k"
    assert format_ups(15200) == "15.2k"
    assert format_ups(1200000) == "1.2m"
    assert format_ups(None) == "0"
    assert format_ups("invalid") == "0"


def test_format_time_ago():
    """Test format_time_ago relative timestamp formatting."""
    assert format_time_ago(None) == "recently"
    assert format_time_ago(0) == "recently"
    assert format_time_ago(-100) == "recently"
    assert format_time_ago("invalid") == "recently"

    now = time.time()
    assert format_time_ago(now - 30) == "just now"
    assert format_time_ago(now - 120) == "2m ago"
    assert format_time_ago(now - 7200) == "2h ago"
    assert format_time_ago(now - 172800) == "2d ago"
    assert format_time_ago(now - 31536000 * 2) == "2y ago"


def test_html_card_renderer_render_html():
    """Test Jinja2 template rendering into HTML string."""
    renderer = HTMLCardRenderer()
    post = RedditPost(
        post_id="p1",
        title="Test Post Title",
        author="test_author",
        subreddit="AskReddit",
        body="Test post body content.",
        ups=15200,
        created_utc=time.time() - 3600,
    )

    html = renderer.render_html(post)
    assert "AskReddit" in html
    assert "test_author" in html
    assert "Test Post Title" in html
    assert "Test post body content." in html
    assert "15.2k" in html

    comment = RedditComment(
        comment_id="c1",
        author="comment_author",
        body="Test comment body content.",
        ups=450,
        created_utc=time.time() - 1800,
    )
    c_html = renderer.render_html(comment, subreddit="AskReddit")
    assert "comment_author" in c_html
    assert "Test comment body content." in c_html
    assert "450" in c_html


def test_html_card_renderer_png_rendering(tmp_path):
    """Test PNG image generation via HTMLCardRenderer."""
    renderer = HTMLCardRenderer()
    post = RedditPost(
        post_id="p_test",
        title="Rendering Test Title",
        author="creator",
        subreddit="AskReddit",
        body="Body text for PNG rendering.",
        ups=1200,
    )
    out_file = str(tmp_path / "post_card.png")
    res_path = renderer.render(post, out_file)
    assert os.path.exists(res_path)

    # Verify PNG image properties
    img = Image.open(res_path)
    assert img.format == "PNG"
    assert img.width > 0
    assert img.height > 0


def test_pillow_fallback_renderer(tmp_path):
    """Test pure-Python Pillow card image fallback generator."""
    post = RedditPost(
        post_id="p_pil",
        title="Pillow Fallback Title",
        author="pil_author",
        subreddit="AskReddit",
        body="Pillow body content text.",
        ups=999,
    )
    out_file = str(tmp_path / "pillow_card.png")
    res_path = _render_with_pillow(post, out_file, subreddit="AskReddit")
    assert os.path.exists(res_path)

    img = Image.open(res_path)
    assert img.format == "PNG"
    assert img.width == 1920


def test_render_card_image_functional_interface(tmp_path):
    """Test render_card_image helper function."""
    out_path = str(tmp_path / "func_card.png")
    res = render_card_image(
        title="Func Title",
        body="Func Body",
        author="func_user",
        subreddit="AskReddit",
        ups=888,
        output_path=out_path,
        card_type="post",
    )
    assert os.path.exists(res)


def test_post_and_comment_filtering():
    """Test filtering logic for NSFW, stickied, deleted, and bot posts/comments."""
    assert is_bot_or_deleted_author(None) is True
    assert is_bot_or_deleted_author("[deleted]") is True
    assert is_bot_or_deleted_author("AutoModerator") is True
    assert is_bot_or_deleted_author("RemindMeBot") is True
    assert is_bot_or_deleted_author("normal_user") is False

    # Valid post
    assert is_valid_post_data({
        "title": "Good Title",
        "author": "real_user",
        "over_18": False,
        "stickied": False,
    }) is True

    # NSFW post
    assert is_valid_post_data({
        "title": "NSFW Post",
        "author": "real_user",
        "over_18": True,
    }) is False

    # Stickied post
    assert is_valid_post_data({
        "title": "Pinned Mod Post",
        "author": "real_user",
        "stickied": True,
    }) is False

    # Deleted author post
    assert is_valid_post_data({
        "title": "Some Title",
        "author": "[deleted]",
    }) is False

    # Valid comment
    assert is_valid_comment_data({
        "body": "Insightful answer.",
        "author": "commenter1",
        "stickied": False,
    }) is True

    # AutoModerator comment
    assert is_valid_comment_data({
        "body": "Please follow rules.",
        "author": "AutoModerator",
    }) is False

    # Deleted comment body
    assert is_valid_comment_data({
        "body": "[deleted]",
        "author": "commenter2",
    }) is False


def test_json_api_parsing_and_sorting(tmp_path):
    """Test JSON API response parsing, filtering, and comment sorting."""
    async def _run():
        scraper = RedditScraper(force_json_api=True)

        mock_top_payload = {
            "kind": "Listing",
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": "post_nsfw",
                            "title": "NSFW Post Title",
                            "author": "user1",
                            "over_18": True,
                        },
                    },
                    {
                        "kind": "t3",
                        "data": {
                            "id": "post_valid",
                            "title": "Valid Post Title",
                            "author": "good_user",
                            "selftext": "Valid post body text.",
                            "ups": 1500,
                            "created_utc": 1600000000.0,
                            "over_18": False,
                            "stickied": False,
                        },
                    },
                ]
            },
        }

        mock_comments_payload = [
            {"kind": "Listing", "data": {"children": [{"data": mock_top_payload["data"]["children"][1]["data"]}]}},
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "t1",
                            "data": {
                                "id": "c_bot",
                                "author": "AutoModerator",
                                "body": "Bot warning",
                                "ups": 999,
                            },
                        },
                        {
                            "kind": "t1",
                            "data": {
                                "id": "c_low",
                                "author": "user_a",
                                "body": "Lower upvoted comment.",
                                "ups": 50,
                                "created_utc": 1600000100.0,
                            },
                        },
                        {
                            "kind": "t1",
                            "data": {
                                "id": "c_high",
                                "author": "user_b",
                                "body": "Higher upvoted comment.",
                                "ups": 500,
                                "created_utc": 1600000200.0,
                            },
                        },
                    ]
                },
            },
        ]

        with patch("requests.get") as mock_get:
            resp_top = MagicMock()
            resp_top.status_code = 200
            resp_top.json.return_value = mock_top_payload

            resp_comments = MagicMock()
            resp_comments.status_code = 200
            resp_comments.json.return_value = mock_comments_payload

            mock_get.side_effect = [resp_top, resp_comments]

            scraped = await scraper._scrape_json_api("AskReddit")

            assert scraped.post.post_id == "post_valid"
            assert scraped.post.title == "Valid Post Title"
            assert len(scraped.comments) == 2
            # Verify ordering: c_high (500 ups) comes before c_low (50 ups)
            assert scraped.comments[0].comment_id == "c_high"
            assert scraped.comments[1].comment_id == "c_low"

    asyncio.run(_run())


def test_playwright_fallback_to_json_api():
    """Test automatic fallback from Playwright error to JSON API."""
    async def _run():
        scraper = RedditScraper(force_json_api=False)

        sample_post = RedditPost(
            post_id="p_fb", title="Fallback Post", author="user_fb", subreddit="AskReddit", body="Text"
        )
        sample_scraped = ScrapedContent(post=sample_post, comments=[])

        with patch.object(scraper, "_scrape_playwright", side_effect=RuntimeError("Playwright error simulation")):
            with patch.object(scraper, "_scrape_json_api", return_value=sample_scraped) as mock_json_api:
                result = await scraper.scrape("AskReddit")

                assert result == sample_scraped
                mock_json_api.assert_called_once_with("AskReddit", None, max_comments=2)

    asyncio.run(_run())


def test_fetch_reddit_post_e2e(tmp_path):
    """Test end-to-end fetch_reddit_post execution."""
    async def _run():
        sample_post = RedditPost(
            post_id="p_e2e",
            title="End-to-End Post Title",
            author="e2e_author",
            subreddit="AskReddit",
            body="End to end test body text.",
            ups=3000,
        )
        sample_comments = [
            RedditComment(comment_id="c1", author="comm1", body="Comment 1 text", ups=400),
            RedditComment(comment_id="c2", author="comm2", body="Comment 2 text", ups=200),
        ]

        with patch("src.scraper.reddit_scraper.RedditScraper.scrape") as mock_scrape:
            mock_scraped = ScrapedContent(
                post=sample_post,
                comments=sample_comments,
                op_card_image_path=str(tmp_path / "op.png"),
                comment_card_image_paths=[str(tmp_path / "c1.png"), str(tmp_path / "c2.png")],
            )
            mock_scrape.return_value = mock_scraped

            result = await fetch_reddit_post("AskReddit")
            assert result.post.title == "End-to-End Post Title"
            assert len(result.comments) == 2
            assert result.op_card_image_path.endswith("op.png")
            assert result.comment1_card_image_path.endswith("c1.png")
            assert result.comment2_card_image_path.endswith("c2.png")

    asyncio.run(_run())
