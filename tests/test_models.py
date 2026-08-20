"""Unit tests for src/models.py data models, validation, and serialization."""

import pytest
from src.models import (
    RedditComment,
    RedditPost,
    ScrapedContent,
    AudioClip,
    VideoRenderConfig,
    UploadMetadata,
    calculate_total_audio_duration,
    calculate_clip_timeline,
)


def test_reddit_comment_creation_and_serialization():
    """Test RedditComment instantiation, to_dict, and from_dict."""
    comment = RedditComment(
        comment_id="c101",
        author="user_a",
        body="Great story!",
        ups=42,
        created_utc=1600000000.0,
        parent_id="p001",
    )
    assert comment.comment_id == "c101"
    assert comment.ups == 42

    d = comment.to_dict()
    assert d["comment_id"] == "c101"
    assert d["author"] == "user_a"

    reconstructed = RedditComment.from_dict(d)
    assert reconstructed == comment


def test_reddit_comment_validation():
    """Test validation in RedditComment.from_dict."""
    with pytest.raises(ValueError, match="comment_id"):
        RedditComment.from_dict({"comment_id": "", "body": "test"})


def test_reddit_post_creation_and_serialization():
    """Test RedditPost instantiation, to_dict, and from_dict."""
    post = RedditPost(
        post_id="p101",
        title="What is your favourite programming language?",
        author="coder123",
        subreddit="AskReddit",
        body="Interested in hearing opinions.",
        ups=1200,
        num_comments=350,
    )
    assert post.post_id == "p101"
    assert post.title == "What is your favourite programming language?"

    d = post.to_dict()
    reconstructed = RedditPost.from_dict(d)
    assert reconstructed == post


def test_reddit_post_validation():
    """Test validation in RedditPost.from_dict."""
    with pytest.raises(ValueError, match="post_id"):
        RedditPost.from_dict({"post_id": "", "title": "Test Title"})

    with pytest.raises(ValueError, match="title"):
        RedditPost.from_dict({"post_id": "p1", "title": ""})


def test_scraped_content_properties_and_serialization():
    """Test ScrapedContent convenience properties and serialization."""
    post = RedditPost(
        post_id="p1", title="Sample Title", author="author1", subreddit="AskReddit", body="Post text"
    )
    comment1 = RedditComment(comment_id="c1", author="comm1", body="Comment text 1")
    comment2 = RedditComment(comment_id="c2", author="comm2", body="Comment text 2")

    scraped = ScrapedContent(
        post=post,
        comments=[comment1, comment2],
        op_card_image_path="assets/op_card.png",
        comment_card_image_paths=["assets/c1_card.png", "assets/c2_card.png"],
    )

    assert scraped.post_title == "Sample Title"
    assert scraped.post_body == "Post text"
    assert scraped.author == "author1"
    assert scraped.subreddit == "AskReddit"
    assert scraped.comment1_card_image_path == "assets/c1_card.png"
    assert scraped.comment2_card_image_path == "assets/c2_card.png"

    d = scraped.to_dict()
    reconstructed = ScrapedContent.from_dict(d)
    assert reconstructed.post.post_id == "p1"
    assert len(reconstructed.comments) == 2
    assert reconstructed.comment1_card_image_path == "assets/c1_card.png"


def test_audio_clip_validation_and_duration():
    """Test AudioClip creation, negative duration validation, and math."""
    clip = AudioClip(
        clip_id="op_clip",
        file_path="temp/op.mp3",
        duration_seconds=5.5,
        text="Sample text",
    )
    assert clip.duration_seconds == 5.5

    with pytest.raises(ValueError, match="duration_seconds"):
        AudioClip(clip_id="bad", file_path="temp/bad.mp3", duration_seconds=-1.0)


def test_calculate_total_audio_duration_and_timeline():
    """Test audio clip duration accumulation and timeline calculation."""
    clip1 = AudioClip(clip_id="op", file_path="temp/1.mp3", duration_seconds=4.0)
    clip2 = AudioClip(clip_id="comment_1", file_path="temp/2.mp3", duration_seconds=6.5)

    clips_list = [clip1, clip2]
    clips_dict = {"op": clip1, "comment_1": clip2}

    assert calculate_total_audio_duration(clips_list) == 10.5
    assert calculate_total_audio_duration(clips_dict) == 10.5

    timeline = calculate_clip_timeline(clips_dict)
    assert len(timeline) == 2
    assert timeline[0]["start_time"] == 0.0
    assert timeline[0]["end_time"] == 4.0
    assert timeline[1]["start_time"] == 4.0
    assert timeline[1]["end_time"] == 10.5


def test_video_render_config():
    """Test VideoRenderConfig defaults and validation."""
    config = VideoRenderConfig()
    assert config.width == 1080
    assert config.height == 1920
    assert config.fps == 30

    d = config.to_dict()
    reconstructed = VideoRenderConfig.from_dict(d)
    assert reconstructed == config

    with pytest.raises(ValueError, match="positive integers"):
        VideoRenderConfig(width=0)


def test_upload_metadata():
    """Test UploadMetadata creation and empty title validation."""
    meta = UploadMetadata(title="My Reddit Short", description="Description here", tags=["reddit"])
    assert meta.title == "My Reddit Short"
    assert meta.privacy_status == "unlisted"

    d = meta.to_dict()
    reconstructed = UploadMetadata.from_dict(d)
    assert reconstructed == meta

    with pytest.raises(ValueError, match="title cannot be empty"):
        UploadMetadata(title="  ")
