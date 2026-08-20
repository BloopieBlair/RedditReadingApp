"""
Unit tests for TTS Voice Engine module (src/tts/voice_engine.py).
"""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from src.models import AudioClip, RedditComment, RedditPost, ScrapedContent
from src.tts.voice_engine import (
    TTSVoiceEngine,
    clean_text_for_tts,
    estimate_speech_duration,
    generate_fallback_audio,
    generate_voiceover,
    get_audio_duration,
)


def test_clean_text_for_tts_complex():
    """Test text cleaning logic with complex markdown, URLs, tags, HTML entities, and emojis."""
    raw_text = (
        "Hello **world** &amp; [click here](https://example.com/test)! "
        "Visit r/AskReddit and u/spez.\n"
        ">!Secret spoiler!< > Quoted line\n"
        "Emoji test 😂🔥! Check https://reddit.com for details."
    )
    cleaned = clean_text_for_tts(raw_text)

    assert "https://" not in cleaned
    assert "**" not in cleaned
    assert "r slash AskReddit" in cleaned
    assert "user spez" in cleaned
    assert "&" in cleaned
    assert "Secret spoiler" in cleaned
    assert "Quoted line" in cleaned
    assert "😂" not in cleaned
    assert "🔥" not in cleaned
    assert "Hello world" in cleaned


def test_clean_text_for_tts_edge_cases():
    """Test clean_text_for_tts with empty, whitespace, and plain inputs."""
    assert clean_text_for_tts("") == ""
    assert clean_text_for_tts("   \n\t ") == ""
    assert clean_text_for_tts("Plain simple text") == "Plain simple text"


def test_estimate_speech_duration():
    """Test estimate_speech_duration helper."""
    assert estimate_speech_duration("") == 1.5
    # 10 words -> 10 / 2.5 = 4.0 seconds
    ten_words = "one two three four five six seven eight nine ten"
    assert pytest.approx(estimate_speech_duration(ten_words), 0.1) == 4.0


def test_generate_fallback_audio_wav(tmp_path: Path):
    """Test offline fallback audio generation for WAV files."""
    wav_path = tmp_path / "test_synth.wav"
    duration_returned = generate_fallback_audio(wav_path, target_duration=2.0)

    assert wav_path.exists()
    assert wav_path.stat().st_size > 0
    assert pytest.approx(duration_returned, abs=0.05) == 2.0

    duration_extracted = get_audio_duration(wav_path)
    assert pytest.approx(duration_extracted, abs=0.05) == 2.0


def test_generate_fallback_audio_mp3(tmp_path: Path):
    """Test offline fallback audio generation for MP3 files."""
    mp3_path = tmp_path / "test_synth.mp3"
    duration_returned = generate_fallback_audio(mp3_path, target_duration=3.0)

    assert mp3_path.exists()
    assert mp3_path.stat().st_size > 0
    assert duration_returned > 0.0

    duration_extracted = get_audio_duration(mp3_path)
    assert pytest.approx(duration_extracted, abs=0.2) == 3.0


def test_get_audio_duration_file_not_found():
    """Test get_audio_duration raises FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        get_audio_duration("non_existent_audio_file.mp3")


def test_tts_voice_engine_generate_audio(tmp_path: Path):
    """Test TTSVoiceEngine generate_audio returns valid AudioClip."""
    async def _runner():
        engine = TTSVoiceEngine(allow_offline_fallback=True)
        out_file = tmp_path / "output_test.mp3"

        clip = await engine.generate_audio("Hello unit test voice synthesis", str(out_file))

        assert isinstance(clip, AudioClip)
        assert clip.clip_id == "output_test"
        assert clip.file_path == str(out_file)
        assert clip.duration_seconds > 0.0
        assert Path(clip.file_path).exists()

    asyncio.run(_runner())


def test_tts_voice_engine_offline_forced(tmp_path: Path):
    """Test TTSVoiceEngine handles network errors by falling back to synthetic audio."""
    async def _runner():
        engine = TTSVoiceEngine(allow_offline_fallback=True)
        out_file = tmp_path / "forced_fallback.mp3"

        with patch("edge_tts.Communicate.save", side_effect=ConnectionError("Simulated offline status")):
            clip = await engine.generate_audio("Offline test sentence", str(out_file))

            assert isinstance(clip, AudioClip)
            assert clip.clip_id == "forced_fallback"
            assert clip.file_path == str(out_file)
            assert clip.duration_seconds > 0.0
            assert out_file.exists()

    asyncio.run(_runner())


def test_generate_voiceover_scraped_content(tmp_path: Path):
    """Test generate_voiceover creates op.mp3, comment_1.mp3, comment_2.mp3 and returns correct Dict."""
    async def _runner():
        post = RedditPost(
            post_id="p123",
            title="What is your favourite programming language?",
            author="coder_1",
            subreddit="AskReddit",
            body="Tell us why you love it.",
        )
        comments = [
            RedditComment(comment_id="c1", author="python_fan", body="Python because of readability."),
            RedditComment(comment_id="c2", author="rust_fan", body="Rust because of memory safety."),
        ]
        scraped_content = ScrapedContent(post=post, comments=comments)

        output_dir = tmp_path / "voiceovers"
        clips = await generate_voiceover(scraped_content, str(output_dir))

        assert "op" in clips
        assert "comment_1" in clips
        assert "comment_2" in clips

        assert clips["op"].clip_id == "op"
        assert clips["op"].file_path == str(output_dir / "op.mp3")
        assert clips["op"].duration_seconds > 0.0
        assert (output_dir / "op.mp3").exists()

        assert clips["comment_1"].clip_id == "comment_1"
        assert clips["comment_1"].file_path == str(output_dir / "comment_1.mp3")
        assert clips["comment_1"].duration_seconds > 0.0
        assert (output_dir / "comment_1.mp3").exists()

        assert clips["comment_2"].clip_id == "comment_2"
        assert clips["comment_2"].file_path == str(output_dir / "comment_2.mp3")
        assert clips["comment_2"].duration_seconds > 0.0
        assert (output_dir / "comment_2.mp3").exists()

    asyncio.run(_runner())


def test_clean_text_for_tts_complex_zwj_emojis():
    """Test clean_text_for_tts removes ZWJ family emojis, shrugging, flags, and leaves no orphan control codes."""
    raw_text = "Hello world! 😀 🔥 👨‍👩‍👧‍👦 🇺🇸 🤷‍♂️ 🪓 🫥 🫨"
    cleaned = clean_text_for_tts(raw_text)

    assert cleaned == "Hello world!"
    assert "\u200D" not in cleaned
    assert "\uFE0F" not in cleaned
    assert "\uFE0E" not in cleaned
    assert "\u200B" not in cleaned


def test_generate_audio_invalid_filename_characters(tmp_path: Path):
    """Test generate_audio sanitizes filenames containing invalid Windows path characters like ':' or '?'."""
    async def _runner():
        engine = TTSVoiceEngine(allow_offline_fallback=True)
        # File path stem with invalid characters ':' and '?'
        invalid_path = tmp_path / "audio:test?file.mp3"

        clip = await engine.generate_audio("Sanitization test audio", str(invalid_path))

        assert isinstance(clip, AudioClip)
        assert ":" not in clip.clip_id
        assert "?" not in clip.clip_id
        assert clip.clip_id == "audio_test_file"
        assert ":" not in Path(clip.file_path).name
        assert "?" not in Path(clip.file_path).name
        assert Path(clip.file_path).exists()

    asyncio.run(_runner())

