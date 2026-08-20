"""
Voice & Text-to-Speech Engine for Reddit Reading YouTube Shorts.
Uses edge-tts with automatic text cleaning, duration extraction, and offline fallback.
"""

import html
import logging
import re
import unicodedata
import wave
from pathlib import Path
from typing import Dict, Optional, Union

import edge_tts
from edge_tts.exceptions import NoAudioReceived
from src.models import AudioClip, ScrapedContent

logger = logging.getLogger(__name__)

# Silent MP3 MPEG-1 Layer III frame template (128 kbps, 44.1 kHz, 417 bytes)
_SILENT_MP3_FRAME = b"\xff\xfb\x90\xc4" + b"\x00" * 413
_MP3_FRAME_DURATION = 1152.0 / 44100.0


def clean_text_for_tts(raw_text: str) -> str:
    """Clean and format raw text for natural TTS output."""
    if not raw_text:
        return ""

    # 1. Unescape HTML entities (&amp; -> &, &lt; -> <, etc.)
    text = html.unescape(raw_text)

    # 2. Strip Reddit spoiler tags (>!text!<)
    text = re.sub(r">!|!<", "", text)

    # 3. Convert Markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # 4. Strip standalone URLs (http/https/www)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 5. Format Reddit user & subreddit tags for natural pronunciation
    text = re.sub(r"\br/([A-Za-z0-9_]+)", r"r slash \1", text)
    text = re.sub(r"\bu/([A-Za-z0-9_-]+)", r"user \1", text)

    # 6. Strip Markdown formatting symbols (*, _, ~~, `, #, >)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s*[>#\-\*]+\s*", "", text, flags=re.MULTILINE)

    # 7. Strip Emojis, format control characters, non-spacing marks, and non-BMP symbols
    text = "".join(
        c
        for c in text
        if not (
            unicodedata.category(c) in ("So", "Cs", "Cf", "Mn")
            or ord(c) > 0x1F000
            or 0x1F600 <= ord(c) <= 0x1F64F
            or 0x1F300 <= ord(c) <= 0x1F5FF
            or 0x1F680 <= ord(c) <= 0x1F6FF
            or 0x2600 <= ord(c) <= 0x27BF
            or c in ("\u200D", "\uFE0F", "\uFE0E", "\u200B", "\u200C", "\u200E", "\u200F")
        )
    )

    # 8. Normalize spaces and newlines
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def estimate_speech_duration(text: str) -> float:
    """Estimate speech duration in seconds based on word count (~2.5 words/sec)."""
    words = text.split()
    if not words:
        return 1.5
    return max(1.5, len(words) / 2.5)


def generate_fallback_audio(output_path: Union[str, Path], target_duration: float) -> float:
    """Generate a silent MP3 or WAV file when offline or TTS network fails."""
    path = Path(output_path)
    sanitized_stem = re.sub(r'[\\/*?:"<>|]', "_", path.stem)
    path = path.parent / f"{sanitized_stem}{path.suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".wav":
        sample_rate = 44100
        num_samples = int(sample_rate * max(0.5, target_duration))
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * num_samples)
        return num_samples / float(sample_rate)
    else:
        # Default MP3 silent frame synthesis
        num_frames = max(1, int(round(max(0.5, target_duration) / _MP3_FRAME_DURATION)))
        with open(path, "wb") as f:
            f.write(_SILENT_MP3_FRAME * num_frames)
        return num_frames * _MP3_FRAME_DURATION


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """Extract audio duration in seconds using mutagen, moviepy, wave, or size estimation."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # WAV extraction via standard library
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return frames / float(rate)
        except Exception:
            pass

    # Mutagen extraction for MP3 / general audio
    try:
        import mutagen.mp3
        audio = mutagen.mp3.MP3(str(path))
        if audio.info and audio.info.length:
            return float(audio.info.length)
    except Exception:
        pass

    # MoviePy fallback
    try:
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        with AudioFileClip(str(path)) as clip:
            return float(clip.duration)
    except Exception:
        pass

    # Size-based fallback (approx 16,000 bytes/sec for 128 kbps audio)
    size_bytes = path.stat().st_size
    return max(0.5, size_bytes / 16000.0)


class TTSVoiceEngine:
    """Voice generator class using edge-tts with offline fallback and metadata tracking."""

    def __init__(
        self,
        voice: str = "en-US-ChristopherNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        allow_offline_fallback: bool = True,
    ):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.allow_offline_fallback = allow_offline_fallback

    async def generate_audio(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: Optional[str] = None,
    ) -> AudioClip:
        """Synthesize text to speech audio and return AudioClip object."""
        import asyncio

        out = Path(output_path)
        sanitized_stem = re.sub(r'[\\/*?:"<>|]', "_", out.stem)
        out = out.parent / f"{sanitized_stem}{out.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)

        cleaned_text = clean_text_for_tts(text) or "(No text)"
        selected_voice = voice or self.voice

        # Pre-flight check: edge-tts raises NoAudioReceived for punctuation/whitespace-only
        # strings (e.g. ".", "...", ". . ."). Pad those with a spoken placeholder so we
        # avoid the empty-exception warning and still produce valid audio.
        _printable = re.sub(r"[^\w]", "", cleaned_text)
        if not _printable:
            logger.debug(
                "Text reduced to punctuation/whitespace after cleaning (%r); "
                "substituting placeholder before TTS.",
                cleaned_text,
            )
            cleaned_text = "(content unavailable)"

        try:
            communicate = edge_tts.Communicate(
                cleaned_text,
                selected_voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch,
            )
            await asyncio.wait_for(communicate.save(str(out)), timeout=30.0)
        except NoAudioReceived:
            # edge-tts returned no audio bytes — usually means the text was too short or
            # contained only symbols the service cannot pronounce.  Retry once with an
            # explicit spoken placeholder before falling back to silence.
            if not self.allow_offline_fallback:
                raise
            retry_text = "(content not available)"
            logger.warning(
                "edge-tts returned no audio for %r — retrying with placeholder text.",
                cleaned_text[:80],
            )
            try:
                communicate = edge_tts.Communicate(
                    retry_text,
                    selected_voice,
                    rate=self.rate,
                    volume=self.volume,
                    pitch=self.pitch,
                )
                await asyncio.wait_for(communicate.save(str(out)), timeout=30.0)
                cleaned_text = retry_text
            except Exception as retry_err:
                logger.warning(
                    "edge-tts retry also failed (%s). Using offline fallback.",
                    type(retry_err).__name__,
                )
                est_duration = estimate_speech_duration(cleaned_text)
                generate_fallback_audio(out, est_duration)
        except asyncio.TimeoutError:
            if not self.allow_offline_fallback:
                raise
            logger.warning(
                "edge-tts synthesis timed out for %r. Using offline fallback.",
                cleaned_text[:80],
            )
            est_duration = estimate_speech_duration(cleaned_text)
            generate_fallback_audio(out, est_duration)
        except Exception as e:
            if not self.allow_offline_fallback:
                raise
            logger.warning(
                "edge-tts synthesis failed (%s: %s). Using offline fallback.",
                type(e).__name__,
                e,
            )
            est_duration = estimate_speech_duration(cleaned_text)
            generate_fallback_audio(out, est_duration)

        duration = get_audio_duration(out)
        clip_id = out.stem

        return AudioClip(
            clip_id=clip_id,
            file_path=str(out),
            duration_seconds=duration,
            text=cleaned_text,
        )

    async def generate(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: Optional[str] = None,
    ) -> AudioClip:
        """Alias for generate_audio to maintain interface flexibility."""
        return await self.generate_audio(text=text, output_path=output_path, voice=voice)


async def generate_voiceover(
    scraped_content: ScrapedContent,
    output_dir: Union[str, Path],
    voice: str = "en-US-ChristopherNeural",
) -> Dict[str, AudioClip]:
    """
    Generate voiceover audio clips for OP and top comments in ScrapedContent.
    Returns map with keys: 'op', 'comment_1', 'comment_2'.
    """
    engine = TTSVoiceEngine(voice=voice)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, AudioClip] = {}

    # 1. OP Audio (Title + Body if present)
    op_text = scraped_content.post.title
    if scraped_content.post.body and scraped_content.post.body.strip():
        op_text = f"{op_text}. {scraped_content.post.body.strip()}"

    op_path = out_dir / "op.mp3"
    op_clip = await engine.generate_audio(text=op_text, output_path=str(op_path))
    result["op"] = op_clip

    # 2. Comments Audio
    for idx, comment in enumerate(scraped_content.comments, start=1):
        if not comment or not comment.body:
            continue
        c_path = out_dir / f"comment_{idx}.mp3"
        c_clip = await engine.generate_audio(text=comment.body, output_path=str(c_path))
        result[f"comment_{idx}"] = c_clip

    return result
