"""Background music generator for Reddit Reading YouTube Shorts using Meta MusicGen (local) and DSP Lofi synthesis."""

import os
import sys
import gc
import logging
from pathlib import Path
from typing import Optional

from src.config import ASSETS_DIR, ensure_directories

logger = logging.getLogger(__name__)

# Optional custom HuggingFace cache directory (respects HF_HOME if set)
if "HF_HOME" not in os.environ and os.path.exists("e:/Gemini Things/HF_Cache"):
    os.environ["HF_HOME"] = "e:/Gemini Things/HF_Cache"

MUSIC_DIR = ASSETS_DIR / "music"
MUSIC_DIR.mkdir(parents=True, exist_ok=True)

MUSIC_PROMPTS = {
    "lofi": "lofi ambient chill relaxing beat",
    "funny": "upbeat energetic funny comedy beat",
    "dramatic": "dramatic mysterious storytelling music",
    "acoustic": "happy acoustic ukulele upbeat music",
    "synthwave": "synthwave cyberpunk dark retro pulse",
}


def write_wav_file(path: str, sample_rate: int, audio_data) -> None:
    """Writes 1D float audio numpy array to 16-bit PCM WAV file."""
    import wave
    import numpy as np
    data_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(data_int16.tobytes())


def generate_dsp_lofi(output_path: str, duration_sec: float = 30.0) -> str:
    """Synthesize clean DSP Lofi ambient track locally without GPU requirements."""
    import numpy as np
    sample_rate = 44100
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Base Pads
    chords_frequencies = [
        [220.0, 261.63, 329.63, 392.00],  # Am7
        [174.61, 220.00, 261.63, 349.23], # Fmaj7
        [130.81, 164.81, 196.00, 261.63], # Cmaj7
        [196.00, 246.94, 293.66, 392.00]  # G6
    ]

    chord_duration = 7.5
    samples_per_chord = int(sample_rate * chord_duration)
    pad_wave = np.zeros(num_samples)

    for i in range(4):
        start_idx = i * samples_per_chord
        end_idx = min(start_idx + samples_per_chord, num_samples)
        chord_len = end_idx - start_idx
        chord_t = np.linspace(0, chord_duration, chord_len, endpoint=False)
        chord_signal = np.zeros(chord_len)

        for f in chords_frequencies[i]:
            chord_signal += 0.20 * np.sin(2 * np.pi * f * chord_t)
            chord_signal += 0.04 * np.sin(2 * np.pi * (2 * f) * chord_t)

        envelope = np.ones(chord_len)
        att = int(sample_rate * 1.0)
        rel = int(sample_rate * 1.0)
        envelope[:att] = np.linspace(0, 1, att)
        envelope[-rel:] = np.linspace(1, 0, rel)
        pad_wave[start_idx:end_idx] = chord_signal * envelope

    # Tape Hiss & Vinyl Crackle
    tape_hiss = np.random.normal(0, 0.004, num_samples)
    combined = pad_wave + tape_hiss
    combined = np.clip(combined, -0.9, 0.9)

    write_wav_file(output_path, sample_rate, combined)
    return output_path


def generate_musicgen_ai(prompt_key: str = "lofi", duration_seconds: int = 15) -> str:
    """
    Generates original background music using Meta's MusicGen (facebook/musicgen-small).
    Loads 100% locally from e:/Gemini Things/HF_Cache.
    """
    prompt = MUSIC_PROMPTS.get(prompt_key, MUSIC_PROMPTS["lofi"])
    output_path = str(MUSIC_DIR / f"musicgen_{prompt_key}.wav")

    if Path(output_path).exists() and Path(output_path).stat().st_size > 10000:
        logger.info(f"Using cached Meta MusicGen audio: {output_path}")
        return output_path

    logger.info(f"Generating Meta MusicGen track for prompt: '{prompt}'...")

    try:
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        model_id = "facebook/musicgen-small"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        try:
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            model = MusicgenForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, local_files_only=True
            ).to(device)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=False)
            model = MusicgenForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, local_files_only=False
            ).to(device)

        max_tokens = min(1500, max(256, int(duration_seconds * 50)))

        inputs = processor(
            text=[f"instrumental background music, {prompt}, no vocals, clean audio"],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            audio_values = model.generate(**inputs, max_new_tokens=max_tokens)

        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()

        write_wav_file(output_path, sampling_rate, audio_data)
        logger.info(f"Meta MusicGen AI soundtrack generated: {output_path}")

        del model, processor, inputs, audio_values
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_path
    except Exception as e:
        logger.warning(f"Meta MusicGen generation failed ({e}). Falling back to DSP Lofi beat.")
        return generate_dsp_lofi(output_path, duration_sec=float(duration_seconds))


def get_background_music(music_style: Optional[str] = "lofi") -> Optional[str]:
    """Resolves or generates background music file path based on selected music style."""
    if not music_style or music_style.lower() in ("none", "off", "disabled"):
        return None

    style = music_style.lower().strip()
    if style == "dsp":
        dsp_path = str(MUSIC_DIR / "dsp_lofi.wav")
        if not Path(dsp_path).exists():
            generate_dsp_lofi(dsp_path, duration_sec=30.0)
        return dsp_path

    return generate_musicgen_ai(prompt_key=style, duration_seconds=15)
