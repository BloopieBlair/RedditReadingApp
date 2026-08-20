"""9:16 vertical video composer and dynamic card overlay engine."""

import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
)
from src.config import (
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    get_background_video_path,
)
from src.models import (
    AudioClip,
    ScrapedContent,
    VideoRenderConfig,
    calculate_clip_timeline,
    calculate_total_audio_duration,
)
from src.video.background_manager import BackgroundManager

logger = logging.getLogger(__name__)


def apply_card_shadow_and_corners(
    image_path: Union[str, Path, Image.Image],
    max_width: int = 960,
    max_height: Optional[int] = None,
    corner_radius: int = 24,
    shadow_blur: int = 20,
) -> Image.Image:
    """
    Helper to crop rounded corners and add a soft drop shadow canvas around a card image.
    Accepts file path (str/Path) or an existing PIL Image object.
    Supports max_width and max_height scaling to prevent extra long posts from overflowing vertical video bounds.
    """
    if isinstance(image_path, (str, Path)):
        img = Image.open(str(image_path)).convert("RGBA")
    elif isinstance(image_path, Image.Image):
        img = image_path.convert("RGBA")
    else:
        raise TypeError("image_path must be a file path or PIL Image instance")

    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        return img

    # Scale down proportionally if width > max_width OR height > max_height
    scale_w = max_width / float(orig_w) if orig_w > max_width else 1.0
    scale_h = max_height / float(orig_h) if (max_height and orig_h > max_height) else 1.0
    scale = min(scale_w, scale_h)

    if scale < 1.0:
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    w, h = img.size

    # 1. Rounded corner mask
    mask = Image.new("L", (w, h), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rounded_rectangle((0, 0, w, h), radius=corner_radius, fill=255)

    orig_alpha = img.split()[3]
    combined_alpha = ImageChops.multiply(orig_alpha, mask)
    img.putalpha(combined_alpha)

    # 2. Soft drop shadow canvas
    pad = shadow_blur * 2
    canvas_w = w + pad * 2
    canvas_h = h + pad * 2

    shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (pad, pad + 8, pad + w, pad + 8 + h), radius=corner_radius, fill=(0, 0, 0, 160)
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))

    # 3. Paste rounded card over shadow
    shadow.paste(img, (pad, pad), img)
    return shadow


def create_fallback_card(
    title: str, text: str, width: int = 860, height: int = 400
) -> Image.Image:
    """Generate a clean synthetic PNG card image when screenshot is missing."""
    img = Image.new("RGBA", (width, height), (255, 255, 255, 245))
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([(0, 0), (width, 60)], fill=(240, 242, 245, 255))
    draw.text((24, 18), f"r/AskReddit • {title[:35]}", fill=(30, 30, 30))

    # Text lines
    words = (text or "").split()
    lines = []
    curr = []
    for w in words:
        curr.append(w)
        if len(" ".join(curr)) > 42:
            lines.append(" ".join(curr[:-1]))
            curr = [w]
    if curr:
        lines.append(" ".join(curr))

    y_pos = 80
    for line in lines[:8]:
        draw.text((24, y_pos), line, fill=(40, 40, 40))
        y_pos += 32

    return img


def get_ffmpeg_vcodec(use_gpu: bool = True) -> str:
    """Return hardware encoder if GPU acceleration is requested and available, else 'libx264'."""
    if use_gpu:
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            res = subprocess.run(
                [exe, "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if "h264_nvenc" in res.stdout:
                logger.info("GPU Acceleration enabled: Using h264_nvenc hardware encoder (NVIDIA).")
                return "h264_nvenc"
            elif "h264_amf" in res.stdout:
                logger.info("GPU Acceleration enabled: Using h264_amf hardware encoder (AMD).")
                return "h264_amf"
            elif "h264_qsv" in res.stdout:
                logger.info("GPU Acceleration enabled: Using h264_qsv hardware encoder (Intel).")
                return "h264_qsv"
        except Exception as e:
            logger.warning(f"GPU encoder check failed ({e}). Falling back to CPU software encoder libx264.")
    return "libx264"


def get_gpu_status() -> Dict[str, Any]:
    """Detects available GPU hardware acceleration (NVIDIA NVENC, AMD AMF, Intel QSV, or CPU fallback)."""
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        res = subprocess.run(
            [exe, "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out = res.stdout or ""
        if "h264_nvenc" in out:
            return {
                "has_gpu": True,
                "encoder": "h264_nvenc",
                "vendor": "NVIDIA",
                "label": "GPU Hardware Acceleration",
                "detail": "NVIDIA NVENC Hardware Encoder"
            }
        elif "h264_amf" in out:
            return {
                "has_gpu": True,
                "encoder": "h264_amf",
                "vendor": "AMD",
                "label": "GPU Hardware Acceleration",
                "detail": "AMD AMF Hardware Encoder"
            }
        elif "h264_qsv" in out:
            return {
                "has_gpu": True,
                "encoder": "h264_qsv",
                "vendor": "Intel",
                "label": "GPU Hardware Acceleration",
                "detail": "Intel QuickSync Hardware Encoder"
            }
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")

    return {
        "has_gpu": False,
        "encoder": "libx264",
        "vendor": "CPU",
        "label": "CPU Software Encoding",
        "detail": "libx264 Software Encoder"
    }


class ShortsVideoComposer:
    """Compositor for Reddit Reading YouTube Shorts using MoviePy 2.x and synthetic fallback."""

    def __init__(self, config: Optional[VideoRenderConfig] = None, use_gpu: bool = True):
        self.config = config or VideoRenderConfig(
            width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=VIDEO_FPS
        )
        self.use_gpu = use_gpu
        self.bg_manager = BackgroundManager()

    def compose(
        self,
        scraped_content: ScrapedContent,
        audio_clips: Dict[str, AudioClip],
        output_path: str,
        background_video_path: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        music_style: Optional[str] = "lofi",
    ) -> str:
        """Renders final 9:16 vertical video with card overlays, composite audio, and background music."""
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)

        if use_gpu is not None:
            self.use_gpu = use_gpu

        timeline = calculate_clip_timeline(audio_clips) if audio_clips else []
        total_duration = timeline[-1]["end_time"] if timeline else 5.0
        total_duration = max(1.0, float(total_duration))

        try:
            return self._compose_ffmpeg_native(
                scraped_content=scraped_content,
                audio_clips=audio_clips,
                timeline=timeline,
                total_duration=total_duration,
                output_path=str(output_p),
                background_video_path=background_video_path,
                music_style=music_style,
            )
        except Exception as e:
            logger.warning(
                f"Native FFmpeg GPU composition failed ({e}). Falling back to MoviePy."
            )
            try:
                return self._compose_moviepy(
                    scraped_content=scraped_content,
                    audio_clips=audio_clips,
                    timeline=timeline,
                    total_duration=total_duration,
                    output_path=str(output_p),
                    background_video_path=background_video_path,
                )
            except Exception as e2:
                logger.warning(
                    f"MoviePy composition failed ({e2}). Triggering synthetic fallback renderer."
                )
                return self._render_synthetic_fallback(
                    scraped_content=scraped_content,
                    audio_clips=audio_clips,
                    timeline=timeline,
                    total_duration=total_duration,
                    output_path=str(output_p),
                    background_video_path=background_video_path,
                )

    def _compose_ffmpeg_native(
        self,
        scraped_content: ScrapedContent,
        audio_clips: Dict[str, AudioClip],
        timeline: List[Dict],
        total_duration: float,
        output_path: str,
        background_video_path: Optional[str] = None,
        music_style: Optional[str] = "lofi",
    ) -> str:
        """Native FFmpeg C++/GPU Filter pipeline.
        Executes video cropping, PNG image overlays, and background music mixing directly inside FFmpeg GPU hardware engine.
        """
        logger.info("Executing native FFmpeg GPU filter pipeline...")
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)

        # Use a short random hex token so the temp path stays well under Windows'
        # 260-char MAX_PATH limit regardless of how long the output filename is.
        temp_dir = output_p.parent / f"_tmp_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Prepare PNG Cards
        is_long_op = False
        enable_subs = getattr(self.config, "enable_subtitles", True)
        sub_style = getattr(self.config, "subtitle_style", "yellow_pill")
        op_max_h = 1320 if enable_subs else 1600

        op_card_path = scraped_content.op_card_image_path
        if op_card_path and Path(op_card_path).exists():
            op_card_pil = apply_card_shadow_and_corners(op_card_path, max_width=960, max_height=op_max_h)
        else:
            op_title = scraped_content.post.title if scraped_content.post else "Reddit Post"
            op_body = scraped_content.post.body if scraped_content.post else ""
            op_card_pil = apply_card_shadow_and_corners(
                create_fallback_card(op_title, op_body), max_width=960, max_height=op_max_h
            )

        op_png = temp_dir / "op.png"
        op_card_pil.save(op_png)

        comment_inputs = []
        for idx, entry in enumerate(timeline):
            key = entry["key"]
            if key == "op":
                continue

            card_path = None
            if key in ("comment_1", "comment1") or idx == 1:
                card_path = scraped_content.comment1_card_image_path
            elif key in ("comment_2", "comment2") or idx == 2:
                card_path = scraped_content.comment2_card_image_path
            else:
                c_idx = idx - 1
                if c_idx < len(scraped_content.comment_card_image_paths):
                    card_path = scraped_content.comment_card_image_paths[c_idx]

            if card_path and Path(card_path).exists():
                c_pil = apply_card_shadow_and_corners(card_path, max_width=960)
            else:
                body = (
                    scraped_content.comments[idx - 1].body
                    if (idx - 1 < len(scraped_content.comments))
                    else f"Comment {idx}"
                )
                c_pil = apply_card_shadow_and_corners(
                    create_fallback_card(f"Comment {idx}", body), max_width=960
                )

            c_png = temp_dir / f"comment_{idx}.png"
            c_pil.save(c_png)
            comment_inputs.append(
                {
                    "png_path": str(c_png),
                    "start": entry["start_time"],
                    "end": entry["end_time"],
                }
            )

        # Calculate OP duration & check if OP post is too long (intersects comment area)
        op_h = op_card_pil.height
        op_end_time = next((entry["end_time"] for entry in timeline if entry["key"] == "op"), 0.0)
        if op_end_time <= 0 and timeline:
            op_end_time = timeline[0]["end_time"]

        is_long_op = (op_h > 750) or (140 + op_h > 940)
        op_y = max(60, min(140, VIDEO_HEIGHT - op_h - 40))

        # 1.5 Prepare Animated Subtitle PNG Overlays
        sub_inputs = []
        if enable_subs:
            from src.video.subtitles import generate_caption_timeline, render_caption_image

            raw_sub_items = []
            op_entry = next((e for e in timeline if e["key"] == "op"), None)
            if op_entry and scraped_content.post:
                op_dur = op_entry["end_time"] - op_entry["start_time"]
                op_text = (scraped_content.post.title + " " + scraped_content.post.body).strip()
                raw_sub_items.extend(
                    generate_caption_timeline(op_text, duration=op_dur, start_time=op_entry["start_time"])
                )

            for idx, entry in enumerate(timeline):
                if entry["key"] == "op":
                    continue
                c_idx = idx - 1
                if c_idx < len(scraped_content.comments):
                    c_text = scraped_content.comments[c_idx].body.strip()
                    c_dur = entry["end_time"] - entry["start_time"]
                    raw_sub_items.extend(
                        generate_caption_timeline(c_text, duration=c_dur, start_time=entry["start_time"])
                    )

            for s_idx, item in enumerate(raw_sub_items):
                sub_img = render_caption_image(item["phrase"], canvas_w=VIDEO_WIDTH, style=sub_style)
                sub_png = temp_dir / f"sub_{s_idx}.png"
                sub_img.save(sub_png)

                # Non-overlapping Y position calculation
                is_op_phase = item["start"] < op_end_time
                if is_op_phase:
                    sub_y = 1520 if is_long_op else max(1350, min(1620, op_y + op_h + 40))
                else:
                    sub_y = 1380

                sub_inputs.append({
                    "png_path": str(sub_png),
                    "start": item["start"],
                    "end": item["end"],
                    "y": sub_y
                })

        # 2. Concatenate audio clips & resolve background music
        temp_audio_path = temp_dir / "temp_audio.wav"
        has_audio = self._concatenate_audio_clips(timeline, temp_audio_path)

        from src.music_generator import get_background_music
        bg_music_file = get_background_music(music_style=music_style) if music_style else None
        has_music = bool(bg_music_file and Path(bg_music_file).exists())

        # 3. Build FFmpeg command with native GPU hardware acceleration & thread cap
        bg_source = background_video_path or str(get_background_video_path())
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [ffmpeg_exe, "-y", "-threads", "4"]
        if self.use_gpu:
            cmd += ["-hwaccel", "cuda"]

        if bg_source and Path(bg_source).exists():
            cmd += ["-ss", "0", "-stream_loop", "-1", "-i", str(bg_source), "-t", str(total_duration)]
        else:
            cmd += [
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:d={total_duration}:r={VIDEO_FPS}",
            ]

        # Add card image inputs (Subtitle images are loaded via filter script movie filter to keep cmd line short)
        cmd += ["-i", str(op_png)]
        for ci in comment_inputs:
            cmd += ["-i", ci["png_path"]]

        # Add audio input if present
        if has_audio and temp_audio_path.exists():
            cmd += ["-i", str(temp_audio_path)]

        # Add background music input if present
        if has_music:
            cmd += ["-stream_loop", "-1", "-i", str(bg_music_file)]

        # Build filter_complex string
        filter_parts = [f"[0:v]crop=ih*9/16:ih,scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}[bg]"]
        if is_long_op:
            logger.info(f"Long OP post detected (height={op_h}px). Hiding OP card after t={op_end_time:.2f}s.")
            filter_parts.append(f"[bg][1:v]overlay=(W-w)/2:{op_y}:enable='between(t,0,{op_end_time})'[v1]")
        else:
            filter_parts.append(f"[bg][1:v]overlay=(W-w)/2:{op_y}[v1]")

        curr_stream = "[v1]"

        # Add comment card overlays
        total_card_overlays = len(comment_inputs)
        total_sub_overlays = len(sub_inputs)
        total_visual_overlays = total_card_overlays + total_sub_overlays

        for i, ci in enumerate(comment_inputs):
            input_idx = i + 2
            next_idx = i + 2
            is_final = (i == total_card_overlays - 1) and (total_sub_overlays == 0)
            next_stream = "[outv]" if is_final else f"[v{next_idx}]"

            if is_long_op:
                c_y = max(300, (VIDEO_HEIGHT - 450) // 2)
            else:
                c_y = max(640, min(1280, 160 + op_h + 40))

            filter_parts.append(
                f"{curr_stream}[{input_idx}:v]overlay=(W-w)/2:{c_y}:enable='between(t,{ci['start']},{ci['end']})'{next_stream}"
            )
            curr_stream = next_stream

        # Add subtitle card overlays using FFmpeg movie filter to prevent Windows command line length limit (WinError 206)
        for j, si in enumerate(sub_inputs):
            next_idx = total_card_overlays + j + 2
            is_final = (j == total_sub_overlays - 1)
            next_stream = "[outv]" if is_final else f"[v{next_idx}]"

            safe_sub_path = si["png_path"].replace("\\", "/").replace(":", "\\:")
            filter_parts.append(f"movie='{safe_sub_path}'[sub{j}]")
            filter_parts.append(
                f"{curr_stream}[sub{j}]overlay=0:{si['y']}:enable='between(t,{si['start']},{si['end']})'{next_stream}"
            )
            curr_stream = next_stream

        if total_visual_overlays == 0:
            filter_parts[-1] = filter_parts[-1].replace("[v1]", "[outv]")

        # Mix voiceover audio and background music
        audio_map_target = None
        if has_audio and temp_audio_path.exists():
            voice_idx = 1 + 1 + total_card_overlays
            if has_music:
                music_idx = voice_idx + 1
                filter_parts.append(
                    f"[{voice_idx}:a]volume=1.0[voice]; [{music_idx}:a]volume=0.14[music]; [voice][music]amix=inputs=2:duration=first[outa]"
                )
                audio_map_target = "[outa]"
            else:
                audio_map_target = f"{voice_idx}:a"

        filter_complex_str = "; ".join(filter_parts)
        filter_script_path = temp_dir / "filter_script.txt"
        with open(filter_script_path, "w", encoding="utf-8") as f:
            f.write(filter_complex_str)

        vcodec = get_ffmpeg_vcodec(use_gpu=self.use_gpu)

        cmd += [
            "-filter_complex_script",
            str(filter_script_path),
            "-map",
            "[outv]",
        ]

        if audio_map_target:
            cmd += ["-map", audio_map_target, "-c:a", "aac"]

        cmd += [
            "-t",
            str(total_duration),
            "-c:v",
            vcodec,
        ]

        if "nvenc" in vcodec:
            cmd += ["-preset", "p1"]
        elif "amf" in vcodec:
            cmd += ["-quality", "speed"]
        elif "qsv" in vcodec:
            cmd += ["-preset", "veryfast"]
        else:
            cmd += ["-preset", "ultrafast"]

        cmd += [
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_p),
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return str(output_p)

    def _compose_moviepy(
        self,
        scraped_content: ScrapedContent,
        audio_clips: Dict[str, AudioClip],
        timeline: List[Dict],
        total_duration: float,
        output_path: str,
        background_video_path: Optional[str] = None,
    ) -> str:
        # 1. Background Video Clip
        is_long_op = False
        bg_clip = self.bg_manager.get_background_clip(
            target_duration=total_duration, video_source_path=background_video_path
        )

        # 2. Audio Composite
        audio_moviepy_clips = []
        for entry in timeline:
            fp = entry["file_path"]
            if Path(fp).exists():
                a_clip = AudioFileClip(fp).with_start(entry["start_time"])
                audio_moviepy_clips.append(a_clip)

        composite_audio = (
            CompositeAudioClip(audio_moviepy_clips) if audio_moviepy_clips else None
        )

        # 3. Card Overlays
        video_clips = [bg_clip]

        # 3a. OP Card (Hides after voiceover if very long, else persists for full video)
        op_card_path = scraped_content.op_card_image_path
        if op_card_path and Path(op_card_path).exists():
            op_card_pil = apply_card_shadow_and_corners(op_card_path, max_width=960, max_height=1600)
        else:
            op_title = scraped_content.post.title if scraped_content.post else "Reddit Post"
            op_body = scraped_content.post.body if scraped_content.post else ""
            op_card_pil = apply_card_shadow_and_corners(
                create_fallback_card(op_title, op_body), max_width=960, max_height=1600
            )

        enable_subs = getattr(self.config, "enable_subtitles", True)
        sub_style = getattr(self.config, "subtitle_style", "yellow_pill")

        op_h = op_card_pil.height
        op_end_time = next((entry["end_time"] for entry in timeline if entry["key"] == "op"), 0.0)
        if op_end_time <= 0 and timeline:
            op_end_time = timeline[0]["end_time"]

        is_long_op = (op_h > 750) or (140 + op_h > 940)
        op_dur = op_end_time if is_long_op else total_duration
        op_y = max(60, min(140, VIDEO_HEIGHT - op_h - 40))

        sub_inputs = []
        if enable_subs:
            from src.video.subtitles import generate_caption_timeline, render_caption_image

            temp_mpy_dir = Path(output_path).parent / f"_tmp_mpy_{uuid.uuid4().hex[:8]}"
            temp_mpy_dir.mkdir(parents=True, exist_ok=True)

            raw_sub_items = []
            op_entry = next((e for e in timeline if e["key"] == "op"), None)
            if op_entry and scraped_content.post:
                op_dur = op_entry["end_time"] - op_entry["start_time"]
                op_text = (scraped_content.post.title + " " + scraped_content.post.body).strip()
                raw_sub_items.extend(
                    generate_caption_timeline(op_text, duration=op_dur, start_time=op_entry["start_time"])
                )

            for idx, entry in enumerate(timeline):
                if entry["key"] == "op":
                    continue
                c_idx = idx - 1
                if c_idx < len(scraped_content.comments):
                    c_text = scraped_content.comments[c_idx].body.strip()
                    c_dur = entry["end_time"] - entry["start_time"]
                    raw_sub_items.extend(
                        generate_caption_timeline(c_text, duration=c_dur, start_time=entry["start_time"])
                    )

            for s_idx, item in enumerate(raw_sub_items):
                sub_img = render_caption_image(item["phrase"], canvas_w=VIDEO_WIDTH, style=sub_style)
                sub_png = temp_mpy_dir / f"sub_{s_idx}.png"
                sub_img.save(sub_png)

                is_op_phase = item["start"] < op_end_time
                sub_y = 1520 if (is_op_phase and is_long_op) else (1520 if is_op_phase else 1380)
                sub_inputs.append({
                    "png_path": str(sub_png),
                    "start": item["start"],
                    "end": item["end"],
                    "y": sub_y
                })

        op_card_clip = (
            ImageClip(np.array(op_card_pil))
            .with_start(0.0)
            .with_duration(op_dur)
            .with_position(("center", op_y))
        )
        video_clips.append(op_card_clip)

        # 3b. Sequential Comment Cards
        for idx, entry in enumerate(timeline):
            key = entry["key"]
            if key == "op":
                continue  # OP audio plays while OP card is already pinned at top

            start_t = entry["start_time"]
            dur = entry["duration"]

            card_path = None
            if key in ("comment_1", "comment1") or idx == 1:
                card_path = scraped_content.comment1_card_image_path
            elif key in ("comment_2", "comment2") or idx == 2:
                card_path = scraped_content.comment2_card_image_path
            else:
                c_idx = idx - 1
                if c_idx < len(scraped_content.comment_card_image_paths):
                    card_path = scraped_content.comment_card_image_paths[c_idx]

            if card_path and Path(card_path).exists():
                card_pil = apply_card_shadow_and_corners(card_path, max_width=960)
            else:
                body = (
                    scraped_content.comments[idx - 1].body
                    if (idx - 1 < len(scraped_content.comments))
                    else f"Comment {idx}"
                )
                fb_img = create_fallback_card(f"Comment {idx}", body)
                card_pil = apply_card_shadow_and_corners(fb_img, max_width=960)

            c_y = (VIDEO_HEIGHT - card_pil.height) // 2 if is_long_op else (160 + op_h + 40)

            c_clip = (
                ImageClip(np.array(card_pil))
                .with_start(start_t)
                .with_duration(dur)
                .with_position(("center", c_y))
            )
            video_clips.append(c_clip)

        # 3c. Add Animated Subtitle Overlay Clips
        if enable_subs:
            for si in sub_inputs:
                sub_img = Image.open(si["png_path"]).convert("RGBA")
                sub_clip = (
                    ImageClip(np.array(sub_img))
                    .with_start(si["start"])
                    .with_duration(si["end"] - si["start"])
                    .with_position((0, si["y"]))
                )
                video_clips.append(sub_clip)

        # 4. Composite & Export
        final_video = CompositeVideoClip(
            video_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
        )
        if composite_audio:
            final_video = final_video.with_audio(composite_audio)

        vcodec = get_ffmpeg_vcodec(use_gpu=self.use_gpu)
        preset_val = "p1" if "nvenc" in vcodec else "ultrafast"
        temp_audio = f"{output_path}.temp_audio.m4a"

        final_video.write_videofile(
            output_path,
            fps=VIDEO_FPS,
            codec=vcodec,
            audio_codec="aac",
            temp_audiofile=temp_audio,
            remove_temp=True,
            preset=preset_val,
            threads=6,
            ffmpeg_params=["-vf", "format=yuv420p", "-movflags", "+faststart"],
            logger=None,
        )

        final_video.close()
        bg_clip.close()
        if composite_audio:
            composite_audio.close()
        for a_clip in audio_moviepy_clips:
            a_clip.close()

        return output_path

    def _render_synthetic_fallback(
        self,
        scraped_content: ScrapedContent,
        audio_clips: Dict[str, AudioClip],
        timeline: List[Dict],
        total_duration: float,
        output_path: str,
        background_video_path: Optional[str] = None,
    ) -> str:
        """
        Synthetic fallback video renderer (OpenCV + Pillow + wave/imageio_ffmpeg).
        Guarantees MP4 video output even if MoviePy environment fails.
        """
        logger.info("Executing synthetic fallback video renderer...")
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)

        is_long_op = False
        syn_token = uuid.uuid4().hex[:8]
        temp_syn_dir = output_p.parent / f"_tmp_syn_{syn_token}"
        temp_syn_dir.mkdir(parents=True, exist_ok=True)

        temp_video_path = temp_syn_dir / "temp_no_audio.mp4"

        # 1. Build list of active card images & positioning
        op_card_pil = None
        op_card_path = scraped_content.op_card_image_path
        if op_card_path and Path(op_card_path).exists():
            op_card_pil = apply_card_shadow_and_corners(op_card_path, max_width=960, max_height=1600)
        else:
            op_title = scraped_content.post.title if scraped_content.post else "Reddit Post"
            op_body = scraped_content.post.body if scraped_content.post else ""
            op_card_pil = apply_card_shadow_and_corners(
                create_fallback_card(op_title, op_body), max_width=960, max_height=1600
            )

        enable_subs = getattr(self.config, "enable_subtitles", True)
        sub_style = getattr(self.config, "subtitle_style", "yellow_pill")

        op_h = op_card_pil.height if op_card_pil else 400
        op_end_time = next((entry["end_time"] for entry in timeline if entry["key"] == "op"), 0.0)
        if op_end_time <= 0 and timeline:
            op_end_time = timeline[0]["end_time"]

        is_long_op = (op_h > 750) or (140 + op_h > 940)

        sub_inputs = []
        if enable_subs:
            from src.video.subtitles import generate_caption_timeline, render_caption_image

            raw_sub_items = []
            op_entry = next((e for e in timeline if e["key"] == "op"), None)
            if op_entry and scraped_content.post:
                op_dur = op_entry["end_time"] - op_entry["start_time"]
                op_text = (scraped_content.post.title + " " + scraped_content.post.body).strip()
                raw_sub_items.extend(
                    generate_caption_timeline(op_text, duration=op_dur, start_time=op_entry["start_time"])
                )

            for idx, entry in enumerate(timeline):
                if entry["key"] == "op":
                    continue
                c_idx = idx - 1
                if c_idx < len(scraped_content.comments):
                    c_text = scraped_content.comments[c_idx].body.strip()
                    c_dur = entry["end_time"] - entry["start_time"]
                    raw_sub_items.extend(
                        generate_caption_timeline(c_text, duration=c_dur, start_time=entry["start_time"])
                    )

            for s_idx, item in enumerate(raw_sub_items):
                sub_img = render_caption_image(item["phrase"], canvas_w=VIDEO_WIDTH, style=sub_style)
                sub_png = temp_syn_dir / f"sub_{s_idx}.png"
                sub_img.save(sub_png)

                is_op_phase = item["start"] < op_end_time
                sub_y = 1520 if (is_op_phase and is_long_op) else (1520 if is_op_phase else 1380)
                sub_inputs.append({
                    "png_path": str(sub_png),
                    "start": item["start"],
                    "end": item["end"],
                    "y": sub_y
                })

        comment_cards = []
        for idx, entry in enumerate(timeline):
            key = entry["key"]
            if key == "op":
                continue

            card_path = None
            if key in ("comment_1", "comment1") or idx == 1:
                card_path = scraped_content.comment1_card_image_path
            elif key in ("comment_2", "comment2") or idx == 2:
                card_path = scraped_content.comment2_card_image_path
            else:
                c_idx = idx - 1
                if c_idx < len(scraped_content.comment_card_image_paths):
                    card_path = scraped_content.comment_card_image_paths[c_idx]

            if card_path and Path(card_path).exists():
                c_pil = apply_card_shadow_and_corners(card_path, max_width=960)
            else:
                body = (
                    scraped_content.comments[idx - 1].body
                    if (idx - 1 < len(scraped_content.comments))
                    else f"Comment {idx}"
                )
                fb_img = create_fallback_card(f"Comment {idx}", body)
                c_pil = apply_card_shadow_and_corners(fb_img, max_width=960)

            comment_cards.append(
                {
                    "start": entry["start_time"],
                    "end": entry["end_time"],
                    "card_pil": c_pil,
                    "pos_y": 740,
                }
            )

        # 2. Render background + overlay frame-by-frame with OpenCV & Pillow
        bg_source = background_video_path or str(get_background_video_path())
        cap = (
            cv2.VideoCapture(bg_source)
            if bg_source and Path(bg_source).exists()
            else None
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(temp_video_path), fourcc, float(VIDEO_FPS), (VIDEO_WIDTH, VIDEO_HEIGHT)
        )
        total_frames = int(total_duration * VIDEO_FPS)

        for frame_idx in range(total_frames):
            t = frame_idx / float(VIDEO_FPS)

            frame_bgr = None
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    target_ar = 9.0 / 16.0
                    curr_ar = w / float(h) if h > 0 else 1.0
                    if curr_ar > target_ar:
                        crop_w = int(h * target_ar)
                        x1 = (w - crop_w) // 2
                        cropped = frame[:, x1 : x1 + crop_w]
                    elif curr_ar < target_ar:
                        crop_h = int(w / target_ar)
                        y1 = (h - crop_h) // 2
                        cropped = frame[y1 : y1 + crop_h, :]
                    else:
                        cropped = frame
                    frame_bgr = cv2.resize(cropped, (VIDEO_WIDTH, VIDEO_HEIGHT))

            if frame_bgr is None:
                # Dynamic gradient frame
                t_ratio = frame_idx / float(total_frames) if total_frames > 0 else 0.0
                b = np.linspace(
                    30 + 20 * np.sin(t_ratio * np.pi * 2),
                    60,
                    VIDEO_HEIGHT,
                    dtype=np.uint8,
                )
                g = np.linspace(
                    15,
                    40 + 20 * np.cos(t_ratio * np.pi * 2),
                    VIDEO_HEIGHT,
                    dtype=np.uint8,
                )
                r = np.linspace(
                    40 + 30 * np.sin(t_ratio * np.pi * 4),
                    80,
                    VIDEO_HEIGHT,
                    dtype=np.uint8,
                )
                frame_bgr = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
                frame_bgr[:, :, 0] = b[:, None]
                frame_bgr[:, :, 1] = g[:, None]
                frame_bgr[:, :, 2] = r[:, None]

            # Composite cards onto background using Pillow
            bg_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert(
                "RGBA"
            )

            # 2a. Paste OP card pinned at top-center (hide after op_end_time if long OP)
            if op_card_pil and (not is_long_op or t <= op_end_time):
                x = (VIDEO_WIDTH - op_card_pil.width) // 2
                op_y = max(60, min(140, VIDEO_HEIGHT - op_card_pil.height - 40))
                bg_pil.paste(op_card_pil, (x, op_y), op_card_pil)

            # 2b. Paste active comment card
            for cc in comment_cards:
                if cc["start"] <= t <= cc["end"]:
                    c_pil = cc["card_pil"]
                    x = (VIDEO_WIDTH - c_pil.width) // 2
                    y = (VIDEO_HEIGHT - c_pil.height) // 2 if is_long_op else (160 + op_h + 40)
                    bg_pil.paste(c_pil, (x, y), c_pil)
                    break

            # 2c. Paste active subtitle card if present
            if enable_subs:
                for si in sub_inputs:
                    if si["start"] <= t <= si["end"]:
                        s_pil = Image.open(si["png_path"]).convert("RGBA")
                        bg_pil.paste(s_pil, (0, si["y"]), s_pil)
                        break

            out_frame = cv2.cvtColor(np.array(bg_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            writer.write(out_frame)

        writer.release()
        if cap:
            cap.release()

        # 3. Concatenate audio clips and mux with FFmpeg
        temp_audio_path = temp_syn_dir / "temp_audio.wav"
        has_audio = self._concatenate_audio_clips(timeline, temp_audio_path)

        vcodec = get_ffmpeg_vcodec(use_gpu=self.use_gpu)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        if (
            has_audio
            and temp_audio_path.exists()
            and temp_audio_path.stat().st_size > 0
        ):
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(temp_video_path),
                "-i",
                str(temp_audio_path),
                "-c:v",
                vcodec,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                "-shortest",
                str(output_p),
            ]
            subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
        else:
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(temp_video_path),
                "-c:v",
                vcodec,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_p),
            ]
            subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )

        import shutil
        shutil.rmtree(temp_syn_dir, ignore_errors=True)

        return str(output_p)

    def _concatenate_audio_clips(
        self, timeline: List[Dict], output_wav: Path
    ) -> bool:
        """Concatenates audio clips in timeline to a WAV file for fallback encoding."""
        try:
            clips = []
            for entry in timeline:
                fp = entry["file_path"]
                if Path(fp).exists():
                    clips.append(AudioFileClip(fp).with_start(entry["start_time"]))
            if not clips:
                return False
            comp = CompositeAudioClip(clips)
            comp.write_audiofile(str(output_wav), fps=44100, logger=None)
            comp.close()
            for c in clips:
                c.close()
            return True
        except Exception as e:
            logger.warning(f"Audio concatenation in fallback failed: {e}")
            return False


def compose_shorts_video(
    scraped_content: ScrapedContent,
    audio_clips: Dict[str, AudioClip],
    output_path: str,
    background_video_path: Optional[str] = None,
    use_gpu: bool = True,
    music_style: Optional[str] = "lofi",
    config: Optional[VideoRenderConfig] = None,
) -> str:
    """Functional interface for composing 9:16 YouTube Shorts video."""
    composer = ShortsVideoComposer(config=config, use_gpu=use_gpu)
    return composer.compose(
        scraped_content=scraped_content,
        audio_clips=audio_clips,
        output_path=output_path,
        background_video_path=background_video_path,
        use_gpu=use_gpu,
        music_style=music_style,
    )
