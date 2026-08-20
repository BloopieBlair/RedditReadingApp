"""End-to-End Reddit Reading Shorts execution pipeline with manifest tracking."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import OUTPUT_DIR, ensure_directories
from src.models import ScrapedContent
from src.scraper import fetch_reddit_post
from src.tts import generate_voiceover
from src.uploader import upload_short
from src.video import compose_shorts_video

logger = logging.getLogger(__name__)


class RedditReadingPipeline:
    """Integrated pipeline manager for generating Reddit Reading YouTube Shorts."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        voice: str = "en-US-ChristopherNeural",
        background_video: Optional[str] = None,
        dry_run: bool = False,
        use_gpu: bool = True,
        music_style: Optional[str] = "lofi",
        privacy_status: str = "unlisted",
        max_comments: int = 2,
        enable_subtitles: bool = True,
        subtitle_style: str = "yellow_pill",
    ):
        ensure_directories()
        self.output_dir = output_dir or str(OUTPUT_DIR)
        self.voice = voice
        self.background_video = background_video
        self.dry_run = dry_run
        self.use_gpu = use_gpu
        self.music_style = music_style
        self.privacy_status = privacy_status
        self.max_comments = max_comments
        self.enable_subtitles = enable_subtitles
        self.subtitle_style = subtitle_style

    async def execute(
        self,
        subreddit: str = "AskReddit",
        post_id: Optional[str] = None,
        output_video_path: Optional[str] = None,
        upload: bool = False,
    ) -> Dict[str, Any]:
        """Execute end-to-end Reddit Reading video pipeline."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        target_video_path = output_video_path or str(
            out_dir / f"{subreddit}_short.mp4"
        )

        # Stage 1: Scrape
        logger.info(f"Stage 1/4: Scraping r/{subreddit} (max comments: {self.max_comments})...")
        scraped_content = await fetch_reddit_post(
            subreddit=subreddit, post_id=post_id, max_comments=self.max_comments
        )

        # Stage 2: TTS
        logger.info("Stage 2/4: Generating TTS voiceover...")
        audio_clips = await generate_voiceover(
            scraped_content=scraped_content,
            output_dir=self.output_dir,
            voice=self.voice,
        )

        # Stage 3: Video Composition
        from src.models import VideoRenderConfig
        render_cfg = VideoRenderConfig(
            enable_subtitles=self.enable_subtitles,
            subtitle_style=self.subtitle_style
        )
        logger.info(f"Stage 3/4: Compositing video (subtitles={self.enable_subtitles}, style={self.subtitle_style})...")
        rendered_video = compose_shorts_video(
            scraped_content=scraped_content,
            audio_clips=audio_clips,
            output_path=target_video_path,
            background_video_path=self.background_video,
            use_gpu=self.use_gpu,
            music_style=self.music_style,
            config=render_cfg,
        )

        # Stage 3.5: AI Title & Description Generation
        from src.ai_generator import generate_video_metadata, save_video_folder

        logger.info("Generating AI title and description...")
        metadata = generate_video_metadata(
            post_title=scraped_content.post.title,
            post_body=scraped_content.post.body,
            subreddit=subreddit,
        )

        output_folder = save_video_folder(
            output_base_dir=self.output_dir,
            subreddit=subreddit,
            scraped_content=scraped_content,
            video_file_path=rendered_video,
            metadata=metadata,
        )
        saved_video_path = output_video_path or str(Path(output_folder) / f"{subreddit}_short.mp4")

        # Stage 4: Upload
        upload_result = None
        if upload:
            logger.info(f"Stage 4/4: Uploading to YouTube ({self.privacy_status.upper()} privacy status)...")
            upload_result = upload_short(
                video_path=saved_video_path,
                title=metadata["title"],
                description=metadata["description"],
                tags=["reddit", "shorts", subreddit, "minecraft"],
                privacy_status=self.privacy_status,
                dry_run=self.dry_run,
            )
        else:
            logger.info("Stage 4/4: Upload skipped (dry run / test mode).")

        # Save manifest
        manifest = {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "subreddit": subreddit,
            "post_id": scraped_content.post.post_id,
            "post_title": scraped_content.post.title,
            "video_path": saved_video_path,
            "output_folder": output_folder,
            "title": metadata["title"],
            "description": metadata["description"],
            "voice": self.voice,
            "dry_run": self.dry_run,
            "upload_result": upload_result,
            "scraped_content": scraped_content.to_dict(),
            "audio_clips": {k: v.to_dict() for k, v in audio_clips.items()},
        }

        manifest_path = Path(output_folder) / "manifest.json"
        legacy_manifest_path = out_dir / f"manifest_{scraped_content.post.post_id}.json"
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, default=str)
            with open(legacy_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, default=str)
            logger.info(f"Manifest saved to {manifest_path}")
        except Exception as e:
            logger.warning(f"Failed to save manifest: {e}")

        return manifest


async def run_batch_pipeline(
    subreddits: Optional[list] = None,
    batch_count: int = 5,
    voice: str = "en-US-ChristopherNeural",
    background_video: Optional[str] = None,
    music_style: Optional[str] = "lofi",
    privacy_status: str = "unlisted",
    max_comments: int = 2,
    upload: bool = False,
    dry_run: bool = False,
    use_gpu: bool = True,
    enable_subtitles: bool = True,
    subtitle_style: str = "yellow_pill",
    status_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute end-to-end batch processing pipeline across multiple subreddits."""
    subs = [s.strip().replace("r/", "") for s in (subreddits or ["AskReddit"]) if s.strip()]
    if not subs:
        subs = ["AskReddit"]

    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_folder_name = f"batch_{now_str}"
    batch_dir = Path(OUTPUT_DIR) / batch_folder_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    results = []
    logger.info(f"Starting Batch Pipeline ({batch_count} videos, {max_comments} comments/video) across subreddits: {subs}")

    for i in range(batch_count):
        sub = subs[i % len(subs)]
        item_folder = batch_dir / f"short_{i+1}_{sub}"

        if status_callback:
            await status_callback(
                stage=f"Processing Video {i+1} of {batch_count}",
                message=f"Scraping & rendering r/{sub} ({i+1}/{batch_count})...",
                current=i + 1,
                total=batch_count,
            )

        pipeline = RedditReadingPipeline(
            output_dir=str(item_folder),
            voice=voice,
            background_video=background_video,
            dry_run=dry_run,
            use_gpu=use_gpu,
            music_style=music_style,
            privacy_status=privacy_status,
            max_comments=max_comments,
            enable_subtitles=enable_subtitles,
            subtitle_style=subtitle_style,
        )

        try:
            res = await pipeline.execute(subreddit=sub, upload=upload)
            res["batch_index"] = i + 1
            results.append(res)
        except Exception as e:
            logger.error(f"Batch item {i+1} (r/{sub}) failed: {e}", exc_info=True)
            results.append({
                "status": "error",
                "batch_index": i + 1,
                "subreddit": sub,
                "error": str(e),
            })

    successful_count = sum(1 for r in results if r.get("status") == "success")
    summary_path = batch_dir / "batch_summary.json"
    summary_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_folder": batch_folder_name,
        "total_requested": batch_count,
        "successful_count": successful_count,
        "subreddits": subs,
        "items": results,
    }

    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save batch summary: {e}")

    first_video = ""
    for r in results:
        if r.get("video_path") and Path(r["video_path"]).exists():
            first_video = r["video_path"]
            break

    return {
        "is_batch": True,
        "batch_folder": batch_folder_name,
        "total_requested": batch_count,
        "successful_count": successful_count,
        "video_path": first_video,
        "output_folder": str(batch_dir),
        "results": results,
    }


async def run_pipeline(
    subreddit: str = "AskReddit",
    post_id: Optional[str] = None,
    output_video_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    upload: bool = False,
    voice: str = "en-US-ChristopherNeural",
    background_video: Optional[str] = None,
    dry_run: bool = False,
    use_gpu: bool = True,
    music_style: Optional[str] = "lofi",
    privacy_status: str = "unlisted",
    max_comments: int = 2,
    enable_subtitles: bool = True,
    subtitle_style: str = "yellow_pill",
) -> Dict[str, Any]:
    """Functional wrapper for end-to-end pipeline execution."""
    pipeline = RedditReadingPipeline(
        output_dir=output_dir,
        voice=voice,
        background_video=background_video,
        dry_run=dry_run,
        use_gpu=use_gpu,
        music_style=music_style,
        privacy_status=privacy_status,
        max_comments=max_comments,
        enable_subtitles=enable_subtitles,
        subtitle_style=subtitle_style,
    )
    return await pipeline.execute(
        subreddit=subreddit,
        post_id=post_id,
        output_video_path=output_video_path,
        upload=upload,
    )

