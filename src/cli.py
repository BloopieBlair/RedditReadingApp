"""CLI entrypoint with subcommands for Reddit Reading YouTube Shorts automation.

Subcommands:
  scrape     - Fetch Reddit post + comments and render card PNGs.
  tts        - Generate TTS audio from text or a scraped post JSON file.
  composite  - Composite card images + audio into a 9:16 video.
  upload     - Upload a video file to YouTube.
  pipeline   - Run the full end-to-end automation pipeline.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.config import OUTPUT_DIR, ensure_directories

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="reddit-shorts",
        description="Reddit Reading YouTube Shorts Automation CLI",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── scrape ────────────────────────────────────────────────────────────
    scrape_parser = subparsers.add_parser(
        "scrape", help="Scrape a Reddit post and render card images"
    )
    scrape_parser.add_argument(
        "--subreddit", type=str, default="AskReddit",
        help="Target subreddit name (default: AskReddit)",
    )
    scrape_parser.add_argument(
        "--post-id", type=str, default=None,
        help="Specific Reddit post ID to scrape",
    )
    scrape_parser.add_argument(
        "--limit", type=int, default=10,
        help="Max number of top posts to scan (default: 10)",
    )
    scrape_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save scraped JSON + card images (default: output/)",
    )

    # ── tts ───────────────────────────────────────────────────────────────
    tts_parser = subparsers.add_parser(
        "tts", help="Generate TTS audio from text or a scraped post JSON"
    )
    tts_group = tts_parser.add_mutually_exclusive_group(required=True)
    tts_group.add_argument(
        "--text", type=str, default=None,
        help="Raw text to convert to speech",
    )
    tts_group.add_argument(
        "--post-json", type=str, default=None,
        help="Path to scraped post JSON file for full voiceover generation",
    )
    tts_parser.add_argument(
        "--voice", type=str, default="en-US-ChristopherNeural",
        help="Edge-TTS voice name (default: en-US-ChristopherNeural)",
    )
    tts_parser.add_argument(
        "--output", type=str, default=None,
        help="Output audio file path (for --text mode)",
    )

    # ── composite ─────────────────────────────────────────────────────────
    composite_parser = subparsers.add_parser(
        "composite", help="Composite card images + audio into a 9:16 video"
    )
    composite_parser.add_argument(
        "--post-json", type=str, required=True,
        help="Path to scraped post JSON containing card image paths",
    )
    composite_parser.add_argument(
        "--audio-dir", type=str, required=True,
        help="Directory containing TTS audio files (op.mp3, comment_1.mp3, etc.)",
    )
    composite_parser.add_argument(
        "--background", type=str, default=None,
        help="Path to background video file (Minecraft parkour)",
    )
    composite_parser.add_argument(
        "--output", type=str, default=None,
        help="Output MP4 video file path",
    )
    composite_parser.add_argument(
        "--gpu", action=argparse.BooleanOptionalAction, default=True,
        help="Use GPU hardware acceleration for video encoding (default: enabled)",
    )

    # ── upload ────────────────────────────────────────────────────────────
    upload_parser = subparsers.add_parser(
        "upload", help="Upload a video to YouTube"
    )
    upload_parser.add_argument(
        "--video", type=str, required=True,
        help="Path to the MP4 video file to upload",
    )
    upload_parser.add_argument(
        "--title", type=str, required=True,
        help="YouTube video title",
    )
    upload_parser.add_argument(
        "--description", type=str, default="",
        help="YouTube video description",
    )
    upload_parser.add_argument(
        "--tags", type=str, nargs="*", default=None,
        help="Tags for the YouTube video",
    )
    upload_parser.add_argument(
        "--privacy", type=str, default="unlisted",
        choices=["public", "unlisted", "private"],
        help="Privacy status (default: unlisted)",
    )
    upload_parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate upload without actually uploading",
    )

    # ── pipeline (full end-to-end) ────────────────────────────────────────
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run the full end-to-end pipeline (scrape → TTS → composite → upload)"
    )
    pipeline_parser.add_argument(
        "--subreddit", type=str, default="AskReddit",
        help="Target subreddit name (default: AskReddit)",
    )
    pipeline_parser.add_argument(
        "--post-id", type=str, default=None,
        help="Specific Reddit post ID",
    )
    pipeline_parser.add_argument(
        "--voice", type=str, default="en-US-ChristopherNeural",
        help="Edge-TTS voice name (default: en-US-ChristopherNeural)",
    )
    pipeline_parser.add_argument(
        "--background", type=str, default=None,
        help="Path to background video file (Minecraft parkour)",
    )
    pipeline_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for generated files",
    )
    pipeline_parser.add_argument(
        "--output-path", type=str, default=None,
        help="Specific output video file path",
    )
    pipeline_parser.add_argument(
        "--upload", action="store_true",
        help="Upload the generated video to YouTube after compositing",
    )
    pipeline_parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate upload without actually uploading to YouTube",
    )
    # ── web (Web UI server) ───────────────────────────────────────────────
    web_parser = subparsers.add_parser(
        "web", help="Start the Reddit Reading Web UI server (http://localhost:8000)"
    )
    web_parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host address for web server (default: 127.0.0.1)",
    )
    web_parser.add_argument(
        "--port", type=int, default=8000,
        help="Port number for web server (default: 8000)",
    )

    return parser


def cmd_web(args: argparse.Namespace) -> None:
    """Handle the 'web' subcommand."""
    from src.web.app import start_server
    start_server(host=args.host, port=args.port)


def _configure_logging(verbose: bool = False) -> None:
    """Set up logging with appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def cmd_scrape(args: argparse.Namespace) -> None:
    """Handle the 'scrape' subcommand."""
    from src.scraper import fetch_reddit_post

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scraping r/{args.subreddit}...")
    scraped = await fetch_reddit_post(
        subreddit=args.subreddit, post_id=args.post_id
    )

    # Save scraped content as JSON
    json_path = out_dir / f"scraped_{scraped.post.post_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scraped.to_dict(), f, indent=2, default=str)

    print(f"[OK] Scraped post: {scraped.post.title[:80]}")
    print(f"  Comments: {len(scraped.comments)}")
    print(f"  OP card:  {scraped.op_card_image_path}")
    for i, path in enumerate(scraped.comment_card_image_paths, 1):
        print(f"  Comment {i} card: {path}")
    print(f"  Saved JSON: {json_path}")


async def cmd_tts(args: argparse.Namespace) -> None:
    """Handle the 'tts' subcommand."""
    from src.tts import TTSVoiceEngine, generate_voiceover
    from src.models import ScrapedContent

    if args.text:
        # Single text mode
        engine = TTSVoiceEngine(voice=args.voice)
        out_path = args.output or str(OUTPUT_DIR / "tts_output.mp3")
        clip = await engine.generate_audio(text=args.text, output_path=out_path)
        print(f"[OK] Generated TTS audio: {clip.file_path}")
        print(f"  Duration: {clip.duration_seconds:.2f}s")
    else:
        # Post JSON mode — generate voiceover for full post
        post_json_path = Path(args.post_json)
        if not post_json_path.exists():
            print(f"Error: Post JSON file not found: {args.post_json}", file=sys.stderr)
            sys.exit(1)

        with open(post_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scraped = ScrapedContent.from_dict(data)
        out_dir = str(post_json_path.parent)
        clips = await generate_voiceover(
            scraped_content=scraped, output_dir=out_dir, voice=args.voice
        )

        print(f"[OK] Generated {len(clips)} TTS audio clips:")
        for key, clip in clips.items():
            print(f"  {key}: {clip.file_path} ({clip.duration_seconds:.2f}s)")


async def cmd_composite(args: argparse.Namespace) -> None:
    """Handle the 'composite' subcommand."""
    from src.models import ScrapedContent, AudioClip
    from src.video import compose_shorts_video
    from src.tts.voice_engine import get_audio_duration

    # Load scraped content
    post_json_path = Path(args.post_json)
    if not post_json_path.exists():
        print(f"Error: Post JSON file not found: {args.post_json}", file=sys.stderr)
        sys.exit(1)

    with open(post_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scraped = ScrapedContent.from_dict(data)

    # Load audio clips from directory
    audio_dir = Path(args.audio_dir)
    audio_clips = {}
    for key, filename in [("op", "op.mp3"), ("comment_1", "comment_1.mp3"), ("comment_2", "comment_2.mp3")]:
        audio_path = audio_dir / filename
        if audio_path.exists():
            duration = get_audio_duration(str(audio_path))
            audio_clips[key] = AudioClip(
                clip_id=key,
                file_path=str(audio_path),
                duration_seconds=duration,
            )

    if not audio_clips:
        print("Error: No audio clips found in the specified directory", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(OUTPUT_DIR / "composite_output.mp4")
    print(f"Compositing video with {len(audio_clips)} audio clips...")
    result = compose_shorts_video(
        scraped_content=scraped,
        audio_clips=audio_clips,
        output_path=output_path,
        background_video_path=args.background,
        use_gpu=getattr(args, "gpu", True),
    )
    print(f"[OK] Video composited: {result}")


def cmd_upload(args: argparse.Namespace) -> None:
    """Handle the 'upload' subcommand."""
    from src.uploader import upload_short

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    print(f"Uploading {args.video}...")
    result = upload_short(
        video_path=args.video,
        title=args.title,
        description=args.description,
        tags=args.tags,
        privacy_status=args.privacy,
        dry_run=args.dry_run,
    )
    print(f"[OK] Upload result: {result.get('status')}")
    if result.get("video_url"):
        print(f"  Video URL: {result['video_url']}")
    if result.get("dry_run"):
        print("  (dry run — no actual upload performed)")


async def cmd_pipeline(args: argparse.Namespace) -> None:
    """Handle the 'pipeline' subcommand (full end-to-end)."""
    from src.pipeline import run_pipeline

    print(f"Running full pipeline for r/{args.subreddit}...")
    result = await run_pipeline(
        subreddit=args.subreddit,
        post_id=args.post_id,
        output_video_path=args.output_path,
        output_dir=args.output_dir,
        upload=args.upload,
        voice=args.voice,
        background_video=args.background,
        dry_run=args.dry_run,
        use_gpu=getattr(args, "gpu", True),
    )

    print(f"\n[OK] Pipeline completed!")
    print(f"  Status:     {result.get('status')}")
    print(f"  Post:       {result.get('post_title', 'N/A')[:80]}")
    print(f"  Video:      {result.get('video_path')}")
    if result.get("upload_result"):
        ur = result["upload_result"]
        print(f"  Upload:     {ur.get('status')} — {ur.get('video_url', 'N/A')}")
        if ur.get("dry_run"):
            print("  (dry run — no actual upload performed)")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI execution entrypoint."""
    ensure_directories()
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(verbose=getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "scrape":
            asyncio.run(cmd_scrape(args))
        elif args.command == "tts":
            asyncio.run(cmd_tts(args))
        elif args.command == "composite":
            asyncio.run(cmd_composite(args))
        elif args.command == "upload":
            cmd_upload(args)
        elif args.command == "pipeline":
            asyncio.run(cmd_pipeline(args))
        elif args.command == "web":
            cmd_web(args)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Command '{args.command}' failed: {e}", exc_info=True)
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
