"""AI OP Post Watcher and End-to-End Autonomous Workflow Orchestrator."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
import requests

from src.config import AI_OP_DEFAULT_MIN_COMMENTS, AI_OP_DEFAULT_POLL_INTERVAL
from src.models import AIPostRecord
from src.pipeline import RedditReadingPipeline
from src.poster.ai_poster import AIOPGenerator
from src.poster.reddit_client import RedditPosterClient
from src.poster.tracker import AIPostTracker
from src.scraper.reddit_scraper import is_bot_or_deleted_author, is_valid_comment_data

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
HEADERS = {"User-Agent": USER_AGENT}


class AIPostWatcher:
    """Monitors active AI OP Reddit posts, tracks comments, and auto-triggers video rendering."""

    def __init__(
        self,
        tracker: Optional[AIPostTracker] = None,
        generator: Optional[AIOPGenerator] = None,
        poster_client: Optional[RedditPosterClient] = None,
    ):
        self.tracker = tracker or AIPostTracker()
        self.generator = generator or AIOPGenerator()
        self.poster_client = poster_client or RedditPosterClient()

    async def check_post_comments(self, post_id: str) -> Dict[str, Any]:
        """
        Check live Reddit comments for a tracked AI OP post.
        Filters out bots and stickied AutoModerator comments.
        """
        clean_id = str(post_id).replace("t3_", "").strip()
        record = self.tracker.get_post(clean_id)
        if not record:
            return {"status": "error", "error": f"Post [{clean_id}] not found in tracker"}

        # If it's a simulated post, check if it has simulated comments or generate mock comment count
        if record.is_simulated or clean_id.startswith("sim_"):
            now_iso = datetime.now(timezone.utc).isoformat()
            # For simulated posts, simulate meatbag comments arriving
            sim_count = max(record.current_comments_count + 1, record.min_comments_target)
            previews = [
                "Human Meatbag #1: This is the most unhinged thing I've read all week.",
                "Human Meatbag #2: Literally happened to my cousin last Tuesday.",
            ]
            self.tracker.update_comments(
                post_id=clean_id,
                comments_count=sim_count,
                previews=previews,
                last_checked_at=now_iso,
            )
            return {
                "status": "success",
                "post_id": clean_id,
                "subreddit": record.subreddit,
                "current_comments_count": sim_count,
                "min_comments_target": record.min_comments_target,
                "is_ready": sim_count >= record.min_comments_target,
                "previews": previews,
                "last_checked_at": now_iso,
            }

        # Live Reddit check via public JSON endpoint
        sub = record.subreddit.replace("r/", "").strip()
        url = f"https://www.reddit.com/r/{sub}/comments/{clean_id}.json"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=10.0)
            if resp.status_code != 200:
                logger.warning(f"Reddit JSON fetch returned status {resp.status_code} for post {clean_id}")
                return {
                    "status": "error",
                    "post_id": clean_id,
                    "error": f"HTTP status {resp.status_code} from Reddit",
                }

            data = resp.json()
            if not isinstance(data, list) or len(data) < 2:
                return {
                    "status": "error",
                    "post_id": clean_id,
                    "error": "Unexpected JSON response structure from Reddit",
                }

            comments_listing = data[1].get("data", {}).get("children", [])
            valid_comments = []
            previews = []

            for child in comments_listing:
                c_data = child.get("data", {})
                if not is_valid_comment_data(c_data):
                    continue

                author = c_data.get("author", "anonymous")
                body = c_data.get("body", "").strip()
                ups = c_data.get("ups", 0)

                if body and not is_bot_or_deleted_author(author):
                    valid_comments.append(c_data)
                    preview_text = f"u/{author} ({ups} pts): {body[:80]}..." if len(body) > 80 else f"u/{author} ({ups} pts): {body}"
                    previews.append(preview_text)

            valid_count = len(valid_comments)
            now_iso = datetime.now(timezone.utc).isoformat()

            updated_record = self.tracker.update_comments(
                post_id=clean_id,
                comments_count=valid_count,
                previews=previews[:5],
                last_checked_at=now_iso,
            )

            is_ready = valid_count >= record.min_comments_target
            logger.info(
                f"Post [{clean_id}] in r/{sub}: {valid_count}/{record.min_comments_target} "
                f"quality meatbag comments found (ready={is_ready})."
            )

            return {
                "status": "success",
                "post_id": clean_id,
                "subreddit": sub,
                "current_comments_count": valid_count,
                "min_comments_target": record.min_comments_target,
                "is_ready": is_ready,
                "previews": previews[:5],
                "last_checked_at": now_iso,
            }

        except Exception as e:
            logger.error(f"Failed to check comments for post {clean_id}: {e}")
            return {
                "status": "error",
                "post_id": clean_id,
                "error": str(e),
            }

    async def render_post_video(
        self,
        post_id: str,
        voice: str = "en-US-ChristopherNeural",
        background_video: Optional[str] = None,
        music_style: Optional[str] = "lofi",
        privacy_status: str = "unlisted",
        upload: bool = False,
        dry_run: bool = False,
        use_gpu: bool = True,
        enable_subtitles: bool = True,
        subtitle_style: str = "yellow_pill",
    ) -> Dict[str, Any]:
        """
        Execute video rendering pipeline for a specific AI OP post.
        """
        clean_id = str(post_id).replace("t3_", "").strip()
        record = self.tracker.get_post(clean_id)
        if not record:
            raise ValueError(f"Post [{clean_id}] not found in AI OP tracker")

        logger.info(f"Rendering AI OP video for post [{clean_id}] in r/{record.subreddit}...")

        pipeline = RedditReadingPipeline(
            voice=voice,
            background_video=background_video,
            dry_run=dry_run,
            use_gpu=use_gpu,
            music_style=music_style,
            privacy_status=privacy_status,
            max_comments=record.min_comments_target,
            enable_subtitles=enable_subtitles,
            subtitle_style=subtitle_style,
        )

        result = await pipeline.execute(
            subreddit=record.subreddit,
            post_id=None if record.is_simulated else clean_id,
            upload=upload,
        )

        video_path = result.get("video_path", "")
        output_folder = result.get("output_folder", "")

        # Mark rendered in tracker
        self.tracker.mark_rendered(
            post_id=clean_id,
            video_path=video_path,
            output_folder=output_folder,
        )

        return result

    async def run_ai_op_workflow(
        self,
        subreddit: str = "AskReddit",
        theme: Optional[str] = None,
        style: str = "comedic",
        min_comments: int = AI_OP_DEFAULT_MIN_COMMENTS,
        poll_interval: int = AI_OP_DEFAULT_POLL_INTERVAL,
        max_wait_seconds: int = 3600,
        voice: str = "en-US-ChristopherNeural",
        background_video: Optional[str] = None,
        music_style: Optional[str] = "lofi",
        privacy_status: str = "unlisted",
        upload: bool = False,
        dry_run: bool = False,
        use_gpu: bool = True,
        enable_subtitles: bool = True,
        subtitle_style: str = "yellow_pill",
        status_callback: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        """
        Complete end-to-end autonomous AI OP workflow:
        1. AI ideates a funny post for the subreddit.
        2. Submits the post to Reddit via dedicated bot account.
        3. Tracks the post in AI OP manifest.
        4. Waits / polls until human comments arrive.
        5. Automatically executes video rendering and output saving.
        """
        clean_sub = subreddit.replace("r/", "").strip()

        # ── Stage 1: AI Post Ideation ──
        if status_callback:
            await status_callback(
                stage="Generating AI Post",
                message=f"AI is crafting a hilarious post idea for r/{clean_sub}...",
                progress=10,
            )

        post_idea = self.generator.generate_post(
            subreddit=clean_sub,
            theme=theme,
            style=style,
        )
        logger.info(f"AI OP Generated Post: '{post_idea['title']}'")

        # ── Stage 2: Submit to Reddit ──
        if status_callback:
            await status_callback(
                stage="Submitting Post to Reddit",
                message=f"Posting to r/{clean_sub} from AI bot account...",
                progress=25,
            )

        submit_result = self.poster_client.submit_post(
            subreddit=clean_sub,
            title=post_idea["title"],
            body=post_idea.get("body", ""),
            dry_run=dry_run,
        )

        post_id = submit_result["post_id"]
        post_url = submit_result["url"]

        # ── Stage 3: Register in Tracker ──
        record = AIPostRecord(
            post_id=post_id,
            subreddit=clean_sub,
            title=post_idea["title"],
            body=post_idea.get("body", ""),
            url=post_url,
            author=submit_result.get("author", "AI_OP_Bot"),
            created_utc=time.time(),
            status="waiting_for_comments",
            min_comments_target=min_comments,
            current_comments_count=0,
            is_simulated=submit_result.get("is_simulated", False),
        )
        self.tracker.add_post(record)

        # ── Stage 4: Wait / Poll for Meatbag Comments ──
        start_time = time.time()
        is_ready = False

        while time.time() - start_time < max_wait_seconds:
            elapsed = int(time.time() - start_time)
            if status_callback:
                await status_callback(
                    stage="Waiting for Meatbags",
                    message=f"Waiting for comments on Reddit post [{post_id}] ({elapsed}s elapsed, target: {min_comments})...",
                    progress=35 + min(30, int(elapsed / max_wait_seconds * 30)),
                )

            check_res = await self.check_post_comments(post_id)
            current_count = check_res.get("current_comments_count", 0)

            if current_count >= min_comments:
                is_ready = True
                logger.info(f"AI OP post [{post_id}] reached target ({current_count}/{min_comments} comments). Ready to render!")
                break

            # If simulated, we don't need to loop forever in test runs
            if submit_result.get("is_simulated", False):
                is_ready = True
                break

            await asyncio.sleep(poll_interval)

        if not is_ready:
            msg = f"Timed out waiting for comments on post [{post_id}] after {max_wait_seconds}s"
            logger.warning(msg)
            self.tracker.update_post(post_id, status="timed_out", error=msg)
            return {
                "status": "timed_out",
                "post_id": post_id,
                "url": post_url,
                "title": post_idea["title"],
                "error": msg,
            }

        # ── Stage 5: Render Video Pipeline ──
        if status_callback:
            await status_callback(
                stage="Rendering Shorts Video",
                message=f"Human comments received! Compositing 9:16 Shorts video...",
                progress=75,
            )

        render_result = await self.render_post_video(
            post_id=post_id,
            voice=voice,
            background_video=background_video,
            music_style=music_style,
            privacy_status=privacy_status,
            upload=upload,
            dry_run=dry_run,
            use_gpu=use_gpu,
            enable_subtitles=enable_subtitles,
            subtitle_style=subtitle_style,
        )

        if status_callback:
            await status_callback(
                stage="Completed!",
                message="AI OP workflow completed successfully!",
                progress=100,
            )

        return {
            "status": "success",
            "post_id": post_id,
            "url": post_url,
            "title": post_idea["title"],
            "subreddit": clean_sub,
            "comments_count": record.current_comments_count,
            "video_path": render_result.get("video_path"),
            "output_folder": render_result.get("output_folder"),
            "manifest": render_result,
        }
