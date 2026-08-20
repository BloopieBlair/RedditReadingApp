"""Reddit scraper implementation featuring Playwright dark-mode scraping and public JSON API fallback."""

import asyncio
import logging
from typing import Optional, List, Dict, Any
import requests

import html
from src.models import RedditPost, RedditComment, ScrapedContent
from src.scraper.html_renderer import HTMLCardRenderer
from src.config import TEMP_DIR, ensure_directories
from src.history import PostHistoryManager

logger = logging.getLogger(__name__)

USER_AGENT = "RedditReadingApp/1.0"
HEADERS = {"User-Agent": USER_AGENT}


def extract_image_url(post_data: Dict[str, Any]) -> Optional[str]:
    """Extract image URL from Reddit post data if present."""
    if not isinstance(post_data, dict):
        return None

    # Direct image URL in link posts
    url = post_data.get("url_overridden_by_dest") or post_data.get("url")
    if url and isinstance(url, str):
        url_lower = url.lower()
        if any(url_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
            return url
        if "i.redd.it" in url_lower or "i.imgur.com" in url_lower:
            return url

    # Preview image
    try:
        preview_images = post_data.get("preview", {}).get("images", [])
        if preview_images and isinstance(preview_images, list):
            src_url = preview_images[0].get("source", {}).get("url")
            if src_url and isinstance(src_url, str):
                return html.unescape(src_url)
    except Exception:
        pass

    return None


def is_bot_or_deleted_author(author: Optional[str]) -> bool:
    """Check if author string indicates a bot, automoderator, or deleted user."""
    if not author:
        return True
    a_lower = str(author).strip().lower()
    if a_lower in ("[deleted]", "[removed]", "automoderator"):
        return True
    if "bot" in a_lower or a_lower.endswith("bot"):
        return True
    return False


def is_valid_post_data(data: Dict[str, Any]) -> bool:
    """Validate post dictionary for NSFW, stickied, deleted, or bot attributes."""
    if not data or not isinstance(data, dict):
        return False
    if data.get("over_18", False):
        return False
    if data.get("stickied", False) or data.get("pinned", False):
        return False
    title = data.get("title")
    if not title or str(title).strip() in ("[deleted]", "[removed]"):
        return False
    author = data.get("author")
    if is_bot_or_deleted_author(author):
        return False
    return True


def is_valid_comment_data(data: Dict[str, Any]) -> bool:
    """Validate comment dictionary for stickied, deleted, or bot attributes."""
    if not data or not isinstance(data, dict):
        return False
    if data.get("stickied", False):
        return False
    body = data.get("body")
    if not body or str(body).strip() in ("[deleted]", "[removed]"):
        return False
    author = data.get("author")
    if is_bot_or_deleted_author(author):
        return False
    return True


class RedditScraper:
    """Dual-path scraper with Playwright web browser primary and public JSON API fallback."""

    def __init__(self, use_headless: bool = True, force_json_api: bool = False):
        self.use_headless = use_headless
        self.force_json_api = force_json_api
        self.renderer = HTMLCardRenderer()
        self.history = PostHistoryManager()
        ensure_directories()

    async def scrape(
        self, subreddit: str = "AskReddit", post_id: Optional[str] = None, max_comments: int = 2
    ) -> ScrapedContent:
        """Main scraping method attempting Playwright first, falling back to JSON API."""
        clean_sub = subreddit.replace("r/", "").strip()
        if not self.force_json_api:
            try:
                scraped = await self._scrape_playwright(clean_sub, post_id, max_comments=max_comments)
                if scraped and scraped.post and len(scraped.comments) >= 1:
                    return scraped
            except Exception as e:
                logger.warning(f"Playwright scrape failed: {e}. Falling back to Reddit JSON API.")

        return await self._scrape_json_api(clean_sub, post_id, max_comments=max_comments)

    async def _scrape_playwright(
        self, subreddit: str, post_id: Optional[str] = None, max_comments: int = 2
    ) -> ScrapedContent:
        """Attempt scraping live Reddit post and comments using Playwright browser page context."""
        from playwright.async_api import async_playwright

        clean_sub = subreddit.replace("r/", "").strip()
        base_url = f"https://www.reddit.com/r/{clean_sub}/"

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.use_headless)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                color_scheme="dark",
            )
            page = await context.new_page()

            resp = await page.goto(base_url, wait_until="domcontentloaded", timeout=10000)
            await page.wait_for_timeout(1500)
            if not resp or resp.status >= 400:
                await browser.close()
                raise RuntimeError(f"Playwright received HTTP status {resp.status if resp else 'None'} for {base_url}")

            seen_ids = list(self.history._seen_ids)

            # Execute fetch in page context to obtain JSON data without Cloudflare 403 block
            js_script = """async (args) => {
                const targetPostId = args.targetPostId;
                const seenIds = args.seenIds || [];
                let pId = targetPostId;
                let postData = null;
                const cleanSub = '""" + clean_sub + """';

                if (!pId) {
                    const topResp = await fetch('/r/' + cleanSub + '/top.json?limit=25&t=day');
                    if (topResp.status === 200) {
                        const topJson = await topResp.json();
                        const children = topJson?.data?.children || [];
                        for (const child of children) {
                            const p = child.data;
                            if (p && !p.over_18 && !p.stickied && p.title && p.title !== '[deleted]' && p.title !== '[removed]') {
                                if (seenIds.includes(p.id)) continue;
                                pId = p.id;
                                postData = p;
                                break;
                            }
                        }

                        // Fallback if all posts were seen
                        if (!pId && children.length > 0) {
                            pId = children[0]?.data?.id;
                            postData = children[0]?.data;
                        }
                    }
                }

                if (!pId) return null;

                const commUrl = '/r/' + cleanSub + '/comments/' + pId + '.json';
                const commResp = await fetch(commUrl);
                if (commResp.status !== 200) return null;
                const commJson = await commResp.json();

                if (!postData && commJson?.[0]?.data?.children?.[0]?.data) {
                    postData = commJson[0].data.children[0].data;
                }

                const rawComments = commJson?.[1]?.data?.children || [];
                return { postData, rawComments };
            }"""

            eval_result = await page.evaluate(js_script, {"targetPostId": post_id, "seenIds": seen_ids})
            await browser.close()

            if not eval_result or not eval_result.get("postData"):
                raise RuntimeError(f"Playwright fetch returned no valid post data for r/{clean_sub}")

            post_data = eval_result["postData"]
            raw_comments = eval_result.get("rawComments", [])

            if not is_valid_post_data(post_data):
                raise RuntimeError(f"Post {post_data.get('id')} failed post validation rules")

            return await self._build_scraped_content(clean_sub, post_data, raw_comments, max_comments=max_comments)

    async def _scrape_json_api(
        self, subreddit: str, post_id: Optional[str] = None, max_comments: int = 2
    ) -> ScrapedContent:
        """Fetch post and comments using standard requests to public Reddit JSON API."""
        clean_sub = subreddit.replace("r/", "").strip()
        loop = asyncio.get_event_loop()

        def fetch_json(url_str: str) -> Any:
            resp = requests.get(url_str, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            return resp.json()

        target_post_id = post_id
        target_post_data = None
        raw_comments = []

        if target_post_id:
            comments_url = f"https://www.reddit.com/r/{clean_sub}/comments/{target_post_id}.json"
            json_data = await loop.run_in_executor(None, fetch_json, comments_url)
            if isinstance(json_data, list) and len(json_data) > 0:
                post_children = json_data[0].get("data", {}).get("children", [])
                if post_children:
                    target_post_data = post_children[0].get("data")
                if len(json_data) > 1:
                    raw_comments = json_data[1].get("data", {}).get("children", [])
        else:
            top_url = f"https://www.reddit.com/r/{clean_sub}/top.json?limit=25&t=day"
            top_json = await loop.run_in_executor(None, fetch_json, top_url)
            children = top_json.get("data", {}).get("children", [])

            for child in children:
                p_data = child.get("data", {})
                if is_valid_post_data(p_data):
                    cand_id = p_data.get("id")
                    if not post_id and self.history.is_seen(cand_id):
                        continue
                    comments_url = f"https://www.reddit.com/r/{clean_sub}/comments/{cand_id}.json"
                    try:
                        comments_json = await loop.run_in_executor(None, fetch_json, comments_url)
                        if isinstance(comments_json, list) and len(comments_json) > 1:
                            cand_raw = comments_json[1].get("data", {}).get("children", [])
                            valid_c = [
                                c.get("data") for c in cand_raw
                                if c.get("kind") == "t1" and is_valid_comment_data(c.get("data", {}))
                            ]
                            if len(valid_c) >= 1:
                                target_post_id = cand_id
                                target_post_data = p_data
                                raw_comments = cand_raw
                                break
                    except Exception:
                        continue

            # Fallback if all top posts were already seen: pick first valid post
            if not target_post_data and children:
                for child in children:
                    p_data = child.get("data", {})
                    if is_valid_post_data(p_data):
                        target_post_id = p_data.get("id")
                        target_post_data = p_data
                        break

        if not target_post_data or not is_valid_post_data(target_post_data):
            raise RuntimeError(f"No valid Reddit post found for r/{clean_sub}")

        if target_post_id and not raw_comments:
            comments_url = f"https://www.reddit.com/r/{clean_sub}/comments/{target_post_id}.json"
            try:
                comments_json = await loop.run_in_executor(None, fetch_json, comments_url)
                if isinstance(comments_json, list) and len(comments_json) > 1:
                    raw_comments = comments_json[1].get("data", {}).get("children", [])
            except Exception:
                pass

        return await self._build_scraped_content(clean_sub, target_post_data, raw_comments, max_comments=max_comments)

    async def _build_scraped_content(
        self, subreddit: str, post_data: Dict[str, Any], raw_comments: List[Dict[str, Any]], max_comments: int = 2
    ) -> ScrapedContent:
        """Helper to construct RedditPost, top RedditComments, and render card screenshot PNGs."""
        img_url = extract_image_url(post_data)
        post_obj = RedditPost(
            post_id=str(post_data.get("id", "sample_post")),
            title=str(post_data.get("title", "")),
            author=str(post_data.get("author", "anonymous")),
            subreddit=subreddit,
            body=str(post_data.get("selftext", "")),
            ups=int(post_data.get("ups", 0)),
            created_utc=float(post_data.get("created_utc", 0.0)),
            num_comments=int(post_data.get("num_comments", 0)),
            url=str(post_data.get("url", "")),
            image_url=img_url,
        )

        self.history.mark_seen(post_obj.post_id)

        candidate_comments: List[RedditComment] = []
        for c_item in raw_comments:
            if c_item.get("kind") != "t1":
                continue
            c_data = c_item.get("data", {})
            if not is_valid_comment_data(c_data):
                continue

            comment_obj = RedditComment(
                comment_id=str(c_data.get("id")),
                author=str(c_data.get("author", "anonymous")),
                body=str(c_data.get("body", "")),
                ups=int(c_data.get("ups", 0)),
                created_utc=float(c_data.get("created_utc", 0.0)),
                parent_id=c_data.get("parent_id"),
            )
            candidate_comments.append(comment_obj)

        # Sort candidate top-level comments by upvotes descending
        candidate_comments.sort(key=lambda c: c.ups, reverse=True)
        selected_comments = candidate_comments[:max_comments]

        # Render screenshot cards using async_render
        op_card_path = str(TEMP_DIR / f"op_card_{post_obj.post_id}.png")
        await self.renderer.async_render(post_obj, op_card_path, subreddit=subreddit)

        comment_card_paths: List[str] = []
        for idx, comment in enumerate(selected_comments, start=1):
            c_card_path = str(TEMP_DIR / f"comment_{idx}_{post_obj.post_id}.png")
            await self.renderer.async_render(comment, c_card_path, subreddit=subreddit)
            comment_card_paths.append(c_card_path)

        return ScrapedContent(
            post=post_obj,
            comments=selected_comments,
            op_card_image_path=op_card_path,
            comment_card_image_paths=comment_card_paths,
        )


async def fetch_reddit_post(
    subreddit: str = "AskReddit", post_id: Optional[str] = None, max_comments: int = 2
) -> ScrapedContent:
    """Functional interface for scraping Reddit posts."""
    scraper = RedditScraper()
    return await scraper.scrape(subreddit=subreddit, post_id=post_id, max_comments=max_comments)
