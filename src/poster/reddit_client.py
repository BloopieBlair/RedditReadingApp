"""Reddit Client supporting Sock Puppet user accounts via Playwright browser automation and OAuth2 API."""

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import requests

from src.config import REDDIT_SESSION_FILE, get_reddit_credentials, ensure_directories

logger = logging.getLogger(__name__)

REDDIT_OAUTH_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_BASE_URL = "https://oauth.reddit.com"
BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"


class RedditAPIError(Exception):
    """Custom exception for Reddit API or browser submission errors."""
    pass


class RedditPosterClient:
    """
    Dual-mode Reddit poster:
    1. Browser Automation Mode (Primary): Uses Playwright with a persistent logged-in Sock Puppet
       user session (undetectable by anti-bot filters, works on all subreddits).
    2. OAuth API Mode (Fallback): Uses standard Reddit script application credentials.
    3. Dry-Run / Simulation Mode: Creates simulated posts for offline pipeline testing.
    """

    def __init__(
        self,
        session_file: Optional[Path] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        user_agent: Optional[str] = None,
        dry_run: bool = False,
    ):
        creds = get_reddit_credentials()
        self.session_file = Path(session_file) if session_file is not None else REDDIT_SESSION_FILE
        self.client_id = (client_id if client_id is not None else creds.get("client_id", "")).strip()
        self.client_secret = (client_secret if client_secret is not None else creds.get("client_secret", "")).strip()
        self.username = (username if username is not None else creds.get("username", "")).strip()
        self.password = (password if password is not None else creds.get("password", "")).strip()
        self.user_agent = (user_agent if user_agent is not None else creds.get("user_agent", "python:ai-op-shorts-generator:v1.0")).strip()
        self.dry_run = dry_run

        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    @property
    def has_session(self) -> bool:
        """Check if a saved Playwright sock puppet browser session exists."""
        if not self.session_file.exists():
            return False
        try:
            with open(self.session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return isinstance(data, dict) and bool(data.get("cookies") or data.get("origins"))
        except Exception:
            return False

    @property
    def has_credentials(self) -> bool:
        """Check if all required Reddit API credentials are provided."""
        return bool(self.client_id and self.client_secret and self.username and self.password)

    def get_session_status(self) -> Dict[str, Any]:
        """Inspect saved session file and return connection details."""
        if not self.has_session:
            return {
                "has_session": False,
                "username": self.username or "Not configured",
                "message": "No sock puppet session found. Click 'Log in to Reddit' to connect.",
            }

        try:
            with open(self.session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                cookies = data.get("cookies", [])
                session_cookie = next((c for c in cookies if c.get("name") in ("reddit_session", "token_v2")), None)
                username_hint = self.username or "SockPuppet"

                return {
                    "has_session": True,
                    "username": username_hint,
                    "cookies_count": len(cookies),
                    "has_session_cookie": session_cookie is not None,
                    "session_file": str(self.session_file),
                    "message": f"Sock puppet account active (Session file saved).",
                }
        except Exception as e:
            return {
                "has_session": False,
                "error": str(e),
                "message": f"Error reading session file: {e}",
            }

    def login_browser_interactive(self, timeout_seconds: int = 120) -> Dict[str, Any]:
        """
        Launches a visible Playwright browser window pointing to reddit.com/login.
        Allows the user to log into their sock puppet account manually.
        Saves storage_state (cookies + localStorage) upon successful login.
        """
        from playwright.sync_api import sync_playwright

        ensure_directories()
        self.session_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Opening browser window for Reddit sock puppet login...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                user_agent=BROWSER_USER_AGENT,
                color_scheme="dark",
            )
            page = context.new_page()

            page.goto("https://www.reddit.com/login", wait_until="domcontentloaded")

            start_t = time.time()
            logged_in = False

            while time.time() - start_t < timeout_seconds:
                time.sleep(1.5)
                # Check cookies for reddit_session or token
                cookies = context.cookies()
                has_auth = any(c["name"] in ("reddit_session", "token_v2", "loid") and c.get("value") for c in cookies)
                
                # Check if navigated away from /login
                curr_url = page.url.lower()
                if has_auth and ("/login" not in curr_url or curr_url == "https://www.reddit.com/"):
                    logged_in = True
                    break

            if logged_in:
                # Save session state
                context.storage_state(path=str(self.session_file))
                browser.close()
                logger.info(f"Successfully saved Reddit sock puppet session to {self.session_file}")
                return {
                    "status": "success",
                    "message": "Sock puppet login successful! Session saved.",
                    "session_file": str(self.session_file),
                }
            else:
                browser.close()
                raise RedditAPIError("Login timed out or was not completed in the browser.")

    def submit_post_browser(
        self,
        subreddit: str,
        title: str,
        body: str = "",
        headless: bool = True,
    ) -> Dict[str, Any]:
        """
        Submit a post to Reddit as the logged-in sock puppet user using Playwright browser.
        Handles both title-only posts (e.g. AskReddit, Showerthoughts) and text body posts.
        """
        from playwright.sync_api import sync_playwright

        clean_sub = subreddit.replace("r/", "").strip()
        submit_url = f"https://www.reddit.com/r/{clean_sub}/submit"

        logger.info(f"Launching Playwright browser to submit post to r/{clean_sub}...")

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            )

            # Load storage state if available
            ctx_kwargs = {
                "user_agent": BROWSER_USER_AGENT,
                "color_scheme": "dark",
                "viewport": {"width": 1280, "height": 800},
            }
            if self.has_session:
                ctx_kwargs["storage_state"] = str(self.session_file)

            context = browser.new_context(**ctx_kwargs)
            page = context.new_page()

            try:
                resp = page.goto(submit_url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(2000)
            except Exception as e:
                browser.close()
                raise RedditAPIError(f"Failed to load Reddit submit page for r/{clean_sub}: {e}")

            # Verify we are not redirected to login page
            if "/login" in page.url.lower():
                browser.close()
                raise RedditAPIError(
                    f"Not logged in to Reddit. Please connect your sock puppet account via 'ai-op login' first."
                )

            # 1. Fill Title
            title_filled = False
            title_selectors = [
                'textarea[placeholder*="Title"]',
                'textarea[name="title"]',
                'input[name="title"]',
                '[data-testid="post-title-input"]',
                'faceplate-textarea-input[name="title"] textarea',
                'faceplate-textarea-input[name="title"]',
            ]

            for sel in title_selectors:
                try:
                    if page.locator(sel).first.is_visible():
                        page.locator(sel).first.click()
                        page.locator(sel).first.fill(title)
                        title_filled = True
                        break
                except Exception:
                    continue

            if not title_filled:
                # Fallback: find any visible textarea on page
                try:
                    first_textarea = page.locator("textarea").first
                    if first_textarea.is_visible():
                        first_textarea.fill(title)
                        title_filled = True
                except Exception:
                    pass

            if not title_filled:
                browser.close()
                raise RedditAPIError(f"Could not find title input field on r/{clean_sub}/submit page.")

            # 2. Fill Body if present and not empty
            if body and body.strip():
                page.wait_for_timeout(500)
                body_selectors = [
                    'div[contenteditable="true"]',
                    'textarea[placeholder*="Text"]',
                    'textarea[name="body"]',
                    '[data-testid="post-body-input"]',
                    'faceplate-textarea-input[name="body"] textarea',
                ]
                for sel in body_selectors:
                    try:
                        if page.locator(sel).first.is_visible():
                            page.locator(sel).first.click()
                            page.locator(sel).first.fill(body)
                            break
                    except Exception:
                        continue

            page.wait_for_timeout(1000)

            # 3. Click Post / Submit Button
            post_btn_selectors = [
                'button:has-text("Post")',
                'button[type="submit"]:has-text("Post")',
                '[data-testid="post-button"]',
                'button[data-reddit-event-action="post"]',
                'button.button-brand:has-text("Post")',
            ]

            post_clicked = False
            for sel in post_btn_selectors:
                try:
                    btn = page.locator(sel).first
                    if btn.is_visible() and btn.is_enabled():
                        btn.click()
                        post_clicked = True
                        break
                except Exception:
                    continue

            if not post_clicked:
                browser.close()
                raise RedditAPIError(f"Post button not found or disabled on r/{clean_sub}/submit.")

            # 4. Wait for redirection to created post URL (/comments/{id}/)
            post_id = ""
            post_url = ""

            try:
                # Wait up to 15 seconds for URL to transition to post page
                start_w = time.time()
                while time.time() - start_w < 15:
                    page.wait_for_timeout(1000)
                    curr_url = page.url
                    match = re.search(r"/comments/([a-zA-Z0-9_]+)", curr_url)
                    if match:
                        post_id = match.group(1)
                        post_url = curr_url
                        break
            except Exception:
                pass

            # Save updated storage state (cookies might have refreshed)
            try:
                context.storage_state(path=str(self.session_file))
            except Exception:
                pass

            browser.close()

            if not post_id:
                raise RedditAPIError(
                    f"Post submitted to r/{clean_sub}, but timed out waiting for redirect to post page URL."
                )

            logger.info(f"Successfully posted to r/{clean_sub} via browser: [{post_id}] {post_url}")

            return {
                "post_id": post_id,
                "url": post_url,
                "subreddit": clean_sub,
                "title": title,
                "body": body,
                "author": self.username or "SockPuppetAccount",
                "created_utc": time.time(),
                "is_simulated": False,
            }

    # ── OAuth API Fallback ───────────────────────────────────────────────────

    def authenticate(self) -> str:
        """Authenticate with Reddit API using OAuth2 password grant (API fallback)."""
        if not self.has_credentials:
            raise RedditAPIError(
                "Missing Reddit API credentials. Please configure REDDIT_CLIENT_ID, "
                "REDDIT_CLIENT_SECRET, REDDIT_USERNAME, and REDDIT_PASSWORD."
            )

        if self._access_token and time.time() < (self._token_expires_at - 60):
            return self._access_token

        headers = {"User-Agent": self.user_agent}
        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
        }

        try:
            resp = requests.post(
                REDDIT_OAUTH_TOKEN_URL,
                auth=(self.client_id, self.client_secret),
                data=data,
                headers=headers,
                timeout=12.0,
            )
        except Exception as e:
            raise RedditAPIError(f"Network error connecting to Reddit OAuth endpoint: {e}")

        if resp.status_code != 200:
            raise RedditAPIError(f"Reddit OAuth authentication failed (HTTP {resp.status_code}): {resp.text}")

        token_data = resp.json()
        if "error" in token_data:
            error_msg = token_data.get("error_description") or token_data.get("error")
            raise RedditAPIError(f"Reddit OAuth error: {error_msg}")

        access_token = token_data.get("access_token")
        if not access_token:
            raise RedditAPIError(f"Reddit OAuth returned empty access token: {token_data}")

        expires_in = token_data.get("expires_in", 3600)
        self._access_token = access_token
        self._token_expires_at = time.time() + float(expires_in)
        return self._access_token

    def get_me(self) -> Dict[str, Any]:
        """Fetch profile info of authenticated bot user."""
        token = self.authenticate()
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": self.user_agent,
        }

        try:
            resp = requests.get(f"{REDDIT_API_BASE_URL}/api/v1/me", headers=headers, timeout=10.0)
        except Exception as e:
            raise RedditAPIError(f"Failed to reach Reddit /api/v1/me: {e}")

        if resp.status_code != 200:
            raise RedditAPIError(f"Reddit /api/v1/me failed (HTTP {resp.status_code}): {resp.text}")

        data = resp.json()
        return {
            "username": data.get("name", self.username),
            "id": data.get("id"),
            "total_karma": data.get("total_karma", 0),
            "link_karma": data.get("link_karma", 0),
            "comment_karma": data.get("comment_karma", 0),
            "created_utc": data.get("created_utc", 0),
            "is_verified": data.get("has_verified_email", False),
        }

    def submit_post(
        self,
        subreddit: str,
        title: str,
        body: str = "",
        dry_run: Optional[bool] = None,
        use_browser: bool = True,
    ) -> Dict[str, Any]:
        """
        Main submission method:
        - If dry_run: returns simulated post.
        - If use_browser and has_session: posts via Playwright sock puppet browser.
        - Else if has_credentials: posts via Reddit OAuth2 API.
        - Else: falls back to simulation mode.
        """
        clean_sub = subreddit.replace("r/", "").strip()
        is_dry = self.dry_run if dry_run is None else dry_run

        if is_dry or (not self.has_session and not self.has_credentials):
            if not self.has_session and not self.has_credentials and not is_dry:
                logger.warning("No Reddit session or credentials found. Simulating post submission.")
            sim_id = f"sim_{uuid.uuid4().hex[:6]}"
            sim_url = f"https://www.reddit.com/r/{clean_sub}/comments/{sim_id}/"
            logger.info(f"[SIMULATION] Created simulated AI OP post {sim_id} in r/{clean_sub}: {title}")
            return {
                "post_id": sim_id,
                "url": sim_url,
                "subreddit": clean_sub,
                "title": title,
                "body": body,
                "author": self.username or "SockPuppetAccount",
                "created_utc": time.time(),
                "is_simulated": True,
            }

        # 1. Primary: Browser submission with Sock Puppet account
        if use_browser and self.has_session:
            return self.submit_post_browser(subreddit=clean_sub, title=title, body=body, headless=True)

        # 2. Fallback: API OAuth submission
        if self.has_credentials:
            token = self.authenticate()
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": self.user_agent,
            }
            data = {
                "api_type": "json",
                "sr": clean_sub,
                "kind": "self",
                "title": title,
                "text": body or "",
                "resubmit": "true",
                "sendreplies": "true",
            }
            try:
                resp = requests.post(f"{REDDIT_API_BASE_URL}/api/submit", headers=headers, data=data, timeout=15.0)
            except Exception as e:
                raise RedditAPIError(f"Network error submitting post to r/{clean_sub}: {e}")

            if resp.status_code != 200:
                raise RedditAPIError(f"Reddit submit failed (HTTP {resp.status_code}): {resp.text}")

            resp_json = resp.json()
            json_data = resp_json.get("json", {})
            errors = json_data.get("errors", [])
            if errors:
                err_details = ", ".join([f"{e[0]}: {e[1]}" if len(e) >= 2 else str(e) for e in errors])
                raise RedditAPIError(f"Reddit rejected submission to r/{clean_sub}: {err_details}")

            post_data = json_data.get("data", {})
            post_url = post_data.get("url", "")
            post_id = post_data.get("id", "")
            clean_post_id = post_id.replace("t3_", "") if post_id else ""
            if not clean_post_id and post_url:
                parts = [p for p in post_url.split("/") if p]
                if "comments" in parts:
                    idx = parts.index("comments")
                    if idx + 1 < len(parts):
                        clean_post_id = parts[idx + 1]

            return {
                "post_id": clean_post_id,
                "url": post_url or f"https://www.reddit.com/r/{clean_sub}/comments/{clean_post_id}/",
                "subreddit": clean_sub,
                "title": title,
                "body": body,
                "author": self.username,
                "created_utc": time.time(),
                "is_simulated": False,
            }

        # If we reached here, simulate
        sim_id = f"sim_{uuid.uuid4().hex[:6]}"
        return {
            "post_id": sim_id,
            "url": f"https://www.reddit.com/r/{clean_sub}/comments/{sim_id}/",
            "subreddit": clean_sub,
            "title": title,
            "body": body,
            "author": self.username or "SockPuppetAccount",
            "created_utc": time.time(),
            "is_simulated": True,
        }
