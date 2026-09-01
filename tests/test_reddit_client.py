"""Unit tests for RedditPosterClient."""

import pytest
from unittest.mock import patch, MagicMock
from src.poster.reddit_client import RedditPosterClient, RedditAPIError


class TestRedditPosterClient:
    """Test Reddit OAuth authentication, posting, and simulation mode."""

    def test_has_credentials_check(self):
        client_empty = RedditPosterClient(client_id="", client_secret="", username="", password="")
        assert client_empty.has_credentials is False

        client_full = RedditPosterClient(
            client_id="cid123",
            client_secret="sec456",
            username="botUser",
            password="botPassword",
        )
        assert client_full.has_credentials is True

    def test_dry_run_simulation_mode(self):
        client = RedditPosterClient(dry_run=True)
        res = client.submit_post(
            subreddit="AskReddit",
            title="What is the weirdest habit your pet has?",
            body="",
        )
        assert res["is_simulated"] is True
        assert res["post_id"].startswith("sim_")
        assert res["subreddit"] == "AskReddit"
        assert "weirdest habit" in res["title"]

    def test_authenticate_success(self):
        client = RedditPosterClient(
            client_id="cid123",
            client_secret="sec456",
            username="botUser",
            password="botPassword",
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "access_token": "fake_token_abc_123",
                "token_type": "bearer",
                "expires_in": 3600,
            }

            token = client.authenticate()
            assert token == "fake_token_abc_123"
            assert client._access_token == "fake_token_abc_123"

    def test_authenticate_failure_raises_error(self):
        client = RedditPosterClient(
            client_id="cid123",
            client_secret="sec456",
            username="botUser",
            password="wrongPassword",
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 401
            mock_post.return_value.text = "Unauthorized"

            with pytest.raises(RedditAPIError) as excinfo:
                client.authenticate()
            assert "HTTP 401" in str(excinfo.value)

    def test_submit_post_live_success(self):
        client = RedditPosterClient(
            client_id="cid123",
            client_secret="sec456",
            username="botUser",
            password="botPassword",
        )

        with patch.object(client, "authenticate", return_value="fake_token"), \
             patch("requests.post") as mock_post:
            
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "json": {
                    "errors": [],
                    "data": {
                        "id": "t3_1abcxyz",
                        "name": "t3_1abcxyz",
                        "url": "https://www.reddit.com/r/AskReddit/comments/1abcxyz/funny_title/",
                    }
                }
            }

            res = client.submit_post(
                subreddit="AskReddit",
                title="Funny Question?",
                body="",
                dry_run=False,
            )

            assert res["post_id"] == "1abcxyz"
            assert res["subreddit"] == "AskReddit"
            assert res["is_simulated"] is False
            assert "comments/1abcxyz" in res["url"]

    def test_submit_post_reddit_error_handling(self):
        client = RedditPosterClient(
            client_id="cid123",
            client_secret="sec456",
            username="botUser",
            password="botPassword",
        )

        with patch.object(client, "authenticate", return_value="fake_token"), \
             patch("requests.post") as mock_post:
            
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "json": {
                    "errors": [["RATELIMIT", "You are doing that too much. Try again in 5 minutes.", "ratelimit"]],
                    "data": {}
                }
            }

            with pytest.raises(RedditAPIError) as excinfo:
                client.submit_post(subreddit="AskReddit", title="Funny?", body="")
            assert "RATELIMIT" in str(excinfo.value)

    def test_has_session_check_and_status(self, tmp_path):
        sess_file = tmp_path / "reddit_user_session.json"
        client = RedditPosterClient(session_file=sess_file)
        assert client.has_session is False
        status_before = client.get_session_status()
        assert status_before["has_session"] is False

        # Write valid Playwright session data
        import json
        with open(sess_file, "w", encoding="utf-8") as f:
            json.dump({"cookies": [{"name": "reddit_session", "value": "xyz_session_token"}]}, f)

        assert client.has_session is True
        status_after = client.get_session_status()
        assert status_after["has_session"] is True
        assert status_after["has_session_cookie"] is True

    def test_submit_post_browser_mode(self, tmp_path):
        sess_file = tmp_path / "reddit_user_session.json"
        import json
        with open(sess_file, "w", encoding="utf-8") as f:
            json.dump({"cookies": [{"name": "reddit_session", "value": "xyz"}]}, f)

        client = RedditPosterClient(session_file=sess_file)

        mock_browser_result = {
            "post_id": "browser_post_999",
            "url": "https://www.reddit.com/r/AskReddit/comments/browser_post_999/",
            "subreddit": "AskReddit",
            "title": "Browser Posted Title?",
            "body": "",
            "author": "SockPuppetUser",
            "created_utc": 1234567.0,
            "is_simulated": False,
        }

        with patch.object(client, "submit_post_browser", return_value=mock_browser_result) as mock_submit:
            res = client.submit_post(subreddit="AskReddit", title="Browser Posted Title?", body="")
            assert res["post_id"] == "browser_post_999"
            assert res["is_simulated"] is False
            mock_submit.assert_called_once()
