"""Unit tests for AI OP FastAPI Web API endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.web.app import app
from src.models import AIPostRecord


@pytest.fixture
def client():
    return TestClient(app)


class TestAIOPWebAPI:
    """Test AI OP Web UI API endpoints."""

    def test_api_ai_op_generate(self, client):
        with patch("src.poster.ai_poster.AIOPGenerator.generate_post") as mock_gen:
            mock_gen.return_value = {
                "title": "What is the most bizarre fact you know?",
                "body": "",
                "subreddit": "AskReddit",
                "rationale": "People love trivia",
                "is_fallback": True,
            }

            resp = client.post("/api/ai-op/generate", json={
                "subreddit": "AskReddit",
                "style": "comedic",
            })

            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert "bizarre fact" in data["post"]["title"]

    def test_api_ai_op_post_and_list(self, client, tmp_path):
        test_file = tmp_path / "web_test_posts.json"
        with patch("src.poster.tracker.AI_OP_POSTS_FILE", test_file), \
             patch("src.config.AI_OP_POSTS_FILE", test_file):

            resp = client.post("/api/ai-op/post", json={
                "subreddit": "AskReddit",
                "title": "What is a weird talent you possess?",
                "body": "",
                "min_comments": 2,
                "dry_run": True,
            })

            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert data["record"]["post_id"].startswith("sim_")

            # Check list endpoint
            list_resp = client.get("/api/ai-op/posts")
            assert list_resp.status_code == 200
            list_data = list_resp.json()
            assert len(list_data["posts"]) >= 1

    def test_api_ai_op_credentials_status(self, client):
        resp = client.get("/api/ai-op/credentials")
        assert resp.status_code == 200
        data = resp.json()
        assert "has_credentials" in data

    def test_api_ai_op_save_credentials(self, client, tmp_path):
        with patch("src.config.REDDIT_CREDENTIALS_FILE", tmp_path / "reddit_creds.json"):
            resp = client.post("/api/ai-op/credentials", json={
                "client_id": "cid_test",
                "client_secret": "csec_test",
                "username": "bot_test",
                "password": "pwd_test",
                "user_agent": "python:test:v1.0",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"

    def test_api_ai_op_browser_login(self, client):
        with patch("src.poster.reddit_client.RedditPosterClient.login_browser_interactive") as mock_login:
            mock_login.return_value = {
                "status": "success",
                "message": "Sock puppet login successful! Session saved.",
                "session_file": "assets/reddit_user_session.json",
            }
            resp = client.post("/api/ai-op/browser-login?timeout=60")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
