"""Unit tests for AI OP (Original Poster) post generation."""

import pytest
from unittest.mock import patch, MagicMock
from src.poster.ai_poster import AIOPGenerator, FALLBACK_TEMPLATES


class TestAIOPGenerator:
    """Test AIOPGenerator ideation, formatting, and fallbacks."""

    def test_fallback_templates_exist_for_common_subreddits(self):
        assert "AskReddit" in FALLBACK_TEMPLATES
        assert "Showerthoughts" in FALLBACK_TEMPLATES
        assert "unpopularopinion" in FALLBACK_TEMPLATES
        assert "AmItheAsshole" in FALLBACK_TEMPLATES
        assert "tifu" in FALLBACK_TEMPLATES
        assert len(FALLBACK_TEMPLATES["AskReddit"]) > 0

    def test_fallback_generation_askreddit(self):
        generator = AIOPGenerator()
        # With ollama offline, it falls back to templates
        with patch("src.poster.ai_poster.ensure_ollama_running", return_value=False):
            res = generator.generate_post(subreddit="AskReddit", style="comedic")
            assert res["is_fallback"] is True
            assert res["subreddit"] == "AskReddit"
            assert isinstance(res["title"], str)
            assert len(res["title"]) > 10

    def test_fallback_generation_with_theme(self):
        generator = AIOPGenerator()
        with patch("src.poster.ai_poster.ensure_ollama_running", return_value=False):
            res = generator.generate_post(subreddit="AskReddit", theme="dating disasters")
            assert res["subreddit"] == "AskReddit"
            assert "dating disasters" in res["title"].lower() or len(res["title"]) > 10

    def test_ollama_generation_success(self):
        generator = AIOPGenerator()
        mock_response = {
            "response": '{"title": "What is something people do in public that immediately marks them as a lunatic?", "body": "", "rationale": "People love sharing weird public encounters."}'
        }

        with patch("src.poster.ai_poster.ensure_ollama_running", return_value=True), \
             patch("src.poster.ai_poster.get_available_ollama_models", return_value=["gemma3:4b"]), \
             patch("requests.post") as mock_post:
            
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response

            res = generator.generate_post(subreddit="AskReddit", style="comedic")
            assert res["is_fallback"] is False
            assert res["subreddit"] == "AskReddit"
            assert res["title"].endswith("?")
            assert res["body"] == ""  # AskReddit disallows selftext
            assert "lunatic" in res["title"].lower()

    def test_askreddit_question_mark_enforcement(self):
        generator = AIOPGenerator()
        mock_response = {
            "response": '{"title": "Name a movie that is 10/10 until the final 5 minutes", "body": "some body", "rationale": "Passionate debates"}'
        }

        with patch("src.poster.ai_poster.ensure_ollama_running", return_value=True), \
             patch("src.poster.ai_poster.get_available_ollama_models", return_value=["gemma3:4b"]), \
             patch("requests.post") as mock_post:
            
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response

            res = generator.generate_post(subreddit="AskReddit", style="comedic")
            assert res["title"].endswith("?")
            assert res["body"] == ""

    def test_custom_subreddit_guidelines(self):
        generator = AIOPGenerator()
        guidelines = generator._get_subreddit_guidelines("AskReddit")
        assert "question mark" in guidelines.lower()
        guidelines_shower = generator._get_subreddit_guidelines("Showerthoughts")
        assert "epiphany" in guidelines_shower.lower()
