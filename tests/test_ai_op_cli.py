"""Unit tests for AI OP CLI subcommands."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.cli import build_parser, main


class TestAIOPCLI:
    """Test AI OP CLI argument parser and commands."""

    def test_parser_ai_op_generate(self):
        parser = build_parser()
        args = parser.parse_args(["ai-op", "generate", "--subreddit", "Showerthoughts", "--style", "absurd"])
        assert args.command == "ai-op"
        assert args.ai_op_action == "generate"
        assert args.subreddit == "Showerthoughts"
        assert args.style == "absurd"

    def test_parser_ai_op_post(self):
        parser = build_parser()
        args = parser.parse_args(["ai-op", "post", "--subreddit", "AskReddit", "--dry-run", "--min-comments", "3"])
        assert args.command == "ai-op"
        assert args.ai_op_action == "post"
        assert args.subreddit == "AskReddit"
        assert args.dry_run is True
        assert args.min_comments == 3

    def test_parser_ai_op_list(self):
        parser = build_parser()
        args = parser.parse_args(["ai-op", "list", "--status", "rendered", "--subreddit", "AskReddit"])
        assert args.command == "ai-op"
        assert args.ai_op_action == "list"
        assert args.status == "rendered"
        assert args.subreddit == "AskReddit"

    def test_parser_ai_op_check(self):
        parser = build_parser()
        args = parser.parse_args(["ai-op", "check", "--post-id", "xyz123"])
        assert args.command == "ai-op"
        assert args.ai_op_action == "check"
        assert args.post_id == "xyz123"

    def test_parser_ai_op_run_all(self):
        parser = build_parser()
        args = parser.parse_args(["ai-op", "run-all", "--subreddit", "funny", "--min-comments", "2", "--dry-run"])
        assert args.command == "ai-op"
        assert args.ai_op_action == "run-all"
        assert args.subreddit == "funny"
        assert args.min_comments == 2
        assert args.dry_run is True

    def test_main_dispatch_ai_op_generate(self):
        with patch("src.poster.ai_poster.AIOPGenerator.generate_post") as mock_gen:
            mock_gen.return_value = {
                "title": "Mock Title?",
                "body": "",
                "subreddit": "AskReddit",
                "rationale": "Mock",
                "is_fallback": True,
            }
            main(["ai-op", "generate", "--subreddit", "AskReddit"])
            mock_gen.assert_called_once()

    def test_parser_ai_op_login(self):
        parser = build_parser()
        args = parser.parse_args(["ai-op", "login", "--timeout", "60"])
        assert args.command == "ai-op"
        assert args.ai_op_action == "login"
        assert args.timeout == 60

    def test_main_dispatch_ai_op_login(self):
        with patch("src.poster.reddit_client.RedditPosterClient.login_browser_interactive") as mock_login:
            mock_login.return_value = {"status": "success", "message": "Saved"}
            main(["ai-op", "login", "--timeout", "30"])
            mock_login.assert_called_once_with(timeout_seconds=30)
