"""Unit tests for CLI argument parsing and subcommand dispatch."""

import argparse
import sys
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.cli import build_parser, main


class TestBuildParser:
    """Tests for CLI argument parser construction."""

    def test_parser_returns_argparse_instance(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_prog_name(self):
        parser = build_parser()
        assert parser.prog == "reddit-shorts"

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--verbose", "pipeline"])
        assert args.verbose is True

    def test_verbose_flag_default(self):
        parser = build_parser()
        args = parser.parse_args(["pipeline"])
        assert args.verbose is False


class TestScrapeSubcommand:
    """Tests for 'scrape' subcommand parsing."""

    def test_scrape_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["scrape"])
        assert args.command == "scrape"
        assert args.subreddit == "AskReddit"
        assert args.post_id is None
        assert args.limit == 10
        assert args.output_dir is None

    def test_scrape_with_subreddit(self):
        parser = build_parser()
        args = parser.parse_args(["scrape", "--subreddit", "funny"])
        assert args.subreddit == "funny"

    def test_scrape_with_post_id(self):
        parser = build_parser()
        args = parser.parse_args(["scrape", "--post-id", "abc123"])
        assert args.post_id == "abc123"

    def test_scrape_with_limit(self):
        parser = build_parser()
        args = parser.parse_args(["scrape", "--limit", "5"])
        assert args.limit == 5

    def test_scrape_with_output_dir(self):
        parser = build_parser()
        args = parser.parse_args(["scrape", "--output-dir", "/tmp/out"])
        assert args.output_dir == "/tmp/out"


class TestTTSSubcommand:
    """Tests for 'tts' subcommand parsing."""

    def test_tts_with_text(self):
        parser = build_parser()
        args = parser.parse_args(["tts", "--text", "Hello world"])
        assert args.command == "tts"
        assert args.text == "Hello world"
        assert args.post_json is None

    def test_tts_with_post_json(self):
        parser = build_parser()
        args = parser.parse_args(["tts", "--post-json", "post.json"])
        assert args.command == "tts"
        assert args.post_json == "post.json"
        assert args.text is None

    def test_tts_requires_text_or_json(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["tts"])

    def test_tts_text_and_json_mutually_exclusive(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["tts", "--text", "hello", "--post-json", "f.json"])

    def test_tts_voice_default(self):
        parser = build_parser()
        args = parser.parse_args(["tts", "--text", "test"])
        assert args.voice == "en-US-ChristopherNeural"

    def test_tts_custom_voice(self):
        parser = build_parser()
        args = parser.parse_args(["tts", "--text", "test", "--voice", "en-US-AnaNeural"])
        assert args.voice == "en-US-AnaNeural"

    def test_tts_output_path(self):
        parser = build_parser()
        args = parser.parse_args(["tts", "--text", "test", "--output", "out.mp3"])
        assert args.output == "out.mp3"


class TestCompositeSubcommand:
    """Tests for 'composite' subcommand parsing."""

    def test_composite_required_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "composite", "--post-json", "post.json", "--audio-dir", "/audio"
        ])
        assert args.command == "composite"
        assert args.post_json == "post.json"
        assert args.audio_dir == "/audio"

    def test_composite_missing_post_json(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["composite", "--audio-dir", "/audio"])

    def test_composite_missing_audio_dir(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["composite", "--post-json", "post.json"])

    def test_composite_optional_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "composite",
            "--post-json", "post.json",
            "--audio-dir", "/audio",
            "--background", "bg.mp4",
            "--output", "out.mp4",
        ])
        assert args.background == "bg.mp4"
        assert args.output == "out.mp4"


class TestUploadSubcommand:
    """Tests for 'upload' subcommand parsing."""

    def test_upload_required_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "upload", "--video", "short.mp4", "--title", "My Short"
        ])
        assert args.command == "upload"
        assert args.video == "short.mp4"
        assert args.title == "My Short"

    def test_upload_missing_video(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["upload", "--title", "test"])

    def test_upload_missing_title(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["upload", "--video", "test.mp4"])

    def test_upload_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "upload", "--video", "v.mp4", "--title", "T"
        ])
        assert args.description == ""
        assert args.tags is None
        assert args.privacy == "unlisted"
        assert args.dry_run is False

    def test_upload_dry_run(self):
        parser = build_parser()
        args = parser.parse_args([
            "upload", "--video", "v.mp4", "--title", "T", "--dry-run"
        ])
        assert args.dry_run is True

    def test_upload_privacy_choices(self):
        parser = build_parser()
        for privacy in ["public", "unlisted", "private"]:
            args = parser.parse_args([
                "upload", "--video", "v.mp4", "--title", "T",
                "--privacy", privacy,
            ])
            assert args.privacy == privacy

    def test_upload_invalid_privacy(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "upload", "--video", "v.mp4", "--title", "T",
                "--privacy", "draft",
            ])

    def test_upload_tags(self):
        parser = build_parser()
        args = parser.parse_args([
            "upload", "--video", "v.mp4", "--title", "T",
            "--tags", "reddit", "shorts", "minecraft",
        ])
        assert args.tags == ["reddit", "shorts", "minecraft"]


class TestPipelineSubcommand:
    """Tests for 'pipeline' subcommand parsing."""

    def test_pipeline_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["pipeline"])
        assert args.command == "pipeline"
        assert args.subreddit == "AskReddit"
        assert args.post_id is None
        assert args.voice == "en-US-ChristopherNeural"
        assert args.background is None
        assert args.output_dir is None
        assert args.output_path is None
        assert args.upload is False
        assert args.dry_run is False

    def test_pipeline_all_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "pipeline",
            "--subreddit", "memes",
            "--post-id", "xyz789",
            "--voice", "en-US-AnaNeural",
            "--background", "parkour.mp4",
            "--output-dir", "/out",
            "--output-path", "/out/video.mp4",
            "--upload",
            "--dry-run",
        ])
        assert args.subreddit == "memes"
        assert args.post_id == "xyz789"
        assert args.voice == "en-US-AnaNeural"
        assert args.background == "parkour.mp4"
        assert args.output_dir == "/out"
        assert args.output_path == "/out/video.mp4"
        assert args.upload is True
        assert args.dry_run is True


class TestMainEntrypoint:
    """Tests for the main() function dispatch."""

    def test_no_command_prints_help(self, capsys):
        """No subcommand should print help and exit 0."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    @patch("src.cli.asyncio.run")
    def test_pipeline_dispatch(self, mock_asyncio_run):
        """'pipeline' command should call asyncio.run with cmd_pipeline."""
        mock_asyncio_run.return_value = None
        try:
            main(["pipeline", "--subreddit", "funny", "--dry-run"])
        except SystemExit:
            pass
        mock_asyncio_run.assert_called_once()

    @patch("src.cli.asyncio.run")
    def test_scrape_dispatch(self, mock_asyncio_run):
        """'scrape' command should call asyncio.run with cmd_scrape."""
        mock_asyncio_run.return_value = None
        try:
            main(["scrape", "--subreddit", "memes"])
        except SystemExit:
            pass
        mock_asyncio_run.assert_called_once()

    @patch("src.cli.asyncio.run")
    def test_tts_dispatch(self, mock_asyncio_run):
        """'tts' command should call asyncio.run with cmd_tts."""
        mock_asyncio_run.return_value = None
        try:
            main(["tts", "--text", "hello"])
        except SystemExit:
            pass
        mock_asyncio_run.assert_called_once()

    @patch("src.cli.cmd_upload")
    def test_upload_dispatch(self, mock_cmd_upload):
        """'upload' command should call cmd_upload (sync)."""
        mock_cmd_upload.return_value = None
        try:
            main(["upload", "--video", "v.mp4", "--title", "T"])
        except SystemExit:
            pass
        mock_cmd_upload.assert_called_once()
