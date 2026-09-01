"""Root entrypoint script for Reddit Reading YouTube Shorts Generator.

Usage:
    python main.py pipeline --subreddit AskReddit --dry-run
    python main.py ai-op generate --subreddit AskReddit --theme "dating"
    python main.py ai-op post --subreddit AskReddit --dry-run
    python main.py ai-op run-all --subreddit AskReddit --min-comments 2
    python main.py scrape --subreddit funny
    python main.py tts --text "Hello world"
    python main.py upload --video output/short.mp4 --title "My Short"
    python main.py web --port 8000
    python main.py --help
"""

from src.cli import main

if __name__ == "__main__":
    main()
