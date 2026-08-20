"""Root entrypoint script for Reddit Reading YouTube Shorts Generator.

Usage:
    python main.py pipeline --subreddit AskReddit --dry-run
    python main.py scrape --subreddit funny
    python main.py tts --text "Hello world"
    python main.py upload --video output/short.mp4 --title "My Short"
    python main.py --help
"""

from src.cli import main

if __name__ == "__main__":
    main()
