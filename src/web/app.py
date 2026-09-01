"""FastAPI Web UI Server for Reddit Reading YouTube Shorts Generator."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional

import shutil
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import OUTPUT_DIR, ensure_directories
from src.pipeline import run_pipeline, run_batch_pipeline

logger = logging.getLogger(__name__)
ensure_directories()

SUBREDDIT_SEARCH_CACHE: Dict[str, Dict[str, Any]] = {}

# Thread pool for blocking OAuth operations
_auth_executor = ThreadPoolExecutor(max_workers=1)
_auth_status: Dict[str, Any] = {"running": False, "error": None}

app = FastAPI(title="Reddit Reading YouTube Shorts Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global status tracking for current job
job_status: Dict[str, Any] = {
    "status": "idle",  # idle, running, completed, error
    "stage": "",
    "message": "",
    "result": None,
    "error": None,
}


class PipelineRequest(BaseModel):
    subreddit: str = "AskReddit"
    subreddits: Optional[List[str]] = None
    post_id: Optional[str] = None
    voice: str = "en-US-ChristopherNeural"
    mode: str = "test"  # "test" for dry-run, "upload" for actual upload
    use_gpu: bool = True
    background_video: Optional[str] = None
    music_style: Optional[str] = "lofi"
    privacy_status: str = "unlisted"
    max_comments: int = 2
    is_batch: bool = False
    batch_count: int = 5
    enable_subtitles: bool = True
    subtitle_style: str = "yellow_pill"


async def task_runner(req: PipelineRequest):
    global job_status
    dry_run = True if req.mode == "test" else False
    upload = True if req.mode == "upload" else False

    if req.is_batch:
        sub_list = req.subreddits if req.subreddits else [req.subreddit]
        job_status.update({
            "status": "running",
            "stage": f"Starting Batch ({req.batch_count} videos)...",
            "message": f"Processing batch across subreddits: {', '.join(sub_list)}",
            "result": None,
            "error": None,
        })

        async def _batch_status_cb(stage, message, current, total):
            job_status.update({
                "status": "running",
                "stage": stage,
                "message": message,
            })

        try:
            result = await run_batch_pipeline(
                subreddits=sub_list,
                batch_count=req.batch_count,
                voice=req.voice,
                background_video=req.background_video,
                music_style=req.music_style,
                privacy_status=req.privacy_status,
                max_comments=req.max_comments,
                upload=upload,
                dry_run=dry_run,
                use_gpu=req.use_gpu,
                enable_subtitles=req.enable_subtitles,
                subtitle_style=req.subtitle_style,
                status_callback=_batch_status_cb,
            )
            job_status.update({
                "status": "completed",
                "stage": "Batch Completed!",
                "message": f"Successfully generated {result['successful_count']} of {req.batch_count} videos",
                "result": result,
                "error": None,
            })
        except Exception as e:
            logger.error(f"Batch pipeline web task failed: {e}", exc_info=True)
            job_status.update({
                "status": "error",
                "stage": "Batch Failed",
                "message": str(e),
                "result": None,
                "error": str(e),
            })
    else:
        job_status.update({
            "status": "running",
            "stage": "Starting Pipeline...",
            "message": f"Processing r/{req.subreddit} in {req.mode.upper()} mode",
            "result": None,
            "error": None,
        })

        try:
            result = await run_pipeline(
                subreddit=req.subreddit,
                post_id=req.post_id,
                voice=req.voice,
                background_video=req.background_video,
                music_style=req.music_style,
                privacy_status=req.privacy_status,
                max_comments=req.max_comments,
                upload=upload,
                dry_run=dry_run,
                use_gpu=req.use_gpu,
                enable_subtitles=req.enable_subtitles,
                subtitle_style=req.subtitle_style,
            )
            job_status.update({
                "status": "completed",
                "stage": "Finished!",
                "message": f"Successfully created short for r/{req.subreddit}",
                "result": result,
                "error": None,
            })
        except Exception as e:
            logger.error(f"Pipeline web task failed: {e}", exc_info=True)
            job_status.update({
                "status": "error",
                "stage": "Failed",
                "message": str(e),
                "result": None,
                "error": str(e),
            })


@app.get("/api/status")
def get_status():
    return job_status


@app.post("/api/pipeline")
async def trigger_pipeline(req: PipelineRequest):
    global job_status
    if job_status["status"] == "running":
        raise HTTPException(status_code=400, detail="A video pipeline job is already running.")

    asyncio.create_task(task_runner(req))
    return {"status": "started", "subreddit": req.subreddit, "mode": req.mode}


@app.get("/api/videos")
def list_videos():
    """List all generated date-timestamped video folders in output/."""
    out_dir = Path(OUTPUT_DIR)
    folders = []

    if out_dir.exists():
        for p in sorted(out_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_dir():
                continue

            batch_summary = p / "batch_summary.json"
            if batch_summary.exists():
                try:
                    import json
                    with open(batch_summary, "r", encoding="utf-8") as f:
                        bdata = json.load(f)
                    for item in bdata.get("items", []):
                        if item.get("video_path") and Path(item["video_path"]).exists():
                            vpath = Path(item["video_path"])
                            rel_url = f"/output/{p.name}/{vpath.parent.name}/{vpath.name}"
                            folders.append({
                                "folder_name": f"{p.name}/{vpath.parent.name}",
                                "subreddit": item.get("subreddit", "Reddit"),
                                "title": f"📦 Batch #{item.get('batch_index', 1)}: {item.get('title') or item.get('post_title') or vpath.stem}",
                                "description": item.get("description", ""),
                                "video_url": rel_url,
                            })
                except Exception:
                    pass
                continue

            meta_json = p / "metadata.json"
            v_file = list(p.glob("*.mp4"))
            if meta_json.exists():
                try:
                    import json
                    with open(meta_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["folder_name"] = p.name
                        data["video_url"] = f"/output/{p.name}/{data.get('video_file', 'video.mp4')}"
                        folders.append(data)
                except Exception:
                    pass
            elif v_file:
                folders.append({
                    "folder_name": p.name,
                    "subreddit": p.name.split("_")[-1] if "_" in p.name else "Reddit",
                    "title": p.name,
                    "description": "",
                    "video_url": f"/output/{p.name}/{v_file[0].name}",
                })
    return folders


@app.get("/output/{full_path:path}")
def serve_output_file(full_path: str):
    file_path = Path(OUTPUT_DIR) / full_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/api/backgrounds")
def list_backgrounds():
    """List all saved pre-cropped background video assets in assets/backgrounds/ and assets/."""
    from src.config import ASSETS_DIR, BACKGROUNDS_DIR, get_background_video_path
    BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)

    default_bg = get_background_video_path()
    items = []
    seen_paths = set()

    # 1. Add default background video first
    if default_bg.exists():
        seen_paths.add(default_bg.resolve())
        size_mb = round(default_bg.stat().st_size / (1024 * 1024), 2)
        items.append({
            "filename": default_bg.name,
            "display_name": f"{default_bg.stem.replace('_', ' ').title()} (Default)",
            "file_path": str(default_bg),
            "size_mb": size_mb,
            "is_default": True
        })

    # 2. Scan BACKGROUNDS_DIR for user uploaded & cropped 9:16 backgrounds
    for p in sorted(BACKGROUNDS_DIR.glob("*.mp4")):
        if p.is_file() and p.resolve() not in seen_paths:
            seen_paths.add(p.resolve())
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)
            display_name = p.stem.replace("_916", "").replace("_", " ").title()
            items.append({
                "filename": p.name,
                "display_name": f"🎬 {display_name} ({size_mb} MB)",
                "file_path": str(p),
                "size_mb": size_mb,
                "is_default": False
            })

    return items


POPULAR_SUBREDDITS_INDEX = [
    # General & Stories
    {"name": "AskReddit", "subscribers": 45000000, "title": "Stories, questions, and discussion", "over18": False},
    {"name": "AskRedditOver30", "subscribers": 500000, "title": "Questions for adults over 30", "over18": False},
    {"name": "tifu", "subscribers": 18000000, "title": "Today I Fucked Up", "over18": False},
    {"name": "AmItheAsshole", "subscribers": 12000000, "title": "A cathartic subreddit to judge others", "over18": False},
    {"name": "showerthoughts", "subscribers": 28000000, "title": "Miniature epiphanies", "over18": False},
    {"name": "confession", "subscribers": 3500000, "title": "Get it off your chest", "over18": False},
    {"name": "offmychest", "subscribers": 3200000, "title": "A place to vent", "over18": False},
    {"name": "trueoffmychest", "subscribers": 2500000, "title": "A real place to vent", "over18": False},
    {"name": "unpopularopinion", "subscribers": 4000000, "title": "Got an unpopular opinion?", "over18": False},
    {"name": "casualiama", "subscribers": 1200000, "title": "Ask me anything, casual edition", "over18": False},
    {"name": "AskHistorians", "subscribers": 2000000, "title": "History questions answered by experts", "over18": False},
    {"name": "AskScience", "subscribers": 24000000, "title": "Ask science questions", "over18": False},
    # Funny & Memes
    {"name": "funny", "subscribers": 60000000, "title": "Welcome to r/funny", "over18": False},
    {"name": "memes", "subscribers": 30000000, "title": "Memes, the DNA of the soul", "over18": False},
    {"name": "dankmemes", "subscribers": 6000000, "title": "Dankest memes on reddit", "over18": False},
    {"name": "wholesomememes", "subscribers": 15000000, "title": "Memes that make you smile", "over18": False},
    {"name": "me_irl", "subscribers": 7000000, "title": "Selfies of the soul", "over18": False},
    {"name": "AdviceAnimals", "subscribers": 10000000, "title": "Image Macros and Advice Animals", "over18": False},
    {"name": "facepalm", "subscribers": 8000000, "title": "A subreddit for facepalms", "over18": False},
    {"name": "therewasanattempt", "subscribers": 5000000, "title": "There was an attempt...", "over18": False},
    {"name": "clevercomebacks", "subscribers": 3000000, "title": "Clever comebacks", "over18": False},
    {"name": "terriblefacebookmemes", "subscribers": 3500000, "title": "Terrible memes found on Facebook", "over18": False},
    # Gaming
    {"name": "gaming", "subscribers": 38000000, "title": "A subreddit for (almost) anything related to games", "over18": False},
    {"name": "Games", "subscribers": 4000000, "title": "Informative game news and discussion", "over18": False},
    {"name": "PCMasterRace", "subscribers": 11000000, "title": "PC Enthusiasts Subreddit", "over18": False},
    {"name": "Minecraft", "subscribers": 7500000, "title": "Official Minecraft Subreddit", "over18": False},
    {"name": "LeagueOfLegends", "subscribers": 7000000, "title": "League of Legends Community", "over18": False},
    {"name": "Valorant", "subscribers": 2000000, "title": "Riot Games Tactical Shooter", "over18": False},
    {"name": "Fortnite", "subscribers": 3000000, "title": "Fortnite Battle Royale", "over18": False},
    {"name": "buildapc", "subscribers": 7000000, "title": "PC Building Community", "over18": False},
    {"name": "GameDeals", "subscribers": 1500000, "title": "Deals on PC & console games", "over18": False},
    {"name": "steam", "subscribers": 2200000, "title": "Valve's Steam platform", "over18": False},
    {"name": "nintendo", "subscribers": 2500000, "title": "Nintendo Games & Systems", "over18": False},
    {"name": "GTA6", "subscribers": 1200000, "title": "Grand Theft Auto VI", "over18": False},
    {"name": "Pokemon", "subscribers": 4500000, "title": "Gotta catch 'em all!", "over18": False},
    {"name": "Overwatch", "subscribers": 5000000, "title": "Overwatch 2 Community", "over18": False},
    {"name": "ApexLegends", "subscribers": 2500000, "title": "Apex Legends Battle Royale", "over18": False},
    # Animals & Cute
    {"name": "aww", "subscribers": 34000000, "title": "Things that make you go AWW!", "over18": False},
    {"name": "Eyebleach", "subscribers": 4000000, "title": "A wholesome palate cleanser", "over18": False},
    {"name": "rarepuppers", "subscribers": 2500000, "title": "All the goodest dogs", "over18": False},
    {"name": "CatsWithDogs", "subscribers": 500000, "title": "Cats and dogs hanging out", "over18": False},
    {"name": "AnimalsBeingBros", "subscribers": 6000000, "title": "Animals helping each other", "over18": False},
    {"name": "NatureIsFuckingLit", "subscribers": 10000000, "title": "Nature is amazing", "over18": False},
    # Tech & Science
    {"name": "technology", "subscribers": 15000000, "title": "Subreddit for technology news", "over18": False},
    {"name": "science", "subscribers": 31000000, "title": "New research and peer-reviewed studies", "over18": False},
    {"name": "space", "subscribers": 23000000, "title": "Share and discuss space exploration", "over18": False},
    {"name": "gadgets", "subscribers": 22000000, "title": "Consumer technology & electronics", "over18": False},
    {"name": "hardware", "subscribers": 3500000, "title": "Computer hardware news & specs", "over18": False},
    {"name": "programming", "subscribers": 5500000, "title": "Computer programming discussion", "over18": False},
    {"name": "artificial", "subscribers": 1000000, "title": "Artificial Intelligence discussions", "over18": False},
    {"name": "ChatGPT", "subscribers": 5000000, "title": "OpenAI ChatGPT discussion", "over18": False},
    # Movies & Shows
    {"name": "movies", "subscribers": 32000000, "title": "News & discussion about major feature films", "over18": False},
    {"name": "television", "subscribers": 18000000, "title": "Subreddit for TV shows & news", "over18": False},
    {"name": "marvelstudios", "subscribers": 3500000, "title": "Marvel Cinematic Universe", "over18": False},
    {"name": "StarWars", "subscribers": 3000000, "title": "Star Wars universe", "over18": False},
    {"name": "anime", "subscribers": 9000000, "title": "Anime discussion", "over18": False},
    # Lifestyle
    {"name": "fitness", "subscribers": 11000000, "title": "Health & fitness goals", "over18": False},
    {"name": "lifehacks", "subscribers": 7000000, "title": "Tips to improve everyday life", "over18": False},
    {"name": "personalfinance", "subscribers": 18000000, "title": "Learn to manage your money", "over18": False},
    {"name": "EatCheapAndHealthy", "subscribers": 4500000, "title": "Eat healthy on a budget", "over18": False},
    {"name": "DIY", "subscribers": 23000000, "title": "Do It Yourself projects", "over18": False},
    {"name": "travel", "subscribers": 9000000, "title": "Travel advice and pictures", "over18": False},
]


def clean_subreddit_name(text: str) -> str:
    """Extract clean subreddit name from text or URL (e.g. https://www.reddit.com/r/vtubers/ -> vtubers)."""
    if not text:
        return ""
    text = text.strip()
    import re
    m = re.search(r'/r/([A-Za-z0-9_]+)', text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'^r/([A-Za-z0-9_]+)', text, re.IGNORECASE)
    if m:
        return m.group(1)
    text = re.sub(r'^https?://', '', text)
    return text.replace("r/", "").strip("/")


@app.get("/api/search_subreddits")
def search_subreddits(q: str = Query(..., min_length=2)):
    """
    Search active subreddits with local index first, online query fallback,
    and automatic custom query fallback for any unindexed subreddit or URL.
    """
    cleaned_q = clean_subreddit_name(q)
    if not cleaned_q or len(cleaned_q) < 2:
        return []

    # 1. Match local index first
    local_matches = []
    for sub in POPULAR_SUBREDDITS_INDEX:
        name_lower = sub["name"].lower()
        if name_lower.startswith(cleaned_q.lower()) or cleaned_q.lower() in name_lower:
            local_matches.append(sub)

    local_matches.sort(
        key=lambda x: (
            0 if x["name"].lower().startswith(cleaned_q.lower()) else 1,
            -x["subscribers"],
        )
    )

    results = list(local_matches[:5])

    # Check if exact match is already present in results
    has_exact = any(item["name"].lower() == cleaned_q.lower() for item in results)

    # If unindexed or pasted URL, include clean subreddit name as custom option #1
    if not has_exact:
        custom_item = {
            "name": cleaned_q,
            "subscribers": 0,
            "title": f"Use custom subreddit r/{cleaned_q}",
            "over18": False,
            "is_custom": True,
        }
        results.insert(0, custom_item)

    return results[:5]


@app.post("/api/upload_background")
async def upload_background(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    from src.config import ASSETS_DIR, BACKGROUNDS_DIR
    BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix.lower()
    clean_stem = Path(file.filename).stem.replace(" ", "_").replace("-", "_")
    cropped_filename = f"{clean_stem}_916.mp4"
    cropped_path = BACKGROUNDS_DIR / cropped_filename
    raw_path = ASSETS_DIR / f"raw_{file.filename}"

    # Check if pre-cropped background already exists in backgrounds library
    if cropped_path.exists() and cropped_path.stat().st_size > 1000:
        logger.info(f"Background '{cropped_filename}' already exists in backgrounds library. Reusing existing file.")
        return {
            "status": "success",
            "file_path": str(cropped_path),
            "filename": cropped_filename,
            "cached": True,
            "message": f"Reusing saved background: {cropped_filename}"
        }

    try:
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        import imageio_ffmpeg, subprocess
        from src.video.composer import get_ffmpeg_vcodec

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        vcodec = get_ffmpeg_vcodec(use_gpu=True)
        preset_val = "p1" if "nvenc" in vcodec else "fast"

        base_cmd = [ffmpeg_exe, "-y", "-threads", "2"]
        if "nvenc" in vcodec:
            base_cmd += ["-hwaccel", "cuda"]

        # Handle static/animated WebP and images
        if ext in (".webp", ".png", ".jpg", ".jpeg"):
            cmd = base_cmd + [
                "-loop", "1", "-i", str(raw_path),
                "-vf", "crop=ih*9/16:ih,scale=1080:1920",
                "-t", "30", "-c:v", vcodec, "-preset", preset_val,
                "-pix_fmt", "yuv420p", "-an", str(cropped_path)
            ]
        else:
            cmd = base_cmd + [
                "-i", str(raw_path),
                "-vf", "crop=ih*9/16:ih,scale=1080:1920",
                "-c:v", vcodec, "-preset", preset_val,
                "-pix_fmt", "yuv420p", "-an", str(cropped_path)
            ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if raw_path.exists():
            raw_path.unlink()

        return {
            "status": "success",
            "file_path": str(cropped_path),
            "filename": cropped_filename,
            "cached": False,
            "message": f"Cropped & saved new background: {cropped_filename}"
        }
    except Exception as e:
        logger.warning(f"FFmpeg crop/convert failed ({e}). Using raw background file.")
        return {"status": "success", "file_path": str(raw_path), "filename": file.filename, "cached": False}


class DownloadYouTubeBgRequest(BaseModel):
    url: str


@app.post("/api/download_youtube_background")
async def download_youtube_background_endpoint(req: DownloadYouTubeBgRequest):
    """
    Downloads a background video from a YouTube URL via yt-dlp,
    auto-crops it to 9:16 aspect ratio (1080x1920) using GPU FFmpeg,
    saves the cropped MP4 asset to backgrounds library,
    and automatically cleans up the raw uncropped download.
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="No YouTube URL provided")

    import re, subprocess, imageio_ffmpeg, yt_dlp
    from src.config import ASSETS_DIR, BACKGROUNDS_DIR
    from src.video.composer import get_ffmpeg_vcodec

    BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    raw_template = str(ASSETS_DIR / "raw_yt_%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": raw_template,
        "ffmpeg_location": ffmpeg_exe,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web", "mweb", "tv"],
            }
        },
        "nocheckcertificate": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        loop = asyncio.get_event_loop()
        def _download():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                return info, filename

        info, raw_file_path = await loop.run_in_executor(None, _download)
    except Exception as e:
        logger.error(f"yt-dlp download failed for URL '{url}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download YouTube video: {str(e)}")

    video_id = info.get("id", "video")
    video_title = info.get("title", "yt_video")
    clean_title = re.sub(r'[^a-zA-Z0-9_]', '_', video_title)[:40].strip("_")
    cropped_filename = f"yt_{video_id}_{clean_title}_916.mp4"
    cropped_path = BACKGROUNDS_DIR / cropped_filename
    raw_p = Path(raw_file_path)

    # Re-use existing cropped background if already present
    if cropped_path.exists() and cropped_path.stat().st_size > 1000:
        if raw_p.exists():
            try:
                raw_p.unlink()
            except Exception:
                pass
        return {
            "status": "success",
            "file_path": str(cropped_path),
            "filename": cropped_filename,
            "title": video_title,
            "cached": True,
            "message": f"Reusing saved background: {video_title}"
        }

    try:
        vcodec = get_ffmpeg_vcodec(use_gpu=True)
        preset_val = "p1" if "nvenc" in vcodec else "fast"

        base_cmd = [ffmpeg_exe, "-y", "-threads", "2"]
        if "nvenc" in vcodec:
            base_cmd += ["-hwaccel", "cuda"]

        cmd = base_cmd + [
            "-i", str(raw_p),
            "-vf", "crop=ih*9/16:ih,scale=1080:1920",
            "-c:v", vcodec, "-preset", preset_val,
            "-pix_fmt", "yuv420p", "-an", str(cropped_path)
        ]

        def _run_ffmpeg():
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        await loop.run_in_executor(None, _run_ffmpeg)
    except Exception as e:
        logger.error(f"FFmpeg cropping failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video cropping failed: {str(e)}")
    finally:
        # Clean up raw downloaded uncropped video to preserve disk space
        if raw_p.exists():
            try:
                raw_p.unlink()
            except Exception:
                pass

    return {
        "status": "success",
        "file_path": str(cropped_path),
        "filename": cropped_filename,
        "title": video_title,
        "cached": False,
        "message": f"Downloaded, cropped & saved background: {video_title}"
    }


@app.get("/api/auth/status")
def get_auth_status():
    """Returns current YouTube authentication status and channel info."""
    from src.uploader.youtube_uploader import YouTubeUploader
    uploader = YouTubeUploader()
    status = uploader.get_channel_status()
    status["auth_running"] = _auth_status["running"]
    status["auth_error"] = _auth_status["error"]
    return status


class SaveYouTubeCredentialsRequest(BaseModel):
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    json_content: Optional[str] = None


@app.post("/api/auth/save_credentials")
def save_youtube_credentials(req: SaveYouTubeCredentialsRequest):
    """
    Saves Google OAuth2 client secrets directly to assets/client_secret.json.
    Supports either client_id + client_secret fields or raw JSON string.
    """
    import json
    from src.config import ASSETS_DIR
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = ASSETS_DIR / "client_secret.json"

    if req.json_content and req.json_content.strip():
        try:
            data = json.loads(req.json_content.strip())
            client_info = data.get("installed", {}) or data.get("web", {})
            cid = client_info.get("client_id", "").strip()
            csec = client_info.get("client_secret", "").strip()
            if not cid or not csec:
                raise ValueError("JSON must contain 'client_id' and 'client_secret'.")
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return {"status": "success", "message": "Saved client_secret.json successfully."}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid client_secret.json format: {str(e)}")

    if not req.client_id or not req.client_secret:
        raise HTTPException(status_code=400, detail="OAuth Client ID and Client Secret are required.")

    cid = req.client_id.strip()
    csecret = req.client_secret.strip()

    secret_data = {
        "installed": {
            "client_id": cid,
            "project_id": "reddit-reading-app",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": csecret,
            "redirect_uris": ["http://localhost:8080/", "http://127.0.0.1:8080/"],
        }
    }
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(secret_data, f, indent=2)
    return {"status": "success", "message": "Saved YouTube API credentials successfully."}


@app.post("/api/auth/upload_client_secrets")
async def upload_client_secrets_file(file: UploadFile = File(...)):
    """Uploads a client_secret.json file directly from Google Cloud Console."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    import json
    from src.config import ASSETS_DIR
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = ASSETS_DIR / "client_secret.json"

    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
        client_info = data.get("installed", {}) or data.get("web", {})
        if not client_info.get("client_id") or not client_info.get("client_secret"):
            raise ValueError("Uploaded JSON does not contain client_id and client_secret.")
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return {"status": "success", "message": "Uploaded and saved client_secret.json successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid client_secret.json file: {str(e)}")


@app.post("/api/auth/youtube")
async def start_youtube_auth():
    """Launches the Google OAuth2 browser login flow in a background thread."""
    global _auth_status
    if _auth_status["running"]:
        raise HTTPException(status_code=400, detail="Authentication already in progress.")

    def _do_auth():
        global _auth_status
        _auth_status = {"running": True, "error": None}
        try:
            from src.uploader.youtube_uploader import YouTubeUploader
            uploader = YouTubeUploader()
            uploader.authenticate_user(port=8080)
            _auth_status = {"running": False, "error": None}
            logger.info("YouTube OAuth authentication completed successfully.")
        except Exception as e:
            logger.error(f"YouTube OAuth failed: {e}")
            _auth_status = {"running": False, "error": str(e)}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_auth_executor, _do_auth)
    return {"status": "started", "message": "OAuth browser window opened. Complete login in the browser."}


@app.post("/api/auth/logout")
def logout_youtube_account():
    """Revokes cached token to allow logging into a different YouTube account."""
    from src.uploader.youtube_uploader import YouTubeUploader
    uploader = YouTubeUploader()
    uploader.revoke_credentials()
    return {"status": "success", "message": "Logged out of YouTube account."}


@app.get("/api/system/gpu")
def get_system_gpu_status():
    """Returns GPU hardware acceleration detection info."""
    from src.video.composer import get_gpu_status
    return get_gpu_status()


@app.get("/output/{folder_name}/{filename}")
def serve_output_file(folder_name: str, filename: str):
    file_path = Path(OUTPUT_DIR) / folder_name / filename
    if not file_path.exists():
        # Fallback to direct output dir file
        file_path = Path(OUTPUT_DIR) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# ── AI OP (Auto-Poster & Watcher) Endpoints ──────────────────────────────
class AIOPGenerateRequest(BaseModel):
    subreddit: str = "AskReddit"
    theme: Optional[str] = None
    style: str = "comedic"


class AIOPPostRequest(BaseModel):
    subreddit: str = "AskReddit"
    title: str
    body: Optional[str] = ""
    style: Optional[str] = "comedic"
    min_comments: int = 2
    dry_run: bool = False


class SaveRedditCredentialsRequest(BaseModel):
    client_id: str
    client_secret: str
    username: str
    password: str
    user_agent: Optional[str] = None


@app.post("/api/ai-op/generate")
def api_ai_op_generate(req: AIOPGenerateRequest):
    """Generate a funny post tailored for the target subreddit."""
    from src.poster import AIOPGenerator
    gen = AIOPGenerator()
    try:
        post = gen.generate_post(subreddit=req.subreddit, theme=req.theme, style=req.style)
        return {"status": "success", "post": post}
    except Exception as e:
        logger.error(f"AI OP generate failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai-op/post")
def api_ai_op_post(req: AIOPPostRequest):
    """Post an AI-crafted submission to Reddit and register in tracker."""
    import time
    from src.poster import RedditPosterClient, AIPostTracker
    from src.models import AIPostRecord

    client = RedditPosterClient(dry_run=req.dry_run)
    tracker = AIPostTracker()

    try:
        res = client.submit_post(
            subreddit=req.subreddit,
            title=req.title,
            body=req.body or "",
            dry_run=req.dry_run,
        )

        record = AIPostRecord(
            post_id=res["post_id"],
            subreddit=req.subreddit,
            title=req.title,
            body=req.body or "",
            url=res["url"],
            author=res.get("author", "AI_OP_Bot"),
            created_utc=time.time(),
            status="waiting_for_comments",
            min_comments_target=req.min_comments,
            is_simulated=res.get("is_simulated", False),
        )
        tracker.add_post(record)
        return {"status": "success", "record": record.to_dict()}
    except Exception as e:
        logger.error(f"AI OP post failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/ai-op/posts")
def api_ai_op_list_posts(status: Optional[str] = None, subreddit: Optional[str] = None):
    """Return all tracked AI OP submissions."""
    from src.poster import AIPostTracker
    tracker = AIPostTracker()
    posts = tracker.list_posts(status=status, subreddit=subreddit)
    return {"status": "success", "posts": [p.to_dict() for p in posts]}


@app.post("/api/ai-op/check/{post_id}")
async def api_ai_op_check_post(post_id: str):
    """Check live Reddit comments for a tracked AI OP post."""
    from src.poster import AIPostWatcher
    watcher = AIPostWatcher()
    res = await watcher.check_post_comments(post_id)
    return res


@app.post("/api/ai-op/render/{post_id}")
async def api_ai_op_render_post(post_id: str, background_tasks: BackgroundTasks):
    """Trigger video rendering for a tracked AI OP post."""
    global job_status
    from src.poster import AIPostWatcher, AIPostTracker
    tracker = AIPostTracker()
    record = tracker.get_post(post_id)
    if not record:
        raise HTTPException(status_code=404, detail="Post not found")

    job_status.update({
        "status": "running",
        "stage": "Rendering AI OP Post",
        "message": f"Rendering video for post [{record.post_id}] in r/{record.subreddit}...",
        "result": None,
        "error": None,
    })

    async def _do_render():
        global job_status
        watcher = AIPostWatcher()
        try:
            res = await watcher.render_post_video(post_id=post_id)
            job_status.update({
                "status": "completed",
                "stage": "AI OP Video Rendered!",
                "message": f"Successfully rendered video: {res.get('video_path')}",
                "result": res,
                "error": None,
            })
        except Exception as e:
            logger.error(f"AI OP render failed: {e}", exc_info=True)
            job_status.update({
                "status": "error",
                "stage": "Render Failed",
                "message": str(e),
                "result": None,
                "error": str(e),
            })

    background_tasks.add_task(_do_render)
    return {"status": "started", "message": f"Rendering video for post [{post_id}]"}


@app.delete("/api/ai-op/posts/{post_id}")
def api_ai_op_delete_post(post_id: str):
    """Delete a tracked AI OP post."""
    from src.poster import AIPostTracker
    tracker = AIPostTracker()
    deleted = tracker.delete_post(post_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Post not found")
    return {"status": "success", "message": "Post deleted"}


@app.get("/api/ai-op/credentials")
def api_ai_op_get_credentials():
    """Check status of configured Reddit sock puppet session and bot credentials."""
    from src.config import get_reddit_credentials
    from src.poster import RedditPosterClient
    client = RedditPosterClient()
    creds = get_reddit_credentials()
    cid = creds.get("client_id", "")
    session_status = client.get_session_status()
    return {
        "has_session": client.has_session,
        "has_credentials": client.has_credentials,
        "username": creds.get("username", "") or session_status.get("username", ""),
        "client_id_preview": f"{cid[:4]}...{cid[-4:]}" if len(cid) > 8 else (cid or "Not set"),
        "user_agent": creds.get("user_agent", ""),
        "session_status": session_status,
    }


@app.post("/api/ai-op/browser-login")
def api_ai_op_browser_login(timeout: int = 120):
    """Launch Playwright browser window to log in to Reddit Sock Puppet account."""
    from src.poster import RedditPosterClient
    client = RedditPosterClient()
    try:
        res = client.login_browser_interactive(timeout_seconds=timeout)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai-op/credentials")
def api_ai_op_save_credentials(req: SaveRedditCredentialsRequest):
    """Save Reddit bot account credentials."""
    from src.config import save_reddit_credentials
    saved = save_reddit_credentials(
        client_id=req.client_id,
        client_secret=req.client_secret,
        username=req.username,
        password=req.password,
        user_agent=req.user_agent,
    )
    if not saved:
        raise HTTPException(status_code=500, detail="Failed to save credentials")
    return {"status": "success", "message": "Reddit credentials saved successfully"}


@app.post("/api/ai-op/test-credentials")
def api_ai_op_test_credentials(req: Optional[SaveRedditCredentialsRequest] = None):
    """Test Reddit OAuth authentication and fetch bot account info."""
    from src.poster import RedditPosterClient
    if req and req.client_id and req.client_secret and req.username and req.password:
        client = RedditPosterClient(
            client_id=req.client_id,
            client_secret=req.client_secret,
            username=req.username,
            password=req.password,
            user_agent=req.user_agent,
        )
    else:
        client = RedditPosterClient()

    if not client.has_credentials:
        raise HTTPException(status_code=400, detail="Reddit credentials are not configured")

    try:
        me = client.get_me()
        return {
            "status": "success",
            "message": f"Connected to Reddit as /u/{me['username']} (Total Karma: {me['total_karma']})",
            "profile": me,
        }
    except Exception as e:
        logger.warning(f"Reddit credentials test failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))



# Mount Web UI HTML Page
HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reddit Reading Shorts Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #090a0f;
            --bg-surface: #11131a;
            --bg-surface-elevated: #161922;
            --bg-surface-hover: #1c202c;
            --bg-input: #0d0f15;
            
            --border-subtle: rgba(255, 255, 255, 0.07);
            --border-strong: rgba(255, 255, 255, 0.13);
            --border-focus: #ff4500;
            
            --brand-primary: #ff4500;
            --brand-primary-hover: #e03d00;
            --brand-gradient: linear-gradient(135deg, #ff4500 0%, #ff6b35 100%);
            
            --accent-blue: #3b82f6;
            --accent-blue-hover: #2563eb;
            --accent-purple: #8b5cf6;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-tertiary: #64748b;
            
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            
            --shadow-card: 0 20px 40px -15px rgba(0, 0, 0, 0.7), 0 0 0 1px var(--border-subtle);
            --shadow-btn: 0 4px 14px 0 rgba(255, 69, 0, 0.35);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(circle at 50% 0%, rgba(255, 69, 0, 0.06) 0%, transparent 50%),
                radial-gradient(circle at 100% 100%, rgba(59, 130, 246, 0.04) 0%, transparent 40%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Custom Scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 999px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.25);
        }

        /* Header Bar */
        header {
            padding: 0.9rem 2rem;
            background: rgba(17, 19, 26, 0.85);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            text-decoration: none;
        }

        .brand-logo {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 14px rgba(255, 69, 0, 0.4);
            flex-shrink: 0;
            overflow: hidden;
        }

        .brand-logo svg {
            width: 36px;
            height: 36px;
            display: block;
        }

        .brand-text {
            display: flex;
            flex-direction: column;
        }

        .brand-title {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .brand-subtitle {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-tertiary);
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .gpu-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.25);
            color: #34d399;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.85); }
        }

        .btn-yt-login {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: #cc0000;
            background: linear-gradient(180deg, #e50914 0%, #b8050e 100%);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 999px;
            padding: 0.4rem 1rem;
            font-size: 0.8rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 2px 8px rgba(204, 0, 0, 0.3);
        }

        .btn-yt-login:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(204, 0, 0, 0.45);
        }

        .yt-channel-badge-container {
            display: none;
            align-items: center;
            gap: 0.5rem;
        }

        .yt-connected-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(16, 185, 129, 0.12);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 999px;
            padding: 0.35rem 0.85rem;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .btn-switch-account {
            background: var(--bg-surface-elevated);
            color: var(--text-secondary);
            border: 1px solid var(--border-strong);
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .btn-switch-account:hover {
            background: var(--bg-surface-hover);
            color: #ffffff;
            border-color: rgba(255, 255, 255, 0.25);
        }

        /* Layout Grid */
        main {
            flex: 1;
            max-width: 1480px;
            width: 95%;
            margin: 1.5rem auto;
            padding: 0 0.5rem;
            display: grid;
            grid-template-columns: 1.12fr 0.88fr;
            gap: 1.5rem;
            align-items: start;
        }

        @media (max-width: 1080px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        /* Studio Panels */
        .panel {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xl);
            padding: 1.75rem;
            box-shadow: var(--shadow-card);
            position: relative;
        }

        .panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 1.1rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .panel-title-wrapper {
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }

        .panel-icon-box {
            width: 32px;
            height: 32px;
            border-radius: var(--radius-sm);
            background: rgba(255, 69, 0, 0.1);
            border: 1px solid rgba(255, 69, 0, 0.25);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--brand-primary);
        }

        .panel-icon-box.preview-icon {
            background: rgba(59, 130, 246, 0.1);
            border-color: rgba(59, 130, 246, 0.25);
            color: var(--accent-blue);
        }

        .panel-title {
            font-size: 1.08rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.01em;
        }

        /* Form Grid */
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.15rem;
        }

        @media (max-width: 720px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            margin-bottom: 0.1rem;
            position: relative;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        label {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.74rem;
            font-weight: 700;
            color: var(--text-secondary);
            margin-bottom: 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        /* Inputs & Selects */
        .input-wrapper {
            position: relative;
        }

        select, input[type="text"], input[type="number"] {
            width: 100%;
            padding: 0.72rem 0.95rem;
            background: var(--bg-input);
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            font-size: 0.88rem;
            font-weight: 500;
            outline: none;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3);
        }

        select {
            appearance: none;
            -webkit-appearance: none;
            padding-right: 2.4rem;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 0.85rem center;
            cursor: pointer;
        }

        select:focus, input[type="text"]:focus, input[type="number"]:focus {
            border-color: var(--brand-primary);
            box-shadow: 0 0 0 3px rgba(255, 69, 0, 0.18), inset 0 1px 2px rgba(0, 0, 0, 0.3);
            background: #11141c;
        }

        select:hover, input[type="text"]:hover, input[type="number"]:hover {
            border-color: rgba(255, 255, 255, 0.22);
        }

        option {
            background: #161922;
            color: #f8fafc;
            padding: 8px 12px;
        }

        /* Batch Mode Box */
        .batch-card {
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-lg);
            padding: 1.15rem;
            transition: border-color 0.2s ease;
        }

        .batch-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .batch-toggle-label {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            cursor: pointer;
            text-transform: none;
            font-size: 0.92rem;
            font-weight: 600;
            color: #ffffff;
            margin: 0;
            letter-spacing: normal;
        }

        .custom-switch {
            position: relative;
            display: inline-block;
            width: 40px;
            height: 22px;
        }

        .custom-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .slider-round {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: #27272a;
            transition: .25s cubic-bezier(0.16, 1, 0.3, 1);
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }

        .slider-round:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 2px;
            bottom: 2px;
            background-color: #ffffff;
            transition: .25s cubic-bezier(0.16, 1, 0.3, 1);
            border-radius: 50%;
            box-shadow: 0 1px 4px rgba(0,0,0,0.4);
        }

        input:checked + .slider-round {
            background-color: var(--brand-primary);
            border-color: var(--brand-primary);
        }

        input:checked + .slider-round:before {
            transform: translateX(18px);
        }

        .batch-sub-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            gap: 0.45rem;
            background: var(--bg-input);
            padding: 0.75rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-subtle);
            max-height: 180px;
            overflow-y: auto;
        }

        .sub-chip-label {
            text-transform: none;
            color: var(--text-primary);
            font-size: 0.8rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            cursor: pointer;
            padding: 0.35rem 0.55rem;
            border-radius: var(--radius-sm);
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid transparent;
            transition: all 0.15s ease;
            margin: 0;
            letter-spacing: normal;
        }

        .sub-chip-label:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: var(--border-subtle);
        }

        .sub-checkbox {
            accent-color: var(--brand-primary);
            width: 14px;
            height: 14px;
            cursor: pointer;
        }

        .mini-btn {
            padding: 0.25rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-secondary);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .mini-btn:hover {
            background: rgba(255, 255, 255, 0.12);
            color: #ffffff;
        }

        /* Range Slider */
        .range-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: #27272a;
            border-radius: 999px;
            outline: none;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.5);
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--brand-primary);
            cursor: pointer;
            box-shadow: 0 0 10px rgba(255, 69, 0, 0.6);
            transition: transform 0.1s ease;
            border: 2px solid #ffffff;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.15);
        }

        .badge-pill {
            background: rgba(255, 69, 0, 0.12);
            color: #ff6b35;
            border: 1px solid rgba(255, 69, 0, 0.3);
            font-weight: 700;
            font-size: 0.75rem;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            letter-spacing: 0.02em;
            white-space: nowrap;
        }

        /* Buttons & Actions */
        .btn-group {
            display: grid;
            grid-template-columns: 1fr 1.25fr;
            gap: 0.9rem;
            margin-top: 1.5rem;
            grid-column: 1 / -1;
        }

        button.main-action-btn {
            padding: 0.95rem 1.4rem;
            border: none;
            border-radius: var(--radius-md);
            font-weight: 700;
            font-size: 0.92rem;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.55rem;
            letter-spacing: -0.01em;
        }

        .btn-test {
            background: var(--bg-surface-elevated);
            color: #e2e8f0;
            border: 1px solid var(--border-strong);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }

        .btn-test:hover {
            background: var(--bg-surface-hover);
            border-color: rgba(255, 255, 255, 0.3);
            color: #ffffff;
            transform: translateY(-1px);
        }

        .btn-upload {
            background: var(--brand-gradient);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: var(--shadow-btn);
            position: relative;
            overflow: hidden;
        }

        .btn-upload:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px 0 rgba(255, 69, 0, 0.5);
            background: linear-gradient(135deg, #ff5511 0%, #ff7b45 100%);
        }

        .btn-upload:active, .btn-test:active {
            transform: translateY(1px);
        }

        button:disabled {
            opacity: 0.45;
            cursor: not-allowed !important;
            transform: none !important;
            box-shadow: none !important;
        }

        /* Status & Progress Bar */
        .status-box {
            background: var(--bg-surface-elevated);
            border-radius: var(--radius-md);
            padding: 1.1rem 1.25rem;
            margin-top: 1.25rem;
            border: 1px solid var(--border-strong);
            grid-column: 1 / -1;
            display: none;
            position: relative;
            overflow: hidden;
        }

        .status-box::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #ff4500, #3b82f6, #8b5cf6, #ff4500);
            background-size: 200% 100%;
            animation: move-gradient 2s linear infinite;
        }

        @keyframes move-gradient {
            0% { background-position: 0% 0%; }
            100% { background-position: 200% 0%; }
        }

        .status-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-weight: 700;
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
            color: #ffffff;
        }

        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-top-color: var(--brand-primary);
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* Preview Panel */
        .video-preview-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 480px;
            border-radius: var(--radius-lg);
            background: #08090d;
            border: 1px dashed var(--border-strong);
            padding: 1.25rem;
            position: relative;
        }

        video {
            max-height: 520px;
            max-width: 100%;
            border-radius: var(--radius-md);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.1);
            background: #000000;
        }

        .custom-player-bar {
            display: none;
            width: 100%;
            max-width: 320px;
            margin-top: 0.85rem;
            background: rgba(22, 25, 34, 0.95);
            backdrop-filter: blur(10px);
            padding: 0.65rem 0.95rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-strong);
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }

        /* Metadata & Info Cards */
        .metadata-card {
            margin-top: 1.25rem;
            width: 100%;
            background: var(--bg-surface-elevated);
            border-radius: var(--radius-md);
            padding: 1rem 1.15rem;
            border: 1px solid var(--border-subtle);
        }

        .meta-title {
            font-weight: 700;
            font-size: 0.95rem;
            color: #ffffff;
            margin-bottom: 0.4rem;
            line-height: 1.4;
        }

        .meta-desc {
            font-size: 0.8rem;
            color: var(--text-secondary);
            white-space: pre-wrap;
            line-height: 1.5;
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0, 0, 0, 0.25);
            padding: 0.6rem 0.8rem;
            border-radius: var(--radius-sm);
            border: 1px solid rgba(255, 255, 255, 0.04);
        }

        /* History Library */
        .history-list {
            margin-top: 0.75rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.65rem;
            max-height: 240px;
            overflow-y: auto;
            padding-right: 0.25rem;
        }

        @media (max-width: 600px) {
            .history-list {
                grid-template-columns: 1fr;
            }
        }

        .history-item {
            padding: 0.7rem 0.85rem;
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .history-item:hover {
            border-color: rgba(255, 255, 255, 0.2);
            background: var(--bg-surface-hover);
            transform: translateY(-1px);
        }

        .history-meta-sub {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--brand-primary);
            margin-bottom: 0.15rem;
        }

        .history-meta-title {
            font-size: 0.78rem;
            color: var(--text-secondary);
            max-width: 140px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .history-play-btn {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-tertiary);
            display: flex;
            align-items: center;
            gap: 0.25rem;
            background: rgba(255, 255, 255, 0.05);
            padding: 0.3rem 0.55rem;
            border-radius: var(--radius-sm);
        }

        .history-item:hover .history-play-btn {
            color: #ffffff;
            background: var(--brand-primary);
        }

        /* Autocomplete dropdowns */
        .autocomplete-dropdown {
            position: absolute;
            z-index: 100;
            left: 0;
            right: 0;
            top: calc(100% + 4px);
            background: #161922;
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-md);
            max-height: 220px;
            overflow-y: auto;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7);
        }

        /* YouTube API Setup Modal */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            animation: modalFadeIn 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .modal-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-xl);
            padding: 1.75rem;
            width: 100%;
            max-width: 580px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px var(--border-subtle);
            position: relative;
            max-height: 90vh;
            overflow-y: auto;
        }

        .tab-btn.active {
            background: rgba(255, 69, 0, 0.15) !important;
            color: var(--brand-primary) !important;
            border-color: rgba(255, 69, 0, 0.4) !important;
        }

        @keyframes modalFadeIn {
            from { opacity: 0; transform: scale(0.96); }
            to { opacity: 1; transform: scale(1); }
        }

        /* AI OP Mode & Components */
        .studio-mode-switcher {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            background: rgba(22, 25, 34, 0.7);
            padding: 0.35rem;
            border-radius: 999px;
            border: 1px solid var(--border-subtle);
            max-width: 540px;
            margin-left: auto;
            margin-right: auto;
        }
        .studio-mode-tab {
            flex: 1;
            padding: 0.6rem 1.25rem;
            border-radius: 999px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-size: 0.88rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        .studio-mode-tab.active {
            background: var(--brand-primary);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(255, 69, 0, 0.35);
        }
        .studio-mode-tab:hover:not(.active) {
            color: #ffffff;
            background: rgba(255, 255, 255, 0.05);
        }
        .ai-op-preview-box {
            background: #0b0d13;
            border: 1px solid rgba(255, 69, 0, 0.3);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            margin-top: 1.25rem;
        }
        .ai-op-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .status-badge-submitted { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
        .status-badge-waiting { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .status-badge-ready { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .status-badge-rendered { background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
    </style>
</head>
<body>
    <header>
        <div class="brand-container">
            <div class="brand-logo">
                <svg viewBox="0 0 24 24" width="36" height="36">
                    <circle cx="12" cy="12" r="12" fill="#FF4500"/>
                    <path fill="#FFFFFF" d="M17.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.56 12 8 12.56 8 13.25c0 .688.56 1.25 1.25 1.25.688 0 1.25-.562 1.25-1.25 0-.69-.562-1.25-1.25-1.25zm5.5 0c-.688 0-1.25.56-1.25 1.25 0 .688.562 1.25 1.25 1.25.69 0 1.25-.562 1.25-1.25 0-.69-.56-1.25-1.25-1.25zm-5.465 3.99a.577.577 0 0 0-.41.983c.77.77 1.83 1.15 2.875 1.15 1.044 0 2.104-.38 2.874-1.15a.577.577 0 0 0-.82-.816c-.552.55-1.318.82-2.054.82-.736 0-1.502-.27-2.054-.82a.574.574 0 0 0-.411-.167z"/>
                </svg>
            </div>
            <div class="brand-text">
                <div class="brand-title">Reddit Shorts Studio</div>
                <div class="brand-subtitle">AI Video Generation Engine</div>
            </div>
        </div>

        <div class="header-actions">
            <div id="gpuBadge" class="gpu-badge">
                <div class="status-dot"></div>
                <span id="gpuBadgeText">GPU Hardware Acceleration</span>
            </div>
            
            <button id="redditAuthBtn" class="btn-reddit-account" onclick="openRedditAuthModal()" style="display:inline-flex; align-items:center; gap:0.45rem; background:rgba(255, 69, 0, 0.12); border:1px solid rgba(255, 69, 0, 0.35); color:#ff4500; padding:0.45rem 0.9rem; border-radius:999px; font-size:0.8rem; font-weight:700; cursor:pointer; transition:all 0.2s ease;">
                <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.56 12 8 12.56 8 13.25c0 .688.56 1.25 1.25 1.25.688 0 1.25-.562 1.25-1.25 0-.69-.562-1.25-1.25-1.25zm5.5 0c-.688 0-1.25.56-1.25 1.25 0 .688.562 1.25 1.25 1.25.69 0 1.25-.562 1.25-1.25 0-.69-.56-1.25-1.25-1.25zm-5.465 3.99a.577.577 0 0 0-.41.983c.77.77 1.83 1.15 2.875 1.15 1.044 0 2.104-.38 2.874-1.15a.577.577 0 0 0-.82-.816c-.552.55-1.318.82-2.054.82-.736 0-1.502-.27-2.054-.82a.574.574 0 0 0-.411-.167z"/></svg>
                <span id="redditAuthBtnText">Reddit Bot: Setup</span>
            </button>

            <button id="ytAuthBtn" class="btn-yt-login" onclick="openYouTubeAuthModal()">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>
                <span>YouTube API</span>
            </button>

            <div id="ytChannelBadge" class="yt-channel-badge-container">
                <div class="yt-connected-pill">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
                    <span id="ytChannelName">Connected</span>
                </div>
                <button type="button" class="btn-switch-account" onclick="switchYouTubeAccount()">Switch Account</button>
            </div>
        </div>
    </header>

    <main>
        <!-- Studio Mode Switcher -->
        <div class="studio-mode-switcher" style="grid-column: 1 / -1;">
            <button type="button" id="modeBtnScraper" class="studio-mode-tab active" onclick="switchStudioMode('scraper')">
                🎬 Scraper &amp; Render Studio
            </button>
            <button type="button" id="modeBtnAiOp" class="studio-mode-tab" onclick="switchStudioMode('ai-op')">
                🤖 AI OP (Auto-Poster &amp; Meatbag Watcher)
            </button>
        </div>

        <!-- Scraper Control Panel -->
        <div class="panel" id="panelScraperConfig">
            <div class="panel-header">
                <div class="panel-title-wrapper">
                    <div class="panel-icon-box">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                    </div>
                    <div class="panel-title">Video Configuration</div>
                </div>
            </div>

            <div class="form-grid">
                <!-- Batch Mode Card (Full Width) -->
                <div class="form-group full-width batch-card">
                    <div class="batch-card-header">
                        <label class="batch-toggle-label">
                            <label class="custom-switch" style="margin: 0;">
                                <input type="checkbox" id="batchModeCheckbox" onchange="toggleBatchModeUI()">
                                <span class="slider-round"></span>
                            </label>
                            <span>Enable Batch Queue Mode <span style="font-size:0.75rem; color:var(--text-tertiary); font-weight:400;">(Process & Upload Multiple Shorts)</span></span>
                        </label>
                    </div>

                    <div id="batchControlsBox" style="display: none; margin-top: 1.15rem; flex-direction: column; gap: 0.85rem; border-top: 1px solid var(--border-subtle); padding-top: 1rem;">
                        <div style="display:flex; align-items:center; justify-content:space-between;">
                            <label style="margin:0; font-weight:700; color:var(--text-secondary);">Target Subreddit Pool</label>
                            <div style="display:flex; gap:0.4rem;">
                                <button type="button" class="mini-btn" onclick="selectAllSubs(true)">Select All</button>
                                <button type="button" class="mini-btn" onclick="selectAllSubs(false)">Clear All</button>
                            </div>
                        </div>

                        <div id="subChecklist" class="batch-sub-grid">
                            <label class="sub-chip-label"><input type="checkbox" class="sub-checkbox" value="AskReddit" checked onchange="updateBatchButtonText()"> r/AskReddit</label>
                            <label class="sub-chip-label"><input type="checkbox" class="sub-checkbox" value="funny" checked onchange="updateBatchButtonText()"> r/funny</label>
                            <label class="sub-chip-label"><input type="checkbox" class="sub-checkbox" value="memes" checked onchange="updateBatchButtonText()"> r/memes</label>
                            <label class="sub-chip-label"><input type="checkbox" class="sub-checkbox" value="tifu" onchange="updateBatchButtonText()"> r/tifu</label>
                            <label class="sub-chip-label"><input type="checkbox" class="sub-checkbox" value="showerthoughts" onchange="updateBatchButtonText()"> r/showerthoughts</label>
                            <label class="sub-chip-label"><input type="checkbox" class="sub-checkbox" value="wholesomememes" onchange="updateBatchButtonText()"> r/wholesomememes</label>
                            <label class="sub-chip-label"><input type="checkbox" class="sub-checkbox" value="AskRedditOver30" onchange="updateBatchButtonText()"> r/AskRedditOver30</label>
                        </div>

                        <div style="margin-top: 0.25rem; display: flex; gap: 0.5rem; position: relative;">
                            <input type="text" id="batchCustomSubInput" placeholder="Add custom subreddit (e.g. vtubers, pcgaming)" oninput="onBatchCustomSubInput()" autocomplete="off">
                            <button type="button" onclick="addCustomSubFromBatchInput()" class="mini-btn" style="padding: 0.5rem 1rem; font-weight:700; background:rgba(255, 69, 0, 0.15); color:var(--brand-primary); border-color:rgba(255, 69, 0, 0.3); border-radius:var(--radius-md); white-space:nowrap;">+ Add to Queue</button>
                            <div id="batchSubAutocompleteBox" class="autocomplete-dropdown" style="display:none;"></div>
                        </div>

                        <div style="margin-top: 0.25rem;">
                            <label for="batchCountInput">Total Videos in Batch Queue</label>
                            <input type="number" id="batchCountInput" value="5" min="1" max="50" oninput="updateBatchButtonText()">
                        </div>
                    </div>
                </div>

                <!-- Single Subreddit Selection -->
                <div class="form-group" id="singleSubGroup">
                    <label for="subredditSelect">Select Subreddit</label>
                    <select id="subredditSelect" onchange="toggleCustomSubreddit()">
                        <option value="AskReddit" selected>r/AskReddit</option>
                        <option value="funny">r/funny</option>
                        <option value="memes">r/memes</option>
                        <option value="tifu">r/tifu</option>
                        <option value="showerthoughts">r/showerthoughts</option>
                        <option value="wholesomememes">r/wholesomememes</option>
                        <option value="AskRedditOver30">r/AskRedditOver30</option>
                        <option value="custom">-- Custom Subreddit / URL --</option>
                    </select>
                    <div class="custom-sub-box" id="customSubBox" style="display: none; position: relative; margin-top: 0.5rem;">
                        <input type="text" id="customSubInput" placeholder="Type subreddit or paste Reddit URL..." oninput="onCustomSubredditInput()" autocomplete="off">
                        <div id="subAutocompleteBox" class="autocomplete-dropdown" style="display:none;"></div>
                    </div>
                </div>

                <!-- Comments Slider -->
                <div class="form-group">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.45rem;">
                        <label for="commentsSlider" style="margin-bottom:0;">Comments Per Video (1 to 5)</label>
                        <span id="commentsValBadge" class="badge-pill">2 Comments</span>
                    </div>
                    <div class="range-container" style="height: 42px;">
                        <input type="range" id="commentsSlider" min="1" max="5" value="2" step="1" oninput="document.getElementById('commentsValBadge').innerText = this.value + (this.value == 1 ? ' Comment' : ' Comments')">
                    </div>
                </div>

                <!-- TTS Voice -->
                <div class="form-group">
                    <label for="voiceSelect">Neural TTS Voice</label>
                    <select id="voiceSelect">
                        <option value="en-US-ChristopherNeural" selected>Male (Christopher)</option>
                        <option value="en-US-GuyNeural">Male (Guy)</option>
                        <option value="en-US-AnaNeural">Female (Ana)</option>
                        <option value="en-US-JennyNeural">Female (Jenny)</option>
                    </select>
                </div>

                <!-- Music Style -->
                <div class="form-group">
                    <label for="musicSelect">Background Soundtrack</label>
                    <select id="musicSelect">
                        <option value="lofi" selected>Lofi Chill (Meta MusicGen AI)</option>
                        <option value="funny">Upbeat Funny (Meta MusicGen AI)</option>
                        <option value="dramatic">Dramatic Story (Meta MusicGen AI)</option>
                        <option value="acoustic">Happy Acoustic (Meta MusicGen AI)</option>
                        <option value="synthwave">Synthwave Pulse (Meta MusicGen AI)</option>
                        <option value="dsp">Lofi Beat (Fast Synthetic)</option>
                        <option value="none">No Music (Voiceover Only)</option>
                    </select>
                </div>

                <!-- On-Screen Subtitles -->
                <div class="form-group">
                    <label for="subtitleStyleSelect">On-Screen Dynamic Captions</label>
                    <select id="subtitleStyleSelect">
                        <option value="yellow_pill" selected>Dynamic Yellow Pill (Viral Shorts)</option>
                        <option value="cyan_glow">Neon Cyan Glow</option>
                        <option value="minimal_white">Minimalist White</option>
                        <option value="none">No Captions</option>
                    </select>
                </div>

                <!-- YouTube Visibility -->
                <div class="form-group">
                    <label for="privacySelect">YouTube Visibility</label>
                    <select id="privacySelect">
                        <option value="unlisted" selected>Unlisted (Recommended for Testing)</option>
                        <option value="public">Public (Published Live to YouTube)</option>
                        <option value="private">Private (Restricted to Account Owner)</option>
                    </select>
                </div>

                <!-- Background Video -->
                <div class="form-group full-width">
                    <label for="bgSelect">Background Video Asset</label>
                    <select id="bgSelect" onchange="onBackgroundSelectChange()">
                        <option value="" disabled selected>Loading background assets...</option>
                    </select>
                </div>

                <div class="form-group full-width" id="bgUploadBox" style="display: none;">
                    <input type="file" id="bgFileInput" accept="video/mp4,video/webm,video/quicktime,image/webp,.webp" onchange="uploadBackgroundVideo(this)" style="display: none;">
                    <button type="button" onclick="document.getElementById('bgFileInput').click()" style="width:100%; background:var(--bg-surface-elevated); border:1px dashed var(--border-strong); color:var(--text-secondary); padding:0.85rem; border-radius:var(--radius-md); font-weight:600; font-size:0.85rem; cursor:pointer;">
                        Select File to Upload & Auto-Crop to 9:16
                    </button>
                    <div id="bgStatus" style="font-size:0.8rem; color:#4ade80; margin-top:0.4rem; display:none;"></div>
                </div>

                <div class="form-group full-width" id="bgYoutubeBox" style="display: none;">
                    <div style="display:flex; gap:0.5rem;">
                        <input type="text" id="bgYoutubeUrlInput" placeholder="Paste YouTube Video or Shorts URL (e.g. https://www.youtube.com/watch?v=...)" style="flex:1;">
                        <button type="button" onclick="downloadYoutubeBackground()" style="background:#cc0000; color:white; border:none; padding:0.75rem 1.15rem; border-radius:var(--radius-md); font-weight:700; font-size:0.85rem; cursor:pointer; white-space:nowrap;">
                            Download & Crop
                        </button>
                    </div>
                    <div id="bgYtStatus" style="font-size:0.82rem; color:#4ade80; margin-top:0.5rem; display:none;"></div>
                </div>

                <!-- Action Buttons -->
                <div class="btn-group">
                    <button class="main-action-btn btn-test" id="btnTest" onclick="startPipeline('test')">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.31"/><path d="M14 9.3V2"/><path d="M8.5 2h7"/><path d="M14 9.3a6.5 6.5 0 1 1-4 0"/></svg>
                        <span>Test (Dry Run)</span>
                    </button>
                    <button class="main-action-btn btn-upload" id="btnUpload" onclick="startPipeline('upload')">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                        <span>Generate & Upload</span>
                    </button>
                </div>

                <div class="status-box" id="statusBox">
                    <div class="status-header">
                        <span id="statusStage">Processing Pipeline...</span>
                        <div class="spinner" id="statusSpinner"></div>
                    </div>
                    <div id="statusMessage" style="font-size: 0.82rem; color: var(--text-secondary); font-family: 'JetBrains Mono', monospace;">Executing job...</div>
                </div>
            </div>
        </div>

        <!-- Preview & Output Panel -->
        <div class="panel" id="panelScraperPreview">
            <div class="panel-header">
                <div class="panel-title-wrapper">
                    <div class="panel-icon-box preview-icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
                    </div>
                    <div class="panel-title">Studio Preview & Output</div>
                </div>
            </div>

            <div class="video-preview-container" id="previewContainer">
                <p id="placeholderText" style="color: var(--text-tertiary); font-size: 0.88rem; text-align: center; max-width: 280px; line-height: 1.5;">Select a subreddit configuration and click <b>Test</b> or <b>Generate & Upload</b> to render the short.</p>
                <video id="videoPlayer" controls playsinline style="display: none; cursor: pointer;" onclick="togglePlayPause()"></video>

                <!-- Custom Control Bar -->
                <div id="customVideoControls" class="custom-player-bar">
                    <div style="display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;">
                        <button type="button" id="btnPlayPause" onclick="togglePlayPause()" class="mini-btn" style="background:var(--brand-primary); color:#ffffff; border:none; font-weight:700;">Pause</button>
                        <button type="button" id="btnMute" onclick="toggleMute()" class="mini-btn">Sound</button>
                        <span id="videoTimeText" style="font-size: 0.75rem; font-weight: 600; color: var(--text-secondary); font-family: 'JetBrains Mono', monospace;">0:00 / 0:00</span>
                    </div>
                    <input type="range" id="videoScrubber" min="0" max="100" value="0" step="0.1" oninput="seekVideo(this.value)" style="width: 100%; margin-top: 0.5rem;">
                </div>
            </div>

            <div id="batchPlaylistContainer" style="display: none; margin-top: 1rem;"></div>

            <div class="metadata-card" id="metadataCard" style="display: none;">
                <div class="meta-title" id="metaTitle"></div>
                <div class="meta-desc" id="metaDesc"></div>
            </div>

            <div style="margin-top: 1.4rem;">
                <label style="color: var(--text-secondary);">Recent Generated Outputs</label>
                <div class="history-list" id="historyList"></div>
            </div>
        </div>

        <!-- ==================== AI OP STUDIO PANELS ==================== -->
        <!-- AI OP Creator Panel -->
        <div class="panel" id="panelAiOpCreator" style="display:none;">
            <div class="panel-header">
                <div class="panel-title-wrapper">
                    <div class="panel-icon-box" style="background:rgba(255, 69, 0, 0.12); color:#ff4500; border-color:rgba(255, 69, 0, 0.3);">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
                    </div>
                    <div>
                        <div class="panel-title">AI OP Post Ideation &amp; Poster</div>
                        <div style="font-size:0.75rem; color:var(--text-tertiary);">Craft viral posts engineered to bait human meatbags for funny comments</div>
                    </div>
                </div>
            </div>

            <div class="form-grid" style="margin-top:1.25rem;">
                <div class="form-group">
                    <label for="aiOpSubreddit">Target Subreddit</label>
                    <select id="aiOpSubreddit" onchange="onAiOpSubredditChange()">
                        <option value="AskReddit" selected>r/AskReddit</option>
                        <option value="Showerthoughts">r/Showerthoughts</option>
                        <option value="unpopularopinion">r/unpopularopinion</option>
                        <option value="AmItheAsshole">r/AmItheAsshole</option>
                        <option value="tifu">r/tifu</option>
                        <option value="NoStupidQuestions">r/NoStupidQuestions</option>
                        <option value="mildlyinfuriating">r/mildlyinfuriating</option>
                        <option value="custom">-- Custom Subreddit --</option>
                    </select>
                    <input type="text" id="aiOpCustomSub" placeholder="e.g. funny, memes, CasualConversation" style="display:none; margin-top:0.45rem;">
                </div>

                <div class="form-group">
                    <label for="aiOpStyle">Persona / Tone Style</label>
                    <select id="aiOpStyle">
                        <option value="comedic" selected>Comedic &amp; Witty</option>
                        <option value="absurd">Absurd &amp; Surreal</option>
                        <option value="provocative">Spicy Hot Take</option>
                        <option value="story">Blunder / Dilemma Story</option>
                        <option value="thought-provoking">Thought-Provoking Epiphany</option>
                    </select>
                </div>

                <div class="form-group full-width">
                    <label for="aiOpTheme">Topic / Theme <span style="font-size:0.75rem; color:var(--text-tertiary); font-weight:400;">(Optional - leave empty for AI's choice)</span></label>
                    <input type="text" id="aiOpTheme" placeholder="e.g. 'Job interview red flags', 'Roommate horror stories', 'Superpowers with dumb caveats'">
                </div>

                <div class="form-group">
                    <label for="aiOpMinComments">Comments Target to Trigger Video</label>
                    <input type="number" id="aiOpMinComments" value="2" min="1" max="10">
                </div>

                <div class="form-group" style="display:flex; align-items:center; justify-content:space-between; background:var(--bg-surface-elevated); padding:0.75rem 1rem; border-radius:var(--radius-md); border:1px solid var(--border-subtle); margin-top:0.35rem;">
                    <label style="margin:0; font-weight:700; cursor:pointer;" for="aiOpDryRun">
                        Simulation / Dry-Run Mode
                        <div style="font-size:0.72rem; color:var(--text-tertiary); font-weight:400;">Simulate posting &amp; comments without live Reddit API calls</div>
                    </label>
                    <input type="checkbox" id="aiOpDryRun" style="width:1.2rem; height:1.2rem; cursor:pointer;">
                </div>
            </div>

            <div style="margin-top:1.25rem;">
                <button type="button" class="main-action-btn btn-upload" id="btnAiOpGenerate" onclick="generateAiOpPost()" style="width:100%;">
                    <span>✨ Generate Viral Post Idea</span>
                </button>
            </div>

            <!-- AI OP Live Editable Preview -->
            <div id="aiOpPreviewBox" class="ai-op-preview-box" style="display:none;">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem; border-bottom:1px solid var(--border-subtle); padding-bottom:0.5rem;">
                    <div style="display:flex; align-items:center; gap:0.45rem;">
                        <span class="badge-pill" id="prevSubBadge">r/AskReddit</span>
                        <span style="font-size:0.75rem; color:var(--text-tertiary);">Posted by u/<span id="prevAuthor">AI_OP_Bot</span></span>
                    </div>
                    <span id="prevModelPill" class="badge-pill" style="background:rgba(59, 130, 246, 0.15); color:#60a5fa; border:none; font-size:0.7rem;">Ollama Gemma</span>
                </div>

                <div style="margin-bottom:0.75rem;">
                    <label style="font-size:0.75rem; color:var(--text-secondary);">Post Title (Editable)</label>
                    <input type="text" id="prevTitleInput" style="font-weight:700; font-size:0.95rem; color:#ffffff; background:var(--bg-input);">
                </div>

                <div id="prevBodyGroup" style="margin-bottom:0.75rem;">
                    <label style="font-size:0.75rem; color:var(--text-secondary);">Post Body / Selftext (Editable)</label>
                    <textarea id="prevBodyInput" rows="3" style="width:100%; padding:0.65rem; background:var(--bg-input); border:1px solid var(--border-strong); border-radius:var(--radius-md); color:#ffffff; font-size:0.82rem; resize:vertical;"></textarea>
                </div>

                <div id="prevRationaleBox" style="font-size:0.78rem; color:#93c5fd; background:rgba(59, 130, 246, 0.08); border:1px solid rgba(59, 130, 246, 0.2); padding:0.5rem 0.75rem; border-radius:var(--radius-md); margin-bottom:0.85rem;">
                    💡 <b>Why meatbags will comment:</b> <span id="prevRationaleText">...</span>
                </div>

                <button type="button" class="main-action-btn" id="btnAiOpSubmit" onclick="submitAiOpPost()" style="width:100%; background:linear-gradient(135deg, #10b981 0%, #059669 100%); color:#ffffff;">
                    <span>🚀 Submit Post to Reddit &amp; Start Watching</span>
                </button>
            </div>
        </div>

        <!-- AI OP Tracker Panel -->
        <div class="panel" id="panelAiOpTracker" style="display:none;">
            <div class="panel-header" style="display:flex; justify-content:space-between; align-items:center;">
                <div class="panel-title-wrapper">
                    <div class="panel-icon-box" style="background:rgba(59, 130, 246, 0.12); color:#3b82f6; border-color:rgba(59, 130, 246, 0.3);">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    </div>
                    <div>
                        <div class="panel-title">Tracked AI Posts &amp; Meatbag Monitor</div>
                        <div style="font-size:0.75rem; color:var(--text-tertiary);">Track human comments and auto-trigger video rendering</div>
                    </div>
                </div>
                <button type="button" class="mini-btn" onclick="loadAiOpPosts()" style="padding:0.4rem 0.8rem;">🔄 Refresh</button>
            </div>

            <div id="aiOpPostsContainer" style="margin-top:1rem; max-height:580px; overflow-y:auto;">
                <div style="text-align:center; padding:2rem 1rem; color:var(--text-tertiary); font-size:0.85rem;">
                    No active AI OP posts yet. Generate and submit a post to begin tracking!
                </div>
            </div>
        </div>
    </main>

    <script>
        let pollInterval = null;
        let uploadedBgPath = null;

        function toggleBatchModeUI() {
            const cb = document.getElementById('batchModeCheckbox');
            const batchBox = document.getElementById('batchControlsBox');
            const singleGroup = document.getElementById('singleSubGroup');
            if (!cb || !batchBox || !singleGroup) return;

            if (cb.checked) {
                batchBox.style.display = 'flex';
                singleGroup.style.display = 'none';
            } else {
                batchBox.style.display = 'none';
                singleGroup.style.display = 'block';
            }
            updateBatchButtonText();
        }

        function selectAllSubs(checked) {
            const checkboxes = document.querySelectorAll('.sub-checkbox');
            if (checkboxes) {
                checkboxes.forEach(function(cb) { cb.checked = checked; });
            }
            updateBatchButtonText();
        }

        function updateBatchButtonText() {
            const cb = document.getElementById('batchModeCheckbox');
            const btnTest = document.getElementById('btnTest');
            const btnUpload = document.getElementById('btnUpload');
            const inputCount = document.getElementById('batchCountInput');
            if (!cb || !btnTest || !btnUpload) return;

            const isBatch = cb.checked;
            const count = inputCount ? (parseInt(inputCount.value) || 5) : 5;

            if (isBatch) {
                btnTest.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg><span>Batch Test (' + count + ' Videos)</span>';
                btnUpload.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>Batch Generate & Upload (' + count + ' Videos)</span>';
            } else {
                btnTest.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 2v7.31"/><path d="M14 9.3V2"/><path d="M8.5 2h7"/><path d="M14 9.3a6.5 6.5 0 1 1-4 0"/></svg><span>Test (Dry Run)</span>';
                btnUpload.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>Generate & Upload</span>';
            }
        }

        function getSelectedSubreddits() {
            const checkboxes = document.querySelectorAll('.sub-checkbox:checked');
            if (!checkboxes || checkboxes.length === 0) return ['AskReddit'];
            const list = [];
            checkboxes.forEach(function(cb) { list.push(cb.value); });
            return list.length > 0 ? list : ['AskReddit'];
        }

        async function loadBackgrounds(selectedPath) {
            try {
                const resp = await fetch('/api/backgrounds');
                const items = await resp.json();
                const select = document.getElementById('bgSelect');
                if (!select) return;
                select.innerHTML = '';

                items.forEach(function(item) {
                    const opt = document.createElement('option');
                    opt.value = item.file_path;
                    opt.textContent = item.display_name || item.filename;
                    select.appendChild(opt);
                });

                const ytOpt = document.createElement('option');
                ytOpt.value = 'youtube';
                ytOpt.textContent = '📥 Download from YouTube URL...';
                select.appendChild(ytOpt);

                const uploadOpt = document.createElement('option');
                uploadOpt.value = 'upload';
                uploadOpt.textContent = '➕ Upload Local Video Asset...';
                select.appendChild(uploadOpt);

                if (selectedPath) {
                    select.value = selectedPath;
                } else if (items.length > 0) {
                    select.value = items[0].file_path;
                }
                onBackgroundSelectChange();
            } catch (e) {
                console.error('Failed to load backgrounds:', e);
            }
        }

        function onBackgroundSelectChange() {
            const select = document.getElementById('bgSelect');
            const uploadBox = document.getElementById('bgUploadBox');
            const ytBox = document.getElementById('bgYoutubeBox');
            if (!select) return;
            if (select.value === 'upload') {
                if (uploadBox) uploadBox.style.display = 'block';
                if (ytBox) ytBox.style.display = 'none';
                uploadedBgPath = null;
            } else if (select.value === 'youtube') {
                if (ytBox) ytBox.style.display = 'block';
                if (uploadBox) uploadBox.style.display = 'none';
                uploadedBgPath = null;
            } else {
                if (uploadBox) uploadBox.style.display = 'none';
                if (ytBox) ytBox.style.display = 'none';
                uploadedBgPath = select.value;
            }
        }

        async function uploadBackgroundVideo(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const statusDiv = document.getElementById('bgStatus');
            if (statusDiv) {
                statusDiv.style.display = 'block';
                statusDiv.style.color = '#94a3b8';
                statusDiv.innerText = '⏳ Cropping ' + file.name + ' to 9:16 vertical...';
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                const resp = await fetch('/api/upload_background', {
                    method: 'POST',
                    body: formData
                });
                const data = await resp.json();
                if (data.status === 'success') {
                    uploadedBgPath = data.file_path;
                    if (statusDiv) {
                        statusDiv.style.color = '#4ade80';
                        statusDiv.innerText = data.cached ? '✓ Reusing saved background: ' + file.name : '✓ Cropped & saved: ' + file.name;
                    }
                    await loadBackgrounds(data.file_path);
                } else if (statusDiv) {
                    statusDiv.style.color = '#ef4444';
                    statusDiv.innerText = '❌ Failed to process background video';
                }
            } catch (e) {
                if (statusDiv) {
                    statusDiv.style.color = '#ef4444';
                    statusDiv.innerText = '❌ Upload error: ' + e.message;
                }
            }
        }

        async function downloadYoutubeBackground() {
            const input = document.getElementById('bgYoutubeUrlInput');
            const statusDiv = document.getElementById('bgYtStatus');
            if (!input || !input.value.trim()) {
                alert('Please enter a YouTube video URL.');
                return;
            }

            const url = input.value.trim();
            if (statusDiv) {
                statusDiv.style.display = 'block';
                statusDiv.style.color = '#94a3b8';
                statusDiv.innerText = '⏳ Downloading YouTube video & auto-cropping to 9:16 vertical...';
            }

            try {
                const resp = await fetch('/api/download_youtube_background', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url })
                });
                const data = await resp.json();
                if (resp.ok && data.status === 'success') {
                    uploadedBgPath = data.file_path;
                    if (statusDiv) {
                        statusDiv.style.color = '#4ade80';
                        statusDiv.innerText = data.cached ? '✓ Reusing saved background: ' + (data.title || data.filename) : '✓ Downloaded, cropped & saved: ' + (data.title || data.filename);
                    }
                    await loadBackgrounds(data.file_path);
                } else {
                    if (statusDiv) {
                        statusDiv.style.color = '#ef4444';
                        statusDiv.innerText = '❌ Failed: ' + (data.detail || 'Download error');
                    }
                }
            } catch (e) {
                if (statusDiv) {
                    statusDiv.style.color = '#ef4444';
                    statusDiv.innerText = '❌ Network error: ' + e.message;
                }
            }
        }

        let searchDebounceTimer = null;

        function onCustomSubredditInput() {
            const input = document.getElementById('customSubInput');
            const autoBox = document.getElementById('subAutocompleteBox');
            if (!input || !autoBox) return;

            const q = input.value.trim().replace(/^r\//i, '');
            if (q.length < 2) {
                autoBox.style.display = 'none';
                return;
            }

            clearTimeout(searchDebounceTimer);
            searchDebounceTimer = setTimeout(async function() {
                try {
                    const resp = await fetch('/api/search_subreddits?q=' + encodeURIComponent(q));
                    if (!resp.ok) return;
                    const items = await resp.json();

                    if (!items || items.length === 0) {
                        autoBox.style.display = 'none';
                        return;
                    }

                    autoBox.innerHTML = '';
                    items.forEach(function(item) {
                        const row = document.createElement('div');
                        row.style.cssText = 'padding:0.6rem 0.85rem; display:flex; justify-content:space-between; align-items:center; cursor:pointer; border-bottom:1px solid rgba(255,255,255,0.05); transition:background 0.15s ease;';
                        row.onmouseover = function() { row.style.background = 'rgba(255,255,255,0.08)'; };
                        row.onmouseout = function() { row.style.background = 'transparent'; };

                        let subsFormatted = item.subscribers ? (item.subscribers > 1000000 ? (item.subscribers/1000000).toFixed(1) + 'M' : Math.round(item.subscribers/1000) + 'k') + ' members' : '';
                        if (item.is_custom) {
                            subsFormatted = '<span style="color:#ff6b35; font-weight:700;">Custom</span>';
                        }
                        const nsfwBadge = item.over18 ? ' <span style="background:#ef4444; color:#fff; font-size:0.65rem; padding:1px 4px; border-radius:3px; font-weight:700;">18+</span>' : '';

                        row.innerHTML = '<div><strong style="color:#ff6b35; font-size:0.85rem;">r/' + item.name + '</strong>' + nsfwBadge + '<div style="font-size:0.75rem; color:#94a3b8; max-width:240px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">' + (item.title || '') + '</div></div><span style="font-size:0.75rem; color:#64748b; font-weight:600;">' + subsFormatted + '</span>';

                        row.onclick = function() {
                            input.value = item.name;
                            autoBox.style.display = 'none';
                        };
                        autoBox.appendChild(row);
                    });

                    autoBox.style.display = 'block';
                } catch (e) {
                    console.log('Autosearch error:', e);
                }
            }, 300);
        }

        function getSavedCustomSubreddits() {
            try {
                const raw = localStorage.getItem('reddit_custom_subreddits');
                return raw ? JSON.parse(raw) : [];
            } catch (e) {
                return [];
            }
        }

        function saveCustomSubreddit(subName) {
            if (!subName) return;
            const cleaned = subName.trim().replace(/^https?:\/\/[^\/]+/i, '').replace(/^\/r\//i, '').replace(/^r\//i, '').replace(/\/$/, '').trim();
            if (!cleaned) return;

            let list = getSavedCustomSubreddits();
            if (!list.some(function(s) { return s.toLowerCase() === cleaned.toLowerCase(); })) {
                list.push(cleaned);
                try {
                    localStorage.setItem('reddit_custom_subreddits', JSON.stringify(list));
                } catch (e) {}
            }
            renderCustomSubredditsInBatchList();
        }

        function removeCustomSubreddit(subName) {
            let list = getSavedCustomSubreddits();
            list = list.filter(function(s) { return s.toLowerCase() !== subName.toLowerCase(); });
            try {
                localStorage.setItem('reddit_custom_subreddits', JSON.stringify(list));
            } catch (e) {}
            renderCustomSubredditsInBatchList();
        }

        function renderCustomSubredditsInBatchList() {
            const checklist = document.getElementById('subChecklist');
            if (!checklist) return;

            const oldCustoms = checklist.querySelectorAll('.custom-sub-label');
            oldCustoms.forEach(function(el) { el.remove(); });

            const saved = getSavedCustomSubreddits();
            saved.forEach(function(sub) {
                const label = document.createElement('label');
                label.className = 'custom-sub-label sub-chip-label';
                label.style.cssText = 'color:#ff8555; background:rgba(255, 69, 0, 0.1); border-color:rgba(255, 69, 0, 0.25); justify-content:space-between;';

                const leftSpan = document.createElement('span');
                leftSpan.style.cssText = 'display:flex; align-items:center; gap:0.45rem; overflow:hidden; text-overflow:ellipsis;';
                
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.className = 'sub-checkbox';
                cb.value = sub;
                cb.checked = true;
                cb.onchange = updateBatchButtonText;

                leftSpan.appendChild(cb);
                leftSpan.appendChild(document.createTextNode('r/' + sub));

                const delBtn = document.createElement('span');
                delBtn.innerHTML = '&times;';
                delBtn.title = 'Remove';
                delBtn.style.cssText = 'color:#ef4444; font-weight:700; font-size:1rem; line-height:1; cursor:pointer; padding:0 3px;';
                delBtn.onclick = function(ev) {
                    ev.preventDefault();
                    ev.stopPropagation();
                    removeCustomSubreddit(sub);
                };

                label.appendChild(leftSpan);
                label.appendChild(delBtn);
                checklist.appendChild(label);
            });

            updateBatchButtonText();
        }

        function addCustomSubFromBatchInput() {
            const input = document.getElementById('batchCustomSubInput');
            if (!input || !input.value.trim()) return;
            const val = input.value.trim();
            saveCustomSubreddit(val);
            input.value = '';
            const autoBox = document.getElementById('batchSubAutocompleteBox');
            if (autoBox) autoBox.style.display = 'none';
        }

        let batchSearchDebounceTimer = null;
        function onBatchCustomSubInput() {
            const input = document.getElementById('batchCustomSubInput');
            const autoBox = document.getElementById('batchSubAutocompleteBox');
            if (!input || !autoBox) return;

            const q = input.value.trim().replace(/^r\//i, '');
            if (q.length < 2) {
                autoBox.style.display = 'none';
                return;
            }

            clearTimeout(batchSearchDebounceTimer);
            batchSearchDebounceTimer = setTimeout(async function() {
                try {
                    const resp = await fetch('/api/search_subreddits?q=' + encodeURIComponent(q));
                    if (!resp.ok) return;
                    const items = await resp.json();

                    if (!items || items.length === 0) {
                        autoBox.style.display = 'none';
                        return;
                    }

                    autoBox.innerHTML = '';
                    items.forEach(function(item) {
                        const row = document.createElement('div');
                        row.style.cssText = 'padding:0.55rem 0.8rem; display:flex; justify-content:space-between; align-items:center; cursor:pointer; border-bottom:1px solid rgba(255,255,255,0.05); transition:background 0.15s ease;';
                        row.onmouseover = function() { row.style.background = 'rgba(255,255,255,0.08)'; };
                        row.onmouseout = function() { row.style.background = 'transparent'; };

                        let subsFormatted = item.subscribers ? (item.subscribers > 1000000 ? (item.subscribers/1000000).toFixed(1) + 'M' : Math.round(item.subscribers/1000) + 'k') + ' members' : '';
                        if (item.is_custom) {
                            subsFormatted = '<span style="color:#ff6b35; font-weight:700;">Custom</span>';
                        }
                        const nsfwBadge = item.over18 ? ' <span style="background:#ef4444; color:#fff; font-size:0.65rem; padding:1px 4px; border-radius:3px; font-weight:700;">18+</span>' : '';

                        row.innerHTML = '<div><strong style="color:#ff6b35; font-size:0.85rem;">r/' + item.name + '</strong>' + nsfwBadge + '</div><span style="font-size:0.75rem; color:#64748b; font-weight:600;">' + subsFormatted + '</span>';

                        row.onclick = function() {
                            input.value = item.name;
                            autoBox.style.display = 'none';
                            addCustomSubFromBatchInput();
                        };
                        autoBox.appendChild(row);
                    });

                    autoBox.style.display = 'block';
                } catch (e) {
                    console.log('Batch autosearch error:', e);
                }
            }, 300);
        }

        document.addEventListener('click', function(e) {
            const autoBox = document.getElementById('subAutocompleteBox');
            const input = document.getElementById('customSubInput');
            if (autoBox && input && !autoBox.contains(e.target) && e.target !== input) {
                autoBox.style.display = 'none';
            }

            const batchAutoBox = document.getElementById('batchSubAutocompleteBox');
            const batchInput = document.getElementById('batchCustomSubInput');
            if (batchAutoBox && batchInput && !batchAutoBox.contains(e.target) && e.target !== batchInput) {
                batchAutoBox.style.display = 'none';
            }
        });

        function toggleCustomSubreddit() {
            const select = document.getElementById('subredditSelect');
            const customBox = document.getElementById('customSubBox');
            if (customBox && select) {
                customBox.style.display = select.value === 'custom' ? 'block' : 'none';
            }
        }

        function getSelectedSubreddit() {
            const select = document.getElementById('subredditSelect');
            if (select && select.value === 'custom') {
                const customVal = document.getElementById('customSubInput').value.trim();
                if (!customVal) return 'AskReddit';
                const m = customVal.match(/\/r\/([A-Za-z0-9_]+)/i) || customVal.match(/^r\/([A-Za-z0-9_]+)/i);
                let cleaned = 'AskReddit';
                if (m) cleaned = m[1];
                else cleaned = customVal.replace(/^https?:\/\/[^\/]+/i, '').replace(/^\/r\//i, '').replace(/^r\//i, '').replace(/\/$/, '').trim() || 'AskReddit';
                saveCustomSubreddit(cleaned);
                return cleaned;
            }
            return select ? select.value : 'AskReddit';
        }

        async function startPipeline(mode) {
            const cb = document.getElementById('batchModeCheckbox');
            const isBatch = cb ? cb.checked : false;
            const subreddit = getSelectedSubreddit();
            const subreddits = isBatch ? getSelectedSubreddits() : [subreddit];
            const countInput = document.getElementById('batchCountInput');
            const batch_count = (isBatch && countInput) ? (parseInt(countInput.value) || 5) : 1;

            const voiceSelect = document.getElementById('voiceSelect');
            const voice = voiceSelect ? voiceSelect.value : 'en-US-ChristopherNeural';

            const musicSelect = document.getElementById('musicSelect');
            const music_style = musicSelect ? musicSelect.value : 'lofi';

            const privacySelect = document.getElementById('privacySelect');
            const privacy_status = privacySelect ? privacySelect.value : 'unlisted';

            const commentsSlider = document.getElementById('commentsSlider');
            const max_comments = commentsSlider ? (parseInt(commentsSlider.value) || 2) : 2;

            const subtitleStyleSelect = document.getElementById('subtitleStyleSelect');
            const subStyle = subtitleStyleSelect ? subtitleStyleSelect.value : 'yellow_pill';
            const enableSubs = subStyle !== 'none';

            if (mode === 'upload') {
                const authResp = await fetch('/api/auth/status');
                const authData = await authResp.json();
                if (!authData.authenticated) {
                    openYouTubeAuthModal();
                    return;
                }
            }

            const btnTest = document.getElementById('btnTest');
            const btnUpload = document.getElementById('btnUpload');
            if (btnTest) btnTest.disabled = true;
            if (btnUpload) btnUpload.disabled = true;

            const statusBox = document.getElementById('statusBox');
            const statusStage = document.getElementById('statusStage');
            const statusMessage = document.getElementById('statusMessage');
            const statusSpinner = document.getElementById('statusSpinner');

            if (statusBox) statusBox.style.display = 'block';
            if (statusSpinner) statusSpinner.style.display = 'block';
            if (statusStage) statusStage.innerText = isBatch ? ('Initializing Batch Queue (' + batch_count + ' videos)...') : 'Initializing Pipeline...';
            if (statusMessage) statusMessage.innerText = isBatch ? ('Batching ' + batch_count + ' videos across ' + subreddits.length + ' subreddits...') : ('Running r/' + subreddit + ' (' + mode.toUpperCase() + ' mode)...');

            try {
                const resp = await fetch('/api/pipeline', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        subreddit: subreddit,
                        subreddits: subreddits,
                        voice: voice,
                        mode: mode,
                        use_gpu: true,
                        background_video: uploadedBgPath,
                        music_style: music_style,
                        privacy_status: privacy_status,
                        max_comments: max_comments,
                        is_batch: isBatch,
                        batch_count: batch_count,
                        enable_subtitles: enableSubs,
                        subtitle_style: subStyle
                    })
                });

                if (!resp.ok) {
                    const err = await resp.json();
                    alert(err.detail || 'Failed to start pipeline');
                    resetUI();
                    return;
                }

                pollInterval = setInterval(checkStatus, 2000);
            } catch (e) {
                alert('Connection error: ' + e.message);
                resetUI();
            }
        }

        async function checkStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();

                const statusStage = document.getElementById('statusStage');
                const statusMessage = document.getElementById('statusMessage');
                const statusSpinner = document.getElementById('statusSpinner');

                if (statusStage) statusStage.innerText = data.stage || 'Processing...';
                if (statusMessage) statusMessage.innerText = data.message || '';

                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    if (statusSpinner) statusSpinner.style.display = 'none';
                    if (statusStage) statusStage.innerText = 'Completed Successfully';
                    displayVideo(data.result);
                    resetUI(false);
                    loadHistory();
                } else if (data.status === 'error') {
                    clearInterval(pollInterval);
                    if (statusSpinner) statusSpinner.style.display = 'none';
                    if (statusStage) statusStage.innerText = 'Job Failed';
                    if (statusMessage) statusMessage.innerText = data.error || 'An error occurred';
                    resetUI(false);
                }
            } catch (e) {
                console.error(e);
            }
        }

        function setupCustomVideoControls() {
            const player = document.getElementById('videoPlayer');
            const btnPlayPause = document.getElementById('btnPlayPause');
            const timeText = document.getElementById('videoTimeText');
            const scrubber = document.getElementById('videoScrubber');

            if (!player || player._hasControlsSetup) return;
            player._hasControlsSetup = true;

            player.addEventListener('timeupdate', function() {
                if (player.duration) {
                    const pct = (player.currentTime / player.duration) * 100;
                    if (scrubber) scrubber.value = pct;
                    const curM = Math.floor(player.currentTime / 60);
                    const curS = Math.floor(player.currentTime % 60).toString().padStart(2, '0');
                    const durM = Math.floor(player.duration / 60);
                    const durS = Math.floor(player.duration % 60).toString().padStart(2, '0');
                    if (timeText) timeText.innerText = curM + ':' + curS + ' / ' + durM + ':' + durS;
                }
            });

            player.addEventListener('play', function() {
                if (btnPlayPause) btnPlayPause.innerText = 'Pause';
            });

            player.addEventListener('pause', function() {
                if (btnPlayPause) btnPlayPause.innerText = 'Play';
            });

            player.addEventListener('ended', function() {
                if (btnPlayPause) btnPlayPause.innerText = 'Play';
            });
        }

        function togglePlayPause() {
            const player = document.getElementById('videoPlayer');
            if (!player) return;
            if (player.paused) {
                player.play().catch(function(e) { console.log('Play error:', e); });
            } else {
                player.pause();
            }
        }

        function toggleMute() {
            const player = document.getElementById('videoPlayer');
            const btnMute = document.getElementById('btnMute');
            if (!player) return;
            player.muted = !player.muted;
            if (btnMute) btnMute.innerText = player.muted ? 'Muted' : 'Sound';
        }

        function seekVideo(val) {
            const player = document.getElementById('videoPlayer');
            if (!player || !player.duration) return;
            player.currentTime = (val / 100) * player.duration;
        }

        function displayVideo(result, forceReload = false) {
            if (!result) return;
            const player = document.getElementById('videoPlayer');
            const placeholder = document.getElementById('placeholderText');
            const metaCard = document.getElementById('metadataCard');
            const batchPlaylist = document.getElementById('batchPlaylistContainer');
            const customCtrl = document.getElementById('customVideoControls');

            if (placeholder) placeholder.style.display = 'none';
            if (player) player.style.display = 'block';
            if (customCtrl) customCtrl.style.display = 'block';
            setupCustomVideoControls();

            let videoUrl = '';
            if (result.video_url) {
                videoUrl = result.video_url;
            } else if (result.video_path) {
                const normPath = result.video_path.replaceAll(String.fromCharCode(92), '/');
                if (normPath.includes('/output/')) {
                    videoUrl = '/output/' + normPath.split('/output/')[1];
                } else {
                    const parts = normPath.split('/');
                    const fname = parts.pop();
                    const subfolder = parts.pop();
                    videoUrl = '/output/' + subfolder + '/' + fname;
                }
            }

            if (player && videoUrl) {
                player.controls = true;
                player.setAttribute('controls', 'controls');
                player.setAttribute('playsinline', 'playsinline');

                const absTarget = new URL(videoUrl, window.location.href).href;
                if (player.src !== absTarget || forceReload) {
                    player.src = videoUrl;
                    player.load();
                    const p = player.play();
                    if (p && p.catch) {
                        p.catch(function(e) { console.log('Autoplay handled:', e); });
                    }
                }
            }

            if (metaCard) metaCard.style.display = 'block';
            const mTitle = document.getElementById('metaTitle');
            const mDesc = document.getElementById('metaDesc');
            if (mTitle) mTitle.innerText = result.title || result.post_title || 'Generated Video';
            if (mDesc) mDesc.innerText = result.description || 'Saved in: ' + (result.output_folder || result.video_path);

            if (batchPlaylist) {
                if (result.is_batch && result.results && result.results.length > 0) {
                    batchPlaylist.style.display = 'block';
                    batchPlaylist.innerHTML = '<div style="font-weight:700; font-size:0.8rem; color:var(--text-secondary); margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.04em;">Batch Playlist (' + result.results.length + ' Videos)</div>';
                    const listDiv = document.createElement('div');
                    listDiv.style.cssText = 'display:flex; flex-wrap:wrap; gap:0.4rem;';

                    result.results.forEach(function(item, idx) {
                        if (item.status !== 'success') return;
                        const btn = document.createElement('button');
                        btn.type = 'button';
                        btn.className = 'mini-btn';
                        btn.style.cssText = 'padding:0.35rem 0.75rem; font-size:0.8rem; font-weight:600;';
                        btn.innerText = '#' + (idx + 1) + ' r/' + item.subreddit;
                        btn.onclick = function() {
                            displayVideo(item, true);
                        };
                        listDiv.appendChild(btn);
                    });
                    batchPlaylist.appendChild(listDiv);
                } else if (!result.is_batch_item) {
                    batchPlaylist.style.display = 'none';
                }
            }
        }

        function resetUI(hideStatus = true) {
            const btnTest = document.getElementById('btnTest');
            const btnUpload = document.getElementById('btnUpload');
            const statusBox = document.getElementById('statusBox');
            if (btnTest) btnTest.disabled = false;
            if (btnUpload) btnUpload.disabled = false;
            if (hideStatus && statusBox) {
                statusBox.style.display = 'none';
            }
        }

        async function loadHistory() {
            try {
                const resp = await fetch('/api/videos');
                const videos = await resp.json();
                const list = document.getElementById('historyList');
                if (!list) return;
                list.innerHTML = '';

                videos.forEach(function(v) {
                    const item = document.createElement('div');
                    item.className = 'history-item';
                    item.innerHTML = '<div><span class="history-meta-sub">r/' + (v.subreddit || 'Reddit') + '</span><div class="history-meta-title" title="' + (v.title || v.folder_name) + '">' + (v.title ? v.title.slice(0, 40) + '...' : v.folder_name) + '</div></div><span class="history-play-btn"><svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Play</span>';
                    item.onclick = function() {
                        displayVideo({
                            video_path: v.video_url,
                            output_folder: v.folder_name,
                            subreddit: v.subreddit,
                            title: v.title,
                            description: v.description
                        }, true);
                    };
                    list.appendChild(item);
                });
            } catch (e) {
                console.error(e);
            }
        }

        async function checkGpuStatus() {
            try {
                const resp = await fetch('/api/system/gpu');
                const data = await resp.json();
                const badge = document.getElementById('gpuBadge');
                const text = document.getElementById('gpuBadgeText');
                if (!badge || !text) return;

                if (data.has_gpu) {
                    badge.style.display = 'inline-flex';
                    badge.style.background = 'rgba(16, 185, 129, 0.08)';
                    badge.style.borderColor = 'rgba(16, 185, 129, 0.25)';
                    badge.style.color = '#34d399';
                    text.textContent = 'GPU Hardware Acceleration';
                    badge.title = (data.detail || data.vendor) + ' (' + data.encoder + ')';
                    const dot = badge.querySelector('.status-dot');
                    if (dot) {
                        dot.style.background = '#10b981';
                        dot.style.boxShadow = '0 0 8px #10b981';
                    }
                } else {
                    badge.style.display = 'inline-flex';
                    badge.style.background = 'rgba(148, 163, 184, 0.08)';
                    badge.style.borderColor = 'rgba(148, 163, 184, 0.25)';
                    badge.style.color = '#94a3b8';
                    text.textContent = 'CPU Software Encoding';
                    badge.title = 'No supported GPU encoder found; using CPU libx264';
                    const dot = badge.querySelector('.status-dot');
                    if (dot) {
                        dot.style.background = '#94a3b8';
                        dot.style.boxShadow = 'none';
                    }
                }
            } catch (e) {
                console.error('GPU check failed:', e);
            }
        }

        loadHistory();
        loadBackgrounds();
        checkYouTubeAuthStatus();
        checkGpuStatus();

        let activeModalTab = 'manual';

        function switchModalTab(tab) {
            activeModalTab = tab;
            const tabBtnManual = document.getElementById('tabBtnManual');
            const tabBtnUpload = document.getElementById('tabBtnUpload');
            const tabBtnJson = document.getElementById('tabBtnJson');
            const tabContentManual = document.getElementById('tabContentManual');
            const tabContentUpload = document.getElementById('tabContentUpload');
            const tabContentJson = document.getElementById('tabContentJson');

            if (tabBtnManual) tabBtnManual.className = tab === 'manual' ? 'mini-btn tab-btn active' : 'mini-btn tab-btn';
            if (tabBtnUpload) tabBtnUpload.className = tab === 'upload' ? 'mini-btn tab-btn active' : 'mini-btn tab-btn';
            if (tabBtnJson) tabBtnJson.className = tab === 'json' ? 'mini-btn tab-btn active' : 'mini-btn tab-btn';

            if (tabContentManual) tabContentManual.style.display = tab === 'manual' ? 'block' : 'none';
            if (tabContentUpload) tabContentUpload.style.display = tab === 'upload' ? 'block' : 'none';
            if (tabContentJson) tabContentJson.style.display = tab === 'json' ? 'block' : 'none';
        }

        async function openYouTubeAuthModal() {
            const modal = document.getElementById('ytAuthModal');
            const notice = document.getElementById('ytConfigNotice');
            if (!modal) return;

            try {
                const resp = await fetch('/api/auth/status');
                const data = await resp.json();
                if (notice) {
                    if (data.authenticated) {
                        notice.style.display = 'block';
                        notice.style.background = 'rgba(16, 185, 129, 0.1)';
                        notice.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                        notice.style.color = '#6ee7b7';
                        notice.innerHTML = '✓ Currently connected to YouTube channel: <b>' + (data.channel_name || 'Active Channel') + '</b>. You can re-authorize or enter new credentials below.';
                    } else if (data.has_client_secrets) {
                        notice.style.display = 'block';
                        notice.style.background = 'rgba(59, 130, 246, 0.08)';
                        notice.style.borderColor = 'rgba(59, 130, 246, 0.25)';
                        notice.style.color = '#93c5fd';
                        notice.innerHTML = '✓ API Credentials configured in <code>assets/client_secret.json</code>. Click <b>Connect &amp; Authorize YouTube</b> to log in, or enter new credentials below.';
                    } else {
                        notice.style.display = 'block';
                        notice.style.background = 'rgba(245, 158, 11, 0.08)';
                        notice.style.borderColor = 'rgba(245, 158, 11, 0.25)';
                        notice.style.color = '#fcd34d';
                        notice.innerHTML = '👉 Enter your Google YouTube API OAuth Client credentials below to enable auto-upload.';
                    }
                }
            } catch (e) {
                console.error(e);
            }

            modal.style.display = 'flex';
        }

        function closeYouTubeAuthModal() {
            const modal = document.getElementById('ytAuthModal');
            if (modal) modal.style.display = 'none';
        }

        async function saveYouTubeCredentialsAndAuth() {
            const saveBtn = document.getElementById('modalSaveAuthBtn');
            const cid = document.getElementById('modalClientId') ? document.getElementById('modalClientId').value.trim() : '';
            const csecret = document.getElementById('modalClientSecret') ? document.getElementById('modalClientSecret').value.trim() : '';
            const rawJson = document.getElementById('modalRawJson') ? document.getElementById('modalRawJson').value.trim() : '';

            let hasNewInput = false;
            let payload = {};

            if (activeModalTab === 'manual' && (cid || csecret)) {
                if (!cid || !csecret) {
                    alert('Please enter both OAuth Client ID and Client Secret.');
                    return;
                }
                payload = { client_id: cid, client_secret: csecret };
                hasNewInput = true;
            } else if (activeModalTab === 'json' && rawJson) {
                payload = { json_content: rawJson };
                hasNewInput = true;
            }

            if (hasNewInput) {
                if (saveBtn) {
                    saveBtn.disabled = true;
                    saveBtn.innerText = 'Saving credentials...';
                }
                try {
                    const resp = await fetch('/api/auth/save_credentials', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const resData = await resp.json();
                    if (!resp.ok) {
                        alert(resData.detail || 'Failed to save credentials');
                        if (saveBtn) {
                            saveBtn.disabled = false;
                            saveBtn.innerText = 'Connect & Authorize YouTube';
                        }
                        return;
                    }
                } catch (e) {
                    alert('Error saving credentials: ' + e.message);
                    if (saveBtn) {
                        saveBtn.disabled = false;
                        saveBtn.innerText = 'Connect & Authorize YouTube';
                    }
                    return;
                }
            }

            closeYouTubeAuthModal();
            startYouTubeAuth();
        }

        async function uploadSecretsFileFromModal(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const statusDiv = document.getElementById('modalUploadStatus');
            if (statusDiv) {
                statusDiv.style.display = 'block';
                statusDiv.style.color = '#94a3b8';
                statusDiv.innerText = '⏳ Uploading ' + file.name + '...';
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                const resp = await fetch('/api/auth/upload_client_secrets', {
                    method: 'POST',
                    body: formData
                });
                const data = await resp.json();
                if (resp.ok && data.status === 'success') {
                    if (statusDiv) {
                        statusDiv.style.color = '#4ade80';
                        statusDiv.innerText = '✓ Saved ' + file.name + ' successfully!';
                    }
                    setTimeout(function() {
                        closeYouTubeAuthModal();
                        startYouTubeAuth();
                    }, 800);
                } else {
                    if (statusDiv) {
                        statusDiv.style.color = '#ef4444';
                        statusDiv.innerText = '❌ Failed: ' + (data.detail || 'Upload error');
                    }
                }
            } catch (e) {
                if (statusDiv) {
                    statusDiv.style.color = '#ef4444';
                    statusDiv.innerText = '❌ Upload error: ' + e.message;
                }
            }
        }

        async function checkYouTubeAuthStatus() {
            try {
                const resp = await fetch('/api/auth/status');
                const data = await resp.json();
                const btn = document.getElementById('ytAuthBtn');
                const badge = document.getElementById('ytChannelBadge');
                const channelName = document.getElementById('ytChannelName');
                if (!btn) return;

                if (data.auth_running) {
                    btn.textContent = 'Waiting for login...';
                    btn.disabled = true;
                    setTimeout(checkYouTubeAuthStatus, 3000);
                    return;
                }

                if (data.authenticated) {
                    btn.style.display = 'none';
                    if (badge) badge.style.display = 'flex';
                    if (channelName) channelName.textContent = data.channel_name || 'YouTube Connected';
                } else {
                    btn.style.display = 'inline-flex';
                    if (badge) badge.style.display = 'none';
                    btn.disabled = false;
                    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg><span>Enter YouTube API for Auto-Upload</span>';
                    if (data.auth_error) {
                        btn.title = 'Auth error: ' + data.auth_error;
                    }
                }
            } catch (e) {
                console.error('Auth status check failed:', e);
            }
        }

        async function switchYouTubeAccount() {
            if (!confirm("Do you want to log out of the current YouTube account and select a different email?")) return;
            try {
                await fetch('/api/auth/logout', { method: 'POST' });
                const btn = document.getElementById('ytAuthBtn');
                const badge = document.getElementById('ytChannelBadge');
                if (btn) {
                    btn.style.display = 'inline-flex';
                    btn.disabled = true;
                    btn.innerHTML = 'Opening browser...';
                }
                if (badge) badge.style.display = 'none';
                startYouTubeAuth();
            } catch (e) {
                alert('Error switching account: ' + e.message);
            }
        }

        async function startYouTubeAuth() {
            const btn = document.getElementById('ytAuthBtn');
            if (!btn) return;
            btn.disabled = true;
            btn.innerHTML = 'Opening browser...';
            try {
                const resp = await fetch('/api/auth/youtube', { method: 'POST' });
                const data = await resp.json();
                if (resp.ok) {
                    btn.innerHTML = 'Complete login in browser...';
                    const poll = setInterval(async function() {
                        const s = await fetch('/api/auth/status');
                        const sd = await s.json();
                        if (!sd.auth_running) {
                            clearInterval(poll);
                            checkYouTubeAuthStatus();
                        }
                    }, 3000);
                } else {
                    alert(data.detail || 'Failed to start authentication');
                    btn.disabled = false;
                    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg><span>Enter YouTube API for Auto-Upload</span>';
                }
            } catch (e) {
                alert('Error: ' + e.message);
                btn.disabled = false;
                btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg><span>Enter YouTube API for Auto-Upload</span>';
            }
        }

        // ==================== AI OP STUDIO JAVASCRIPT ====================
        let currentAiOpPost = null;

        function switchStudioMode(mode) {
            const btnScraper = document.getElementById('modeBtnScraper');
            const btnAiOp = document.getElementById('modeBtnAiOp');
            const panelScraperConfig = document.getElementById('panelScraperConfig');
            const panelScraperPreview = document.getElementById('panelScraperPreview');
            const panelAiOpCreator = document.getElementById('panelAiOpCreator');
            const panelAiOpTracker = document.getElementById('panelAiOpTracker');

            if (mode === 'ai-op') {
                if (btnScraper) btnScraper.classList.remove('active');
                if (btnAiOp) btnAiOp.classList.add('active');
                if (panelScraperConfig) panelScraperConfig.style.display = 'none';
                if (panelScraperPreview) panelScraperPreview.style.display = 'none';
                if (panelAiOpCreator) panelAiOpCreator.style.display = 'block';
                if (panelAiOpTracker) panelAiOpTracker.style.display = 'block';
                loadAiOpPosts();
            } else {
                if (btnScraper) btnScraper.classList.add('active');
                if (btnAiOp) btnAiOp.classList.remove('active');
                if (panelScraperConfig) panelScraperConfig.style.display = 'block';
                if (panelScraperPreview) panelScraperPreview.style.display = 'block';
                if (panelAiOpCreator) panelAiOpCreator.style.display = 'none';
                if (panelAiOpTracker) panelAiOpTracker.style.display = 'none';
            }
        }

        function onAiOpSubredditChange() {
            const sel = document.getElementById('aiOpSubreddit');
            const custom = document.getElementById('aiOpCustomSub');
            if (sel && custom) {
                custom.style.display = sel.value === 'custom' ? 'block' : 'none';
            }
        }

        async function generateAiOpPost() {
            const btn = document.getElementById('btnAiOpGenerate');
            const sel = document.getElementById('aiOpSubreddit');
            const custom = document.getElementById('aiOpCustomSub');
            const style = document.getElementById('aiOpStyle');
            const theme = document.getElementById('aiOpTheme');
            const previewBox = document.getElementById('aiOpPreviewBox');

            let sub = sel ? sel.value : 'AskReddit';
            if (sub === 'custom') {
                sub = custom ? custom.value.trim() : 'AskReddit';
            }
            if (!sub) sub = 'AskReddit';

            if (btn) {
                btn.disabled = true;
                btn.innerText = '✨ Crafting viral post idea...';
            }

            try {
                const resp = await fetch('/api/ai-op/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        subreddit: sub,
                        theme: theme ? theme.value.trim() : '',
                        style: style ? style.value : 'comedic'
                    })
                });
                const data = await resp.json();
                if (resp.ok && data.post) {
                    currentAiOpPost = data.post;
                    if (previewBox) previewBox.style.display = 'block';
                    const prevSub = document.getElementById('prevSubBadge');
                    const prevTitle = document.getElementById('prevTitleInput');
                    const prevBody = document.getElementById('prevBodyInput');
                    const prevBodyGroup = document.getElementById('prevBodyGroup');
                    const prevRationale = document.getElementById('prevRationaleText');
                    const prevModel = document.getElementById('prevModelPill');

                    if (prevSub) prevSub.innerText = 'r/' + data.post.subreddit;
                    if (prevTitle) prevTitle.value = data.post.title;
                    if (prevBody) prevBody.value = data.post.body || '';
                    if (prevBodyGroup) prevBodyGroup.style.display = (data.post.body || sub.toLowerCase() === 'amitheasshole' || sub.toLowerCase() === 'tifu' || sub.toLowerCase() === 'unpopularopinion') ? 'block' : 'none';
                    if (prevRationale) prevRationale.innerText = data.post.rationale || 'Engineered for high comment engagement.';
                    if (prevModel) prevModel.innerText = data.post.is_fallback ? 'Template Fallback' : (data.post.model_used || 'Ollama Gemma');
                } else {
                    alert('Failed to generate post: ' + (data.detail || 'Unknown error'));
                }
            } catch (e) {
                alert('Error generating post: ' + e.message);
            } finally {
                if (btn) {
                    btn.disabled = false;
                    btn.innerText = '✨ Generate Viral Post Idea';
                }
            }
        }

        async function submitAiOpPost() {
            const btn = document.getElementById('btnAiOpSubmit');
            const titleInput = document.getElementById('prevTitleInput');
            const bodyInput = document.getElementById('prevBodyInput');
            const minCommentsInput = document.getElementById('aiOpMinComments');
            const dryRunInput = document.getElementById('aiOpDryRun');

            if (!titleInput || !titleInput.value.trim()) {
                alert('Please enter or generate a post title first.');
                return;
            }

            const title = titleInput.value.trim();
            const body = bodyInput ? bodyInput.value.trim() : '';
            const sub = currentAiOpPost ? currentAiOpPost.subreddit : 'AskReddit';
            const minComments = minCommentsInput ? parseInt(minCommentsInput.value, 10) : 2;
            const isDryRun = dryRunInput ? dryRunInput.checked : false;

            if (btn) {
                btn.disabled = true;
                btn.innerText = '🚀 Posting to Reddit...';
            }

            try {
                const resp = await fetch('/api/ai-op/post', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        subreddit: sub,
                        title: title,
                        body: body,
                        min_comments: minComments,
                        dry_run: isDryRun
                    })
                });
                const data = await resp.json();
                if (resp.ok) {
                    alert('✓ AI OP post submitted successfully! Now monitoring for comments.');
                    loadAiOpPosts();
                } else {
                    alert('Submission failed: ' + (data.detail || 'Check your Reddit API credentials'));
                }
            } catch (e) {
                alert('Error submitting post: ' + e.message);
            } finally {
                if (btn) {
                    btn.disabled = false;
                    btn.innerText = '🚀 Submit Post to Reddit & Start Watching';
                }
            }
        }

        async function loadAiOpPosts() {
            const container = document.getElementById('aiOpPostsContainer');
            if (!container) return;

            try {
                const resp = await fetch('/api/ai-op/posts');
                const data = await resp.json();
                if (!resp.ok || !data.posts || data.posts.length === 0) {
                    container.innerHTML = '<div style="text-align:center; padding:2rem 1rem; color:var(--text-tertiary); font-size:0.85rem;">No active AI OP posts yet. Generate and submit a post to begin tracking!</div>';
                    return;
                }

                let html = '<div style="display:flex; flex-direction:column; gap:0.85rem;">';
                for (const p of data.posts) {
                    let badgeClass = 'status-badge-waiting';
                    let badgeText = '⏳ WAITING FOR MEATBAGS';
                    if (p.status === 'submitted') { badgeClass = 'status-badge-submitted'; badgeText = 'SUBMITTED'; }
                    else if (p.status === 'ready_to_render') { badgeClass = 'status-badge-ready'; badgeText = '⚡ READY TO RENDER'; }
                    else if (p.status === 'rendered') { badgeClass = 'status-badge-rendered'; badgeText = '🎬 RENDERED'; }

                    html += `
                    <div style="background:var(--bg-surface-elevated); border:1px solid var(--border-subtle); border-radius:var(--radius-md); padding:1rem; display:flex; flex-direction:column; gap:0.65rem;">
                        <div style="display:flex; align-items:center; justify-content:space-between;">
                            <div style="display:flex; align-items:center; gap:0.5rem;">
                                <span class="badge-pill">r/${p.subreddit}</span>
                                <span style="font-size:0.75rem; color:var(--text-tertiary); font-family:'JetBrains Mono', monospace;">[${p.post_id}]</span>
                            </div>
                            <span class="ai-op-status-badge ${badgeClass}">${badgeText}</span>
                        </div>
                        <div style="font-weight:700; font-size:0.92rem; color:#ffffff;">${p.title}</div>
                        ${p.body ? `<div style="font-size:0.8rem; color:var(--text-secondary); line-height:1.4;">${p.body.substring(0, 140)}${p.body.length > 140 ? '...' : ''}</div>` : ''}
                        
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-top:0.35rem; border-top:1px solid var(--border-subtle); padding-top:0.65rem;">
                            <div style="font-size:0.78rem; color:var(--text-secondary);">
                                💬 Comments: <b style="color:#ffffff;">${p.current_comments_count}</b> / ${p.min_comments_target}
                                ${p.last_checked_at ? `<span style="color:var(--text-tertiary); margin-left:0.5rem; font-size:0.72rem;">(Checked: ${new Date(p.last_checked_at).toLocaleTimeString()})</span>` : ''}
                            </div>
                            <div style="display:flex; gap:0.45rem;">
                                <button type="button" class="mini-btn" onclick="checkAiOpPost('${p.post_id}')">🔄 Check</button>
                                ${p.status !== 'rendered' ? `<button type="button" class="mini-btn" style="background:var(--brand-primary); color:#ffffff; font-weight:700;" onclick="renderAiOpPost('${p.post_id}')">🎬 Render</button>` : ''}
                                ${p.rendered_video_path ? `<button type="button" class="mini-btn" style="background:#10b981; color:#ffffff; font-weight:700;" onclick="playAiOpVideo('${p.rendered_video_path}')">▶ Play</button>` : ''}
                                ${p.url ? `<a href="${p.url}" target="_blank" class="mini-btn" style="text-decoration:none;">🔗 Reddit</a>` : ''}
                                <button type="button" class="mini-btn" style="color:#ef4444;" onclick="deleteAiOpPost('${p.post_id}')">🗑️</button>
                            </div>
                        </div>
                    </div>
                    `;
                }
                html += '</div>';
                container.innerHTML = html;
            } catch (e) {
                console.error('Failed to load AI OP posts:', e);
            }
        }

        async function checkAiOpPost(postId) {
            try {
                const resp = await fetch('/api/ai-op/check/' + postId, { method: 'POST' });
                const data = await resp.json();
                if (resp.ok) {
                    loadAiOpPosts();
                } else {
                    alert('Check failed: ' + (data.error || data.detail || 'Network error'));
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        async function renderAiOpPost(postId) {
            try {
                const resp = await fetch('/api/ai-op/render/' + postId, { method: 'POST' });
                const data = await resp.json();
                if (resp.ok) {
                    alert('Rendering started! Switching to Studio tab to monitor progress.');
                    switchStudioMode('scraper');
                    startPollingStatus();
                } else {
                    alert('Render failed: ' + (data.detail || 'Unknown error'));
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        function playAiOpVideo(videoPath) {
            switchStudioMode('scraper');
            const player = document.getElementById('videoPlayer');
            const placeholder = document.getElementById('placeholderText');
            if (player) {
                player.src = '/' + videoPath.replace(/\\/g, '/');
                player.style.display = 'block';
                if (placeholder) placeholder.style.display = 'none';
                player.play();
            }
        }

        async function deleteAiOpPost(postId) {
            if (!confirm("Are you sure you want to delete this tracked AI OP post?")) return;
            try {
                const resp = await fetch('/api/ai-op/posts/' + postId, { method: 'DELETE' });
                if (resp.ok) {
                    loadAiOpPosts();
                }
            } catch (e) {
                alert('Error deleting post: ' + e.message);
            }
        }

        // ==================== REDDIT SOCK PUPPET & BOT AUTH MODAL ====================
        let activeRedditModalTab = 'browser';

        function switchRedditModalTab(tab) {
            activeRedditModalTab = tab;
            const tabs = ['browser', 'account', 'api'];
            tabs.forEach(t => {
                const btn = document.getElementById('tabBtnReddit' + t.charAt(0).toUpperCase() + t.slice(1));
                const pane = document.getElementById('tabContentReddit' + t.charAt(0).toUpperCase() + t.slice(1));
                if (btn) btn.classList.toggle('active', t === tab);
                if (pane) pane.style.display = (t === tab) ? 'block' : 'none';
            });
        }

        async function openRedditAuthModal() {
            const modal = document.getElementById('redditAuthModal');
            const notice = document.getElementById('redditStatusNotice');
            if (!modal) return;

            try {
                const resp = await fetch('/api/ai-op/credentials');
                const data = await resp.json();
                if (notice) {
                    notice.style.display = 'block';
                    if (data.has_session) {
                        notice.style.background = 'rgba(16, 185, 129, 0.1)';
                        notice.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                        notice.style.color = '#6ee7b7';
                        notice.innerHTML = '✓ <b>Browser Sock Puppet Session Active</b>: Authenticated and ready for undetectable remote posting!';
                    } else if (data.has_credentials) {
                        notice.style.background = 'rgba(59, 130, 246, 0.1)';
                        notice.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                        notice.style.color = '#93c5fd';
                        notice.innerHTML = '✓ Reddit Bot Configured: <b>/u/' + data.username + '</b> (OAuth Script Mode)';
                        const uInput = document.getElementById('redditUsername');
                        if (uInput && !uInput.value) uInput.value = data.username;
                    } else {
                        notice.style.background = 'rgba(245, 158, 11, 0.08)';
                        notice.style.borderColor = 'rgba(245, 158, 11, 0.25)';
                        notice.style.color = '#fcd34d';
                        notice.innerHTML = '👉 Click <b>Launch Chromium to Log In</b> below to connect your sock puppet account, or configure credentials manually.';
                    }
                }
            } catch (e) {
                console.error(e);
            }

            modal.style.display = 'flex';
        }

        function closeRedditAuthModal() {
            const modal = document.getElementById('redditAuthModal');
            if (modal) modal.style.display = 'none';
        }

        async function startRedditBrowserLogin() {
            const btn = document.getElementById('btnRedditBrowserLogin');
            const statusDiv = document.getElementById('redditBrowserLoginStatus');

            if (btn) {
                btn.disabled = true;
                btn.innerText = '🌐 Browser open - Log into Reddit...';
            }
            if (statusDiv) {
                statusDiv.style.display = 'block';
                statusDiv.style.color = '#60a5fa';
                statusDiv.innerHTML = '⏳ Chromium window launched. Please log into your Reddit sock puppet account in the opened window...';
            }

            try {
                const resp = await fetch('/api/ai-op/browser-login', { method: 'POST' });
                const data = await resp.json();
                if (resp.ok && data.status === 'success') {
                    if (statusDiv) {
                        statusDiv.style.color = '#4ade80';
                        statusDiv.innerHTML = '✓ ' + data.message;
                    }
                    setTimeout(function() {
                        closeRedditAuthModal();
                        checkRedditAuthStatus();
                    }, 1200);
                } else {
                    if (statusDiv) {
                        statusDiv.style.color = '#ef4444';
                        statusDiv.innerHTML = '❌ ' + (data.detail || 'Login timed out or was closed.');
                    }
                }
            } catch (e) {
                if (statusDiv) {
                    statusDiv.style.color = '#ef4444';
                    statusDiv.innerHTML = '❌ Error: ' + e.message;
                }
            } finally {
                if (btn) {
                    btn.disabled = false;
                    btn.innerText = '🌐 Launch Chromium to Log In to Reddit';
                }
            }
        }

        async function saveRedditCredentials() {
            const cid = document.getElementById('redditClientId') ? document.getElementById('redditClientId').value.trim() : '';
            const csec = document.getElementById('redditClientSecret') ? document.getElementById('redditClientSecret').value.trim() : '';
            const uname = document.getElementById('redditUsername') ? document.getElementById('redditUsername').value.trim() : '';
            const pwd = document.getElementById('redditPassword') ? document.getElementById('redditPassword').value.trim() : '';
            const ua = document.getElementById('redditUserAgent') ? document.getElementById('redditUserAgent').value.trim() : '';

            if (!uname || !pwd) {
                alert('Please enter at least Username and Password.');
                return;
            }

            try {
                const resp = await fetch('/api/ai-op/credentials', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        client_id: cid,
                        client_secret: csec,
                        username: uname,
                        password: pwd,
                        user_agent: ua
                    })
                });
                const data = await resp.json();
                if (resp.ok) {
                    alert('✓ Reddit credentials saved successfully!');
                    closeRedditAuthModal();
                    checkRedditAuthStatus();
                } else {
                    alert('Failed: ' + (data.detail || 'Error saving credentials'));
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        async function testRedditCredentials() {
            const cid = document.getElementById('redditClientId') ? document.getElementById('redditClientId').value.trim() : '';
            const csec = document.getElementById('redditClientSecret') ? document.getElementById('redditClientSecret').value.trim() : '';
            const uname = document.getElementById('redditUsername') ? document.getElementById('redditUsername').value.trim() : '';
            const pwd = document.getElementById('redditPassword') ? document.getElementById('redditPassword').value.trim() : '';

            let payload = null;
            if (cid && csec && uname && pwd) {
                payload = { client_id: cid, client_secret: csec, username: uname, password: pwd };
            }

            const notice = document.getElementById('redditStatusNotice');
            if (notice) {
                notice.style.display = 'block';
                notice.style.background = 'rgba(59, 130, 246, 0.1)';
                notice.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                notice.style.color = '#93c5fd';
                notice.innerHTML = '⏳ Testing connection...';
            }

            try {
                const resp = await fetch('/api/ai-op/test-credentials', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: payload ? JSON.stringify(payload) : null
                });
                const data = await resp.json();
                if (resp.ok) {
                    if (notice) {
                        notice.style.background = 'rgba(16, 185, 129, 0.1)';
                        notice.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                        notice.style.color = '#6ee7b7';
                        notice.innerHTML = '✓ ' + data.message;
                    }
                } else {
                    if (notice) {
                        notice.style.background = 'rgba(239, 68, 68, 0.1)';
                        notice.style.borderColor = 'rgba(239, 68, 68, 0.3)';
                        notice.style.color = '#fca5a5';
                        notice.innerHTML = '❌ Connection failed: ' + (data.detail || 'Check credentials');
                    }
                }
            } catch (e) {
                if (notice) {
                    notice.style.background = 'rgba(239, 68, 68, 0.1)';
                    notice.style.borderColor = 'rgba(239, 68, 68, 0.3)';
                    notice.style.color = '#fca5a5';
                    notice.innerHTML = '❌ Error: ' + e.message;
                }
            }
        }

        async function checkRedditAuthStatus() {
            try {
                const resp = await fetch('/api/ai-op/credentials');
                const data = await resp.json();
                const btnText = document.getElementById('redditAuthBtnText');
                if (btnText) {
                    if (data.has_session) {
                        btnText.innerText = 'Reddit: Browser Active';
                    } else if (data.has_credentials && data.username) {
                        btnText.innerText = 'Reddit: /u/' + data.username;
                    } else {
                        btnText.innerText = 'Reddit Account: Setup';
                    }
                }
            } catch (e) {
                console.error('Reddit auth check failed:', e);
            }
        }

        window.addEventListener('DOMContentLoaded', function() {
            renderCustomSubredditsInBatchList();
            checkRedditAuthStatus();
            loadAiOpPosts();
        });
    </script>

    <!-- YouTube API Configuration Modal -->
    <div id="ytAuthModal" class="modal-overlay" style="display:none;" onclick="if(event.target===this) closeYouTubeAuthModal()">
        <div class="modal-card">
            <div class="modal-header" style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1.15rem; border-bottom:1px solid var(--border-subtle); padding-bottom:0.85rem;">
                <div style="display:flex; align-items:center; gap:0.65rem;">
                    <div class="panel-icon-box" style="background:rgba(239, 68, 68, 0.12); border-color:rgba(239, 68, 68, 0.3); color:#ef4444;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>
                    </div>
                    <div>
                        <div style="font-weight:800; font-size:1.08rem; color:#ffffff;">Enter YouTube API for Auto-Upload</div>
                        <div style="font-size:0.75rem; color:var(--text-tertiary);">Configure your Google Cloud OAuth credentials to publish Shorts</div>
                    </div>
                </div>
                <button type="button" class="mini-btn" onclick="closeYouTubeAuthModal()" style="font-size:1.2rem; line-height:1; padding:0.2rem 0.55rem;">&times;</button>
            </div>

            <div id="ytConfigNotice" style="margin-bottom: 1rem; padding: 0.75rem 1rem; border-radius: var(--radius-md); font-size: 0.82rem; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); color: #93c5fd; display: none;"></div>

            <div class="tab-nav" style="display:flex; gap:0.45rem; margin-bottom:1.15rem; border-bottom:1px solid var(--border-subtle); padding-bottom:0.65rem;">
                <button type="button" id="tabBtnManual" class="mini-btn tab-btn active" onclick="switchModalTab('manual')">Client ID &amp; Secret</button>
                <button type="button" id="tabBtnUpload" class="mini-btn tab-btn" onclick="switchModalTab('upload')">Upload client_secret.json</button>
                <button type="button" id="tabBtnJson" class="mini-btn tab-btn" onclick="switchModalTab('json')">Paste Raw JSON</button>
            </div>

            <!-- Tab 1: Manual Key / Secret Input -->
            <div id="tabContentManual" class="modal-tab-pane">
                <div style="margin-bottom:0.85rem;">
                    <label for="modalClientId">Google OAuth Client ID</label>
                    <input type="text" id="modalClientId" placeholder="e.g. 1234567890-abcdef.apps.googleusercontent.com" autocomplete="off">
                </div>
                <div style="margin-bottom:0.85rem;">
                    <label for="modalClientSecret">Google OAuth Client Secret</label>
                    <input type="text" id="modalClientSecret" placeholder="e.g. GOCSPX-xxxxxxxxxxxxxxxxxxxxxx" autocomplete="off">
                </div>
            </div>

            <!-- Tab 2: Upload JSON -->
            <div id="tabContentUpload" class="modal-tab-pane" style="display:none;">
                <div style="border: 2px dashed var(--border-strong); border-radius: var(--radius-md); padding: 1.5rem 1rem; text-align: center; background: var(--bg-input);">
                    <p style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:0.75rem;">Select your downloaded <b>client_secret.json</b> from Google Cloud Console</p>
                    <input type="file" id="modalSecretsFileInput" accept=".json" onchange="uploadSecretsFileFromModal(this)" style="display:none;">
                    <button type="button" class="mini-btn" onclick="document.getElementById('modalSecretsFileInput').click()" style="background:var(--brand-primary); color:#ffffff; padding:0.55rem 1.15rem; font-weight:700; border-radius:var(--radius-md);">Choose client_secret.json</button>
                    <div id="modalUploadStatus" style="font-size:0.8rem; margin-top:0.6rem; color:#4ade80; display:none;"></div>
                </div>
            </div>

            <!-- Tab 3: Paste Raw JSON -->
            <div id="tabContentJson" class="modal-tab-pane" style="display:none;">
                <div style="margin-bottom:0.85rem;">
                    <label for="modalRawJson">Paste client_secret.json contents</label>
                    <textarea id="modalRawJson" rows="5" placeholder='{"installed": {"client_id": "...", "client_secret": "..."}}' style="width:100%; padding:0.75rem; background:var(--bg-input); border:1px solid var(--border-strong); border-radius:var(--radius-md); color:#ffffff; font-family:&#39;JetBrains Mono&#39;, monospace; font-size:0.78rem; outline:none; resize:vertical;"></textarea>
                </div>
            </div>

            <!-- Step-by-Step Instructions Collapsible -->
            <details style="margin-top:1rem; background:rgba(255,255,255,0.02); border:1px solid var(--border-subtle); border-radius:var(--radius-md); padding:0.75rem;">
                <summary style="font-size:0.8rem; font-weight:700; color:var(--brand-primary); cursor:pointer; user-select:none;">📖 How to get free Google YouTube Data API credentials (3 steps)</summary>
                <ol style="margin-top:0.65rem; padding-left:1.2rem; font-size:0.78rem; color:var(--text-secondary); line-height:1.6;">
                    <li>Go to <a href="https://console.cloud.google.com/" target="_blank" style="color:#60a5fa; text-decoration:underline;">Google Cloud Console</a> &amp; create a project.</li>
                    <li>Under <b>APIs &amp; Services &gt; Library</b>, enable <b>YouTube Data API v3</b>.</li>
                    <li>Under <b>APIs &amp; Services &gt; Credentials &gt; Create Credentials</b>, choose <b>OAuth Client ID</b> (Desktop app), then paste your Client ID &amp; Secret above!</li>
                </ol>
            </details>

            <div style="display:flex; justify-content:flex-end; align-items:center; gap:0.75rem; margin-top:1.25rem; border-top:1px solid var(--border-subtle); padding-top:1rem;">
                <button type="button" class="mini-btn" onclick="closeYouTubeAuthModal()" style="padding:0.6rem 1.1rem;">Cancel</button>
                <button type="button" id="modalSaveAuthBtn" class="btn-yt-login" onclick="saveYouTubeCredentialsAndAuth()" style="padding:0.6rem 1.25rem; border-radius:var(--radius-md);">
                    <span>Connect &amp; Authorize YouTube</span>
                </button>
            </div>
        </div>
    </div>

    <!-- Reddit Sock Puppet / Bot Account Configuration Modal -->
    <div id="redditAuthModal" class="modal-overlay" style="display:none;" onclick="if(event.target===this) closeRedditAuthModal()">
        <div class="modal-card">
            <div class="modal-header" style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1.15rem; border-bottom:1px solid var(--border-subtle); padding-bottom:0.85rem;">
                <div style="display:flex; align-items:center; gap:0.65rem;">
                    <div class="panel-icon-box" style="background:rgba(255, 69, 0, 0.12); border-color:rgba(255, 69, 0, 0.3); color:#ff4500;">
                        <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.56 12 8 12.56 8 13.25c0 .688.56 1.25 1.25 1.25.688 0 1.25-.562 1.25-1.25 0-.69-.562-1.25-1.25-1.25zm5.5 0c-.688 0-1.25.56-1.25 1.25 0 .688.562 1.25 1.25 1.25.69 0 1.25-.562 1.25-1.25 0-.69-.56-1.25-1.25-1.25zm-5.465 3.99a.577.577 0 0 0-.41.983c.77.77 1.83 1.15 2.875 1.15 1.044 0 2.104-.38 2.874-1.15a.577.577 0 0 0-.82-.816c-.552.55-1.318.82-2.054.82-.736 0-1.502-.27-2.054-.82a.574.574 0 0 0-.411-.167z"/></svg>
                    </div>
                    <div>
                        <div style="font-weight:800; font-size:1.08rem; color:#ffffff;">Reddit Sock Puppet Account Settings</div>
                        <div style="font-size:0.75rem; color:var(--text-tertiary);">Connect your dedicated sock puppet account for AI posting</div>
                    </div>
                </div>
                <button type="button" class="mini-btn" onclick="closeRedditAuthModal()" style="font-size:1.2rem; line-height:1; padding:0.2rem 0.55rem;">&times;</button>
            </div>

            <div id="redditStatusNotice" style="margin-bottom: 1rem; padding: 0.75rem 1rem; border-radius: var(--radius-md); font-size: 0.82rem; display: none;"></div>

            <div class="tab-nav" style="display:flex; gap:0.45rem; margin-bottom:1.15rem; border-bottom:1px solid var(--border-subtle); padding-bottom:0.65rem;">
                <button type="button" id="tabBtnRedditBrowser" class="mini-btn tab-btn active" onclick="switchRedditModalTab('browser')">🌐 Browser Login (Recommended)</button>
                <button type="button" id="tabBtnRedditAccount" class="mini-btn tab-btn" onclick="switchRedditModalTab('account')">🔑 User / Password</button>
                <button type="button" id="tabBtnRedditApi" class="mini-btn tab-btn" onclick="switchRedditModalTab('api')">⚙️ OAuth App API</button>
            </div>

            <!-- Tab 1: Browser Sock Puppet Login -->
            <div id="tabContentRedditBrowser" class="modal-tab-pane">
                <div style="border: 2px dashed rgba(255, 69, 0, 0.35); border-radius: var(--radius-md); padding: 1.5rem 1rem; text-align: center; background: var(--bg-input);">
                    <div style="font-size:1.75rem; margin-bottom:0.5rem;">🕵️‍♂️</div>
                    <div style="font-weight:700; font-size:0.95rem; color:#ffffff; margin-bottom:0.35rem;">Remote Sock Puppet Browser Session</div>
                    <p style="font-size:0.82rem; color:var(--text-secondary); max-width:380px; margin:0 auto 1.15rem; line-height:1.5;">
                        Launches a real Chromium browser window. Simply log in to your dedicated Reddit account once. Your session cookies will be saved locally so the AI can post remotely as a human user without API bot restrictions.
                    </p>
                    <button type="button" id="btnRedditBrowserLogin" class="main-action-btn btn-upload" onclick="startRedditBrowserLogin()" style="margin:0 auto; padding:0.75rem 1.4rem;">
                        <span>🌐 Launch Chromium to Log In to Reddit</span>
                    </button>
                    <div id="redditBrowserLoginStatus" style="font-size:0.82rem; margin-top:0.85rem; display:none;"></div>
                </div>
            </div>

            <!-- Tab 2: Direct Username & Password -->
            <div id="tabContentRedditAccount" class="modal-tab-pane" style="display:none;">
                <div style="margin-bottom:0.85rem;">
                    <label for="redditUsername">Sock Puppet Username</label>
                    <input type="text" id="redditUsername" placeholder="e.g. MyDedicatedSockPuppet">
                </div>
                <div style="margin-bottom:0.85rem;">
                    <label for="redditPassword">Account Password</label>
                    <input type="password" id="redditPassword" placeholder="Account password">
                </div>
            </div>

            <!-- Tab 3: OAuth API (Optional) -->
            <div id="tabContentRedditApi" class="modal-tab-pane" style="display:none;">
                <div style="margin-bottom:0.85rem;">
                    <label for="redditClientId">Reddit Script App Client ID</label>
                    <input type="text" id="redditClientId" placeholder="e.g. 14-char script app ID from reddit.com/prefs/apps">
                </div>
                <div style="margin-bottom:0.85rem;">
                    <label for="redditClientSecret">Reddit Script App Client Secret</label>
                    <input type="password" id="redditClientSecret" placeholder="e.g. secret key string">
                </div>
                <div style="margin-bottom:0.85rem;">
                    <label for="redditUserAgent">User Agent (Optional)</label>
                    <input type="text" id="redditUserAgent" placeholder="python:ai-op-shorts-generator:v1.0">
                </div>
            </div>

            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:1.25rem; border-top:1px solid var(--border-subtle); padding-top:1rem;">
                <button type="button" class="mini-btn" onclick="testRedditCredentials()" style="padding:0.6rem 1rem;">🔌 Test Connection</button>
                <div style="display:flex; gap:0.5rem;">
                    <button type="button" class="mini-btn" onclick="closeRedditAuthModal()" style="padding:0.6rem 1.1rem;">Cancel</button>
                    <button type="button" class="btn-primary" onclick="saveRedditCredentials()" style="padding:0.6rem 1.25rem; font-size:0.85rem;">Save Credentials</button>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT


def start_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    print(f"\nReddit Reading Web UI running at: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
