"""Local Ollama AI / Gemini Title and Description Generator for Reddit Shorts."""

import json
import logging
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, List
import requests

logger = logging.getLogger(__name__)

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"


def ensure_ollama_running(ollama_host: str = OLLAMA_HOST, timeout_seconds: int = 6) -> bool:
    """Checks if local Ollama server is responding. If offline, automatically launches 'ollama serve' background process."""
    try:
        resp = requests.get(f"{ollama_host}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        logger.warning("Ollama executable not found on system PATH. Cannot auto-start Ollama server.")
        return False

    logger.info(f"Ollama server offline. Auto-starting background process: '{ollama_bin} serve'...")
    try:
        subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        start_t = time.time()
        while time.time() - start_t < timeout_seconds:
            time.sleep(0.5)
            try:
                resp = requests.get(f"{ollama_host}/api/tags", timeout=1.0)
                if resp.status_code == 200:
                    logger.info("Ollama server successfully auto-started & connected!")
                    return True
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Failed to auto-start Ollama server ({e}).")

    return False


def get_available_ollama_models(ollama_host: str = OLLAMA_HOST) -> List[str]:
    """Returns a list of model names currently installed on local Ollama server."""
    try:
        resp = requests.get(f"{ollama_host}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception:
        pass
    return []


def generate_video_metadata(
    post_title: str,
    post_body: str = "",
    subreddit: str = "AskReddit",
    ollama_host: str = OLLAMA_HOST,
    model_name: str = OLLAMA_MODEL,
) -> Dict[str, str]:
    """Generate viral YouTube Shorts title, description, and hashtags using local Ollama model.

    Falls back to structured formatting if local Ollama server is offline or unavailable.
    """
    title = f"{post_title[:80]} #Shorts #Reddit #r{subreddit}"
    description = (
        f"Top response from r/{subreddit}: {post_title}\n\n"
        f"{post_body[:300]}\n\n"
        f"#shorts #reddit #{subreddit} #minecraftparkour"
    )

    # 1. Ensure Ollama server is running (auto-starts background process if needed)
    is_running = ensure_ollama_running(ollama_host)
    if not is_running:
        logger.info("Ollama server unavailable. Using template metadata.")
        return {"title": title, "description": description}

    # 2. Select best available installed model (strictly prioritizing Gemma)
    available_models = get_available_ollama_models(ollama_host)
    selected_model = model_name
    if available_models:
        gemma_models = [m for m in available_models if "gemma" in m.lower()]
        if gemma_models:
            if model_name in gemma_models:
                selected_model = model_name
            else:
                selected_model = gemma_models[0]
        else:
            if model_name not in available_models:
                selected_model = available_models[0]
        logger.info(f"Selected Gemma local AI model for viral metadata: '{selected_model}'")

    prompt = (
        f"You are a viral YouTube Shorts creator. Create an engaging title and description for a Reddit video.\n"
        f"Subreddit: r/{subreddit}\n"
        f"Reddit Post Title: {post_title}\n"
        f"Post Body: {post_body[:300]}\n\n"
        f"Respond ONLY with a valid JSON object matching this structure:\n"
        f'{{\n  "title": "Short catchy title under 80 chars #shorts #reddit",\n  "description": "Engaging 2-3 sentence description with hashtags #shorts #{subreddit}"\n}}'
    )

    try:
        url = f"{ollama_host}/api/generate"
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7},
        }
        resp = requests.post(url, json=payload, timeout=12.0)
        if resp.status_code == 200:
            result = resp.json()
            raw_text = result.get("response", "")
            if "{" in raw_text and "}" in raw_text:
                json_str = raw_text[raw_text.find("{") : raw_text.rfind("}") + 1]
                data = json.loads(json_str)
                if data.get("title"):
                    title = data["title"]
                if data.get("description"):
                    description = data["description"]
            logger.info(f"Successfully generated viral AI metadata via local Ollama model ({selected_model}).")
    except Exception as e:
        logger.info(f"Ollama generation failed ({e}). Using template metadata.")

    return {
        "title": title,
        "description": description,
    }


def save_video_folder(
    output_base_dir: str,
    subreddit: str,
    scraped_content: Any,
    video_file_path: str,
    metadata: Dict[str, str],
) -> str:
    """Saves video, title, description, and manifest in a date-timestamped folder inside output/."""
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{now_str}_{subreddit}"
    folder_path = Path(output_base_dir) / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    # Move/Copy output video if needed
    src_video = Path(video_file_path)
    target_video = folder_path / f"{subreddit}_short.mp4"

    if src_video.exists() and src_video.resolve() != target_video.resolve():
        import shutil

        shutil.copy2(src_video, target_video)

    # Save title and description txt file
    meta_txt_path = folder_path / "metadata.txt"
    with open(meta_txt_path, "w", encoding="utf-8") as f:
        f.write(f"TITLE:\n{metadata['title']}\n\n")
        f.write(f"DESCRIPTION:\n{metadata['description']}\n")

    # Save metadata json file
    meta_json_path = folder_path / "metadata.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "folder_name": folder_name,
                "subreddit": subreddit,
                "title": metadata["title"],
                "description": metadata["description"],
                "video_file": str(target_video.name),
                "post_title": scraped_content.post.title if hasattr(scraped_content, "post") else "",
            },
            f,
            indent=2,
        )

    return str(folder_path)
