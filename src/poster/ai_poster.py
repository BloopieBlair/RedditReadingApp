"""AI OP (Original Poster) generator for crafting viral, hilarious Reddit posts."""

import json
import logging
import random
from typing import Dict, Any, Optional, List
import requests

from src.ai_generator import ensure_ollama_running, get_available_ollama_models, OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)

# Fallback humorous post templates grouped by subreddit and tone
FALLBACK_TEMPLATES: Dict[str, List[Dict[str, str]]] = {
    "AskReddit": [
        {
            "title": "What is something completely legal that still feels 100% like a crime when your boss walks past your desk?",
            "body": "",
            "style": "comedic",
        },
        {
            "title": "If animals could talk, which species would immediately be the biggest Karen and why?",
            "body": "",
            "style": "absurd",
        },
        {
            "title": "What is the absolute dumbest lie you told as a child that your parents actually believed?",
            "body": "",
            "style": "comedic",
        },
        {
            "title": "You are given $10 million, but an immortal, mildly irritated snail follows you for the rest of your life. What is your snail-containment strategy?",
            "body": "",
            "style": "absurd",
        },
        {
            "title": "What is an unspoken social rule that everyone follows, but nobody can explain why?",
            "body": "",
            "style": "thought-provoking",
        },
        {
            "title": "What is the most chaotic neutral thing you have ever witnessed a complete stranger do in public?",
            "body": "",
            "style": "comedic",
        },
        {
            "title": "What is something that sounds like an insult, but is actually a compliment?",
            "body": "",
            "style": "witty",
        },
        {
            "title": "If you could replace handshakes with any other greeting gesture for all of humanity, what are we doing now?",
            "body": "",
            "style": "comedic",
        },
    ],
    "Showerthoughts": [
        {
            "title": "Centaurs have two ribcages and nobody talks about how deeply unsettling that is.",
            "body": "",
            "style": "absurd",
        },
        {
            "title": "Your future self is currently watching you through memories and judging your life choices.",
            "body": "",
            "style": "thought-provoking",
        },
        {
            "title": "If you get bitten by a radioactive human, you just turn into a tired guy with back pain and a 9-to-5.",
            "body": "",
            "style": "comedic",
        },
        {
            "title": "Elevators are just small rooms that stay in place while moving the entire rest of the building around you.",
            "body": "",
            "style": "absurd",
        },
    ],
    "unpopularopinion": [
        {
            "title": "Cereal is significantly better with warm milk and everyone drinking ice-cold milk is suffering in silence.",
            "body": "Hear me out. Cold milk makes cereal hard and gives you brain freeze in the morning. Warm milk turns it into a comforting, delicious breakfast bowl. You are all cowards for not trying it.",
            "style": "spicy",
        },
        {
            "title": "Sleeping with socks on is objectively superior and people who hate it have weak circulation and weak resolve.",
            "body": "Feet getting cold at 3 AM is the single worst human experience. Socks fix everything immediately. Stop pretending your toes need to 'breathe'.",
            "style": "comedic",
        },
        {
            "title": "The crust is the single best part of the pizza and anyone leaving crusts behind should lose pizza privileges.",
            "body": "It's literally freshly baked bread with leftover garlic sauce dip. Throwing it away is criminal behavior.",
            "style": "provocative",
        },
    ],
    "AmItheAsshole": [
        {
            "title": "AITA for naming my Roomba after my mother-in-law because 'it never stops complaining and bumps into everything'?",
            "body": "I (28M) bought a robot vacuum. Every time it gets stuck under the couch, the app notification says 'Barbara is in distress'. My wife thinks it's hilarious, but her mother visited yesterday, heard me shout 'Barbara shut up and vacuum the rug', and now family dinner is canceled. AITA?",
            "style": "story",
        },
        {
            "title": "AITA for telling my roommate that microwaving fish at 2 AM is a declaration of war?",
            "body": "My roommate came home at 2:15 AM after a party and decided the ideal midnight snack was leftover salmon. The apartment smelled like a Victorian fish market for 48 hours. I changed the WiFi password to 'NoFishAt2AM'. AITA?",
            "style": "story",
        },
    ],
    "tifu": [
        {
            "title": "TIFU by accidentally waving back at a stranger who was actually waving at someone behind me, and doubling down for 5 whole minutes.",
            "body": "It started with a cheerful wave across a crowded coffee shop. By the time I realized my mistake, I had already initiated full finger guns and a head nod. To save face, I had to pretend we were old high school buddies.",
            "style": "story",
        },
    ],
    "NoStupidQuestions": [
        {
            "title": "Why do we feel the overwhelming urge to say 'Big stretch!' every single time a dog stretches in front of us?",
            "body": "Is it a biological imperative? What happens to the universe if we witness a dog stretch and remain silent?",
            "style": "comedic",
        },
    ],
    "mildlyinfuriating": [
        {
            "title": "When you peel a banana from the top and it snaps in half, leaving you holding an empty peel like a fool.",
            "body": "Start your day with ruined potassium and instant disappointment.",
            "style": "comedic",
        },
    ],
}


class AIOPGenerator:
    """Generates funny, provocative, and high-engagement Reddit posts."""

    def __init__(
        self,
        ollama_host: str = OLLAMA_HOST,
        model_name: str = OLLAMA_MODEL,
    ):
        self.ollama_host = ollama_host
        self.model_name = model_name

    def _get_subreddit_guidelines(self, subreddit: str) -> str:
        """Returns specific formatting rules and style hints for target subreddit."""
        sub = subreddit.lower().replace("r/", "").strip()
        if sub == "askreddit":
            return (
                "- Must be an open-ended, engaging, thought-provoking, or hilarious question.\n"
                "- Must end with a question mark (?).\n"
                "- The body MUST be empty (r/AskReddit rule: questions in title only, no text box).\n"
                "- Aim for funny hypotheticals, relatable daily struggles, or absurd scenarios."
            )
        elif sub == "showerthoughts":
            return (
                "- Must be a single mind-blowing, quirky, or hilarious epiphany.\n"
                "- Must be in title only; body MUST be empty.\n"
                "- Avoid common puns or wordplay; aim for funny perspective shifts."
            )
        elif sub in ("unpopularopinion", "the10thdentist"):
            return (
                "- Title must clearly state a comedically spicy, non-hateful hot take.\n"
                "- Body should provide 2-4 sentences of passionate, funny justification.\n"
                "- Good topics: weird food combinations, daily habits, everyday pet peeves."
            )
        elif sub in ("amitheasshole", "aita"):
            return (
                "- Title must start with 'AITA for...?'\n"
                "- Body should be a short, funny 3-5 sentence scenario/dilemma explaining the situation."
            )
        elif sub == "tifu":
            return (
                "- Title must start with 'TIFU by...'\n"
                "- Body should be a funny, relatable 3-5 sentence blunder."
            )
        else:
            return (
                "- Craft a catchy, funny, and engaging title suitable for r/" + subreddit + ".\n"
                "- Keep body short (0-3 sentences) or empty if appropriate."
            )

    def generate_post(
        self,
        subreddit: str = "AskReddit",
        theme: Optional[str] = None,
        style: str = "comedic",
    ) -> Dict[str, Any]:
        """
        Generate a Reddit post tailored for the given subreddit.
        
        Args:
            subreddit: Target subreddit name (e.g. AskReddit, Showerthoughts, unpopularopinion)
            theme: Optional topic or keyword to steer the post (e.g. 'workplace', 'dating', 'roommates')
            style: Tone style ('comedic', 'absurd', 'provocative', 'story', 'thought-provoking')

        Returns:
            Dict with 'title', 'body', 'subreddit', 'style', 'rationale', 'is_fallback'
        """
        clean_sub = subreddit.replace("r/", "").strip()
        guidelines = self._get_subreddit_guidelines(clean_sub)

        # 1. Attempt LLM generation via Ollama
        is_running = ensure_ollama_running(self.ollama_host)
        if is_running:
            available_models = get_available_ollama_models(self.ollama_host)
            selected_model = self.model_name
            if available_models:
                gemma_models = [m for m in available_models if "gemma" in m.lower()]
                if gemma_models:
                    selected_model = self.model_name if self.model_name in gemma_models else gemma_models[0]
                elif self.model_name not in available_models:
                    selected_model = available_models[0]

            theme_clause = f"Theme/Topic: '{theme}'\n" if theme else "Choose a wildly funny and relatable topic.\n"

            prompt = (
                f"You are an expert viral Reddit Original Poster (OP). Your goal is to write a hilarious, high-engagement Reddit post that will make human commenters ('meatbags') rush to reply with funny stories and witty comments.\n\n"
                f"Target Subreddit: r/{clean_sub}\n"
                f"Desired Style: {style}\n"
                f"{theme_clause}"
                f"Subreddit Rules & Guidelines:\n{guidelines}\n\n"
                f"Respond ONLY with a valid JSON object matching this exact structure:\n"
                f'{{\n  "title": "Catchy and funny title here",\n  "body": "Post body text here or empty string if title-only",\n  "rationale": "Brief 1-sentence explanation of why meatbags will comment on this"\n}}'
            )

            try:
                url = f"{self.ollama_host}/api/generate"
                payload = {
                    "model": selected_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.85},
                }
                resp = requests.post(url, json=payload, timeout=15.0)
                if resp.status_code == 200:
                    result = resp.json()
                    raw_text = result.get("response", "")
                    if "{" in raw_text and "}" in raw_text:
                        json_str = raw_text[raw_text.find("{") : raw_text.rfind("}") + 1]
                        data = json.loads(json_str)
                        title = str(data.get("title", "")).strip()
                        body = str(data.get("body", "")).strip()
                        rationale = str(data.get("rationale", "")).strip()

                        # Subreddit specific post-processing
                        if clean_sub.lower() == "askreddit":
                            if not title.endswith("?"):
                                title += "?"
                            body = ""  # AskReddit disallows text box in submissions
                        elif clean_sub.lower() == "showerthoughts":
                            body = ""

                        if title:
                            logger.info(f"Generated AI OP post for r/{clean_sub}: {title}")
                            return {
                                "title": title,
                                "body": body,
                                "subreddit": clean_sub,
                                "style": style,
                                "rationale": rationale or "Engineered for high comment engagement.",
                                "is_fallback": False,
                                "model_used": selected_model,
                            }
            except Exception as e:
                logger.warning(f"Ollama post generation failed ({e}). Falling back to template.")

        # 2. Fallback Template Generation
        return self._get_fallback_post(clean_sub, theme=theme, style=style)

    def _get_fallback_post(
        self,
        subreddit: str,
        theme: Optional[str] = None,
        style: str = "comedic",
    ) -> Dict[str, Any]:
        """Select a curated humorous fallback post for the target subreddit."""
        sub = subreddit.strip()
        matched_pool = None

        # Check exact or case-insensitive match
        for key in FALLBACK_TEMPLATES:
            if key.lower() == sub.lower():
                matched_pool = FALLBACK_TEMPLATES[key]
                break

        if not matched_pool:
            # If subreddit not in template map, use AskReddit templates as generic question pool
            matched_pool = FALLBACK_TEMPLATES["AskReddit"]

        chosen = random.choice(matched_pool)
        title = chosen["title"]
        body = chosen.get("body", "")

        # If theme was provided, try appending a themed spin if appropriate
        if theme and sub.lower() == "askreddit" and not any(w in title.lower() for w in theme.lower().split()):
            title = f"Regarding {theme.strip()}: {title}"

        return {
            "title": title,
            "body": body,
            "subreddit": sub,
            "style": chosen.get("style", style),
            "rationale": "Curated viral engagement template.",
            "is_fallback": True,
            "model_used": "template_fallback",
        }
