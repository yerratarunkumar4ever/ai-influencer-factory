"""
Core pipeline logic — mirrors the n8n workflow:
Form → AI Agent (Claude or GPT) → NanoBanana Pro (image) → Kling 2.6 (video) → Telegram

AI provider priority: ANTHROPIC_API_KEY → OPENAI_API_KEY
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert AI influencer content director. Your job is to create highly detailed,
photorealistic prompts that generate Instagram/TikTok-quality content for AI characters.

## Role
Creative director for an AI influencer — you understand Instagram aesthetics, UGC content,
and what makes posts feel authentic and engaging.

## Output Count
- Image-only posts: generate image_prompt only, set post_type to "image"
- Video posts: generate BOTH image_prompt AND video_prompt, set post_type to "video"
- The image for video posts serves as the first frame that gets animated

## Constraints
1. EVERY image_prompt must describe the character faithfully — same age, ethnicity, hair,
   skin, features across ALL posts
2. Image prompts must be detailed (150-250 words): character appearance, outfit, setting,
   lighting, camera angle, mood, composition
3. Style: "casual iPhone photo" — authentic, amateur, NOT professional studio
4. If a setting image URL is provided: match that setting
5. If an item/product URL is provided: character holds/wears/interacts with it naturally
6. ALL text, logos, labels on products must be preserved exactly

## Image Prompt Guidelines
- Include full character description matching the reference
- Natural everyday settings (apartment, cafe, street, gym, beach, car, bathroom mirror)
- Camera: handheld iPhone, slightly uneven framing, amateur quality
- Include lighting details (golden hour, morning light, warm lamp, etc.)
- Character looks candid and natural, not posed

## Video Prompt Guidelines
- Character in motion: talking to camera, showing product, reacting
- Include a casual dialogue line (like talking to a friend)
- Natural movements: holding something up, turning it, gesturing
- Camera: handheld iPhone feel, slight shake
- Keep under 300 characters
- Video animates what changes from the generated first-frame image

## Output
Return a valid JSON object:
{
  "posts": [
    {
      "title": "2-5 word title",
      "caption": "Instagram caption with hook, emojis, hashtags",
      "post_type": "image" or "video",
      "image_prompt": "detailed 150-250 word prompt",
      "video_prompt": "under 300 char animation prompt, or null for image posts"
    }
  ]
}
"""


@dataclass
class PipelineConfig:
    openai_api_key: str
    kie_ai_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    groq_model: str = "llama-3.3-70b-versatile"

    app_base_url: str = ""

    @property
    def ai_provider(self) -> str:
        """Priority: anthropic → groq → openai → none"""
        if self.anthropic_api_key:
            return "anthropic"
        if self.groq_api_key:
            return "groq"
        if self.openai_api_key:
            return "openai"
        return "none"


# ---------------------------------------------------------------------------
# Step 1 — AI prompt generation (Claude preferred, OpenAI fallback)
# ---------------------------------------------------------------------------

def _build_user_msg(form_data: dict) -> str:
    character_brief = form_data.get("character_brief") or (
        "No character brief provided. Infer personality and style from the creative "
        "direction and image references. Make them relatable, authentic, engaging."
    )
    return (
        f"Generate prompts for the following request. Return valid JSON.\n\n"
        f"Image posts needed: {form_data['num_images']}\n"
        f"Video posts needed: {form_data['num_videos']}\n"
        f"Aspect ratio: {form_data['aspect_ratio']}\n\n"
        f"CREATIVE DIRECTION: {form_data['creative_direction']}\n\n"
        f"IMAGE REFERENCES:\n"
        f"character_image: {form_data['character_url']}\n"
        f"setting_image: {form_data.get('setting_url') or 'Not provided'}\n"
        f"item_image: {form_data.get('item_url') or 'Not provided'}\n\n"
        f"CHARACTER BRIEF:\n{character_brief}"
    )


async def _generate_with_anthropic(user_msg: str, config: PipelineConfig, log: Callable) -> list:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
    log(f"Calling Claude ({config.anthropic_model}) to generate post prompts...")

    message = await client.messages.create(
        model=config.anthropic_model,
        max_tokens=8192,
        system=SYSTEM_PROMPT + "\n\nIMPORTANT: Your entire response must be valid JSON only — no markdown, no explanation.",
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    return result.get("posts", [])


async def _generate_with_openai_compat(
    user_msg: str, api_key: str, base_url: str, model: str, label: str, log: Callable
) -> list:
    """Shared logic for OpenAI-compatible APIs (OpenAI, Groq, etc.)."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    log(f"Calling {label} ({model}) to generate post prompts...")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        max_tokens=8192,
        temperature=0.8,
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("posts", [])


async def generate_prompts(form_data: dict, config: PipelineConfig, log: Callable) -> list:
    if config.ai_provider == "none":
        raise ValueError(
            "No AI key configured. Set GROQ_API_KEY (free), ANTHROPIC_API_KEY, "
            "or OPENAI_API_KEY in Railway Variables."
        )

    user_msg = _build_user_msg(form_data)

    if config.ai_provider == "anthropic":
        posts = await _generate_with_anthropic(user_msg, config, log)
    elif config.ai_provider == "groq":
        posts = await _generate_with_openai_compat(
            user_msg,
            api_key=config.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            model=config.groq_model,
            label="Groq",
            log=log,
        )
    else:
        posts = await _generate_with_openai_compat(
            user_msg,
            api_key=config.openai_api_key,
            base_url="https://api.openai.com/v1",
            model=config.openai_model,
            label="OpenAI",
            log=log,
        )

    log(f"AI generated {len(posts)} post prompts")
    return posts


# ---------------------------------------------------------------------------
# Step 2 — NanoBanana Pro image generation (kie.ai)
# ---------------------------------------------------------------------------

def _jpeg_proxy(url: str, app_base_url: str) -> str:
    """Route any image through our /api/img/input.jpg endpoint.
    The URL ends in .jpg AND returns image/jpeg, satisfying kie.ai's checks.
    Pillow on the server converts WebP/AVIF/HEIC/PNG → JPEG transparently."""
    if not url:
        return url
    import urllib.parse
    encoded = urllib.parse.quote(url, safe="")
    return f"{app_base_url}/api/img/input.jpg?url={encoded}"


async def create_image_task(post: dict, config: PipelineConfig) -> str:
    proxy = lambda u: _jpeg_proxy(u, config.app_base_url)
    image_inputs = [proxy(post["character_url"])]
    if post.get("item_url"):
        image_inputs.append(proxy(post["item_url"]))
    if post.get("setting_url"):
        image_inputs.append(proxy(post["setting_url"]))

    payload = {
        "model": "nano-banana-pro",
        "input": {
            "prompt": (
                "CRITICAL: Preserve ALL text, labels, logos on products EXACTLY. "
                + post["image_prompt"]
            ),
            "image_input": image_inputs,
            "aspect_ratio": post["aspect_ratio"],
            "resolution": "2K",
            "output_format": "jpg",
        },
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.kie.ai/api/v1/jobs/createTask",
            headers={"Authorization": f"Bearer {config.kie_ai_api_key}"},
            json=payload,
        )
        body = resp.json()
        if not resp.is_success:
            raise ValueError(f"kie.ai HTTP {resp.status_code}: {body}")
        data = body.get("data")
        if not data or not data.get("taskId"):
            raise ValueError(f"kie.ai unexpected response: {body}")
        return data["taskId"]


# ---------------------------------------------------------------------------
# Step 3 — Poll until task finishes (images: 45s interval, videos: 120s)
# ---------------------------------------------------------------------------

async def poll_task(
    task_id: str,
    config: PipelineConfig,
    log: Callable,
    poll_interval: int = 45,
    max_attempts: int = 24,
) -> Optional[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, max_attempts + 1):
            resp = await client.get(
                f"https://api.kie.ai/api/v1/jobs/recordInfo?taskId={task_id}",
                headers={"Authorization": f"Bearer {config.kie_ai_api_key}"},
            )
            body = resp.json()
            if not resp.is_success:
                log(f"kie.ai poll error HTTP {resp.status_code}: {body}", "error")
                return None
            data = body.get("data", {})
            state = data.get("state", "failed")

            if state == "success":
                log(f"Task {task_id[:8]}… completed")
                return data

            if "ing" in state.lower():  # queuing / processing / etc.
                log(
                    f"Task {task_id[:8]}… {state} "
                    f"(attempt {attempt}/{max_attempts}, waiting {poll_interval}s)"
                )
                await asyncio.sleep(poll_interval)
            else:
                log(f"Task {task_id[:8]}… failed — state={state}", "error")
                return None

    log(f"Task {task_id[:8]}… timed out after {max_attempts} attempts", "error")
    return None


# ---------------------------------------------------------------------------
# Step 4 — Kling 2.6 video generation (kie.ai)
# ---------------------------------------------------------------------------

async def create_video_task(video_prompt: str, image_url: str, config: PipelineConfig) -> str:
    payload = {
        "model": "kling-2.6/image-to-video",
        "input": {
            "prompt": video_prompt,
            "image_urls": [image_url],
            "duration": "5",
            "sound": True,
        },
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.kie.ai/api/v1/jobs/createTask",
            headers={"Authorization": f"Bearer {config.kie_ai_api_key}"},
            json=payload,
        )
        body = resp.json()
        if not resp.is_success:
            raise ValueError(f"kie.ai HTTP {resp.status_code}: {body}")
        data = body.get("data")
        if not data or not data.get("taskId"):
            raise ValueError(f"kie.ai unexpected response: {body}")
        return data["taskId"]


# ---------------------------------------------------------------------------
# Step 5 — Telegram delivery helpers
# ---------------------------------------------------------------------------

async def _tg(endpoint: str, payload: dict, config: PipelineConfig, timeout: int = 60):
    if not config.telegram_bot_token or not config.telegram_chat_id:
        return
    async with httpx.AsyncClient(timeout=timeout) as client:
        await client.post(
            f"https://api.telegram.org/bot{config.telegram_bot_token}/{endpoint}",
            json=payload,
        )


async def tg_photo(url: str, caption: str, config: PipelineConfig):
    await _tg("sendPhoto", {"chat_id": config.telegram_chat_id, "photo": url, "caption": caption[:1024]}, config)


async def tg_video(url: str, caption: str, config: PipelineConfig):
    await _tg("sendVideo", {"chat_id": config.telegram_chat_id, "video": url, "caption": caption[:1024]}, config, timeout=120)


async def tg_msg(text: str, config: PipelineConfig):
    await _tg("sendMessage", {"chat_id": config.telegram_chat_id, "text": text}, config)


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

async def run_pipeline(
    job_id: str,
    form_data: dict,
    config: PipelineConfig,
    jobs: dict,
    job_logs: dict,
):
    def log(msg: str, level: str = "info"):
        entry = {"time": datetime.now().isoformat(), "msg": msg, "level": level}
        job_logs.setdefault(job_id, []).append(entry)
        logger.info("[%s] [%s] %s", job_id, level.upper(), msg)

    try:
        jobs[job_id]["status"] = "running"
        log("Pipeline started")

        # ── 1. Generate AI prompts ──────────────────────────────────────────
        posts = await generate_prompts(form_data, config, log)
        if not posts:
            raise ValueError("AI returned no posts")

        jobs[job_id]["posts"] = posts
        log(f"Posts to create: {[p['title'] for p in posts]}")

        # ── 2. Process each post ────────────────────────────────────────────
        for i, post in enumerate(posts):
            # Attach form references so we can pass them to the API
            post["character_url"] = form_data["character_url"]
            post["setting_url"] = form_data.get("setting_url", "")
            post["item_url"] = form_data.get("item_url", "")
            post["aspect_ratio"] = form_data["aspect_ratio"]
            post["status"] = "generating_image"
            jobs[job_id]["posts"] = list(posts)  # snapshot

            n = f"[Post {i + 1}/{len(posts)} '{post['title']}']"
            log(f"{n} Starting ({post['post_type']})")

            # ── Image ───────────────────────────────────────────────────────
            log(f"{n} Submitting to NanoBanana Pro…")
            try:
                img_task_id = await create_image_task(post, config)
                log(f"{n} Image task: {img_task_id}")
            except Exception as exc:
                log(f"{n} Failed to submit image task: {exc}", "error")
                posts[i]["status"] = "image_failed"
                jobs[job_id]["posts"] = list(posts)
                await tg_msg(f"Image FAILED: {post['title']}\n{exc}", config)
                continue

            log(f"{n} Waiting 45s before first poll…")
            await asyncio.sleep(45)

            img_data = await poll_task(img_task_id, config, log, poll_interval=45)

            if not img_data:
                log(f"{n} Image generation failed", "error")
                posts[i]["status"] = "image_failed"
                jobs[job_id]["posts"] = list(posts)
                await tg_msg(
                    f"Image FAILED: {post['title']}\n"
                    f"Prompt: {post['image_prompt'][:200]}…",
                    config,
                )
                continue

            try:
                image_url = json.loads(img_data["resultJson"])["resultUrls"][0]
            except Exception as exc:
                log(f"{n} Cannot parse image URL: {exc}", "error")
                posts[i]["status"] = "image_failed"
                jobs[job_id]["posts"] = list(posts)
                continue

            posts[i]["image_url"] = image_url
            posts[i]["status"] = "image_done"
            jobs[job_id]["posts"] = list(posts)
            log(f"{n} Image ready → {image_url}")

            caption = f"{post['title']}\n\n{post['caption']}"
            await tg_photo(image_url, caption, config)
            log(f"{n} Image sent to Telegram")

            # ── Video (only for video posts) ────────────────────────────────
            if post["post_type"] == "video" and post.get("video_prompt"):
                posts[i]["status"] = "generating_video"
                jobs[job_id]["posts"] = list(posts)

                log(f"{n} Submitting to Kling 2.6…")
                try:
                    vid_task_id = await create_video_task(
                        post["video_prompt"], image_url, config
                    )
                    log(f"{n} Video task: {vid_task_id}")
                except Exception as exc:
                    log(f"{n} Failed to submit video task: {exc}", "error")
                    posts[i]["status"] = "video_failed"
                    jobs[job_id]["posts"] = list(posts)
                    await tg_msg(f"Video FAILED: {post['title']}\nImage OK: {image_url}", config)
                    continue

                log(f"{n} Waiting 2min before first poll…")
                await asyncio.sleep(120)

                vid_data = await poll_task(vid_task_id, config, log, poll_interval=120)

                if not vid_data:
                    log(f"{n} Video generation failed", "error")
                    posts[i]["status"] = "video_failed"
                    jobs[job_id]["posts"] = list(posts)
                    await tg_msg(f"Video FAILED: {post['title']}\nImage OK: {image_url}", config)
                    continue

                try:
                    video_url = json.loads(vid_data["resultJson"])["resultUrls"][0]
                except Exception as exc:
                    log(f"{n} Cannot parse video URL: {exc}", "error")
                    posts[i]["status"] = "video_failed"
                    jobs[job_id]["posts"] = list(posts)
                    continue

                posts[i]["video_url"] = video_url
                posts[i]["status"] = "done"
                jobs[job_id]["posts"] = list(posts)
                log(f"{n} Video ready → {video_url}")

                await tg_video(video_url, caption, config)
                log(f"{n} Video sent to Telegram")
                await tg_msg(
                    f"✅ Video post complete!\n\nTitle: {post['title']}\n"
                    f"Image: {image_url}\nVideo: {video_url}",
                    config,
                )
            else:
                posts[i]["status"] = "done"
                jobs[job_id]["posts"] = list(posts)

            log(f"{n} Done!")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        log("All posts complete!", "success")

    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
        log(f"Pipeline failed: {exc}", "error")
        logger.exception("[%s] Unhandled pipeline error", job_id)
