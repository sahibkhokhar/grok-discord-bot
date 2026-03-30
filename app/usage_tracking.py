"""
Persistent per-user token usage and estimated API cost (USD).
Pricing defaults follow public xAI / OpenAI docs as of early 2026; override with env vars.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("discord-bot.usage")

_lock = asyncio.Lock()

# Defaults: xAI Grok 3 Mini (https://docs.x.ai/docs/models)
DEFAULT_XAI_INPUT_PER_MILLION = 0.30
DEFAULT_XAI_OUTPUT_PER_MILLION = 0.50
# OpenAI gpt-5-nano (approximate public list pricing)
DEFAULT_OPENAI_INPUT_PER_MILLION = 0.05
DEFAULT_OPENAI_OUTPUT_PER_MILLION = 0.40
# xAI grok-imagine-image tier (docs / pricing page — standard image)
DEFAULT_IMAGE_COST_USD = 0.02

USAGE_STATS_FILE = Path(os.getenv("USAGE_STATS_FILE", "usage_stats.json")).resolve()

# Optional rough estimate when the API does not return usage (chars → tokens)
CHARS_PER_TOKEN_ESTIMATE = max(2, int(os.getenv("TOKEN_ESTIMATE_CHARS_PER_TOKEN", "4")))


def _per_million_rates() -> tuple[float, float]:
    """(input_usd_per_million, output_usd_per_million) for current text provider."""
    provider = os.getenv("AI_PROVIDER", "xai").lower()
    if provider == "openai":
        inp = float(os.getenv("OPENAI_PRICE_INPUT_PER_MILLION", DEFAULT_OPENAI_INPUT_PER_MILLION))
        out = float(os.getenv("OPENAI_PRICE_OUTPUT_PER_MILLION", DEFAULT_OPENAI_OUTPUT_PER_MILLION))
    else:
        inp = float(os.getenv("XAI_PRICE_INPUT_PER_MILLION", DEFAULT_XAI_INPUT_PER_MILLION))
        out = float(os.getenv("XAI_PRICE_OUTPUT_PER_MILLION", DEFAULT_XAI_OUTPUT_PER_MILLION))
    return inp, out


def image_cost_usd_each() -> float:
    return float(os.getenv("IMAGE_COST_USD_PER_IMAGE", DEFAULT_IMAGE_COST_USD))


def _usd_for_tokens(prompt: int, completion: int) -> float:
    inp_rate, out_rate = _per_million_rates()
    return (prompt / 1_000_000.0) * inp_rate + (completion / 1_000_000.0) * out_rate


def estimate_tokens_from_text(text: str) -> tuple[int, int]:
    """Rough split: treat all as prompt+completion combined half/half if API omitted usage."""
    n = max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)
    half = n // 2
    return half, n - half


@dataclass
class UsageDelta:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    images: int = 0

    def add_json_usage(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        if pt is None and "input_tokens" in usage:
            pt = usage.get("input_tokens")
            ct = usage.get("output_tokens")
        self.prompt_tokens += int(pt or 0)
        self.completion_tokens += int(ct or 0)


def merge_holder_into_delta(holder: dict[str, Any] | None, delta: UsageDelta) -> None:
    if not holder:
        return
    delta.add_json_usage(holder)


async def _read_file() -> dict[str, Any]:
    if not USAGE_STATS_FILE.exists():
        return {"version": 1, "users": {}}
    try:
        text = USAGE_STATS_FILE.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data.get("users"), dict):
            data["users"] = {}
        data.setdefault("version", 1)
        return data
    except Exception:
        logger.exception("Failed to read usage stats; starting fresh")
        return {"version": 1, "users": {}}


def _write_file(data: dict[str, Any]) -> None:
    USAGE_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = USAGE_STATS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(USAGE_STATS_FILE)


async def record_usage(
    user_id: int,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    images: int = 0,
) -> None:
    """Add usage for one user. Safe to call with all zeros (no-op)."""
    if prompt_tokens <= 0 and completion_tokens <= 0 and images <= 0:
        return
    async with _lock:
        data = await _read_file()
        users: dict[str, Any] = data["users"]
        key = str(user_id)
        row = users.get(key) or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "chat_calls": 0,
            "images": 0,
            "estimated_cost_usd": 0.0,
        }
        row["prompt_tokens"] = int(row["prompt_tokens"]) + prompt_tokens
        row["completion_tokens"] = int(row["completion_tokens"]) + completion_tokens
        row["total_tokens"] = int(row["total_tokens"]) + prompt_tokens + completion_tokens
        if prompt_tokens or completion_tokens:
            row["chat_calls"] = int(row["chat_calls"]) + 1
        if images:
            row["images"] = int(row["images"]) + images
        extra_cost = _usd_for_tokens(prompt_tokens, completion_tokens)
        extra_cost += images * image_cost_usd_each()
        row["estimated_cost_usd"] = float(row["estimated_cost_usd"]) + extra_cost
        users[key] = row
        await asyncio.to_thread(_write_file, data)


async def get_user_row(user_id: int) -> dict[str, Any] | None:
    async with _lock:
        data = await _read_file()
        return data["users"].get(str(user_id))


async def leaderboard_rows(limit: int = 15) -> list[tuple[int, dict[str, Any]]]:
    async with _lock:
        data = await _read_file()
        users = data["users"]
        parsed: list[tuple[int, dict[str, Any]]] = []
        for uid_s, row in users.items():
            try:
                uid = int(uid_s)
            except ValueError:
                continue
            parsed.append((uid, row))
        parsed.sort(
            key=lambda x: (
                float(x[1].get("estimated_cost_usd") or 0),
                int(x[1].get("total_tokens") or 0),
            ),
            reverse=True,
        )
        return parsed[: max(1, limit)]


def format_mystats_embed_body(row: dict[str, Any] | None) -> tuple[str, str]:
    """Returns (title, description) for a Discord embed."""
    if not row:
        return (
            "Your usage",
            "No API usage recorded for you yet. Mention the bot or use `/pic` to start.",
        )
    pt = int(row.get("prompt_tokens") or 0)
    ct = int(row.get("completion_tokens") or 0)
    tt = int(row.get("total_tokens") or 0)
    calls = int(row.get("chat_calls") or 0)
    imgs = int(row.get("images") or 0)
    usd = float(row.get("estimated_cost_usd") or 0.0)
    inp_r, out_r = _per_million_rates()
    lines = [
        f"**Chat requests:** {calls}",
        f"**Prompt tokens:** {pt:,}",
        f"**Completion tokens:** {ct:,}",
        f"**Total tokens:** {tt:,}",
        f"**Images generated:** {imgs}",
        f"**Estimated spend:** ${usd:.4f} USD",
        "",
        "_Estimates use your configured model rates "
        f"(input ${inp_r:.2f}/1M, output ${out_r:.2f}/1M tokens; "
        f"${image_cost_usd_each():.2f} per image). Actual invoices may differ._",
    ]
    return "Your Grok usage", "\n".join(lines)
