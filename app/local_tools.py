"""
Local-only tools for the Discord bot. Safe, no web access.
Used via tags in model output: [TIME]...[/TIME], [DICE]...[/DICE], etc.
"""
import logging
import os
import random
import re
import tempfile
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("discord-bot.tools")

# --- Safe calculator (sandboxed) ---
try:
    from math import sqrt
    from simpleeval import SimpleEval
    _calc = SimpleEval()
    _calc.functions["round"] = round
    _calc.functions["abs"] = abs
    _calc.functions["min"] = min
    _calc.functions["max"] = max
    _calc.functions["sqrt"] = sqrt
    HAS_SIMPLEEVAL = True
except ImportError:
    HAS_SIMPLEEVAL = False

# --- Pocket TTS (optional, heavier) ---
_tts_model = None
_tts_voice_states = {}

def _get_tts_model():
    global _tts_model
    if _tts_model is not None:
        return _tts_model
    try:
        from pocket_tts import TTSModel
        import torch
        _tts_model = TTSModel.load_model()
        return _tts_model
    except Exception as e:
        logger.warning("Pocket TTS not available: %s", e)
        return None


def get_time_date(query: str) -> str:
    """Return current time/date info. Query: 'current'|'utc'|'day'|'countdown YYYY-MM-DD'."""
    q = (query or "current").strip().lower()
    now = datetime.now(timezone.utc)
    if q == "utc":
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")
    if q == "current" or q == "":
        # Local time (server time)
        local = datetime.now()
        return local.strftime("%Y-%m-%d %H:%M:%S (server local)")
    if q == "day":
        return now.strftime("%A, %B %d, %Y")
    if q.startswith("countdown ") or q.startswith("countdown:"):
        date_str = q.replace("countdown ", "").replace("countdown:", "").strip()
        try:
            from datetime import datetime as dt
            target = dt.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            delta = target - now
            if delta.total_seconds() < 0:
                return f"{date_str} is in the past."
            d = delta.days
            return f"{d} days until {date_str}."
        except ValueError:
            return "Use countdown YYYY-MM-DD (e.g. countdown 2026-01-01)."
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


def roll_dice(notation: str) -> str:
    """Roll dice. Notation: 1d6, 2d20+3, 4d6, etc. Only d allowed (e.g. 2d6)."""
    notation = (notation or "1d6").strip().lower().replace(" ", "")
    # Allow formats: NdM, NdM+K, NdM-K
    match = re.match(r"^(\d+)d(\d+)([+-]\d+)?$", notation)
    if not match:
        return "Use format like 1d6, 2d20, or 4d6+2."
    n, m, mod = int(match.group(1)), int(match.group(2)), match.group(3)
    if n < 1 or n > 100:
        return "Number of dice must be 1–100."
    if m < 2 or m > 1000:
        return "Sides must be 2–1000."
    modifier = int(mod) if mod else 0
    rolls = [random.randint(1, m) for _ in range(n)]
    total = sum(rolls) + modifier
    detail = " + ".join(str(r) for r in rolls)
    if modifier != 0:
        detail += f" ({modifier:+d})"
    return f"Rolled {notation}: {detail} = **{total}**"


def calculate(expression: str) -> str:
    """Safely evaluate a math expression. Sandboxed (numbers and basic math only)."""
    if not HAS_SIMPLEEVAL:
        return "Calculator unavailable (install simpleeval)."
    expr = (expression or "").strip()
    if not expr:
        return "Provide an expression, e.g. 2+3*4 or sqrt(16)."
    # Remove any characters that could be dangerous
    if re.search(r"[a-zA-Z]", expr) and not re.search(r"\b(sqrt|sin|cos|tan|round|abs|min|max)\b", expr):
        return "Only numbers and operators (+, -, *, /, **, sqrt, etc.) allowed."
    try:
        _calc.names = {}
        result = _calc.eval(expr)
        if result is None:
            return "Could not evaluate."
        return f"`{expr}` = **{result}**"
    except Exception as e:
        return f"Invalid expression: {e}"


def text_tool(spec: str) -> str:
    """Transform text. Spec: action|text. Actions: reverse, rot13, upper, lower, word_count, char_count."""
    if "|" not in spec:
        return "Use format: action|text (e.g. reverse|hello)"
    part = spec.split("|", 1)
    action = (part[0] or "").strip().lower()
    text = (part[1] or "").strip()
    if not action or not text:
        return "Provide both action and text."
    if action == "reverse":
        return text[::-1]
    if action == "rot13":
        import codecs
        return codecs.encode(text, "rot_13")
    if action == "upper":
        return text.upper()
    if action == "lower":
        return text.lower()
    if action == "word_count":
        return str(len(text.split()))
    if action == "char_count":
        return str(len(text))
    return "Actions: reverse, rot13, upper, lower, word_count, char_count"


# Pocket TTS voices (from kyutai-labs/pocket-tts)
TTS_VOICES = ("alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma")


def tts_speak(text: str, voice: str = "alba") -> dict[str, Any]:
    """
    Generate speech from text using Pocket TTS. Returns {"file_path": ..., "filename": ...} or {"error": ...}.
    Only call with content the model itself produced (safe for Discord). Kept short for performance.
    """
    text = (text or "").strip()
    if not text:
        return {"error": "No text to speak."}
    # Limit length to avoid long generation
    if len(text) > 500:
        text = text[:497] + "..."
    voice = (voice or "alba").strip().lower()
    if voice not in TTS_VOICES:
        voice = "alba"
    model = _get_tts_model()
    if model is None:
        return {"error": "TTS not available (pocket-tts not installed or failed to load)."}
    try:
        if voice not in _tts_voice_states:
            _tts_voice_states[voice] = model.get_state_for_audio_prompt(voice)
        voice_state = _tts_voice_states[voice]
        audio = model.generate_audio(voice_state, text)
        import scipy.io.wavfile
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        scipy.io.wavfile.write(path, model.sample_rate, audio.numpy())
        return {"file_path": path, "filename": os.path.basename(path)}
    except Exception as e:
        logger.exception("TTS generation failed")
        return {"error": str(e)}


def run_tool(tag: str, inner: str, tts_enabled: bool = True) -> tuple[str, str | None]:
    """
    Run one tool by tag name. Returns (replacement_text, tts_file_path or None).
    inner is the content between [TAG]...[/TAG].
    """
    inner = (inner or "").strip()
    tts_path = None
    if tag == "TIME":
        out = get_time_date(inner)
    elif tag == "DICE":
        out = roll_dice(inner)
    elif tag == "CALC":
        out = calculate(inner)
    elif tag == "TEXT":
        out = text_tool(inner)
    elif tag == "TTS":
        if not tts_enabled:
            return "🔊 TTS is disabled.", None
        # Optional format: voice|text or just text
        if "|" in inner:
            voice_part, tts_text = inner.split("|", 1)
            voice_part, tts_text = voice_part.strip(), tts_text.strip()
            if voice_part.lower() in TTS_VOICES:
                result = tts_speak(tts_text, voice_part)
            else:
                result = tts_speak(inner)  # whole thing as text
        else:
            result = tts_speak(inner)
        if "error" in result:
            out = f"🔊 TTS error: {result['error']}"
        else:
            out = "🔊 *(audio attached)*"
            tts_path = result.get("file_path")
    else:
        out = f"[Unknown tool: {tag}]"
    return out, tts_path


def process_tool_tags(content: str, tts_enabled: bool = True) -> tuple[str, str | None]:
    """
    Find all [TAG]...[/TAG] in content, run tools, replace with results.
    Returns (new_content, path_to_tts_wav_or_None). Caller must delete the file after sending.
    """
    pattern = re.compile(r"\[(TIME|DICE|CALC|TEXT|TTS)\](.*?)\[/\1\]", re.DOTALL)
    tts_path = None

    def repl(m: re.Match) -> str:
        nonlocal tts_path
        tag, inner = m.group(1), m.group(2)
        out, path = run_tool(tag, inner, tts_enabled=tts_enabled)
        if path:
            tts_path = path
        return out

    new_content = pattern.sub(repl, content)
    return new_content, tts_path
