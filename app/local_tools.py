"""
Local-only tools for the Discord bot. Safe, no web access.
Exposed to the LLM via API function calling (agent loop), not bracket tags.
"""
import json
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


TOOL_TAG_NAMES = ("TIME", "DICE", "CALC", "TEXT", "TTS")

# --- API function calling (OpenAI Responses + xAI chat completions) ---

_LOCAL_TOOL_SPECS_BASE: list[dict[str, Any]] = [
    {
        "name": "local_get_time",
        "description": (
            "Get current date/time on the bot's server. Use when the user asks what time it is, "
            "today's date, or days until a date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "One of: current (server local time), utc, day (weekday and calendar date), "
                        "or countdown YYYY-MM-DD (e.g. countdown 2026-12-25)."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "local_roll_dice",
        "description": "Roll dice. Notation like 1d6, 2d20+3, 4d6.",
        "parameters": {
            "type": "object",
            "properties": {
                "notation": {
                    "type": "string",
                    "description": "Dice notation, e.g. 1d6, 2d20+3",
                },
            },
            "required": ["notation"],
            "additionalProperties": False,
        },
    },
    {
        "name": "local_calculate",
        "description": "Safely evaluate a numeric math expression (no arbitrary code).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression e.g. 2+3*4, sqrt(16)",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    {
        "name": "local_transform_text",
        "description": (
            "Transform text: reverse, rot13, upper, lower, word_count, or char_count. "
            "Pass spec as 'action|text' e.g. reverse|hello."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "string",
                    "description": "Format: action|text (e.g. upper|hello, word_count|foo bar)",
                },
            },
            "required": ["spec"],
            "additionalProperties": False,
        },
    },
]

_TTS_TOOL_SPEC: dict[str, Any] = {
    "name": "local_speak_tts",
    "description": (
        "Generate short spoken audio for the user (attached to the bot's reply). "
        "Only for explicit requests to say/speak/read aloud. Keep phrase to one or two short sentences. "
        "Voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "phrase": {
                "type": "string",
                "description": "Short text for the bot to speak (appropriate for Discord).",
            },
            "voice": {
                "type": "string",
                "description": "Optional voice name; default alba.",
            },
        },
        "required": ["phrase"],
        "additionalProperties": False,
    },
}


def build_responses_api_tools(tts_enabled: bool) -> list[dict[str, Any]]:
    """Tools list for OpenAI Responses API (flat function shape)."""
    specs = list(_LOCAL_TOOL_SPECS_BASE)
    if tts_enabled:
        specs.append(_TTS_TOOL_SPEC)
    return [{"type": "function", **s} for s in specs]


def build_chat_completion_tools(tts_enabled: bool) -> list[dict[str, Any]]:
    """Tools list for xAI / OpenAI chat.completions (nested under function)."""
    out: list[dict[str, Any]] = []
    for spec in _LOCAL_TOOL_SPECS_BASE:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
        )
    if tts_enabled:
        s = _TTS_TOOL_SPEC
        out.append(
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["parameters"],
                },
            }
        )
    return out


def dispatch_local_function(
    name: str,
    arguments_json: str,
    *,
    tts_enabled: bool,
) -> tuple[str, str | None]:
    """
    Run one function call from the model. Returns (output_string_for_model, tts_wav_path_or_none).
    """
    tts_path: str | None = None
    try:
        args = json.loads(arguments_json) if (arguments_json or "").strip() else {}
    except json.JSONDecodeError:
        return "Error: invalid JSON in function arguments.", None

    if name == "local_get_time":
        out = get_time_date(str(args.get("query", "current")))
    elif name == "local_roll_dice":
        out = roll_dice(str(args.get("notation", "1d6")))
    elif name == "local_calculate":
        out = calculate(str(args.get("expression", "")))
    elif name == "local_transform_text":
        out = text_tool(str(args.get("spec", "")))
    elif name == "local_speak_tts":
        if not tts_enabled:
            out = "TTS is disabled."
        else:
            phrase = str(args.get("phrase", "")).strip()
            voice = str(args.get("voice", "alba")).strip()
            result = tts_speak(phrase, voice)
            if "error" in result:
                out = f"TTS error: {result['error']}"
            else:
                out = (
                    "Success: audio was generated and will be attached to your message for the user. "
                    "Briefly acknowledge in text if helpful."
                )
                tts_path = result.get("file_path")
    else:
        out = f"Unknown function: {name}"

    return str(out), tts_path


def tool_tag_displayable_prefix(text: str) -> str:
    """
    Longest prefix of text that does not end inside an unclosed [TIME|DICE|...] tag.
    Avoids showing raw tool tags while they stream in chunk by chunk.
    """
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        matched: str | None = None
        for name in TOOL_TAG_NAMES:
            open_seq = f"[{name}]"
            if text.startswith(open_seq, i):
                matched = name
                break
        if matched is None:
            i += 1
            continue
        close_seq = f"[/{matched}]"
        inner_start = i + len(f"[{matched}]")
        j = text.find(close_seq, inner_start)
        if j == -1:
            return text[:i]
        i = j + len(close_seq)
    return text


def run_tool(
    tag: str,
    inner: str,
    tts_enabled: bool = True,
    defer_tts: bool = False,
) -> tuple[str, str | None]:
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
        if defer_tts:
            return "🔊 *(voice attached below)*", None
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


def process_tool_tags(
    content: str,
    tts_enabled: bool = True,
    defer_tts: bool = False,
) -> tuple[str, str | None]:
    """
    Find all [TAG]...[/TAG] in content, run tools, replace with results.
    Returns (new_content, path_to_tts_wav_or_None). Caller must delete the file after sending.
    If defer_tts is True, TTS tags are replaced with a placeholder and no audio is generated
    (use while streaming; run again with defer_tts=False for the final message).
    """
    pattern = re.compile(r"\[(TIME|DICE|CALC|TEXT|TTS)\](.*?)\[/\1\]", re.DOTALL)
    tts_path = None

    def repl(m: re.Match) -> str:
        nonlocal tts_path
        tag, inner = m.group(1), m.group(2)
        out, path = run_tool(
            tag, inner, tts_enabled=tts_enabled, defer_tts=defer_tts
        )
        if path:
            tts_path = path
        return out

    new_content = pattern.sub(repl, content)
    return new_content, tts_path
