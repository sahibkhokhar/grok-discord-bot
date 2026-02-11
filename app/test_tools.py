#!/usr/bin/env python3
"""
Test local tools and TTS without Discord. Run from app/:
  python test_tools.py
"""
import os
import sys

# run from app/ so local_tools is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_tools import (
    get_time_date,
    roll_dice,
    calculate,
    text_tool,
    tts_speak,
    process_tool_tags,
    TTS_VOICES,
)


def main():
    print("=== Local tools test (no Discord) ===\n")

    # 1) Direct tool calls
    print("1) TIME")
    for q in ["current", "utc", "day", "countdown 2026-01-01"]:
        print(f"   [{q}] -> {get_time_date(q)}")

    print("\n2) DICE")
    for notation in ["1d6", "2d20", "4d6+2"]:
        print(f"   {notation} -> {roll_dice(notation)}")

    print("\n3) CALC")
    for expr in ["2+3*4", "sqrt(144)", "round(3.7)"]:
        print(f"   {expr} -> {calculate(expr)}")

    print("\n4) TEXT")
    print(f"   reverse|hello -> {text_tool('reverse|hello')}")
    print(f"   rot13|hello -> {text_tool('rot13|hello')}")
    print(f"   word_count|one two three -> {text_tool('word_count|one two three')}")

    # 2) Tag parsing (simulate model output)
    print("\n5) process_tool_tags() on sample text")
    sample = (
        "Right now it's [TIME]current[/TIME]. "
        "Lucky roll: [DICE]2d6[/DICE]. "
        "Quick math: [CALC]10*10[/CALC]. "
        "Reversed: [TEXT]reverse|discord[/TEXT]."
    )
    out, tts_path = process_tool_tags(sample, tts_enabled=False)
    print("   Input (excerpt): ... [TIME]current[/TIME] ... [DICE]2d6[/DICE] ...")
    print("   Output:", out[:200] + "..." if len(out) > 200 else out)
    if tts_path:
        print("   (TTS file was generated; would attach in Discord)")

    # 3) TTS (saves to file you can play)
    print("\n6) TTS (Pocket TTS)")
    result = tts_speak("Hello! This is a quick test of the text to speech.", voice="alba")
    if "error" in result:
        print("   TTS error:", result["error"])
        print("   (Install: pip install pocket-tts scipy; first run may download the model.)")
    else:
        path = result["file_path"]
        out_path = os.path.join(os.getcwd(), "tts_test_output.wav")
        try:
            import shutil
            shutil.copy(path, out_path)
            try:
                os.remove(path)
            except Exception:
                pass
            print("   Generated:", os.path.abspath(out_path))
            print("   Play with: open tts_test_output.wav  (macOS) or your media player.")
        except Exception as e:
            print("   Temp file:", path)
            print("   Error saving to cwd:", e)

    print("\nDone.")


if __name__ == "__main__":
    main()
