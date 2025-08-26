#!/usr/bin/env python3
"""
speak_feedback.py
-----------------
Turn a Markdown feedback file (produced by coach_feedback.py) into a short,
friendly voice-based suggestion for the patient.

Backends:
  - mac     : macOS 'say' (no extra deps)
  - pyttsx3 : offline TTS (pip install pyttsx3)
  - eleven  : ElevenLabs TTS (set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID)

Usage examples:
  python speak_feedback.py --md feedback_claspandspread.md --tts mac --out tip.m4a --play
  python speak_feedback.py --md feedback_claspandspread.md --tts eleven --out tip.mp3
  python speak_feedback.py --md feedback_claspandspread.md --tts pyttsx3 --out tip.wav
"""

import os, re, argparse, subprocess

def read_md(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_sections(md: str):
    """Return a dict of key sections if present."""
    sections = {}
    md = md.replace("\r\n", "\n")
    patterns = {
        "summary": r"(?im)^\s*1\)\s*Summary.*?\n(.*?)(?=\n\s*\d\)|\Z)",
        "strengths": r"(?im)^\s*2\)\s*Strengths.*?\n(.*?)(?=\n\s*\d\)|\Z)",
        "corrections": r"(?im)^\s*3\)\s*Top corrections.*?\n(.*?)(?=\n\s*\d\)|\Z)",
        "step_notes": r"(?im)^\s*4\)\s*Step-specific.*?\n(.*?)(?=\n\s*\d\)|\Z)",
        "next": r"(?im)^\s*5\)\s*Next Session Plan.*?\n(.*?)(?=\n\s*\d\)|\Z)",
        "safety": r"(?im)^\s*6\)\s*Safety.*?\n(.*?)(?=\n\s*\d\)|\Z)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, md, flags=re.DOTALL)
        if m:
            sections[key] = m.group(1).strip()
    return sections

def bullet_lines(text: str):
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if re.match(r"^[-*•]\s+", ln):
            ln = re.sub(r"^[-*•]\s+", "", ln).strip()
            lines.append(ln)
        else:
            lines.append(ln)
    return lines

def build_tip_text(md: str, exercise_hint: str = "") -> str:
    s = extract_sections(md)
    lines = []

    # Friendly opening
    if exercise_hint:
        lines.append(f"Nice work on {exercise_hint}!")
    else:
        lines.append("Nice work on your exercise!")

    # Pull summary
    if "summary" in s:
        summary_first = bullet_lines(s["summary"])
        if summary_first:
            lines.append(summary_first[0])

    # 1–2 strengths
    if "strengths" in s:
        st = bullet_lines(s["strengths"])[:2]
        if st:
            lines.append("What went well: " + "; ".join(st))

    # 1–2 corrections
    if "corrections" in s:
        corr = bullet_lines(s["corrections"])[:2]
        if corr:
            lines.append("Quick fix to try: " + " | ".join(corr))

    # Next step
    if "next" in s:
        nxt = bullet_lines(s["next"])[:1]
        if nxt:
            lines.append("Next session focus: " + nxt[0])

    # Gentle close
    lines.append("Keep it slow, pain-free, and breathe with each movement.")

    text = " ".join(lines)
    if len(text) > 400:
        text = text[:400].rsplit(" ", 1)[0] + "..."
    return text

def tts_mac_say(text: str, out_path: str, voice: str = "Samantha", rate: int = 190, play: bool = False):
    base, ext = os.path.splitext(out_path)
    aiff = base + ".aiff"
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", aiff, text], check=True)
    if ext.lower() != ".aiff":
        try:
            subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", aiff, out_path], check=True)
            os.remove(aiff)
        except Exception:
            out_path = aiff
    if play:
        try:
            subprocess.run(["afplay", out_path], check=False)
        except Exception:
            pass
    return out_path

def tts_pyttsx3(text: str, out_path: str):
    try:
        import pyttsx3
    except Exception:
        raise RuntimeError("pyttsx3 not installed. Try: pip install pyttsx3")
    engine = pyttsx3.init()
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path

def tts_elevenlabs(text: str, out_path: str, voice_id: str = None):
    import requests
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set.")
    if not voice_id:
        voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/mpeg",
        "content-type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.7}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True, help="Path to feedback markdown")
    ap.add_argument("--exercise", default="", help="Exercise hint, e.g., 'Clasp and Spread'")
    ap.add_argument("--tts", choices=["mac","pyttsx3","eleven"], default="mac")
    ap.add_argument("--out", default="tip.m4a", help="Output audio file")
    ap.add_argument("--voice", default="Samantha", help="Voice name (mac) or ElevenLabs voice")
    ap.add_argument("--rate", type=int, default=190, help="mac 'say' rate")
    ap.add_argument("--play", action="store_true", help="Auto-play after generation (mac only)")
    args = ap.parse_args()

    md = read_md(args.md)
    text = build_tip_text(md, exercise_hint=args.exercise)

    if args.tts == "mac":
        outp = tts_mac_say(text, args.out, voice=args.voice, rate=args.rate, play=args.play)
    elif args.tts == "pyttsx3":
        outp = tts_pyttsx3(text, args.out)
    else:
        outp = tts_elevenlabs(text, args.out)

    print(f"[OK] Voice tip saved to {outp}")
    print(f"[TIP TEXT]\n{text}\n")

if __name__ == "__main__":
    main()