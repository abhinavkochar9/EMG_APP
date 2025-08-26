#!/usr/bin/env python3
"""
coach_feedback.py  (exercise-aware RAG)
--------------------------------------------------
Now supports --exercise to prioritize the matching TXT description doc(s)
(e.g., Clasp_and_Spread.txt), injecting a concise "Exercise Reference" summary
into the prompt for higher-quality, coach-style feedback.
"""

import os, re, glob, json, argparse, pathlib, textwrap
from datetime import datetime
from typing import List, Tuple, Dict
import pandas as pd

# Optional TF-IDF retriever
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

def load_csv_report(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    header_pairs = []
    for ln in lines[:8]:
        if ',' in ln:
            k, v = ln.split(',', 1)
            header_pairs.append((k.strip(), v.strip()))
    meta = {k: v for k, v in header_pairs if k.lower() not in (',average score by repetition', ',average score by step') and len(k) > 1}

    rep_rows, step_rows, mode = [], [], None
    for ln in lines:
        if ln.lower().startswith(',average score by repetition'):
            mode = 'rep'; continue
        if ln.lower().startswith(',average score by step'):
            mode = 'step'; continue
        parts = [p.strip() for p in ln.split(',')]
        if mode == 'rep' and len(parts) == 2 and parts[0].isdigit():
            try: rep_rows.append({'repetition': int(parts[0]), 'score': float(parts[1])})
            except: pass
        elif mode == 'step' and len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            try: step_rows.append({'repetition': int(parts[0]), 'step': int(parts[1]), 'score': float(parts[2])})
            except: pass

    rep_df  = pd.DataFrame(rep_rows)  if rep_rows  else pd.DataFrame(columns=['repetition','score'])
    step_df = pd.DataFrame(step_rows) if step_rows else pd.DataFrame(columns=['repetition','step','score'])

    overall_avg = None
    if 'Overall Average Score' in meta:
        try: overall_avg = float(str(meta['Overall Average Score']).split()[0])
        except: pass
    if overall_avg is None and not step_df.empty:
        overall_avg = float(step_df['score'].mean())

    weak_steps = []
    if not step_df.empty:
        th = (overall_avg or step_df['score'].mean()) - 10.0
        flags = step_df[step_df['score'] < th].sort_values('score')
        weak_steps = [(int(r['repetition']), int(r['step']), float(r['score'])) for _, r in flags.iterrows()]

    parts = [f"Report file: {os.path.basename(path)}"]
    for k, v in meta.items():
        parts.append(f"{k}: {v}")
    if not rep_df.empty:
        parts.append("Average Score by Repetition: " + "; ".join([f"R{int(r)}={s:.1f}" for r, s in rep_df.values]))
    if not step_df.empty:
        parts.append("Average Score by Step: " + "; ".join([f"R{int(r)}-S{int(s)}={v:.1f}" for r, s, v in step_df.values]))
    blob = "\\n".join(parts)

    return {
        "meta": meta, "rep_df": rep_df, "step_df": step_df,
        "overall_avg": overall_avg, "weak_steps": weak_steps, "blob": blob, "path": path
    }

def normalize_exercise_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.strip().lower())

def collect_corpus(reports_dir: str) -> List[Tuple[str, str]]:
    paths = []
    for ext in ("*.csv", "*.txt", "*.md"):
        paths.extend(glob.glob(os.path.join(reports_dir, ext)))
    docs = []
    for p in paths:
        try:
            if p.lower().endswith(".csv"):
                d = load_csv_report(p)
                docs.append((p, d["blob"]))
            else:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    docs.append((p, f.read()))
        except Exception:
            pass
    return docs

def build_retriever(docs: List[Tuple[str, str]]):
    if not docs:
        return lambda q, k=5: []
    ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]
    if SKLEARN_AVAILABLE:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
        X = vectorizer.fit_transform(texts)
        def query(q: str, k: int = 5):
            from sklearn.metrics.pairwise import cosine_similarity
            qv = vectorizer.transform([q])
            sims = (X @ qv.T).toarray().ravel()
            order = sims.argsort()[::-1][:k]
            return [(ids[i], texts[i], float(sims[i])) for i in order if sims[i] > 0]
        return query
    else:
        def query(q: str, k: int = 5):
            qw = set(re.findall(r"\w+", q.lower()))
            scored = []
            for i, t in enumerate(texts):
                tw = set(re.findall(r"\w+", t.lower()))
                score = len(qw & tw)
                scored.append((ids[i], texts[i], float(score)))
            scored.sort(key=lambda x: x[2], reverse=True)
            return [s for s in scored[:k] if s[2] > 0]
        return query

def find_exercise_docs(docs: List[Tuple[str,str]], exercise: str) -> List[Tuple[str,str]]:
    if not exercise:
        return []
    key = normalize_exercise_name(exercise)
    prefer = []
    for p, t in docs:
        base = normalize_exercise_name(os.path.splitext(os.path.basename(p))[0])
        if key in base:  # e.g., "clasp_and_spread" in "clasp_and_spread"
            prefer.append((p, t))
    return prefer

def summarize_exercise_text(text: str, max_chars: int = 600) -> str:
    # very small heuristic: keep lines with instructions or tips
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    keep = []
    for ln in lines:
        if any(w in ln.lower() for w in ["instruction", "repeat", "tip", "hold", "step", "inhale", "exhale", "elbow", "shoulder", "arm"]):
            keep.append(ln)
    blob = " ".join(keep) if keep else " ".join(lines)
    if len(blob) > max_chars:
        blob = blob[:max_chars].rstrip() + "…"
    return blob

def craft_prompt(report, retrieved, exercise_docs, exercise_name):
    meta = report["meta"]
    overall = report["overall_avg"]
    weak = report["weak_steps"]
    rep_summary = "; ".join([f"R{int(r)}={s:.1f}" for r, s in report["rep_df"].values]) if not report["rep_df"].empty else "N/A"
    step_summary = "; ".join([f"R{int(r)}-S{int(s)}={v:.1f}" for r, s, v in report["step_df"].values]) if not report["step_df"].empty else "N/A"
    weak_summary = ", ".join([f"R{r}-S{s} ({v:.1f})" for r,s,v in weak]) if weak else "None"

    system = textwrap.dedent(f"""
    You are Gideon, an expert lymphatic-rehab coach for breast-cancer survivors.
    Provide supportive, specific, and clinically-informed feedback focused on posture precision, range of motion,
    scapular mechanics, breath coordination, and pain-free execution. Avoid medical diagnosis.
                             
    At the end of the report, write a final 50 word friendly tip for the patient which can be used to send to a tts script to provide real time feedback. 
    """).strip()

    ex_ref = ""
    if exercise_docs:
        ex_ref = "\\n".join([f"[Exercise Doc] {os.path.basename(p)}\\n{summarize_exercise_text(t)}" for p, t in exercise_docs])

    user_context = textwrap.dedent(f"""
    === CURRENT SESSION REPORT ===
    File: {os.path.basename(report['path'])}
    Exercise: {exercise_name or 'Unknown'}
    Overall Average Similarity: {overall:.2f}%
    Averages by Rep: {rep_summary}
    Averages by (Rep,Step): {step_summary}
    Weak spots (below overall-10): {weak_summary}

    === EXERCISE REFERENCE ===
    {ex_ref if ex_ref else '[No exercise description provided]'} 

    === RETRIEVED CONTEXT (prior reports/notes) ===
    """).strip()

    for i, (doc_id, text, score) in enumerate(retrieved, 1):
        snippet = textwrap.shorten(" ".join(text.split()), width=900, placeholder="…")
        user_context += f"\\n[Doc {i} | {os.path.basename(doc_id)} | score={score:.3f}]\\n{snippet}\\n"

    instruction = textwrap.dedent("""
    Write feedback with these sections:
    1) Summary (2–3 lines).
    2) Strengths observed.
    3) Top corrections (≤3), each with: (a) what to fix, (b) a quick cue, (c) why it matters for lymph flow.
    4) Step-specific notes for each weak step (if any).
    5) Next Session Plan (2–3 bullets with a realistic target similarity and a cue to focus on).
    6) Safety note: stop if pain persists and consult a clinician.
    """).strip()

    prompt = system + "\\n\\n" + user_context + "\\n\\n" + instruction
    return prompt

def call_gemini(prompt: str, model: str = "gemini-1.5-flash") -> str:
    import requests
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents":[{"parts":[{"text": prompt}]}]}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return json.dumps(data, indent=2)

def call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    import requests
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":"You are a concise, expert lymphatic rehabilitation coach."},
            {"role":"user","content": prompt}
        ],
        "temperature": 0.2
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to one report CSV")
    ap.add_argument("--reports_dir", default="Exercise_Reports", help="Directory for RAG corpus (CSV/TXT/MD)")
    ap.add_argument("--exercise", default="", help="Exercise name (e.g., 'Clasp And Spread') to prefer matching TXT")
    ap.add_argument("--llm", choices=["gemini","openai"], default="gemini")
    ap.add_argument("--openai-model", default="gpt-4o-mini")
    ap.add_argument("--k", type=int, default=5, help="Top-K retrieved docs")
    ap.add_argument("--out", default="feedback.md", help="Write feedback to this file")
    ap.add_argument("--dry-run", action="store_true", help="Do not call an API; save the constructed prompt instead")
    args = ap.parse_args()

    report = load_csv_report(args.report)
    docs = collect_corpus(args.reports_dir)

    # exercise-aware preference
    exercise_docs = find_exercise_docs(docs, args.exercise)
    # generic retrieval over all docs (including the exercise docs)
    def as_text(t): return " ".join(t.split()) if isinstance(t, str) else str(t)
    retriever_query = f"lymphatic exercise {args.exercise} coaching posture scapular control breath timing EMG activation trend corrections"
    if SKLEARN_AVAILABLE:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
        X = vectorizer.fit_transform([d[1] for d in docs])
        import numpy as np
        qv = vectorizer.transform([retriever_query])
        sims = (X @ qv.T).toarray().ravel()
        order = sims.argsort()[::-1][:args.k]
        retrieved = [(docs[i][0], docs[i][1], float(sims[i])) for i in order if sims[i] > 0]
    else:
        retrieved = docs[:args.k]

    prompt = craft_prompt(report, retrieved, exercise_docs, args.exercise)

    if args.dry_run:
        out_path = args.out if args.out.endswith((".txt",".md")) else "feedback_prompt.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"[DRY-RUN] Prompt written to {out_path}")
        return

    if args.llm == "gemini":
        feedback = call_gemini(prompt)
    else:
        feedback = call_openai(prompt, model=args.openai_model)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(feedback.strip() + "\\n")

    print(f"[OK] Feedback saved to {args.out}")

if __name__ == "__main__":
    main()
