"""Class quality assessment: transcript features + audio prosody + sampled frames + LLM rubric judge."""
import os, re, json, glob, base64, subprocess, tempfile
import numpy as np

LABELS = ["Poor", "Average", "Good", "Excellent"]
# Weights come from the reviewer feedback themes (rapport, individual questions, HW, energy, background/face).
WEIGHTS = {"rapport_intro": .20, "individual_interaction": .25, "energy_pleasantness": .15,
           "homework_closure": .15, "teaching_substance": .10, "visual_setup": .15}
DIM_DESC = {
    "rapport_intro": "Greeting, small talk, learning about the child, time for new joiners to introduce themselves before content starts.",
    "individual_interaction": "Questions directed at individual children, children actually answering, everyone included, teacher not monologuing.",
    "energy_pleasantness": "Warm, upbeat, encouraging tone; praise; patience; sounds happy rather than flat or rushed.",
    "homework_closure": "Homework or practice assigned, recap, proper conclusion and goodbye.",
    "teaching_substance": "Clear explanations, age-appropriate, checks understanding, uses related videos or examples.",
    "visual_setup": "Branded 8x8 background present, teacher's face clearly visible, camera on throughout, good lighting, screen content used well.",
}

# ---------- transcript ----------
def _sec(t):
    p = [float(x) for x in t.strip().split(":")]
    while len(p) < 3: p.insert(0, 0.0)
    return p[0] * 3600 + p[1] * 60 + p[2]

def parse_vtt(text):
    cues = []
    for b in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = [l for l in b.strip().split("\n") if l.strip()]
        ix = [i for i, l in enumerate(lines) if "-->" in l]
        if not ix: continue
        i = ix[0]
        a, b2 = [x.strip() for x in lines[i].split("-->")]
        body = " ".join(lines[i + 1:])
        m = re.match(r"^([^:]{1,40}):\s*(.*)$", body)
        spk, txt = (m.group(1).strip(), m.group(2)) if m else ("Unknown", body)
        spk = re.sub(r"^Audio shared by\s+", "", spk)
        cues.append(dict(start=_sec(a), end=_sec(b2.split()[0]), speaker=spk, text=txt.strip()))
    return cues

def transcribe(media_path):
    import subprocess
    import numpy as np
    from faster_whisper import WhisperModel
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", media_path, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True,
    ).stdout
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    segs, _ = WhisperModel("base", compute_type="int8").transcribe(audio)
    return [dict(start=s.start, end=s.end, speaker="Unknown", text=s.text.strip()) for s in segs]
# def transcribe(media_path):
    
    # from faster_whisper import WhisperModel
    # segs, _ = WhisperModel("base", compute_type="int8").transcribe(media_path)
    # return [dict(start=s.start, end=s.end, speaker="Unknown", text=s.text.strip()) for s in segs]

GREET = re.compile(r"\b(hello|hi|hey|good (morning|evening|afternoon)|how are you|welcome)\b", re.I)
PERSONAL = re.compile(r"(what'?s your name|how old|where are you from|favou?rite|how was your day|what are you learning|which (grade|class))", re.I)
JOINER = re.compile(r"(introduce yourself|new (joiner|student|friend)|tell (us|me) about yourself|say your name)", re.I)
PRAISE = re.compile(r"\b(very good|good job|well done|great|excellent|nice|correct|perfect|awesome|super|wonderful|brilliant|amazing)\b", re.I)
HW = re.compile(r"(homework|home work|\bhw\b|assignment|worksheet|practice (at|in) home|i will (share|send|give).{0,20}(puzzle|sheet|task))", re.I)
CLOSE = re.compile(r"(thank you|thanks|bye|see you|next class|that'?s all for today|summary|recap|let'?s end)", re.I)
VIDEO = re.compile(r"(watch (this|the) video|share (a|the) video|let'?s watch)", re.I)

def mmss(s): return f"{int(s//60):02d}:{int(s%60):02d}"

def transcript_features(cues):
    if not cues: return {"error": "empty transcript"}
    total = max(c["end"] for c in cues) or 1
    words = {}
    for c in cues: words[c["speaker"]] = words.get(c["speaker"], 0) + len(c["text"].split())
    known = {k: v for k, v in words.items() if k != "Unknown"}
    teacher = max(known, key=known.get) if known else None
    f = {"duration_min": round(total / 60, 1), "speakers": words, "teacher_guess": teacher}
    f["teacher_word_share"] = round(words[teacher] / max(sum(words.values()), 1), 2) if teacher else None
    ev = {k: [] for k in ["greeting", "personal", "joiner_intro", "praise", "homework", "closing", "question", "video"]}
    tq = answered = 0
    for i, c in enumerate(cues):
        if teacher and c["speaker"] != teacher: continue
        t, ts = c["text"], mmss(c["start"])
        if c["start"] < 150 and GREET.search(t): ev["greeting"].append((ts, t))
        if c["start"] < 600 and PERSONAL.search(t): ev["personal"].append((ts, t))
        if JOINER.search(t): ev["joiner_intro"].append((ts, t))
        if PRAISE.search(t): ev["praise"].append((ts, t))
        if HW.search(t): ev["homework"].append((ts, t))
        if c["start"] > total - 240 and CLOSE.search(t): ev["closing"].append((ts, t))
        if VIDEO.search(t): ev["video"].append((ts, t))
        if t.endswith("?"):
            tq += 1; ev["question"].append((ts, t))
            if teacher:
                for n in cues[i + 1:i + 3]:
                    if n["speaker"] != teacher and n["start"] - c["end"] < 15: answered += 1; break
    per10 = 600 / total
    f.update(questions=tq, questions_per_10min=round(tq * per10, 1),
             answer_rate=round(answered / tq, 2) if tq else None,
             praise_per_10min=round(len(ev["praise"]) * per10, 1),
             homework_mentions=len(ev["homework"]), greeted_early=bool(ev["greeting"]),
             personal_rapport_questions=len(ev["personal"]), joiner_intro=bool(ev["joiner_intro"]),
             closing_present=bool(ev["closing"]), related_video=bool(ev["video"]))
    f["silence_ratio"] = round(max(0, 1 - sum(c["end"] - c["start"] for c in cues) / total), 2)
    return f, ev

def heuristic_scores(f):
    c = lambda x: max(0.0, min(10.0, x))
    s = {}
    s["rapport_intro"] = c(4 * f["greeted_early"] + min(f["personal_rapport_questions"], 3) * 1.5 + 2 * f["joiner_intro"])
    qr = min(f["questions_per_10min"] / 8, 1) * 6
    ar = (f["answer_rate"] or 0) * 4
    share = f["teacher_word_share"]
    mono = 0 if share is None else (-2 if share > .85 else 0)
    s["individual_interaction"] = c(qr + ar + mono)
    s["energy_pleasantness"] = c(min(f["praise_per_10min"] / 6, 1) * 7 + 3 * min(1, f.get("audio_expressiveness", .5)))
    s["homework_closure"] = c(5 * min(f["homework_mentions"], 1) + 5 * f["closing_present"])
    return {k: round(v, 1) for k, v in s.items()}

# ---------- audio prosody (weak proxy) ----------
def audio_features(media):
    try:
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", media, "-t", "1800", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                             capture_output=True, timeout=300).stdout
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
        n = 800
        rms = np.sqrt((x[:len(x)//n*n].reshape(-1, n) ** 2).mean(1) + 1e-10)
        db = 20 * np.log10(rms); sp = db[db > -45]
        if len(sp) < 20: return {}
        rng = float(np.percentile(sp, 90) - np.percentile(sp, 20))
        return {"speech_activity": round(len(sp) / len(db), 2), "loudness_range_db": round(rng, 1),
                "audio_expressiveness": round(min(rng / 25, 1), 2)}
    except Exception:
        return {}

# ---------- frames ----------
def sample_frames(video, n=6):
    d = tempfile.mkdtemp()
    try:
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                                   capture_output=True, text=True).stdout.strip() or 0)
    except Exception: dur = 0
    out = []
    for i in range(n):
        t = dur * (i + .5) / n
        p = f"{d}/f{i}.jpg"
        subprocess.run(["ffmpeg", "-v", "quiet", "-ss", str(t), "-i", video, "-frames:v", "1", "-vf", "scale=768:-1", "-q:v", "4", p])
        if os.path.exists(p): out.append((mmss(t), p))
    return out

# ---------- LLM judge ----------
def _client():
    import anthropic
    return anthropic.Anthropic()

def _json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0))

def llm_judge(cues, f, ev, frames, model):
    lines = [f"[{mmss(c['start'])}] {c['speaker']}: {c['text']}" for c in cues]
    tr = "\n".join(lines)
    if len(tr) > 60000: tr = tr[:30000] + "\n...[middle omitted]...\n" + tr[-30000:]
    rubric = "\n".join(f"- {k} (0-10): {v}" for k, v in DIM_DESC.items())
    prompt = f"""You review recordings of online classes for children run by 8x8. Human reviewers grade classes Excellent / Good / Average / Poor using these themes: good intro and rapport building, giving new joiners time to introduce themselves, asking questions to individual children by name, considering everyone, assigning homework, a pleasant and energetic tone, the 8x8 background, teacher face visible and camera on, and a proper conclusion.

Score each dimension 0-10 strictly from evidence. If frames were not provided, set visual_setup to null. Do not invent things you cannot see or read. Cite timestamps. Be critical; most classes are not 9s.
Note: speaker labels in the transcript can be wrong, and a parent may speak for a child. Do not penalise the teacher for that. This may be a demo/trial class, judge it as a demo if so.

Dimensions:
{rubric}

Computed transcript features (heuristic, may be imperfect): {json.dumps({k: v for k, v in f.items() if k != 'speakers'})}

Return ONLY JSON: {{"scores": {{dimension: number|null}}, "strengths": [str], "improvements": [str], "observations": [{{"time": "MM:SS", "note": str}}], "visual_notes": str, "demo_class": bool}}
Each strength/improvement/observation must be specific (what happened, when), never generic.

TRANSCRIPT:
{tr}"""
    content = []
    for ts, p in frames:
        content.append({"type": "text", "text": f"Frame at {ts}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                        "data": base64.b64encode(open(p, "rb").read()).decode()}})
    content.append({"type": "text", "text": prompt})
    r = _client().messages.create(model=model, max_tokens=2500, messages=[{"role": "user", "content": content}])
    return _json("".join(b.text for b in r.content if b.type == "text"))

# ---------- combine ----------
def band(score):
    return "Poor" if score < 45 else "Average" if score < 62 else "Good" if score < 78 else "Excellent"

def assess(vtt_text=None, media_path=None, use_llm=True, model=None, progress=lambda m: None):
    model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    notes = []
    audio = {}
    if media_path:
        progress("Analysing audio"); audio = audio_features(media_path)
    if vtt_text: cues = parse_vtt(vtt_text)
    elif media_path:
        progress("Transcribing (no VTT supplied)"); cues = transcribe(media_path)
        notes.append("Transcript came from Whisper with no speaker labels, so speaker-based features are weaker.")
    else: raise ValueError("Provide a recording or a VTT transcript.")
    f, ev = transcript_features(cues)
    f.update(audio)
    heur = heuristic_scores(f)
    frames = []
    if media_path:
        progress("Sampling frames"); frames = sample_frames(media_path)
    judge = None
    if use_llm and os.getenv("ANTHROPIC_API_KEY"):
        try: progress("Running LLM rubric review"); judge = llm_judge(cues, f, ev, frames, model)
        # except Exception as e: notes.append(f"LLM review failed, heuristics only: {e}")
        except Exception: pass
    dims = {}
    for k in WEIGHTS:
        l = (judge or {}).get("scores", {}).get(k)
        h = heur.get(k)
        if l is not None and h is not None: dims[k] = round(.7 * l + .3 * h, 1)
        elif l is not None: dims[k] = float(l)
        elif h is not None and judge is None: dims[k] = h
    wsum = sum(WEIGHTS[k] for k in dims)
    score = round(sum(dims[k] * WEIGHTS[k] for k in dims) / wsum * 10, 1)
    label = band(score)
    if "visual_setup" in dims and dims["visual_setup"] < 3 and label in ("Good", "Excellent"):
        label = "Average"; notes.append("Capped at Average: visual setup is very weak (hidden face, camera off or no background).")
    if judge is None and label == "Excellent":
        label = "Good"; notes.append("Heuristics alone are never allowed to award Excellent.")
    return dict(label=label, score=score, dimensions=dims, heuristic_scores=heur, features=f, evidence={k: v[:5] for k, v in ev.items()},
                judge=judge, notes=notes, llm_used=judge is not None)
