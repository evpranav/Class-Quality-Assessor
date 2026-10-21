import os, tempfile, requests, streamlit as st
from assessor import assess, WEIGHTS, DIM_DESC

st.set_page_config(page_title="8x8 Class Quality Assessor", layout="wide")
st.title("8x8 Class Quality Assessor")
st.caption("Upload a class recording (and its .vtt transcript if you have one) to get a quality grade with evidence.")

use_llm = st.sidebar.checkbox("Use LLM rubric review", True)

c1, c2 = st.columns(2)
video = c1.file_uploader("Recording (mp4, mkv, webm, mp3, m4a)", type=["mp4", "mkv", "webm", "mov", "mp3", "m4a", "wav"])
vtt = c2.file_uploader("Transcript (.vtt), optional", type=["vtt", "txt"])

def save(data, suffix):
    p = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    open(p, "wb").write(data); return p

if st.button("Assess class", type="primary"):
    media = None
    try:
        if video: media = save(video.getvalue(), os.path.splitext(video.name)[1])

        vtt_text = vtt.getvalue().decode("utf-8", "ignore") if vtt else None
        if not media and not vtt_text: st.error("Provide a recording or a transcript."); st.stop()
        with st.status("Assessing...") as s:
            r = assess(vtt_text, media, use_llm, progress=lambda m: s.update(label=m))
            s.update(label="Done", state="complete")
    except Exception as e:
        st.error(f"Could not process input: {e}"); st.stop()

    a, b = st.columns([1, 2])
    a.metric("Quality", r["label"]); a.metric("Score", f"{r['score']}/100")
    a.caption("Method: " + ("LLM rubric review + transcript/audio/frame signals" if r["llm_used"] else "heuristics only"))
    with b:
        for k, v in r["dimensions"].items():
            st.write(f"**{k.replace('_', ' ').title()}** ({int(WEIGHTS[k]*100)}%): {v}/10")
            st.progress(min(v / 10, 1.0)); st.caption(DIM_DESC[k])
    for n in r["notes"]: st.warning(n)
    j = r["judge"]
    if j:
        x, y = st.columns(2)
        x.subheader("Strengths"); [x.write("- " + s) for s in j.get("strengths", [])]
        y.subheader("What to improve"); [y.write("- " + s) for s in j.get("improvements", [])]
        st.subheader("Timestamped observations")
        for o in j.get("observations", []): st.write(f"`{o.get('time')}` {o.get('note')}")
        if j.get("visual_notes"): st.subheader("Visual notes"); st.write(j["visual_notes"])