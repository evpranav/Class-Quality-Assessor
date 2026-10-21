# 8x8 Class Quality Assessor

**Live Demo:** https://gc879r7qecfatmb66uxzs3.streamlit.app/

An application that assesses online class recordings as **Poor, Average, Good, or Excellent**, with supporting observations and actionable feedback.

## How It Works

1. **Transcript analysis:** Uses a supplied VTT/TXT transcript or attempts automatic transcription when only a recording is provided. It examines teacher talk time, questions, student responses, greetings, rapport, introductions for new joiners, homework, and lesson closure.
2. **Audio analysis:** Measures changes in audio loudness as a limited proxy for vocal expressiveness.
3. **Visual analysis:** Samples six video frames to assess teacher visibility, camera presence, lighting, background setup, and screen content.
4. **Rubric-based review:** Evaluates six dimensions informed by the supplied human reviewer feedback: rapport and introduction, individual interaction, energy and pleasantness, homework and closure, teaching substance, and visual setup.
5. **Scoring and feedback:** Combines rubric scores and heuristic signals, applies dimension weights, and produces an overall score, grade, strengths, improvement suggestions, and timestamped observations when available.

### Grade Bands

- **Poor:** Below 45
- **Average:** 45 to 61
- **Good:** 62 to 77
- **Excellent:** 78 and above

The assessment uses hand-set weights and thresholds. Grades should be treated as estimates rather than definitive judgments.

## Inputs

- Video: MP4, MKV, WEBM, MOV
- Audio: MP3, M4A, WAV
- Optional transcript: VTT or TXT

Providing a transcript alongside the recording is recommended because speaker labels can improve speaker-specific analysis.

## Run Locally

Requirements: Python 3.10 or newer and FFmpeg.

```bash
python -m venv venv
```

On Windows:

```powershell
venv\Scripts\activate
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY="your_key"
streamlit run app.py
```

On macOS/Linux:

```bash
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your_key"
streamlit run app.py
```

The application reads the API key from the `ANTHROPIC_API_KEY` environment variable. Never commit API keys to the repository. You can optionally configure `ANTHROPIC_MODEL` to select a different supported model.

An Anthropic API key is required for LLM-based rubric review. Without it, the application falls back to heuristic scoring, subject to the implemented limitations.

## Deployment

The application can be deployed using Streamlit Community Cloud. Configure `ANTHROPIC_API_KEY` in the application's Secrets settings if LLM review is required.

## Project Files

- `app.py`: Streamlit interface
- `assessor.py`: Transcript processing, audio and video analysis, rubric review, and scoring
- `requirements.txt`: Python dependencies
- `packages.txt`: System packages
- `.streamlit/config.toml`: Streamlit configuration

The system is intended to support human review, not replace it.
