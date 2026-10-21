# 8x8 Class Quality Assessor

Grades an online class recording as Poor, Average, Good or Excellent and explains why with timestamped evidence.

## How it works
1. Transcript (.vtt, or Whisper if only media is given): teacher share of talk, questions per 10 min, how often a child answers, praise, greeting and rapport in the first minutes, new-joiner introductions, homework mentions, closing.
2. Audio: loudness range as a weak proxy for vocal expressiveness.
3. Video: 6 frames sampled across the class and judged for 8x8 background, teacher face visible, camera on, lighting, screen content.
4. An LLM scores six rubric dimensions (derived from the human reviewer feedback) citing timestamps. Final dimension score is 0.7 LLM + 0.3 heuristic, weighted into a 0-100 score, then banded.
5. Guards: weak visual setup caps the grade at Average; heuristics alone can never award Excellent.

Folder mapping assumed: Excellent, Good, Average = same; Weak and Known quality issues = Poor. Demo classes are judged as demos.

## Run
```
pip install -r requirements.txt   # also needs ffmpeg
export ANTHROPIC_API_KEY=...      # optional; without it you get heuristics only
streamlit run app.py
```
Deploy on Streamlit Community Cloud or Hugging Face Spaces (packages.txt installs ffmpeg). Put the key in the host's secrets, never in the repo.

## Limits
Thresholds and weights are hand-set from about 10 reviewed classes, not fitted. Tone is judged from words, not voice. Whisper fallback has no speaker labels.
