

"""
AI Interview Prep Pro — Voice-to-Voice Edition
================================================
Flow:
  1. User speaks  → Whisper ASR  → transcript text
  2. Transcript   → GPT-3.5      → interviewer reply text
  3. Reply text   → OpenAI TTS   → audio played back
  4. User audio   → Tone Engine  → scores collected silently
  5. End of interview → full tone + interview report shown
"""

import streamlit as st
from openai import OpenAI
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import numpy as np
import io
import base64
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# ── Tone deps ─────────────────────────────────────────────────────────────────
try:
    import librosa
    import soundfile as sf
    TONE_AVAILABLE = True
except ImportError:
    TONE_AVAILABLE = False

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OPENAI_API_KEY not found in .env file.")
    st.stop()


# =============================================================================
#  TONE ANALYSIS ENGINE  (Layers A → B → C)
# =============================================================================

@dataclass
class AcousticFeatures:
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_range: float = 0.0
    speech_rate: float = 0.0
    pause_frequency: float = 0.0
    pause_duration_mean: float = 0.0
    energy_mean: float = 0.0
    energy_std: float = 0.0
    jitter: float = 0.0
    shimmer: float = 0.0
    hnr: float = 0.0
    mfcc_mean: list = field(default_factory=list)
    speaking_ratio: float = 0.0
    duration_seconds: float = 0.0

@dataclass
class BehavioralScores:
    confidence_score: float = 0.0
    anxiety_score: float = 0.0
    communication_clarity_score: float = 0.0
    engagement_index: float = 0.0
    fluency_score: float = 0.0
    sadness_score: float = 0.0

@dataclass
class ToneResult:
    acoustic: AcousticFeatures = field(default_factory=AcousticFeatures)
    behavioral: BehavioralScores = field(default_factory=BehavioralScores)
    acoustic_confidence: float = 0.0
    linguistic_confidence: float = 0.5
    clarity_score: float = 0.0
    final_confidence_index: float = 0.0
    emotion_label: str = "neutral"
    emotion_scores: dict = field(default_factory=dict)
    question_text: str = ""   # the question that was asked
    answer_text: str = ""     # the transcribed answer


class AcousticExtractor:
    SR = 16000

    def extract(self, audio: np.ndarray, sr: int) -> AcousticFeatures:
        if sr != self.SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SR)
            sr = self.SR
        dur = len(audio) / sr
        f = AcousticFeatures(duration_seconds=dur)

        # Pitch
        f0, voiced_flag, _ = librosa.pyin(audio,
            fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, hop_length=512)
        vf0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        vf0 = vf0[~np.isnan(vf0)]
        if len(vf0) > 2:
            f.pitch_mean  = float(np.mean(vf0))
            f.pitch_std   = float(np.std(vf0))
            f.pitch_range = float(np.percentile(vf0, 95) - np.percentile(vf0, 5))
            periods = 1.0 / (vf0 + 1e-10)
            f.jitter = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-10))

        # Energy
        rms = librosa.feature.rms(y=audio, hop_length=512)[0]
        f.energy_mean = float(np.mean(rms))
        f.energy_std  = float(np.std(rms))
        if len(rms) > 1:
            f.shimmer = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-10))

        # HNR
        harmonic, percussive = librosa.effects.hpss(audio)
        f.hnr = float(10 * np.log10(
            (np.mean(harmonic**2) + 1e-12) / (np.mean(percussive**2) + 1e-12)))

        # Speech/pause
        intervals = librosa.effects.split(audio, top_db=25)
        speech_smp = sum(e - s for s, e in intervals)
        f.speaking_ratio = float(speech_smp / len(audio)) if len(audio) > 0 else 0.0
        f.speech_rate    = f.speaking_ratio
        if len(intervals) > 1 and dur > 0:
            pauses = [(intervals[i][0] - intervals[i-1][1]) / sr
                      for i in range(1, len(intervals))
                      if (intervals[i][0] - intervals[i-1][1]) / sr > 0.15]
            f.pause_frequency     = len(pauses) / dur
            f.pause_duration_mean = float(np.mean(pauses)) if pauses else 0.0

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        f.mfcc_mean = [float(x) for x in np.mean(mfccs, axis=1)]
        return f


class BehavioralScorer:
    # ── Calibrated for real browser/laptop mic recordings ──────────────────
    # Key fixes vs previous version:
    #
    #  jitter:       0.001–0.025 → 0.01–0.08
    #    Browser mics produce jitter of 0.02–0.06 even for confident speakers
    #    due to hardware noise. Old upper bound of 0.025 meant everyone maxed out.
    #
    #  shimmer:      0.01–0.10  → 0.05–0.25
    #    Same reason — natural amplitude variation in real speech is 0.08–0.18.
    #    Old bound of 0.10 treated normal speech as "very shimmery".
    #
    #  energy_std:   0.005–0.04 → 0.005–0.08
    #    Expressive/confident speakers vary volume deliberately (emphasis).
    #    Old bound of 0.04 penalised natural expressiveness as "unstable".
    #
    #  energy_mean:  0.01–0.10  → 0.01–0.07
    #    Browser mics peak at 0.04–0.06 RMS. Old upper bound of 0.10 made
    #    even a loud speaker look like they had 50% energy score.
    #
    #  hnr:          5–22 dB    → 2–18 dB
    #    Browser mic HNR is typically 6–14 dB. Old range pushed most
    #    recordings into the bottom half of the scale.
    #
    #  speaking_ratio: 0.40–0.80 → 0.35–0.75  (slightly more lenient floor)
    # ──────────────────────────────────────────────────────────────────────
    NORMS = {
        "pitch_std":      (15.0,  60.0),
        "pitch_range":    (30.0, 180.0),
        "speech_rate":    (0.35,  0.70),
        "pause_freq":     (0.0,   0.70),   # was 0.80
        "energy_mean":    (0.01,  0.07),   # was 0.10 — browser mics cap here
        "energy_std":     (0.005, 0.08),   # was 0.04 — allow expressive variation
        "jitter":         (0.01,  0.08),   # was 0.001–0.025 — mic noise floor
        "shimmer":        (0.05,  0.25),   # was 0.01–0.10  — real speech range
        "hnr":            (2.0,   18.0),   # was 5–22 — browser mic reality
        "speaking_ratio": (0.35,  0.75),   # was 0.40–0.80
    }

    def _n(self, v, key):
        lo, hi = self.NORMS[key]
        return float(np.clip((v - lo) / (hi - lo + 1e-10), 0, 1))

    def score(self, f: AcousticFeatures):
        n = self._n
        s = BehavioralScores()
        pv  = n(f.pitch_std,        "pitch_std")
        sr_ = n(f.speech_rate,      "speech_rate")
        pf  = n(f.pause_frequency,  "pause_freq")
        en  = n(f.energy_mean,      "energy_mean")
        es  = 1 - n(f.energy_std,   "energy_std")   # inverted: stable = good
        jt  = n(f.jitter,           "jitter")
        sh  = n(f.shimmer,          "shimmer")
        hn  = n(f.hnr,              "hnr")
        sp  = n(f.speaking_ratio,   "speaking_ratio")

        # ── Confidence ────────────────────────────────────────────────────
        # Reduced jitter/shimmer weight; they're mic-noise-sensitive.
        # Increased energy + speaking_ratio weight; these are reliable signals.
        conf = (0.28*en + 0.22*sp + 0.20*sr_ + 0.15*hn
                + 0.08*(1-jt) + 0.07*(1-pf))
        s.confidence_score = round(conf * 100, 1)

        # ── Anxiety ───────────────────────────────────────────────────────
        # Reduced jitter (0.30→0.20) and shimmer (0.25→0.15) weights.
        # Added pause_freq as primary signal (0.20→0.30) — most reliable
        # nervousness marker that isn't affected by mic quality.
        anx = (0.30*pf + 0.20*jt + 0.15*sh
               + 0.20*(1-hn) + 0.15*(1-es))
        s.anxiety_score = round(anx * 100, 1)

        # ── Clarity ──────────────────────────────────────────────────────
        clarity = (0.35*sr_ + 0.30*hn + 0.20*es + 0.15*(1-pf))
        s.communication_clarity_score = round(clarity * 100, 1)

        # ── Engagement ───────────────────────────────────────────────────
        engage = (0.40*pv + 0.30*sp + 0.30*sr_)
        s.engagement_index = round(engage * 100, 1)

        # ── Fluency ──────────────────────────────────────────────────────
        fluency = (0.45*(1-pf) + 0.30*sp + 0.25*(1-jt))
        s.fluency_score = round(fluency * 100, 1)

        # ── Sadness ──────────────────────────────────────────────────────
        sadness = (0.30*(1-en) + 0.25*(1-sr_) + 0.20*(1-pv)
                   + 0.15*pf + 0.10*(1-hn))
        s.sadness_score = round(sadness * 100, 1)

        return s, conf


class EmotionClassifier:
    def classify(self, b: BehavioralScores, f: AcousticFeatures):
        v = {k: 0.0 for k in ["sad","nervous","confident","enthusiastic","calm","disengaged","neutral"]}
        co, ax, sd, eg, fl, cl = (b.confidence_score, b.anxiety_score, b.sadness_score,
                                   b.engagement_index, b.fluency_score, b.communication_clarity_score)

        # ── SAD ──────────────────────────────────────────────────────────────
        if sd > 60:                                    v["sad"] += 40
        if f.energy_mean < 0.018:                      v["sad"] += 20
        if f.speech_rate < 0.42:                       v["sad"] += 15
        if f.pitch_std < 20 and f.pitch_mean < 170:    v["sad"] += 15
        if f.pause_duration_mean > 0.6:                v["sad"] += 10
        if eg < 28:                                    v["sad"] += 10
        if co < 30:                                    v["sad"] += 10

        # ── NERVOUS ──────────────────────────────────────────────────────────
        # Raised thresholds: ax > 60 → 68, jitter/shimmer thresholds raised
        # to not fire on normal browser-mic noise
        if ax > 68:                                    v["nervous"] += 40
        if f.jitter > 0.05:                            v["nervous"] += 20   # was 0.015
        if f.shimmer > 0.18:                           v["nervous"] += 15   # was 0.07
        if f.pause_frequency > 0.65:                   v["nervous"] += 20   # was 0.6, raised weight
        if f.energy_std > 0.04 and f.energy_mean > 0.025: v["nervous"] += 10
        # Prevent nervous if confidence is strong — mutual exclusion
        if co > 58:                                    v["nervous"] -= 25
        if fl > 60:                                    v["nervous"] -= 10

        # ── CONFIDENT ────────────────────────────────────────────────────────
        # Lowered threshold: co > 65 → 52, so mid-range scores still register
        if co > 52:                                    v["confident"] += 40  # was 65
        if f.energy_mean > 0.03:                       v["confident"] += 15  # was 0.05
        if f.speech_rate > 0.50:                       v["confident"] += 15  # was 0.55
        if f.hnr > 10:                                 v["confident"] += 15  # was 14
        if fl > 55:                                    v["confident"] += 10  # was 65
        if ax < 45:                                    v["confident"] += 10  # was 30, wider gate
        if eg > 45:                                    v["confident"] += 5   # bonus for engagement

        # ── ENTHUSIASTIC ─────────────────────────────────────────────────────
        if eg > 68:                                    v["enthusiastic"] += 35
        if f.pitch_std > 42:                           v["enthusiastic"] += 25
        if f.energy_mean > 0.055:                      v["enthusiastic"] += 20
        if f.speech_rate > 0.58:                       v["enthusiastic"] += 10
        if co > 55:                                    v["enthusiastic"] += 10

        # ── CALM ─────────────────────────────────────────────────────────────
        if co > 45 and ax < 40 and sd < 38:            v["calm"] += 30
        if f.energy_std < 0.025:                       v["calm"] += 25      # was 0.012 — too strict
        if 0.42 < f.speech_rate < 0.62:                v["calm"] += 20
        if f.jitter < 0.035:                           v["calm"] += 15      # was 0.008
        if cl > 50:                                    v["calm"] += 10

        # ── DISENGAGED ───────────────────────────────────────────────────────
        if eg < 22 and co < 30:                        v["disengaged"] += 35
        if f.speaking_ratio < 0.38:                    v["disengaged"] += 25
        if f.pitch_std < 12:                           v["disengaged"] += 20
        if f.energy_mean < 0.015:                      v["disengaged"] += 20

        # ── NEUTRAL baseline (always present) ────────────────────────────────
        v["neutral"] += 15

        # Clamp negatives to 0
        v = {k: max(0.0, val) for k, val in v.items()}

        total = sum(v.values()) + 1e-10
        norm = {k: round(val/total*100, 1) for k, val in v.items()}
        return max(norm, key=lambda k: norm[k]), norm


class ToneAnalyser:
    def __init__(self):
        self.extractor  = AcousticExtractor()
        self.scorer     = BehavioralScorer()
        self.classifier = EmotionClassifier()

    def analyse(self, audio: np.ndarray, sr: int, linguistic_conf: float = 0.5,
                question_text: str = "", answer_text: str = "") -> ToneResult:
        r = ToneResult(question_text=question_text, answer_text=answer_text)
        r.acoustic = self.extractor.extract(audio, sr)
        r.behavioral, ac = self.scorer.score(r.acoustic)
        cl = r.behavioral.communication_clarity_score / 100.0
        r.acoustic_confidence   = round(ac, 4)
        r.linguistic_confidence = round(linguistic_conf, 4)
        r.clarity_score         = round(cl, 4)
        r.final_confidence_index = round((0.5*ac + 0.3*linguistic_conf + 0.2*cl) * 100, 1)
        r.emotion_label, r.emotion_scores = self.classifier.classify(r.behavioral, r.acoustic)
        return r


# =============================================================================
#  KNOWLEDGE BASE
# =============================================================================

class CompanyKnowledgeBase:
    def __init__(self):
        self.data = {
            "Google":    {"focus": ["Data structures & algorithms","System design","Googleyness","Problem-solving"], "culture": "Innovation-driven, data-focused, collaborative", "tips": "Show your problem-solving process, not just the answer."},
            "Amazon":    {"focus": ["14 Leadership Principles","System design & scalability","STAR method","Customer obsession"], "culture": "Customer-obsessed, ownership mentality", "tips": "Use STAR method. Demonstrate ownership."},
            "Microsoft": {"focus": ["Technical depth","Collaboration","System design","Growth mindset"], "culture": "Growth mindset, inclusive, collaborative", "tips": "Emphasize learning and continuous improvement."},
            "Meta":      {"focus": ["Coding & algorithms","Scale design","Product sense","Impact-driven"], "culture": "Move fast, be bold, build social value", "tips": "Demonstrate impact-driven thinking at scale."},
            "Apple":     {"focus": ["Technical excellence","Attention to detail","Product design","Innovation"], "culture": "Excellence-driven, detail-oriented", "tips": "Show attention to detail and user-centric thinking."},
        }

    def context(self, company: str) -> str:
        if company not in self.data:
            return ""
        d = self.data[company]
        return (f"Company: {company}\nFocus: {', '.join(d['focus'])}\n"
                f"Culture: {d['culture']}\nTips: {d['tips']}")

    def companies(self):
        return list(self.data.keys())


# =============================================================================
#  HISTORY MANAGER
# =============================================================================

class HistoryManager:
    def __init__(self, path="interview_history.json"):
        self.path = path
        self.data = self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path) as f: return json.load(f)
        except Exception: pass
        return {}

    def _save(self):
        try:
            with open(self.path, "w") as f: json.dump(self.data, f, indent=2)
        except Exception: pass

    def save(self, cid: str, record: dict):
        self.data.setdefault(cid, []).append(record)
        self._save()

    def history(self, cid: str):
        return self.data.get(cid, [])

    def compare(self, cid: str):
        h = self.history(cid)
        if len(h) < 2: return None
        cur, prev = h[-1], h[-2]
        comp = {"current_date": cur["date"], "previous_date": prev["date"],
                "improvements": {}, "declines": {}}
        for cat in cur.get("scores", {}):
            cs, ps = cur["scores"][cat]["score"], prev.get("scores",{}).get(cat,{}).get("score",0)
            d = cs - ps
            if d > 0:  comp["improvements"][cat] = {"current":cs,"previous":ps,"change":d}
            elif d < 0: comp["declines"][cat]    = {"current":cs,"previous":ps,"change":d}
        return comp


# =============================================================================
#  RAG INTERVIEWER
# =============================================================================

class RAGInterviewer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY not found")
        self.client = OpenAI(api_key=api_key)
        self.kb      = CompanyKnowledgeBase()
        self.history = HistoryManager()

    # ── ASR: audio bytes → transcript text ───────────────────────────────────
    def transcribe(self, audio_bytes: bytes) -> str:
        """Send audio to Whisper API and return transcript."""
        try:
            buf = io.BytesIO(audio_bytes)
            buf.name = "audio.wav"
            result = self.client.audio.transcriptions.create(
                model="whisper-1", file=buf, language="en"
            )
            return result.text.strip()
        except Exception as e:
            return f"[Transcription error: {e}]"

    # ── TTS: text → audio bytes (mp3) ────────────────────────────────────────
    def speak(self, text: str, emotion: str = "neutral") -> bytes:
        """
        Convert text to speech via OpenAI TTS.
        Voice adapts slightly based on detected candidate emotion
        (matches doc's adaptive tone strategy).
        """
        # Adaptive voice selection based on candidate's emotion
        voice_map = {
            "nervous":      "nova",    # soft, calm voice to ease candidate
            "sad":          "nova",
            "disengaged":   "onyx",    # deeper, more engaging
            "confident":    "echo",
            "enthusiastic": "echo",
            "calm":         "alloy",
            "neutral":      "alloy",
        }
        voice = voice_map.get(emotion, "alloy")
        try:
            response = self.client.audio.speech.create(
                model="tts-1", voice=voice, input=text
            )
            return response.content
        except Exception as e:
            return b""

    # ── Interview logic ───────────────────────────────────────────────────────
    def start(self, job_role: str, difficulty: str, company: str = None) -> str:
        ctx = self.kb.context(company) if company and company != "General" else ""
        sys = f"""You are a professional AI interviewer for a {difficulty} {job_role} role{f' at {company}' if company and company != "General" else ''}.
{ctx}
Rules:
- Start with a warm welcome and ask how the candidate is doing.
- Then ask them to introduce themselves.
- Mix technical and behavioral questions.
- Ask thoughtful follow-ups.
- After 8-12 questions, let the candidate ask you questions, then wrap up.
- ONLY output the interviewer's words. NEVER write candidate responses."""
        return self._gpt(sys, "Start the interview with a warm greeting.")

    def next_question(self, history: list, job_role: str, difficulty: str,
                      company: str = None, emotion: str = "neutral") -> str:
        recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-8:])
        q_count = sum(1 for m in history if m["role"] == "interviewer")

        last_candidate = next((m["content"].lower() for m in reversed(history)
                                if m["role"] == "candidate"), "")
        end_phrases = ["end the interview","stop the interview","i'm done","that's all",
                       "i'd like to end","can we stop","let's end","i want to stop","i'm finished"]
        if any(p in last_candidate for p in end_phrases):
            return self._gpt(
                "You are concluding an interview. The candidate wants to end. "
                "Thank them warmly, ask if they have final questions, close professionally.",
                "Generate a polite conclusion."
            )

        # Adaptive tone hint from document: if nervous, adjust to calm/supportive
        tone_hint = ""
        if emotion in ["nervous", "sad"]:
            tone_hint = "The candidate seems nervous/stressed. Ask a supportive, encouraging follow-up."
        elif emotion == "disengaged":
            tone_hint = "The candidate seems disengaged. Ask a more engaging, interesting question."
        elif emotion == "confident":
            tone_hint = "The candidate is confident. You may increase difficulty slightly."

        ctx = self.kb.context(company) if company and company != "General" else ""
        sys = f"""You are interviewing for {difficulty} {job_role}{f' at {company}' if company and company != "General" else ''}.
{ctx}
Recent conversation:\n{recent}
Questions asked: {q_count}
{tone_hint}
After 8-12 questions, offer the candidate a chance to ask you questions.
ONLY generate the interviewer's next question. NEVER write the candidate's response."""
        return self._gpt(sys, "Generate the next interviewer question only.")

    def generate_feedback(self, history: list, job_role: str, company: str = None,
                          candidate_id: str = None) -> dict:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        sys = f"""You are a hiring manager evaluating a {job_role} interview{f' at {company}' if company and company != "General" else ''}.
Conversation:\n{text}

Return ONLY valid JSON in this exact format:
{{
  "overall_summary": "2-3 sentence overview",
  "scores": {{
    "technical_skills":    {{"score": 0-10, "strengths": [], "weaknesses": [], "details": ""}},
    "communication":       {{"score": 0-10, "strengths": [], "weaknesses": [], "details": ""}},
    "problem_solving":     {{"score": 0-10, "strengths": [], "weaknesses": [], "details": ""}},
    "cultural_fit":        {{"score": 0-10, "strengths": [], "weaknesses": [], "details": ""}},
    "experience_depth":    {{"score": 0-10, "strengths": [], "weaknesses": [], "details": ""}},
    "behavioral_responses":{{"score": 0-10, "strengths": [], "weaknesses": [], "details": ""}}
  }},
  "overall_score": 0-10,
  "recommendation": "hire/maybe/no_hire",
  "key_highlights": [],
  "improvement_areas": [],
  "actionable_recommendations": []
}}"""
        try:
            raw = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system","content":sys},
                          {"role":"user","content":"Return evaluation JSON only."}],
                temperature=0.4, max_tokens=1500
            ).choices[0].message.content.strip()
            raw = raw.replace("```json","").replace("```","").strip()
            data = json.loads(raw)
            data["date"]    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data["job_role"] = job_role
            data["company"]  = company or "General"
            if candidate_id:
                self.history.save(candidate_id, data)
            return data
        except Exception as e:
            return {"error": str(e)}

    def _gpt(self, system: str, user: str) -> str:
        try:
            return self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system","content":system},
                          {"role":"user","content":user}],
                temperature=0.7, max_tokens=280,
                stop=["candidate:","Candidate:","CANDIDATE:"]
            ).choices[0].message.content.strip()
        except Exception as e:
            return f"[Error: {e}]"


# =============================================================================
#  REPORT CHARTS
# =============================================================================

EMOTION_EMOJI  = {"sad":"😔","nervous":"😰","confident":"💪","enthusiastic":"🔥",
                  "calm":"😌","disengaged":"😑","neutral":"😐"}
EMOTION_COLOR  = {"sad":"#5b8cdb","nervous":"#e8935a","confident":"#4caf89",
                  "enthusiastic":"#f5c842","calm":"#80c7b7","disengaged":"#9e9e9e","neutral":"#b0a8c4"}
SCORE_COLOR    = {"Confidence":"#4caf89","Anxiety":"#e8935a","Clarity":"#5b8cdb",
                  "Engagement":"#f5c842","Fluency":"#80c7b7","Sadness":"#9e85d4"}


def _fig_base():
    return dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), margin=dict(l=0,r=0,t=30,b=0))


def chart_trend(tones: list) -> go.Figure:
    qs = list(range(1, len(tones)+1))
    fig = go.Figure()
    series = {
        "Confidence": ([r.behavioral.confidence_score for r in tones], "#4caf89", "solid", 3),
        "Anxiety":    ([r.behavioral.anxiety_score    for r in tones], "#e8935a", "dash",  2),
        "Clarity":    ([r.behavioral.communication_clarity_score for r in tones], "#5b8cdb","solid",2),
        "Engagement": ([r.behavioral.engagement_index for r in tones], "#f5c842", "dot",   2),
        "Fluency":    ([r.behavioral.fluency_score    for r in tones], "#80c7b7", "dashdot",2),
        "Conf.Index": ([r.final_confidence_index      for r in tones], "#ffffff", "solid", 3),
    }
    for name, (vals, color, dash, width) in series.items():
        fig.add_trace(go.Scatter(x=qs, y=vals, name=name, mode="lines+markers",
            line=dict(color=color, dash=dash, width=width),
            marker=dict(size=7, color=color),
            hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(**_fig_base(), height=340,
        xaxis=dict(title="Answer #", dtick=1, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Score (0–100)", range=[0,108], gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", y=1.08, font=dict(size=10)),
        hovermode="x unified")
    return fig


def chart_radar_tone(tones: list) -> go.Figure:
    avg = lambda fn: round(sum(fn(r) for r in tones)/len(tones), 1)
    cats = ["Confidence","Clarity","Engagement","Fluency","Sadness(inv)"]
    vals = [
        avg(lambda r: r.behavioral.confidence_score),
        avg(lambda r: r.behavioral.communication_clarity_score),
        avg(lambda r: r.behavioral.engagement_index),
        avg(lambda r: r.behavioral.fluency_score),
        avg(lambda r: 100 - r.behavioral.sadness_score),  # invert sadness → positivity
    ]
    fig = go.Figure(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", fillcolor="rgba(76,175,137,0.2)",
        line=dict(color="#4caf89", width=2), marker=dict(size=6, color="#4caf89")))
    fig.update_layout(**_fig_base(), height=300,
        polar=dict(radialaxis=dict(visible=True, range=[0,100],
                   gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#777",size=8)),
                   angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                                    tickfont=dict(color="#ddd",size=10)),
                   bgcolor="rgba(0,0,0,0)"))
    return fig


def chart_emotion_bar(tones: list) -> go.Figure:
    """Stacked bar showing emotion distribution per answer."""
    emotions = ["confident","calm","neutral","nervous","sad","disengaged","enthusiastic"]
    q_labels = [f"A#{i+1}" for i in range(len(tones))]
    fig = go.Figure()
    for em in emotions:
        vals = [r.emotion_scores.get(em, 0) for r in tones]
        fig.add_trace(go.Bar(name=em.capitalize(), x=q_labels, y=vals,
            marker_color=EMOTION_COLOR.get(em, "#888"),
            hovertemplate=f"{em}: %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(**_fig_base(), height=280, barmode="stack",
        xaxis_title="Answer", yaxis=dict(title="Probability %", range=[0,105]),
        legend=dict(orientation="h", y=1.08, font=dict(size=10)))
    return fig


def chart_interview_radar(scores: dict) -> go.Figure:
    cats = [k.replace("_"," ").title() for k in scores]
    vals = [scores[k]["score"] for k in scores]
    fig = go.Figure(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", fillcolor="rgba(67,147,195,0.25)",
        line=dict(color="#4393C3", width=2)))
    fig.update_layout(**_fig_base(), height=360,
        polar=dict(radialaxis=dict(visible=True, range=[0,10])), showlegend=False)
    return fig


def chart_comparison(comp: dict) -> go.Figure:
    cats, cur, prev = [], [], []
    for d in [comp.get("improvements",{}), comp.get("declines",{})]:
        for cat, data in d.items():
            if cat.replace("_"," ").title() not in cats:
                cats.append(cat.replace("_"," ").title())
                cur.append(data["current"]); prev.append(data["previous"])
    fig = go.Figure([
        go.Bar(name="Previous", x=cats, y=prev, marker_color="#5b8cdb"),
        go.Bar(name="Current",  x=cats, y=cur,  marker_color="#4caf89"),
    ])
    fig.update_layout(**_fig_base(), height=320, barmode="group",
        yaxis=dict(range=[0,10]), legend=dict(orientation="h", y=1.08))
    return fig


# =============================================================================
#  AUTO-PLAY AUDIO HELPER
# =============================================================================

def autoplay_audio(audio_bytes: bytes):
    """Embed audio with autoplay in Streamlit via HTML."""
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio autoplay style="width:100%">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        f'</audio>',
        unsafe_allow_html=True
    )


# =============================================================================
#  MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="AI Voice Interviewer", page_icon="🎙️",
                       layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
    html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; }
    h1,h2,h3,h4{ font-family:'Syne',sans-serif; }
    .stApp{ background:#0d0d1a; color:#e2e2e2; }
    section[data-testid="stSidebar"]{ background:#111125 !important; border-right:1px solid rgba(255,255,255,0.05); }
    .stButton>button{ border-radius:8px; font-family:'Syne',sans-serif; font-weight:700;
        font-size:0.82rem; letter-spacing:0.05em; border:1px solid rgba(255,255,255,0.12);
        background:rgba(255,255,255,0.05); color:#e0e0e0; transition:all .2s; }
    .stButton>button:hover{ background:rgba(255,255,255,0.1); border-color:rgba(255,255,255,0.25); }
    div[data-testid="stExpander"]{ background:rgba(255,255,255,0.02);
        border:1px solid rgba(255,255,255,0.06); border-radius:10px; }
    .msg-interviewer{ background:rgba(76,175,137,0.08); border-left:3px solid #4caf89;
        border-radius:0 10px 10px 0; padding:12px 16px; margin:8px 0; }
    .msg-candidate{ background:rgba(91,140,219,0.08); border-left:3px solid #5b8cdb;
        border-radius:0 10px 10px 0; padding:12px 16px; margin:8px 0; }
    .msg-role{ font-size:0.68rem; font-weight:700; letter-spacing:0.1em;
        text-transform:uppercase; margin-bottom:4px; }
    .tone-badge{ display:inline-block; padding:4px 10px; border-radius:20px;
        font-size:0.72rem; font-weight:700; letter-spacing:0.05em; }
    .status-pill{ display:inline-flex; align-items:center; gap:6px; padding:4px 12px;
        border-radius:20px; font-size:0.78rem; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)

    # ── session defaults ──────────────────────────────────────────────────────
    for k, v in {
        "interviewer":    None,
        "tone_analyser":  None,
        "conv":           [],        # [{role, content}]
        "tone_history":   [],        # [ToneResult]
        "active":         False,
        "feedback":       None,
        "candidate_id":   "",
        "last_emotion":   "neutral", # latest detected emotion → adapts TTS voice
        "last_audio_key": 0,         # increments to force st.audio_input reset
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # lazy init
    if st.session_state.interviewer is None:
        try: st.session_state.interviewer = RAGInterviewer()
        except ValueError as e: st.error(str(e)); st.stop()
    if TONE_AVAILABLE and st.session_state.tone_analyser is None:
        st.session_state.tone_analyser = ToneAnalyser()

    iw = st.session_state.interviewer

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Setup")
        if os.getenv("OPENAI_API_KEY"):
            k = os.getenv("OPENAI_API_KEY")
            st.success(f"✅ API: `{k[:5]}…{k[-3:]}`")
        if TONE_AVAILABLE:
            st.success("🧠 Tone Engine: Ready")
        else:
            st.warning("⚠️ librosa not installed")

        st.markdown("---")
        st.session_state.candidate_id = st.text_input(
            "Candidate ID", value=st.session_state.candidate_id,
            help="Reuse to track progress across sessions")

        st.markdown("---")
        itype = st.radio("Interview Type", ["Company-Specific","General"])
        company = None
        kb = CompanyKnowledgeBase()
        if itype == "Company-Specific":
            company = st.selectbox("Company", kb.companies())
        else:
            company = "General"

        job_role   = st.text_input("Job Role", value="Software Engineer")
        difficulty = st.selectbox("Difficulty", ["Beginner","Intermediate","Advanced"])

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Start", use_container_width=True,
                         disabled=st.session_state.active):
                st.session_state.active         = True
                st.session_state.conv           = []
                st.session_state.tone_history   = []
                st.session_state.feedback       = None
                st.session_state.last_emotion   = "neutral"
                st.session_state.last_audio_key += 1

                with st.spinner("Starting interview…"):
                    text = iw.start(job_role, difficulty, company)
                    audio = iw.speak(text, "neutral")

                st.session_state.conv.append({"role":"interviewer","content":text,
                                               "audio": audio})
                st.rerun()

        with c2:
            if st.button("⏹️ End", use_container_width=True,
                         disabled=not st.session_state.active):
                st.session_state.active = False
                if len(st.session_state.conv) > 2:
                    with st.spinner("Generating feedback…"):
                        fb = iw.generate_feedback(
                            [{"role":m["role"],"content":m["content"]}
                             for m in st.session_state.conv],
                            job_role, company,
                            st.session_state.candidate_id or None)
                        st.session_state.feedback = fb
                st.rerun()

        if st.button("🔄 Reset", use_container_width=True):
            for k in ["conv","tone_history","active","feedback","last_emotion"]:
                st.session_state[k] = [] if k in ["conv","tone_history"] else \
                                       False if k == "active" else \
                                       None if k == "feedback" else "neutral"
            st.session_state.last_audio_key += 1
            st.rerun()

        # History sidebar stats
        if st.session_state.candidate_id:
            st.markdown("---")
            hist = iw.history.history(st.session_state.candidate_id)
            if hist:
                st.metric("Past Interviews", len(hist))
                st.metric("Last Score", f"{hist[-1].get('overall_score','N/A')}/10")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style='font-size:1.9rem;margin-bottom:2px;'>
        🎙️ AI Voice Interviewer
        <span style='color:#4caf89;'>Pro</span>
    </h1>
    <p style='color:#555;margin:0 0 12px;font-size:0.9rem;'>
        Voice-to-voice · Whisper ASR · OpenAI TTS · Auto Tone Analysis
    </p>
    """, unsafe_allow_html=True)

    # ── Welcome screen ────────────────────────────────────────────────────────
    if not st.session_state.active and not st.session_state.conv:
        col1, col2, col3 = st.columns(3)
        for col, icon, title, body in [
            (col1,"🎙️","Voice Only",
             "Speak your answers — no typing needed. Whisper transcribes everything automatically."),
            (col2,"🧠","Auto Tone Analysis",
             "Every answer is silently analysed for confidence, anxiety, clarity, emotion and more."),
            (col3,"📊","End-of-Session Report",
             "Full tone + interview report shown once the interview ends. No interruptions during."),
        ]:
            with col:
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                            border-radius:12px;padding:20px;'>
                    <div style='font-size:2rem;'>{icon}</div>
                    <h4 style='margin:8px 0 6px;'>{title}</h4>
                    <p style='color:#666;font-size:0.88rem;margin:0;'>{body}</p>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👈 Configure your interview in the sidebar, then click **▶️ Start**.")
        return

    # ── Active interview ──────────────────────────────────────────────────────
    if st.session_state.conv or st.session_state.active:

        # Status pill
        if st.session_state.active:
            tone_count = len(st.session_state.tone_history)
            last_em    = st.session_state.last_emotion
            em_emoji   = EMOTION_EMOJI.get(last_em, "😐")
            em_color   = EMOTION_COLOR.get(last_em, "#888")
            st.markdown(f"""
            <div style='display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap;'>
                <span class='status-pill' style='background:rgba(76,175,137,0.15);
                    border:1px solid rgba(76,175,137,0.3);color:#4caf89;'>
                    🔴 Interview Live
                </span>
                <span class='status-pill' style='background:rgba(255,255,255,0.05);
                    border:1px solid rgba(255,255,255,0.1);color:#aaa;'>
                    🧠 {tone_count} answer{'s' if tone_count!=1 else ''} analysed
                </span>
                <span class='status-pill' style='background:{em_color}22;
                    border:1px solid {em_color}55;color:{em_color};'>
                    {em_emoji} Last tone: {last_em.capitalize()}
                </span>
            </div>
            """, unsafe_allow_html=True)

        # ── Conversation log ──────────────────────────────────────────────────
        st.markdown("#### 💬 Conversation")
        conv_container = st.container()
        with conv_container:
            for i, msg in enumerate(st.session_state.conv):
                role    = msg["role"]
                content = msg["content"]
                audio   = msg.get("audio")

                if role == "interviewer":
                    st.markdown(f"""
                    <div class='msg-interviewer'>
                        <div class='msg-role' style='color:#4caf89;'>🤖 AI Interviewer</div>
                        {content}
                    </div>""", unsafe_allow_html=True)
                    # Play only the LAST interviewer message automatically
                    if audio and i == len(st.session_state.conv) - 1 and st.session_state.active:
                        autoplay_audio(audio)
                    elif audio:
                        # Replay button for older messages
                        if st.button(f"▶ Replay", key=f"replay_{i}"):
                            autoplay_audio(audio)

                else:  # candidate
                    tone_res = msg.get("tone")
                    em = tone_res.emotion_label if tone_res else None
                    em_badge = ""
                    if em:
                        c = EMOTION_COLOR.get(em,"#888")
                        em_badge = (f"<span class='tone-badge' style='background:{c}22;"
                                    f"border:1px solid {c}55;color:{c};float:right;'>"
                                    f"{EMOTION_EMOJI.get(em,'😐')} {em.capitalize()}</span>")
                    st.markdown(f"""
                    <div class='msg-candidate'>
                        <div class='msg-role' style='color:#5b8cdb;'>
                            👤 You {em_badge}
                        </div>
                        {content}
                    </div>""", unsafe_allow_html=True)

        # ── Voice input area ──────────────────────────────────────────────────
        if st.session_state.active:
            st.markdown("---")
            st.markdown("#### 🎙️ Your Turn — Record Your Answer")
            st.caption("Press the mic, speak your answer, then press stop. Processing is automatic.")

            audio_input = st.audio_input(
                "Speak your answer",
                key=f"mic_{st.session_state.last_audio_key}"
            )

            if audio_input is not None:
                audio_bytes = audio_input.read()

                with st.spinner("🎙️ Transcribing… 🧠 Analysing tone…"):
                    # ── Step 1: Whisper ASR ───────────────────────────────────
                    transcript = iw.transcribe(audio_bytes)

                    # ── Step 2: Tone Analysis (AUTOMATIC — no button needed) ──
                    tone_result = None
                    if TONE_AVAILABLE and st.session_state.tone_analyser:
                        try:
                            audio_np, sr = sf.read(io.BytesIO(audio_bytes))
                            if audio_np.ndim > 1:
                                audio_np = audio_np.mean(axis=1)
                            audio_np = audio_np.astype(np.float32)

                            # Get the last interviewer question for context
                            last_q = next((m["content"] for m in reversed(st.session_state.conv)
                                           if m["role"] == "interviewer"), "")

                            tone_result = st.session_state.tone_analyser.analyse(
                                audio_np, sr,
                                linguistic_conf=0.5,
                                question_text=last_q,
                                answer_text=transcript
                            )
                            st.session_state.tone_history.append(tone_result)
                            st.session_state.last_emotion = tone_result.emotion_label
                        except Exception as e:
                            st.warning(f"Tone analysis skipped: {e}")

                    # ── Step 3: Add candidate turn to conversation ─────────────
                    st.session_state.conv.append({
                        "role":    "candidate",
                        "content": transcript,
                        "tone":    tone_result,
                    })

                    # ── Step 4: Generate next interviewer question ─────────────
                    next_q = iw.next_question(
                        [{"role":m["role"],"content":m["content"]}
                         for m in st.session_state.conv],
                        job_role, difficulty, company,
                        emotion=st.session_state.last_emotion
                    )

                    # ── Step 5: TTS — voice adapts to candidate's emotion ──────
                    next_audio = iw.speak(next_q, st.session_state.last_emotion)

                    st.session_state.conv.append({
                        "role":    "interviewer",
                        "content": next_q,
                        "audio":   next_audio,
                    })

                # Increment key so audio_input resets for next turn
                st.session_state.last_audio_key += 1
                st.rerun()

    # =========================================================================
    #  END-OF-INTERVIEW REPORT  (shown only after interview ends)
    # =========================================================================
    if st.session_state.feedback and not st.session_state.active:
        fb    = st.session_state.feedback
        tones = st.session_state.tone_history   # all ToneResult objects

        st.markdown("---")
        st.markdown("# 📊 Interview Report")

        # ── SECTION 1: TONE ANALYSIS REPORT ──────────────────────────────────
        if tones:
            st.markdown("## 🎙️ Tone & Behavioral Analysis")
            st.caption(f"Based on {len(tones)} recorded answers")

            # Average score metrics
            avg = lambda fn: round(sum(fn(r) for r in tones)/len(tones), 1)
            m_cols = st.columns(6)
            metrics = [
                ("💪 Confidence",  avg(lambda r: r.behavioral.confidence_score)),
                ("😰 Anxiety",     avg(lambda r: r.behavioral.anxiety_score)),
                ("🗣️ Clarity",     avg(lambda r: r.behavioral.communication_clarity_score)),
                ("⚡ Engagement",  avg(lambda r: r.behavioral.engagement_index)),
                ("🌊 Fluency",     avg(lambda r: r.behavioral.fluency_score)),
                ("🎯 Final CI",    avg(lambda r: r.final_confidence_index)),
            ]
            for col, (label, val) in zip(m_cols, metrics):
                col.metric(label, f"{val}/100")

            st.markdown("---")

            # Trend + Radar side by side
            t1, t2 = st.columns([3, 2])
            with t1:
                st.markdown("**Score Trend Across Answers**")
                st.plotly_chart(chart_trend(tones), use_container_width=True, key="trend")
            with t2:
                st.markdown("**Average Behavioral Profile**")
                st.plotly_chart(chart_radar_tone(tones), use_container_width=True, key="tone_radar")

            st.markdown("---")

            # Emotion distribution per answer
            st.markdown("**Emotion Distribution Per Answer**")
            st.plotly_chart(chart_emotion_bar(tones), use_container_width=True, key="em_bar")

            # Emotion journey timeline
            st.markdown("**Emotion Journey**")
            em_cols = st.columns(min(len(tones), 10))
            for i, r in enumerate(tones):
                em = r.emotion_label
                with em_cols[i % len(em_cols)]:
                    c = EMOTION_COLOR.get(em, "#888")
                    st.markdown(f"""
                    <div style='text-align:center;padding:10px 6px;
                                background:{c}15;border:1px solid {c}40;
                                border-radius:10px;'>
                        <div style='font-size:1.5rem;'>{EMOTION_EMOJI.get(em,'😐')}</div>
                        <div style='font-size:0.7rem;color:{c};font-weight:700;'>
                            {em.capitalize()}</div>
                        <div style='font-size:0.62rem;color:#555;'>A#{i+1}</div>
                        <div style='font-size:0.65rem;color:#888;margin-top:3px;'>
                            CI: {r.final_confidence_index:.0f}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # Per-answer detail accordion
            st.markdown("**Per-Answer Breakdown**")
            for i, r in enumerate(tones):
                b, f_ = r.behavioral, r.acoustic
                em = r.emotion_label
                c  = EMOTION_COLOR.get(em, "#888")
                with st.expander(
                    f"Answer #{i+1} — {EMOTION_EMOJI.get(em,'')} {em.capitalize()} "
                    f"| CI: {r.final_confidence_index:.1f}/100"
                ):
                    if r.question_text:
                        st.markdown(f"**Q:** _{r.question_text}_")
                    if r.answer_text:
                        st.markdown(f"**A:** {r.answer_text}")

                    sc1, sc2 = st.columns(2)
                    with sc1:
                        # Horizontal score bars
                        score_fig = go.Figure()
                        labels = ["Confidence","Clarity","Engagement","Fluency","Anxiety","Sadness"]
                        vals_s = [b.confidence_score, b.communication_clarity_score,
                                  b.engagement_index, b.fluency_score,
                                  b.anxiety_score, b.sadness_score]
                        colors_s = [SCORE_COLOR.get(l,"#888") for l in labels]
                        score_fig.add_trace(go.Bar(x=vals_s, y=labels, orientation="h",
                            marker=dict(color=colors_s),
                            text=[f"{v:.0f}" for v in vals_s], textposition="outside",
                            textfont=dict(size=10,color="#ccc"),
                            hovertemplate="%{y}: %{x:.1f}/100<extra></extra>"))
                        score_fig.update_layout(**_fig_base(), height=200,
                            xaxis=dict(range=[0,115],showgrid=False,showticklabels=False),
                            yaxis=dict(showgrid=False), showlegend=False)
                        st.plotly_chart(score_fig, use_container_width=True, key=f"sb_{i}")

                    with sc2:
                        # Emotion donut
                        donut_fig = go.Figure(go.Pie(
                            labels=[k.capitalize() for k in r.emotion_scores],
                            values=list(r.emotion_scores.values()), hole=0.6,
                            marker=dict(colors=[EMOTION_COLOR.get(k,"#888")
                                                for k in r.emotion_scores]),
                            textinfo="label+percent",
                            textfont=dict(size=9,color="#ddd"),
                            sort=True))
                        donut_fig.add_annotation(text=f"{EMOTION_EMOJI.get(em,'')}<br>{em.capitalize()}",
                            x=0.5, y=0.5, font=dict(size=11,color="#eee"), showarrow=False)
                        donut_fig.update_layout(**_fig_base(), height=200, showlegend=False)
                        st.plotly_chart(donut_fig, use_container_width=True, key=f"donut_{i}")

                    # Acoustic details
                    ac1,ac2,ac3,ac4 = st.columns(4)
                    for col, label, val in [
                        (ac1,"🎵 Pitch",    f"{f_.pitch_mean:.0f} Hz"),
                        (ac2,"〰️ Var.",     f"{f_.pitch_std:.1f} Hz"),
                        (ac3,"⚡ Energy",   f"{f_.energy_mean:.4f}"),
                        (ac4,"📻 HNR",      f"{f_.hnr:.1f} dB"),
                    ]:
                        col.metric(label, val)

                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03);border-radius:8px;
                                padding:8px 14px;font-size:0.8rem;color:#888;margin-top:6px;'>
                        <b style='color:#bbb;'>Fusion:</b>
                        {r.acoustic_confidence:.2f}×0.5 (Acoustic) +
                        {r.linguistic_confidence:.2f}×0.3 (Linguistic) +
                        {r.clarity_score:.2f}×0.2 (Clarity)
                        → <b style='color:{c};'>{r.final_confidence_index}/100</b>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")

        # ── SECTION 2: INTERVIEW QUALITY REPORT ──────────────────────────────
        st.markdown("## 🏆 Interview Quality Report")

        if "error" in fb:
            st.error(f"Feedback error: {fb['error']}")
        else:
            st.write(fb.get("overall_summary",""))
            overall = fb.get("overall_score", 0)
            rec     = fb.get("recommendation","N/A")
            rec_e   = {"hire":"✅","maybe":"⚠️","no_hire":"❌"}.get(rec,"❓")

            m1,m2,m3 = st.columns(3)
            m1.metric("Overall Score",    f"{overall}/10")
            m2.metric("Recommendation",   f"{rec_e} {rec.replace('_',' ').title()}")
            m3.metric("Date",             fb.get("date","")[:10])

            st.markdown("---")
            rc1, rc2 = st.columns([1,1])
            with rc1:
                st.markdown("**Score Breakdown**")
                for cat, data in fb.get("scores",{}).items():
                    with st.expander(f"**{cat.replace('_',' ').title()}** — {data['score']}/10"):
                        st.progress(data["score"]/10)
                        for s in data.get("strengths",[]): st.write(f"✅ {s}")
                        for w in data.get("weaknesses",[]): st.write(f"❌ {w}")
                        st.caption(data.get("details",""))
            with rc2:
                st.markdown("**Radar Chart**")
                if fb.get("scores"):
                    st.plotly_chart(chart_interview_radar(fb["scores"]),
                                    use_container_width=True, key="iq_radar")

            st.markdown("---")
            h1, h2 = st.columns(2)
            with h1:
                st.markdown("**🌟 Key Highlights**")
                for h in fb.get("key_highlights",[]): st.success(f"✨ {h}")
            with h2:
                st.markdown("**📈 Areas to Improve**")
                for a in fb.get("improvement_areas",[]): st.warning(f"⚠️ {a}")

            st.markdown("**💡 Actionable Recommendations**")
            for i, r in enumerate(fb.get("actionable_recommendations",[]),1):
                st.info(f"**{i}.** {r}")

            # Progress comparison
            if st.session_state.candidate_id:
                comp = iw.history.compare(st.session_state.candidate_id)
                if comp:
                    st.markdown("---")
                    st.markdown("**📊 Progress vs Previous Interview**")
                    cp1,cp2,cp3 = st.columns(3)
                    cp1.metric("Improved", len(comp.get("improvements",{})))
                    cp2.metric("Declined",  len(comp.get("declines",{})))
                    avg_imp = (sum(d["change"] for d in comp.get("improvements",{}).values())
                               / max(len(comp.get("improvements",{})),1))
                    cp3.metric("Avg Δ", f"{avg_imp:+.1f}")
                    st.plotly_chart(chart_comparison(comp),
                                    use_container_width=True, key="cmp")

        # ── Downloads ─────────────────────────────────────────────────────────
        st.markdown("---")
        tone_summary_txt = ""
        if tones:
            tone_summary_txt = "\n\nTONE ANALYSIS SUMMARY\n" + "="*50 + "\n"
            for i, r in enumerate(tones, 1):
                tone_summary_txt += (
                    f"\nAnswer #{i}\n"
                    f"  Emotion    : {r.emotion_label.upper()}\n"
                    f"  Confidence : {r.behavioral.confidence_score:.1f}/100\n"
                    f"  Anxiety    : {r.behavioral.anxiety_score:.1f}/100\n"
                    f"  Clarity    : {r.behavioral.communication_clarity_score:.1f}/100\n"
                    f"  Engagement : {r.behavioral.engagement_index:.1f}/100\n"
                    f"  Final CI   : {r.final_confidence_index:.1f}/100\n"
                    f"  Q: {r.question_text[:80]}\n"
                    f"  A: {r.answer_text[:120]}\n"
                )

        report_txt = f"""AI VOICE INTERVIEW REPORT
{"="*55}
Date        : {fb.get('date','N/A')}
Role        : {fb.get('job_role','N/A')}
Company     : {fb.get('company','N/A')}
Score       : {fb.get('overall_score','N/A')}/10
Decision    : {fb.get('recommendation','N/A')}
{tone_summary_txt}

INTERVIEW QUALITY
{"="*55}
{fb.get('overall_summary','')}

SCORES
{"─"*40}
"""
        for cat, data in fb.get("scores",{}).items():
            report_txt += (f"{cat.replace('_',' ').title()}: {data['score']}/10\n"
                           f"  + {', '.join(data.get('strengths',[]))}\n"
                           f"  - {', '.join(data.get('weaknesses',[]))}\n\n")
        report_txt += "\nRECOMMENDATIONS\n" + "─"*40 + "\n"
        for i,r in enumerate(fb.get("actionable_recommendations",[]),1):
            report_txt += f"{i}. {r}\n"

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("📥 Download JSON", json.dumps(fb, indent=2),
                f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json", use_container_width=True)
        with dl2:
            st.download_button("📄 Download Full Report", report_txt,
                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain", use_container_width=True)


if __name__ == "__main__":
    main()
