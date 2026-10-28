"""Build the synthetic proxy call set (DR-008 tiers 2+3, construction in DR-016).

Each call = an "agent" (neutral CREMA-D actor) and a "customer" (one CREMA-D actor, one emotion)
taking turns, optionally with background noise at a set SNR, a speaker overlap and a long silence,
then squeezed through a phone channel (300–3400 Hz, 8 kHz, Opus, fake stereo like the trial calls).
Labels are known by construction and written as a meTune manifest (name,result_json), so a split
can be uploaded in the web app as-is.

Splits are speaker-grouped: actors 1001–1045 + ESC-50 folds 1–3 → dev (tune here);
actors 1046–1091 + folds 4–5 → test (report once, never tune).

    uv run python eval/build_proxy.py            # → .data/proxy/{dev,test}/
"""
import argparse
import csv
import json
import random
import subprocess
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from metune_pipeline.decode import decode
from metune_pipeline.vad import speech_probs
from scipy.signal import butter, resample_poly, sosfilt

SR = 16000
ROOT = Path(".data/proxy")
SRC = ROOT / "src"
CREMA_URL = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/{}.wav"
CREMA_META = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/processedResults/summaryTable.csv"
ESC_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{}"
ESC_META = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"

# CREMA-D emotion → meTune tone (DR-016). SAD is left out: it maps to no tone cleanly.
TONE_OF = {"NEU": "neutral", "HAP": "satisfied", "DIS": "frustrated", "ANG": "upset", "FEA": "distressed"}
VOTE_CODE = {"NEU": "N", "HAP": "H", "DIS": "D", "ANG": "A", "FEA": "F"}

# ESC-50 category → noise type label; "background chatter" (babble) and "static" are synthesized.
NOISE_OF = {
    "keyboard_typing": "keyboard typing", "mouse_click": "keyboard typing",
    "engine": "road noise", "car_horn": "road noise", "siren": "road noise",
    "vacuum_cleaner": "appliance hum", "washing_machine": "appliance hum",
    "dog": "dog barking", "crying_baby": "baby crying",
}
NOISE_TYPES = sorted(set(NOISE_OF.values())) + ["background chatter", "static"]
SEVERITY_SNR = {"low": 20.0, "medium": 10.0, "high": 3.0}   # dB, speech-active RMS vs noise RMS
# E31: overlap = both voices active at once for ≥ 0.5 s (VAD on the separate legs). The planned turn shift
# (1.2–2.0 s) often overlaps only the clips' leading/trailing silence (~0.5 s each end).
OVERLAP_MIN_S = 0.5


def fetch(url: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst


def load_audio(path: Path) -> np.ndarray:
    return decode(path, SR)


def intensity_of(levels: list[float]) -> str:
    """Mean perceived (voice-only) intensity of the customer clips, binned (DR-016)."""
    m = sum(levels) / len(levels)
    return "low" if m < 55 else "medium" if m < 70 else "high"


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def synth_static(n: int, rng: np.random.Generator) -> np.ndarray:
    """Line crackle: a faint hiss bed plus Poisson clicks/bursts (~90/min)."""
    x = rng.normal(0, 0.05, n)
    for t in rng.uniform(0, n / SR, rng.poisson(1.5 * n / SR)):
        i, w = int(t * SR), int(rng.integers(20, 400))
        x[i:i + w] += rng.normal(0, 1.0, len(x[i:i + w])) * np.hanning(len(x[i:i + w]))
    return x.astype(np.float32)


def loop_to(x: np.ndarray, n: int) -> np.ndarray:
    return np.tile(x, n // len(x) + 1)[:n]


class Builder:
    def __init__(self, seed: int, overlap_p: float = 0.25, overlap_from_speech: bool = False):
        self.overlap_p, self.overlap_from_speech = overlap_p, overlap_from_speech
        self.rng = random.Random(seed)
        self.nrng = np.random.default_rng(seed)
        meta = fetch(CREMA_META, SRC / "crema" / "summaryTable.csv")
        self.clips = defaultdict(list)      # (actor, emotion) → [(file, level)] agreed by voice-only raters
        for r in csv.DictReader(meta.open()):
            actor, _, emo, _ = r["FileName"].split("_")
            if emo in VOTE_CODE and r["VoiceVote"] == VOTE_CODE[emo]:
                self.clips[(int(actor), emo)].append((r["FileName"], float(r["VoiceLevel"])))
        esc = fetch(ESC_META, SRC / "esc50" / "esc50.csv")
        self.esc = [r for r in csv.DictReader(esc.open()) if r["category"] in NOISE_OF]

    def crema(self, name: str) -> np.ndarray:
        return load_audio(fetch(CREMA_URL.format(name), SRC / "crema" / f"{name}.wav"))

    def noise(self, kind: str, n: int, folds: set[int], actors: list[int]) -> np.ndarray:
        if kind == "static":
            return synth_static(n, self.nrng)
        if kind == "background chatter":     # 4 other voices, overlapping, looped
            voices = []
            for a in self.rng.sample(actors, 4):
                emo = self.rng.choice([e for e in VOTE_CODE if self.clips[(a, e)]])
                v = np.concatenate([self.crema(f) for f, _ in self.rng.sample(self.clips[(a, emo)], min(3, len(self.clips[(a, emo)])))])
                voices.append(np.roll(loop_to(v, n), self.rng.randrange(n)))
            return np.sum(voices, axis=0)
        pool = [r for r in self.esc if NOISE_OF[r["category"]] == kind and int(r["fold"]) in folds]
        picks = self.rng.sample(pool, min(3, len(pool)))
        x = np.concatenate([load_audio(fetch(ESC_URL.format(r["filename"]), SRC / "esc50" / r["filename"])) for r in picks])
        return loop_to(x, n)

    def call(self, tone_emo: str, actors: list[int], folds: set[int],
             wideband_agent: bool = False) -> tuple[np.ndarray, dict, dict]:
        """wideband_agent (E21): like the trial calls — agent leg wideband VoIP and upbeat in ~half the calls,
        customer leg (+ its background noise) band-limited like a phone line. Extra RNG draws only in this mode,
        so the original splits stay reproducible."""
        rng = self.rng
        cust = rng.choice([a for a in actors if len(self.clips[(a, tone_emo)]) >= 3])
        agent_emo = "HAP" if wideband_agent and rng.random() < 0.5 else "NEU"
        agent = rng.choice([a for a in actors if a != cust and len(self.clips[(a, agent_emo)]) >= 3])
        c_clips = rng.sample(self.clips[(cust, tone_emo)], min(len(self.clips[(cust, tone_emo)]), rng.randint(4, 6)))
        a_clips = rng.sample(self.clips[(agent, agent_emo)], min(len(self.clips[(agent, agent_emo)]), 4))

        overlap = rng.random() < self.overlap_p
        long_silence = rng.random() < 0.2
        turns = []                                   # (speaker, audio), agent opens
        for i in range(max(len(c_clips), len(a_clips))):
            if i < len(a_clips):
                turns.append(("agent", self.crema(a_clips[i][0])))
            if i < len(c_clips):
                turns.append(("customer", self.crema(c_clips[i][0])))
        overlap_at = rng.randrange(1, len(turns) - 1) if overlap else -1
        silence_at = rng.randrange(2, len(turns) - 1) if long_silence else -1

        segs, t = [], int(0.8 * SR)
        for i, (who, x) in enumerate(turns):
            if i == overlap_at:
                shift = rng.uniform(1.2, 2.0)
                if self.overlap_from_speech:      # E31: 0.8–1.6 s of real speech on both legs, not clip edges
                    p_start, p_x, _ = segs[-1]
                    t = max(0, p_start + speech_bounds(p_x)[1] - speech_bounds(x)[0] - int((shift - 0.4) * SR))
                else:
                    t = max(0, t - int(shift * SR))   # start before the previous turn ends
            if i == silence_at:
                t += int(rng.uniform(12, 16) * SR)
            segs.append((t, x, who))
            t += len(x) + int(rng.uniform(0.3, 1.2) * SR)
        legs = {"agent": np.zeros(t + SR, np.float32), "customer": np.zeros(t + SR, np.float32)}
        for s, x, who in segs:
            legs[who][s:s + len(x)] += x
        overlap_s = both_talking_s(legs["agent"], legs["customer"])     # no RNG draws: splits stay reproducible
        # original mode: noise on the mixed call; wideband_agent: noise lives on the customer's side of the line
        speech = legs["customer"] if wideband_agent else legs["agent"] + legs["customer"]

        kind, severity, snr = "", "none", None
        if rng.random() < 0.6:
            kind, severity = rng.choice(NOISE_TYPES), rng.choice(list(SEVERITY_SNR))
            snr = SEVERITY_SNR[severity]
            n = self.noise(kind, len(speech), folds, actors)
            active = np.abs(speech) > 1e-3
            gain = rms(speech[active]) / (rms(n) * 10 ** (snr / 20))
            speech = speech + n * gain
        if wideband_agent:
            speech = narrowband(speech) + legs["agent"]
        x = speech / max(1.0, float(np.max(np.abs(speech))) / 0.9)   # headroom, never clip

        label = {
            "emotional_tone": TONE_OF[tone_emo],
            "emotional_intensity": "low" if tone_emo == "NEU" else intensity_of([lv for _, lv in c_clips]),
            "background_noise_present": bool(kind), "background_noise_type": kind,
            "background_noise_severity": severity, "audio_quality": "clear",
            "speaker_overlap_present": overlap_s >= OVERLAP_MIN_S, "long_silence_present": long_silence, "confidence": 1.0,
        }
        meta = {"customer": cust, "agent": agent, "agent_emotion": agent_emo, "crema_emotion": tone_emo,
                "noise": kind, "snr_db": snr,
                "customer_clips": " ".join(f for f, _ in c_clips), "duration_s": round(len(x) / SR, 1),
                "overlap_planned": overlap, "overlap_s": overlap_s}
        return x, label, meta


def speech_bounds(x: np.ndarray) -> tuple[int, int]:
    """First and last speech sample of a clip (Silero VAD > 0.5); the whole clip if none."""
    p, step = speech_probs(x, SR)
    idx = np.where(p > 0.5)[0]
    w = int(step * SR)
    return (int(idx[0] * w), int((idx[-1] + 1) * w)) if len(idx) else (0, len(x))


def both_talking_s(a: np.ndarray, b: np.ndarray) -> float:
    """Seconds where Silero VAD hears speech on both legs at once."""
    pa, step = speech_probs(a, SR)
    pb, _ = speech_probs(b, SR)
    n = min(len(pa), len(pb))
    return round(float(((pa[:n] > 0.5) & (pb[:n] > 0.5)).sum() * step), 2)


def narrowband(x: np.ndarray) -> np.ndarray:
    """Phone-line leg: 300–3400 Hz band-pass and an 8 kHz round trip (wideband_agent calls)."""
    y = sosfilt(butter(8, [300, 3400], btype="bandpass", fs=SR, output="sos"), x)
    return resample_poly(resample_poly(y, 1, 2), 2, 1)[: len(x)].astype(np.float32)


def write_call(x: np.ndarray, dst: Path, channel: str = "phone") -> None:
    """Phone channel: band-limit to 300–3400 Hz at 8 kHz, then Opus in Ogg, fake stereo (like the trial calls).
    Clean channel (E10 control): the same mix as lossless 16 kHz mono FLAC.
    Agent channel (E21): legs already shaped in Builder.call → wideband 16 kHz Opus, fake stereo."""
    if channel == "agent":
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                        "-ac", "2", "-c:a", "libopus", "-b:a", "32k", str(dst)],
                       input=x.astype(np.float32).tobytes(), check=True)
        return
    if channel == "clean":
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", str(dst)],
                       input=x.astype(np.float32).tobytes(), check=True)
        return
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-af", "highpass=f=300,lowpass=f=3400,aresample=8000", "-ac", "2", "-c:a", "libopus", "-b:a", "24k", str(dst)],
        input=x.astype(np.float32).tobytes(), check=True,
    )


def build_split(b: Builder, name: str, actors: list[int], folds: set[int], per_tone: int, channel: str,
                relabel: bool = False, tag: str = "") -> None:
    """relabel: replay an existing split (same seed and args) without writing audio; check every call matches the
    saved meta, then rewrite labels.csv / meta.csv (originals kept as *_v1.csv)."""
    out = ROOT / ((name if channel == "phone" else f"{name}_{channel}") + (f"_{tag}" if tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    old = {r["name"]: r for r in csv.DictReader((out / "meta.csv").open())} if relabel else {}
    plan = [e for e in TONE_OF for _ in range(per_tone)]
    b.rng.shuffle(plan)
    rows, metas = [], []
    for i, emo in enumerate(plan, 1):
        fname = f"{name}_{i:03d}.{'flac' if channel == 'clean' else 'ogg'}"
        x, label, meta = b.call(emo, actors, folds, wideband_agent=channel == "agent")
        if relabel:
            o = old[fname]
            assert (o["customer_clips"], o["noise"], float(o["duration_s"])) == \
                (meta["customer_clips"], meta["noise"], meta["duration_s"]), f"replay differs at {fname}"
        else:
            write_call(x, out / fname, channel)
        rows.append([fname, json.dumps(label)])
        metas.append({"name": fname, **meta})
        print(f"{fname}  {label['emotional_tone']:<10} {label['emotional_intensity']:<6} noise={meta['noise'] or '-':<18} "
              f"overlap={label['speaker_overlap_present']!s:<5} silence={label['long_silence_present']!s:<5} {meta['duration_s']}s")
    if relabel:
        for stem in ("labels", "meta"):
            if not (out / f"{stem}_v1.csv").exists():
                (out / f"{stem}.csv").rename(out / f"{stem}_v1.csv")
    with (out / "labels.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["name", "result_json"], *rows])
    with (out / "meta.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0]))
        w.writeheader()
        w.writerows(metas)


def prefetch(b: Builder) -> None:
    """Download every candidate CREMA-D clip up front, in parallel (≈ 60 MB, cached in .data/proxy/src)."""
    names = {f for clips in b.clips.values() for f, _ in clips}
    todo = [n for n in names if not (SRC / "crema" / f"{n}.wav").exists()]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda n: fetch(CREMA_URL.format(n), SRC / "crema" / f"{n}.wav"), todo))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tone", type=int, default=8, help="calls per tone per split")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--channel", choices=["phone", "clean", "agent"], default="phone",
                    help="clean = same calls without the phone channel (E10 control) → {split}_clean/; "
                         "agent = wideband upbeat agent + phone-line customer (E21) → {split}_agent/")
    ap.add_argument("--val", action="store_true",
                    help="build only a val split from actors 1036–1045 (held out from SER fine-tuning, E16)")
    ap.add_argument("--big", action="store_true",
                    help="build val_big (actors 1036–1045) + test_big (1046–1091) for model selection with less noise;"
                         " use with a new --seed and e.g. --per-tone 24")
    ap.add_argument("--relabel", action="store_true",
                    help="replay existing splits (same args + seed) and rewrite labels only, no audio (E31)")
    ap.add_argument("--overlap-p", type=float, default=0.25, help="share of calls with a planned overlap")
    ap.add_argument("--overlap-from-speech", action="store_true",
                    help="place overlaps on the clips' speech, not their edges (E31; new splits only)")
    ap.add_argument("--tag", default="", help="suffix for the split folders, e.g. ovl → val_big_agent_ovl")
    a = ap.parse_args()
    b = Builder(a.seed, a.overlap_p, a.overlap_from_speech)
    prefetch(b)
    if a.val:
        build_split(b, "val", list(range(1036, 1046)), {1, 2, 3}, a.per_tone, a.channel, a.relabel)
        return
    if a.big:
        build_split(b, "val_big", list(range(1036, 1046)), {1, 2, 3}, a.per_tone, a.channel, a.relabel, a.tag)
        build_split(b, "test_big", list(range(1046, 1092)), {4, 5}, a.per_tone, a.channel, a.relabel, a.tag)
        return
    # Same seed → same calls in both channels: the RNG is consumed by mixing only, never by encoding.
    build_split(b, "dev", list(range(1001, 1046)), {1, 2, 3}, a.per_tone, a.channel, a.relabel)
    build_split(b, "test", list(range(1046, 1092)), {4, 5}, a.per_tone, a.channel, a.relabel)


if __name__ == "__main__":
    main()
