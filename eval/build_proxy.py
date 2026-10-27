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
    def __init__(self, seed: int):
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

    def call(self, tone_emo: str, actors: list[int], folds: set[int]) -> tuple[np.ndarray, dict, dict]:
        rng = self.rng
        cust = rng.choice([a for a in actors if len(self.clips[(a, tone_emo)]) >= 3])
        agent = rng.choice([a for a in actors if a != cust and len(self.clips[(a, "NEU")]) >= 3])
        c_clips = rng.sample(self.clips[(cust, tone_emo)], min(len(self.clips[(cust, tone_emo)]), rng.randint(4, 6)))
        a_clips = rng.sample(self.clips[(agent, "NEU")], min(len(self.clips[(agent, "NEU")]), 4))

        overlap = rng.random() < 0.25
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
        for i, (_, x) in enumerate(turns):
            if i == overlap_at:
                t = max(0, t - int(rng.uniform(1.2, 2.0) * SR))   # start before the previous turn ends
            if i == silence_at:
                t += int(rng.uniform(12, 16) * SR)
            segs.append((t, x))
            t += len(x) + int(rng.uniform(0.3, 1.2) * SR)
        speech = np.zeros(t + SR, np.float32)
        for s, x in segs:
            speech[s:s + len(x)] += x

        kind, severity, snr = "", "none", None
        if rng.random() < 0.6:
            kind, severity = rng.choice(NOISE_TYPES), rng.choice(list(SEVERITY_SNR))
            snr = SEVERITY_SNR[severity]
            n = self.noise(kind, len(speech), folds, actors)
            active = np.abs(speech) > 1e-3
            gain = rms(speech[active]) / (rms(n) * 10 ** (snr / 20))
            speech = speech + n * gain
        x = speech / max(1.0, float(np.max(np.abs(speech))) / 0.9)   # headroom, never clip

        label = {
            "emotional_tone": TONE_OF[tone_emo],
            "emotional_intensity": "low" if tone_emo == "NEU" else intensity_of([lv for _, lv in c_clips]),
            "background_noise_present": bool(kind), "background_noise_type": kind,
            "background_noise_severity": severity, "audio_quality": "clear",
            "speaker_overlap_present": overlap, "long_silence_present": long_silence, "confidence": 1.0,
        }
        meta = {"customer": cust, "agent": agent, "crema_emotion": tone_emo, "noise": kind, "snr_db": snr,
                "customer_clips": " ".join(f for f, _ in c_clips), "duration_s": round(len(x) / SR, 1)}
        return x, label, meta


def write_call(x: np.ndarray, dst: Path, channel: str = "phone") -> None:
    """Phone channel: band-limit to 300–3400 Hz at 8 kHz, then Opus in Ogg, fake stereo (like the trial calls).
    Clean channel (E10 control): the same mix as lossless 16 kHz mono FLAC."""
    if channel == "clean":
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", str(dst)],
                       input=x.astype(np.float32).tobytes(), check=True)
        return
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-af", "highpass=f=300,lowpass=f=3400,aresample=8000", "-ac", "2", "-c:a", "libopus", "-b:a", "24k", str(dst)],
        input=x.astype(np.float32).tobytes(), check=True,
    )


def build_split(b: Builder, name: str, actors: list[int], folds: set[int], per_tone: int, channel: str) -> None:
    out = ROOT / (name if channel == "phone" else f"{name}_{channel}")
    out.mkdir(parents=True, exist_ok=True)
    plan = [e for e in TONE_OF for _ in range(per_tone)]
    b.rng.shuffle(plan)
    rows, metas = [], []
    for i, emo in enumerate(plan, 1):
        fname = f"{name}_{i:03d}.{'ogg' if channel == 'phone' else 'flac'}"
        x, label, meta = b.call(emo, actors, folds)
        write_call(x, out / fname, channel)
        rows.append([fname, json.dumps(label)])
        metas.append({"name": fname, **meta})
        print(f"{fname}  {label['emotional_tone']:<10} {label['emotional_intensity']:<6} noise={meta['noise'] or '-':<18} "
              f"overlap={label['speaker_overlap_present']!s:<5} silence={label['long_silence_present']!s:<5} {meta['duration_s']}s")
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
    ap.add_argument("--channel", choices=["phone", "clean"], default="phone",
                    help="clean = same calls without the phone channel (E10 control) → {split}_clean/")
    a = ap.parse_args()
    b = Builder(a.seed)
    prefetch(b)
    # Same seed → same calls in both channels: the RNG is consumed by mixing only, never by encoding.
    build_split(b, "dev", list(range(1001, 1046)), {1, 2, 3}, a.per_tone, a.channel)
    build_split(b, "test", list(range(1046, 1092)), {4, 5}, a.per_tone, a.channel)


if __name__ == "__main__":
    main()
