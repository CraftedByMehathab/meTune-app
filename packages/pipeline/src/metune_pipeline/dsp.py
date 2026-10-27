"""Audio-health measurements. Great at measuring, bad at meaning: none of these alone decide tone or noise."""
import numpy as np
from scipy.signal import stft

from .config import PipelineConfig


def dsp_features(x: np.ndarray, cfg: PipelineConfig) -> dict:
    sr = cfg.sample_rate
    x = x[int(cfg.skip_start_s * sr):]                          # mask connect tone
    hop = int(cfg.frame_ms / 1000 * sr)
    frames = np.lib.stride_tricks.sliding_window_view(x, hop * 2)[::hop]
    e = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    floor = np.percentile(e, 10)
    active = e > floor + 12
    f, _, Z = stft(x, sr, nperseg=1024, noverlap=512)
    P = np.abs(Z) ** 2
    tot = 10 * np.log10(P.sum(0) + 1e-12)
    act_spec = tot > np.percentile(tot, 10) + 15
    hi = P[(f > 4200) & (f < 7500)].sum(0) / (P.sum(0) + 1e-12)
    minutes = len(x) / sr / 60
    return {
        "clip_pct": round(float(100 * np.mean(np.abs(x) > 0.99)), 4),
        "noise_floor_db": round(float(floor), 1),
        "snr_proxy_db": round(float(np.percentile(e, 95) - floor), 1),
        "spikes_per_min": round(float((np.diff(e) > cfg.spike_jump_db).sum() / minutes), 1),
        "wideband_share": round(float(((hi > 0.003) & act_spec).sum() / max(act_spec.sum(), 1)), 2),
        "energy_std_db": round(float(e[active].std()), 1) if active.any() else 0.0,
        "rms_db": round(float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)), 1),
    }
