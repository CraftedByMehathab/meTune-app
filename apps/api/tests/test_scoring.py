from metune_api.scoring import score

BASE = {"emotional_intensity": "medium", "background_noise_present": False, "background_noise_severity": "none",
        "audio_quality": "clear", "speaker_overlap_present": False, "long_silence_present": False}


def row(tone, **kw):
    return {**BASE, "emotional_tone": tone, **kw}


def test_macro_f1_punishes_always_neutral():
    truth = [row("upset"), row("neutral"), row("satisfied")]
    m = score([(t, row("neutral")) for t in truth])
    assert m.per_field["emotional_tone"].accuracy == round(1 / 3, 3)
    assert m.per_field["emotional_tone"].macro_f1 < 0.34
    assert m.tone_confusion.matrix[0][0] == 1        # neutral → neutral
    assert m.n == 3


def test_empty_returns_none():
    assert score([]) is None
