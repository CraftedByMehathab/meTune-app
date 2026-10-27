"""Auto-scoring when labels are present (DR-012): accuracy + macro F1 per field, tone report + confusion."""
from metune_pipeline.schema import SCORED_FIELDS, TONES
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


class FieldScore(BaseModel):
    accuracy: float
    macro_f1: float


class ClassScore(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class Confusion(BaseModel):
    labels: list[str]
    matrix: list[list[int]]          # rows = true, cols = predicted


class Metrics(BaseModel):
    n: int
    per_field: dict[str, FieldScore]
    tone_report: list[ClassScore]
    tone_confusion: Confusion


def score(pairs: list[tuple[dict, dict]]) -> Metrics | None:
    """pairs = [(label, prediction)]. Macro F1 over classes present in truth or prediction."""
    if not pairs:
        return None
    per_field = {}
    for f in SCORED_FIELDS:
        y = [str(t[f]) for t, _ in pairs]
        p = [str(r[f]) for _, r in pairs]
        per_field[f] = FieldScore(
            accuracy=round(sum(a == b for a, b in zip(y, p)) / len(y), 3),
            macro_f1=round(float(f1_score(y, p, average="macro", zero_division=0)), 3),
        )
    y = [t["emotional_tone"] for t, _ in pairs]
    p = [r["emotional_tone"] for _, r in pairs]
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=TONES, zero_division=0)
    report = [ClassScore(label=lbl, precision=round(float(a), 3), recall=round(float(b), 3),
                         f1=round(float(c), 3), support=int(d))
              for lbl, a, b, c, d in zip(TONES, prec, rec, f1, sup)]
    cm = confusion_matrix(y, p, labels=TONES).tolist()
    return Metrics(n=len(pairs), per_field=per_field, tone_report=report,
                   tone_confusion=Confusion(labels=TONES, matrix=cm))
