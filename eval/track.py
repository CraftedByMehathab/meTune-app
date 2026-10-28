"""Experiment tracking with MLflow, local only (mlflow.db + mlartifacts/, both gitignored; no server, nothing on the VM).

Every eval/runs/<dir>/summary.json becomes one MLflow run in the "meTune" experiment: numeric fields → metrics,
config / split / overrides → params, summary.json and predictions.csv → artifacts, start time from the folder name.
eval/runs/ stays the source of truth; MLflow is the place to compare runs. The Experiment Log keeps the reasoning.

    uv run --group track python eval/track.py backfill        # import every run folder (skips ones already logged)
    make mlflow-ui                                            # http://127.0.0.1:5001

run_eval.py and the e* scripts call log_run() after writing a run folder; without the `track` group it is a no-op.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

RUNS = Path("eval/runs")
TRACKING_URI = "sqlite:///mlflow.db"
ARTIFACTS = Path("mlartifacts")
EXPERIMENT = "meTune"


def _flatten(d: dict, prefix: str = ""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flatten(v, f"{key}.")
        else:
            yield key, v


def _clean(key: str) -> str:
    """MLflow keys: letters, digits, _ - . / and spaces only."""
    return re.sub(r"[^\w\-./ ]", "_", key)[:250]


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    eid = exp.experiment_id if exp else client.create_experiment(EXPERIMENT,
                                                                 artifact_location=ARTIFACTS.absolute().as_uri())
    return client, eid


def log_run(run_dir: Path, quiet: bool = False) -> str | None:
    """Log one run folder; returns the MLflow run id, or None when skipped. Never raises into the caller."""
    try:
        from mlflow.entities import Metric, Param, RunTag
    except ImportError:
        if not quiet:
            print("(MLflow not installed: `uv run --group track …` to log this run)")
        return None
    try:
        client, eid = _client()
        if client.search_runs([eid], f"tags.source_dir = '{run_dir.name}'", max_results=1):
            return None
        s = json.loads((run_dir / "summary.json").read_text())
        stamp = re.match(r"(\d{8}-\d{4})_(.*)", run_dir.name)
        start = int(datetime.strptime(stamp[1], "%Y%m%d-%H%M").timestamp() * 1000) if stamp else None
        name = stamp[2] if stamp else run_dir.name
        exp_id = re.match(r"([A-Z]+\d+[a-z]?)", name)
        tags = {"source_dir": run_dir.name, "experiment_id": exp_id[1] if exp_id else "",
                "invalid": str("INVALID" in name).lower()}
        run = client.create_run(eid, start_time=start, run_name=name,
                                tags={"mlflow.runName": name, **tags})
        params, metrics = {}, []
        for k, v in _flatten(s):
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, (int, float)) and not k.startswith("config."):
                metrics.append(Metric(_clean(k), float(v), start or 0, 0))
            elif isinstance(v, (str, int, float, list)) and v is not None:
                params[_clean(k)] = json.dumps(v) if isinstance(v, list) else str(v)
        client.log_batch(run.info.run_id, metrics=metrics[:1000],
                         params=[Param(k, v[:6000]) for k, v in list(params.items())[:100]],
                         tags=[RunTag(k, v) for k, v in tags.items()])
        for f in ("summary.json", "predictions.csv"):
            if (run_dir / f).exists():
                client.log_artifact(run.info.run_id, str(run_dir / f))
        client.set_terminated(run.info.run_id, end_time=start)
        return run.info.run_id
    except Exception as e:                              # tracking must never break an eval run
        print(f"(MLflow logging skipped for {run_dir.name}: {e})")
        return None


def backfill() -> None:
    dirs = sorted(d for d in RUNS.iterdir() if (d / "summary.json").exists())
    logged = [d for d in dirs if log_run(d, quiet=True)]
    print(f"{len(logged)} new runs logged, {len(dirs) - len(logged)} already there or skipped → {TRACKING_URI}")


if __name__ == "__main__":
    if sys.argv[1:] == ["backfill"]:
        backfill()
    else:
        sys.exit(__doc__)
