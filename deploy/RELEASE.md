# Releasing meTune

A release is a git tag. Pushing a `v*` tag builds both images on GitHub Actions, pushes them to GHCR, and deploys
them to the VM. First-time VM setup is in [DEPLOY.md](DEPLOY.md); this page is for every release after that.

```
git push main ──▶ ci (tests, typecheck)                         nothing deployed
git push tag v* ──▶ deploy: build ──▶ ghcr.io/craftedbymehathab/metune-{api,client}:<sha>
                            deploy ──▶ ssh deploy@VM: /opt/metune/deploy.sh <sha>
                                        pull → write IMAGE_TAG to .env → up -d → wait for /api/healthz
```

Images are tagged with the **short commit SHA**, not the version: `v0.2.0` → `7254332`. The SHA is what runs on
the VM (`IMAGE_TAG` in `/opt/metune/.env`) and what the dashboard shows.

## 1. Before tagging

Changes reach `main` only through a pull request (CI runs on the PR and again on `main`); tag after the merge.

- [ ] **CI is green on `main`** for the commit you will tag: `gh run list --branch main --limit 1`
- [ ] **New or re-exported model files are on the VM.** Models are not in git or the image; the API mounts
      `/opt/metune/models` read-only. A missing file fails every call ("model not found"), by design. Copy it
      **before** the release (the running version ignores files it does not use), then check the checksum:
      ```bash
      scp .data/models/<file>.onnx deploy@129.212.176.75:/opt/metune/models/
      shasum -a 256 .data/models/<file>.onnx
      ssh deploy@129.212.176.75 sha256sum /opt/metune/models/<file>.onnx
      ```
      Currently required: `ser_e20_bag1–5.int8.onnx` (tone) and `ast_audioset.int8.onnx` (noise).
- [ ] **`config.yaml` changes are intended.** It is baked into the image: model ids, vote count, prices and
      thresholds ship with the release. Its hash is stored on every batch.
- [ ] **Cost still fits** ≤ $0.003 per audio minute (the last eval run's `cost_per_min`).
- [ ] **Changed `compose.yaml`, `Caddyfile` or `deploy.sh`?** These live on the VM, not in the image. Copy them
      first: `scp deploy/compose.yaml deploy/scripts/deploy.sh deploy@129.212.176.75:/opt/metune/`
- [ ] The experiment behind the change is in the vault's Experiment Log.

## 2. Release

Pick the version: **minor** (`v0.3.0`) for a model, prompt, config or behaviour change; **patch** (`v0.2.1`)
for a fix that does not change results.

```bash
git switch main && git pull
git tag -a v0.3.0 -m "v0.3.0: <one line: what changes for users>"
git push origin v0.3.0
gh run watch "$(gh run list --workflow deploy.yml --limit 1 --json databaseId -q '.[0].databaseId')"
```

Build ≈ 3 min, deploy ≈ 1 min. The deploy job fails if the API is not healthy within 60 s.

## 3. Verify

```bash
ssh deploy@129.212.176.75 'cd /opt/metune && grep IMAGE_TAG .env && \
  docker compose --env-file .env ps --format "{{.Service}} {{.Image}} {{.Status}}"'
```
- `api` and `client` run the new SHA; `api` is `healthy`.
- Open the app: the dashboard shows the same SHA.
- **Smoke test** one known call inside the live container (~$0.003):
  ```bash
  ssh deploy@129.212.176.75 mkdir -p /tmp/smoke
  scp ../meTune/data/trial/call_003.ogg deploy@129.212.176.75:/tmp/smoke/
  ssh deploy@129.212.176.75 'cd /opt/metune && docker compose cp /tmp/smoke api:/tmp/smoke && \
    docker compose exec -T api metune run /tmp/smoke --out /tmp/smoke/out.csv && \
    docker compose exec -T api cat /tmp/smoke/out.csv; docker compose exec -T api rm -rf /tmp/smoke; rm -rf /tmp/smoke'
  ```
  Expect a full result row (call_003: static noise, medium). An error names the missing model or key.

## 4. Roll back

Every release's images stay in GHCR, so rolling back is deploying an older SHA:
```bash
ssh deploy@129.212.176.75 '/opt/metune/deploy.sh <previous-sha>'
```
Find the previous SHA with `git rev-list -n1 --abbrev-commit v0.2.0` (or in the earlier run's log).
Roll back model files too if the release changed them (keep the old file under another name until the new
release is verified).

Before v0.2.0 the VM built its own images (`REGISTRY=metune`, last one `4b2958e`, still on the VM). To go back
to it: `cp .env.bak-4b2958e .env && docker compose --env-file .env up -d`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `deploy` job: empty `INPUT_HOST`, ssh fails | Repo secrets `VM_HOST`, `VM_USER`, `VM_SSH_KEY` missing — `gh secret list` |
| `deploy.sh`: pull `denied` / `unauthorized` | GHCR package private. Packages → `metune-api` / `metune-client` → Change visibility → Public. A failed pull leaves the running stack and `.env` untouched; re-run the job |
| `✗ api not healthy after deploy` | `ssh deploy@VM 'cd /opt/metune && docker compose logs --tail 100 api'`. Often a missing model file. Roll back (above) |
| Every call fails "model not found" | A file named in `config.yaml` is not in `/opt/metune/models` — copy it, no redeploy needed |
| "cannot be rerun" on an old run | Start a fresh run: Actions → deploy → Run workflow (builds the selected branch's HEAD), or push a new tag |
| Build fails on the image name | GHCR names must be lowercase; the workflow lower-cases the owner — keep that step |

## Access

| What | Where |
|---|---|
| Deploy key | `github-actions-deploy` in `~deploy/.ssh/authorized_keys` on the VM; private half only in the `VM_SSH_KEY` secret. Rotate: add a new key, update the secret, remove the old line |
| Registry | `ghcr.io/craftedbymehathab`, public images (no secrets inside; `.env` and models stay on the VM) |
| VM | `deploy@129.212.176.75`, app in `/opt/metune` |

## Release history

| Version | SHA | Date | What changed |
|---|---|---|---|
| v0.6.0 | `cb1ca29` | 2026-10-28 | Tone from Gemini (E35: MELD macro F1 0.19 → 0.38); confidence = per-tone precision on MELD dev (E36: ECE 0.044); speech-emotion bag kept for `tone_source: ser` |
| v0.5.0 | `787136a` | 2026-10-28 | Per-tone bias on the SER bag (E33): test macro F1 0.50 → 0.61, distressed recall 0.15 → 0.44; confidence calibrator refit out-of-fold (ECE 0.119) |
| v0.4.0 | `7d5ac03` | 2026-10-28 | Calibrated confidence = P(tone right) (E32: ECE 0.28 → 0.09, wrong at ≥ 0.9: 23 → 2 of 113); MLflow tracking (dev only) |
| v0.3.0 | `ef76f18` | 2026-10-28 | Proxy overlap labels fixed (E31); pyannote overlap detector (vendored) ORed with Gemini: recall 0.40 → 0.68 at precision 0.90 |
| v0.2.0 | `7254332` | 2026-10-28 | Gemini 3.8 Flash, 1 vote; local AST noise detector (E25, E28: test_big noise acc 0.74 → 0.95); first CI/CD deploy via GHCR |
| — | `4b2958e` | 2026-10-28 | E21 customer-only tone; built on the VM (pre-CI/CD) |
