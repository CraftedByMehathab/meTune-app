# meTune deployment runbook

One VM, one Docker Compose stack:

```
Internet ──443──▶ caddy ──/api/*──▶ api (FastAPI + worker + pipeline, ffmpeg) ──▶ db (Postgres 17)
                     └────/*─────▶ client (nginx, static React SPA)
                                         api ──HTTPS──▶ Gemini API (only external data flow)
```

| Service | Image | Memory limit | Persistent volume |
|---|---|---|---|
| caddy | `caddy:2.8` | 128 MB | `caddy_data` (TLS certs) |
| client | `metune-client` | 64 MB | — |
| api | `metune-api` | 2 GB | `uploads` (audio, confidential) |
| db | `postgres:17-alpine` | 512 MB | `pg_data` |

## 1. Provision the VM
| Item | Value |
|---|---|
| Size | **4 vCPU / 8 GB RAM / 30+ GB disk** (trial host: DigitalOcean v5 shared `s5-4vcpu-8gb-30gb`, ATL1) |
| OS | Ubuntu 24.04 LTS |
| Access | Your SSH public key at creation (no password) |
| DNS | `A` record `metune.<your-domain>` → VM IP (before first start, so Caddy can get a certificate) |

Why this size: tone runs 5 speech-emotion models on the CPU (~3.5–4.5 s per call) and the API needs up to ~3.5 GB with several files in flight (compose `mem_limit: 5g`). On a 4 GB VM, set `mem_limit: 3g` and `WORKER_CONCURRENCY=2`.

## 2. Bootstrap (once, as root)
```bash
scp deploy/scripts/bootstrap-vm.sh root@<vm>:/root/
ssh root@<vm> 'bash /root/bootstrap-vm.sh deploy'
```
This sets up Docker + compose, a `deploy` user (key-only SSH, root login disabled), ufw (22/80/443 only), 2 GB swap, unattended security upgrades, fail2ban and `/opt/metune`.

## 3. Configure (as `deploy`)
```bash
scp deploy/compose.yaml deploy/Caddyfile deploy/.env.example deploy/scripts/*.sh deploy@<vm>:/opt/metune/
ssh deploy@<vm>
cd /opt/metune && mv .env.example .env && chmod 600 .env && chmod +x *.sh
openssl rand -hex 24   # → PG_PASSWORD
openssl rand -hex 32   # → SESSION_SECRET
```
Generate `ADMIN_PASSWORD_HASH` on your laptop with `make hash-password`, and paste it into `.env` in **single quotes**. Then set `DOMAIN`, `REGISTRY=ghcr.io/<github-user>`, `ADMIN_USER` and `GEMINI_API_KEY` (paid tier).

**Speech-emotion model** (`tone_source: ser`, DR-020): the 5 ONNX files of the E20 bag (8-bit, 5 × 122 MB) are not in git or the image; compose mounts `/opt/metune/models` read-only. Copy them once (and again whenever they are re-exported):
```bash
ssh deploy@<vm> mkdir -p /opt/metune/models
scp .data/models/ser_e20_bag?.int8.onnx deploy@<vm>:/opt/metune/models/   # from the app repo, after eval/export_ser_onnx.py
```
Without it every file fails with `SER model not found` — by design, so tone never silently falls back to the LLM's always-neutral answer.

If the GHCR packages are private, log in once: `echo <PAT with read:packages> | docker login ghcr.io -u <github-user> --password-stdin`

## 4. First deploy
Push a tag (`git tag v0.1.0 && git push --tags`) or run the **deploy** workflow manually. GitHub Actions builds both images, pushes them to GHCR as `:<git-sha>`, then SSHes in and runs `/opt/metune/deploy.sh <sha>`.

Repository secrets: `VM_HOST`, `VM_USER=deploy`, `VM_SSH_KEY` (a dedicated deploy key, added to `~deploy/.ssh/authorized_keys`).

Manual alternative: `./deploy.sh <sha>` on the VM. Without GHCR (first trial deploy): clone the repo to `/opt/metune/src`, `docker build` both images there as `metune/metune-{api,client}:<sha>`, set `REGISTRY=metune` and `IMAGE_TAG=<sha>` in `.env`, then `docker compose --env-file .env up -d` (needs ~3 GB free RAM for the build).

### Verify
```bash
curl -fsS https://metune.<domain>/api/healthz        # {"ok":true,"sha":"<sha>"}
docker compose --env-file .env ps                    # all healthy / running
```
Then in the browser: log in → upload a ZIP containing one corrupt file → batch completes, corrupt file shows `error` → download the CSV. This is the MVP Scope success check.

## 5. Operate
| Task | Command (in `/opt/metune`) |
|---|---|
| Logs | `docker compose --env-file .env logs -f api` |
| Update | Push a new tag (CI deploys), or `./deploy.sh <sha>` |
| **Rollback** | `./deploy.sh <previous-sha>` (the old tag is still in GHCR) |
| Restart api | `docker compose --env-file .env restart api` (queued/running files resume) |
| Change worker concurrency | edit `WORKER_CONCURRENCY` in `.env`, then `docker compose --env-file .env up -d api` |
| psql | `docker compose --env-file .env exec db psql -U metune` |
| Disk | `df -h /` and `docker system df` |

**Monitoring:** add an external uptime check (UptimeRobot, Better Stack) on `https://<domain>/api/healthz` every minute. Caddy and api log JSON to stdout, so `docker compose logs` covers both.

## 6. Backups
```bash
crontab -e
15 3 * * * /opt/metune/backup.sh >> /opt/metune/backups/backup.log 2>&1
```
`backup.sh` writes a Postgres custom-format dump plus a tarball of uploads and keeps 7 days.

- **Off-box:** copy to object storage with client-side encryption only (e.g. `restic` to S3/B2). Never copy to unencrypted buckets, because call audio is confidential.
- **Restore:** `./restore.sh backups/db-<stamp>.dump [backups/uploads-<stamp>.tgz]`.
- **Test it:** run one restore drill before the evaluation.

## 7. Security checklist
- [ ] SSH key-only login, root disabled (bootstrap)
- [ ] ufw: only 22/80/443 open; Postgres is **not** published to the host
- [ ] `.env` is `chmod 600` and never committed
- [ ] `SESSION_SECRET` is random (the api refuses to start with the dev default when `COOKIE_SECURE=true`)
- [ ] HTTPS only (Caddy auto TLS + HSTS); the session cookie is `Secure`, `HttpOnly`, `SameSite=Lax`
- [ ] Login is rate-limited (5/min per IP)
- [ ] Uploads and downloads are behind auth; audio is never served statically
- [ ] Gemini on the paid tier (prompts and audio not used for training); vendor table in the memo

## 8. Cost (trial)
| Item | Monthly |
|---|---|
| VM 4 vCPU / 8 GB (DigitalOcean v5 shared) | ~$73 ($0.100/h, billed per second, no monthly cap) |
| Domain | ~$1 |
| Gemini (measured ~$0.0024/audio-min; tone models run locally, no API cost) | 1,000 audio-min → ~$2.40 |
| GHCR, GitHub Actions | free tier |
