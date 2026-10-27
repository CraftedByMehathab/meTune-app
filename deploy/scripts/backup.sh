#!/usr/bin/env bash
# Nightly backup: Postgres dump + uploaded audio, 7-day retention. Cron (as deploy user):
#   15 3 * * * /opt/metune/backup.sh >> /opt/metune/backups/backup.log 2>&1
# Call audio is confidential: copy off-box only to encrypted storage (see DEPLOY.md §6).
set -euo pipefail
cd "$(dirname "$0")"

STAMP=$(date +%Y%m%d-%H%M%S)
OUT=backups
mkdir -p "$OUT"

docker compose --env-file .env exec -T db pg_dump -U metune -d metune --format=custom > "$OUT/db-$STAMP.dump"
docker run --rm -v metune_uploads:/data:ro -v "$PWD/$OUT":/backup alpine \
  tar czf "/backup/uploads-$STAMP.tgz" -C /data .

find "$OUT" -name 'db-*.dump' -mtime +7 -delete
find "$OUT" -name 'uploads-*.tgz' -mtime +7 -delete
echo "$(date -Is) backup ok: db-$STAMP.dump uploads-$STAMP.tgz"
