#!/usr/bin/env bash
# Restore a backup made by backup.sh:  ./restore.sh backups/db-<stamp>.dump [backups/uploads-<stamp>.tgz]
# Overwrites current data. Stops the api while restoring.
set -euo pipefail
cd "$(dirname "$0")"

DB_DUMP="${1:?usage: restore.sh <db.dump> [uploads.tgz]}"
UPLOADS="${2:-}"

read -r -p "This overwrites the current database${UPLOADS:+ and uploads}. Type 'restore' to continue: " ok
[ "$ok" = "restore" ] || exit 1

docker compose --env-file .env stop api
docker compose --env-file .env exec -T db pg_restore -U metune -d metune --clean --if-exists < "$DB_DUMP"
if [ -n "$UPLOADS" ]; then
  docker run --rm -v metune_uploads:/data -v "$PWD/$(dirname "$UPLOADS")":/backup alpine \
    sh -c "rm -rf /data/* && tar xzf /backup/$(basename "$UPLOADS") -C /data"
fi
docker compose --env-file .env start api
echo "✓ restored"
