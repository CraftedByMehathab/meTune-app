#!/usr/bin/env bash
# Pull images for a tag and restart the stack. Run on the VM in /opt/metune:  ./deploy.sh <tag>
# Rollback = run again with the previous tag.
set -euo pipefail
cd "$(dirname "$0")"

TAG="${1:?usage: deploy.sh <image-tag>}"
# pull first: a failed pull (e.g. private GHCR package) must not leave .env pointing at missing images
IMAGE_TAG="$TAG" docker compose --env-file .env pull api client
sed -i.bak "s/^IMAGE_TAG=.*/IMAGE_TAG=$TAG/" .env
docker compose --env-file .env up -d --remove-orphans
docker image prune -f >/dev/null

for i in $(seq 1 30); do
  if docker compose exec -T api python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/healthz')" 2>/dev/null; then
    echo "✓ deployed $TAG"
    exit 0
  fi
  sleep 2
done
echo "✗ api not healthy after deploy; check: docker compose logs api" >&2
exit 1
