#!/usr/bin/env bash

cd "$(dirname "$0")"

set -e

docker compose up -d --build --remove-orphans

echo 'to stop the container, run: docker compose down'
