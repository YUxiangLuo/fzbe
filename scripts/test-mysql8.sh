#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bun="$project_root/scripts/run-pinned-bun.sh"

image="${MYSQL8_DOCKER_IMAGE:-mysql:8.0}"
host_port="${MYSQL8_DOCKER_PORT:-}"
root_password="${MYSQL8_DOCKER_PASSWORD:-fz_mysql8_test_pw}"
container_name="fz-mysql8-compat-$$-$(date +%s)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to run the MySQL 8 compatibility gate." >&2
  exit 1
fi

cleanup() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

port_mapping="127.0.0.1::3306"
if [[ -n "$host_port" ]]; then
  port_mapping="127.0.0.1:${host_port}:3306"
fi

echo "Starting ephemeral MySQL 8 container (${image})..."
# Keep MySQL 8's default server collation (utf8mb4_0900_ai_ci) so bootstrap and
# migration scripts are forced to set utf8mb4_unicode_ci explicitly.
docker run -d --rm \
  --name "$container_name" \
  -e "MYSQL_ROOT_PASSWORD=${root_password}" \
  -p "$port_mapping" \
  "$image" \
  --max-connections=200 >/dev/null

mapped_port="$(docker port "$container_name" 3306/tcp | head -n 1 | awk -F: '{print $NF}')"
if [[ -z "$mapped_port" ]]; then
  echo "Failed to resolve mapped port for ${container_name}." >&2
  exit 1
fi

echo "Waiting for MySQL 8 on 127.0.0.1:${mapped_port}..."
ready=0
for _ in $(seq 1 90); do
  if docker exec -e "MYSQL_PWD=${root_password}" "$container_name" mysqladmin ping -h 127.0.0.1 -uroot --silent >/dev/null 2>&1; then
    if docker exec -e "MYSQL_PWD=${root_password}" "$container_name" mysql -h 127.0.0.1 -uroot -e "SELECT 1" >/dev/null 2>&1; then
      ready=1
      break
    fi
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "MySQL 8 container failed to become ready within timeout." >&2
  docker logs "$container_name" >&2 || true
  exit 1
fi

version="$(docker exec -e "MYSQL_PWD=${root_password}" "$container_name" mysql -N -s -h 127.0.0.1 -uroot -e "SELECT VERSION()")"
echo "MySQL 8 ready: version=${version} port=${mapped_port}"

run_suite() {
  local script_name="$1"
  local database_name="$2"
  echo ""
  echo "=== Running ${script_name} against ${image} (${database_name}) ==="
  (
    cd "$project_root"
    env \
      E2E_ALLOW_DB_RESET=1 \
      DB_HOST=127.0.0.1 \
      DB_PORT="$mapped_port" \
      DB_USER=root \
      DB_PASSWORD="$root_password" \
      DB_DATABASE="$database_name" \
      DB_TIMEZONE=Z \
      DB_SSL_MODE=disabled \
      JWT_SECRET="${JWT_SECRET:-fz_mysql8_test_jwt_secret_at_least_32_bytes_long}" \
      "$bun" run "$script_name"
  )
}

run_suite "test:system:db-compat" "fz_test_mysql8_compat"
run_suite "test:system:user-phone-canonical" "fz_test_mysql8_phone"
run_suite "test:system:concurrency" "fz_test_mysql8_concurrency"

echo ""
echo "All MySQL 8 compatibility system suites passed on ${version}."
