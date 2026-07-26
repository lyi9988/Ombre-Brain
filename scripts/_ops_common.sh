#!/usr/bin/env bash
set -euo pipefail

ombre_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

ombre_compose_file() {
  if [[ -n "${COMPOSE_FILE:-}" ]]; then
    printf '%s\n' "${COMPOSE_FILE}"
    return
  fi
  for candidate in compose.local.yml compose.hk.yml docker-compose.user.yml docker-compose.yml; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  echo "No compose file found. Set COMPOSE_FILE=/path/to/compose.yml" >&2
  exit 1
}

ombre_compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    echo "Docker Compose not found. Install docker compose first." >&2
    exit 1
  fi
}

ombre_default_health_url() {
  local compose_file="${1}"
  case "${compose_file}" in
    *docker-compose.user.yml) printf '%s\n' "http://127.0.0.1:8000/health" ;;
    *) printf '%s\n' "http://127.0.0.1:18001/health" ;;
  esac
}

ombre_compose_service_exists() {
  local compose_file="${1}"
  local service="${2}"
  ombre_compose -f "${compose_file}" config --services 2>/dev/null | grep -Fxq "${service}"
}

ombre_compose_service_health_url() {
  local compose_file="${1}"
  local service="${2}"
  local internal_port="${3}"
  local fallback_url="${4}"
  local mapped host_port

  mapped="$(ombre_compose -f "${compose_file}" port "${service}" "${internal_port}" 2>/dev/null | tail -n 1 || true)"
  host_port="$(printf '%s\n' "${mapped}" | sed -nE 's/.*:([0-9]+)$/\1/p' | tail -n 1)"
  if [[ -n "${host_port}" ]]; then
    printf 'http://127.0.0.1:%s/health\n' "${host_port}"
  else
    printf '%s\n' "${fallback_url}"
  fi
}

ombre_wait_for_health() {
  local url="${1}"
  local tries="${2:-30}"
  local delay="${3:-2}"

  if ! command -v curl >/dev/null 2>&1; then
    echo "curl not found; skip health check: ${url}"
    return 0
  fi

  echo "Health check: ${url}"
  for ((i = 1; i <= tries; i++)); do
    if curl -fsS "${url}" >/dev/null; then
      echo "Health check OK"
      return 0
    fi
    sleep "${delay}"
  done

  echo "Health check failed after ${tries} tries: ${url}" >&2
  return 1
}

ombre_has_tracked_changes() {
  ! git diff --quiet || ! git diff --cached --quiet
}

ombre_update_git_checkout() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 0
  fi

  local remote="${OMBRE_REMOTE:-origin}"
  local current_branch target_branch target_ref target_sha head_sha backup_branch stamp
  current_branch="$(git branch --show-current 2>/dev/null || true)"
  if [[ -n "${OMBRE_BRANCH:-}" ]]; then
    target_branch="${OMBRE_BRANCH}"
  else
    target_branch="${current_branch:-main}"
    if [[ "${target_branch}" == archive/* ]]; then
      echo "Current branch ${target_branch} is an archive branch; update target defaults to main."
      target_branch="main"
    fi
  fi

  echo "Update code from ${remote}/${target_branch}..."
  git fetch "${remote}" "${target_branch}"
  target_ref="FETCH_HEAD"
  target_sha="$(git rev-parse "${target_ref}")"

  if [[ -n "${current_branch}" && "${current_branch}" != "${target_branch}" ]]; then
    if ombre_has_tracked_changes; then
      echo "Tracked files have local changes; cannot switch branch safely." >&2
      echo "Commit/stash them first, or set OMBRE_BRANCH to the current branch." >&2
      return 1
    fi
    if git show-ref --verify --quiet "refs/heads/${target_branch}"; then
      git switch "${target_branch}"
    else
      git switch -c "${target_branch}" "${target_ref}"
    fi
    current_branch="${target_branch}"
  elif [[ -z "${current_branch}" ]]; then
    if ombre_has_tracked_changes; then
      echo "Detached checkout has tracked local changes; cannot switch branch safely." >&2
      return 1
    fi
    if git show-ref --verify --quiet "refs/heads/${target_branch}"; then
      git switch "${target_branch}"
    else
      git switch -c "${target_branch}" "${target_ref}"
    fi
    current_branch="${target_branch}"
  fi

  head_sha="$(git rev-parse HEAD)"
  if [[ "${head_sha}" == "${target_sha}" ]]; then
    echo "Already up to date."
  elif git merge-base --is-ancestor HEAD "${target_ref}"; then
    git merge --ff-only "${target_ref}"
  elif git merge-base --is-ancestor "${target_ref}" HEAD; then
    echo "Local ${current_branch} is ahead of ${remote}/${target_branch}; skip reset." >&2
    echo "Push or archive local commits before updating this deployment." >&2
    return 1
  else
    if [[ "${OMBRE_ALLOW_DIVERGED_RESET:-1}" != "1" ]]; then
      echo "Local ${current_branch} diverged from ${remote}/${target_branch}." >&2
      echo "Set OMBRE_ALLOW_DIVERGED_RESET=1 to backup and reset this deployment checkout." >&2
      return 1
    fi
    if ombre_has_tracked_changes; then
      echo "Local ${current_branch} diverged and tracked files have local changes." >&2
      echo "Commit/stash tracked changes before updating; untracked .env/state files are fine." >&2
      return 1
    fi
    stamp="$(date +%Y%m%d-%H%M%S)"
    backup_branch="archive/local-${current_branch:-detached}-before-reset-${stamp}"
    git branch "${backup_branch}" HEAD
    echo "Local branch diverged; saved old HEAD as ${backup_branch}."
    git reset --hard "${target_ref}"
  fi

  git branch --set-upstream-to="${remote}/${target_branch}" "${current_branch}" >/dev/null 2>&1 || true
}
