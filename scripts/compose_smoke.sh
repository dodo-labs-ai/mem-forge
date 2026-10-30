#!/usr/bin/env bash
# Fresh-install smoke for the Docker Compose stack.
#
# Builds the current checkout and walks the newcomer path of
# docs/quickstart.md against it: the stack starts, both ports are published on
# loopback only, the API answers and the admin UI is served, the plugins'
# packaged MCP proxy lists and uses the memory tools, models are configured
# through Settings, a local source syncs into memories that search finds, and
# all of it survives a restart. Run it before a release and whenever the
# Dockerfiles, compose.yaml, migrations, startup or configuration defaults
# change, then paste the table it prints into the pull request or the release
# notes.
#
# The stack runs as a Compose project of its own (memforge-smoke-<random>) with
# its own data volume, on free loopback ports, and starts from
# .env.example as a newcomer's .env does; a .env in the checkout is not read.
# Your own `memforge` project, its volume and its ports are never touched.
# Everything the smoke creates (containers, volume, network, the images it
# built and its temporary folder) is removed when it ends, also when a step
# fails or it is interrupted. The shared build cache is kept.
#
# Usage:
#   scripts/compose_smoke.sh [options]
#
# Options:
#   --ollama                A kind without --KIND-provider uses Ollama on this
#                           machine.
#   --ollama-url URL        Ollama as the API container reaches it
#                           (default http://host.docker.internal:11434).
#   --KIND-provider ID      The provider of the generation or the embedding
#                           model, as Settings names it: openrouter, openai,
#                           anthropic (generation only), ollama,
#                           openai_compatible or litellm.
#   --KIND-model ID         The model (defaults below).
#   --KIND-endpoint URL     The provider's base URL; openai_compatible needs one,
#                           ollama defaults to --ollama-url.
#   --KIND-key-env NAME     The environment variable holding the provider's API
#                           key (default <PROVIDER>_API_KEY, e.g.
#                           OPENAI_COMPATIBLE_API_KEY; a local Ollama is sent
#                           a key only when one is named).
#   --no-models             Skip the model steps even when a provider is available.
#   --sync-timeout SECONDS  How long the source sync may take (default 900).
#   --report FILE           Also write the Markdown report to FILE.
#   --quality               Also sync the synthetic documents of
#                           evals/quality/quick, replace the ones it revises with
#                           their second version and sync again, replay the
#                           source-type snapshots of evals/quality/quick/snapshots
#                           through one Source per type, revision 1 then
#                           revision 2, then run the quick offline quality
#                           evaluation of each case kind; the summaries follow
#                           the table.
#   --quality-outputs DIR   With --quality, also save each case kind's case
#                           outputs (the pinned input, rubric and model output
#                           of every case) as DIR/<kind>.json.
#   -h, --help              Show this help.
#
# Models: KIND is generation or embedding, and each kind is configured
# through Settings with its own provider. A kind without --KIND-provider uses
# a local Ollama that has the model pulled with --ollama, and otherwise
# OpenRouter when OPENROUTER_API_KEY is set. Without --KIND-model a kind
# takes its provider's default: Claude Haiku 5.5 for generation on openrouter
# and anthropic, text-embedding-3-small for embedding on openrouter and
# openai, qwen3:0.6b and nomic-embed-text on ollama; any other provider needs
# --KIND-model. Keys are read from the environment or from .env.test at the
# repository root (see scripts/smoke_env.py), sent to the API in request
# bodies and never printed. With no provider for either kind the model, sync
# and retrieval steps are reported as skipped, and the restart check looks for
# the source added through the API, directly and through the MCP proxy.
#
# For example, generation through an OpenAI-compatible endpoint and embedding
# through OpenRouter:
#   scripts/compose_smoke.sh --generation-provider openai_compatible \
#     --generation-endpoint https://llm.example.com/v1 --generation-model my-model \
#     --embedding-provider openrouter
#
# Restricted networks: exported HTTPS_PROXY / HTTP_PROXY are passed to the
# image build and to the API container, with NO_PROXY covering the stack and
# host.docker.internal, as docs/quickstart.md describes. The proxy address
# must be reachable from containers, e.g. http://host.docker.internal:7890.
# Exported MEMFORGE_DOCKERHUB_PREFIX and the other build mirror variables are
# used as in the quickstart. Requests from this machine to the smoke stack
# never go through the proxy.
#
# Requires Docker with Compose 2.24 or newer and python3 3.9 or newer.
# Exits 0 when no step failed, 1 when one did and 130 when interrupted.

set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HELPER="$REPO_ROOT/scripts/compose_smoke.py"
PLUGIN_DIR="$REPO_ROOT/integrations/claude-code/memforge-memory"
PYTHON=${PYTHON:-python3}
DOCS_IN_CONTAINER=/smoke-docs
SNAPSHOTS_DIR="$REPO_ROOT/evals/quality/quick/snapshots"
SNAPSHOTS_IN_CONTAINER=/evaluation-snapshots
SKIPPED=3

no_models=0
run_quality=0
quality_summary=
model_usage=
model_args=()
sync_timeout=900
report_file=
outputs_dir=

usage() {
  sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case $1 in
    --ollama) model_args+=("$1") ;;
    --ollama-url | --generation-provider | --generation-model | --generation-endpoint | --generation-key-env | \
    --embedding-provider | --embedding-model | --embedding-endpoint | --embedding-key-env)
      model_args+=("$1" "${2:?$1 needs a value}"); shift ;;
    --no-models) no_models=1 ;;
    --quality) run_quality=1 ;;
    --sync-timeout) sync_timeout=${2:?--sync-timeout needs seconds}; shift ;;
    --report) report_file=${2:?--report needs a file}; shift ;;
    --quality-outputs) outputs_dir=${2:?--quality-outputs needs a directory}; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

# The keys of .env.test, for each variable the environment leaves unset.
smoke_env=$("$PYTHON" "$REPO_ROOT/scripts/smoke_env.py") || exit 2
eval "$smoke_env"

# Each kind's provider and model: models_label names both for the report,
# generation_label the generation model, and ollama_endpoints the Ollama
# servers the API container has to reach. An empty models_label means no model.
models_label= generation_label= ollama_endpoints=
if (( ! no_models )); then
  model_plan=$("$PYTHON" "$HELPER" model-plan ${model_args[@]+"${model_args[@]}"}) || exit 2
  eval "$model_plan"
fi

for tool in docker "$PYTHON"; do
  command -v "$tool" >/dev/null || { echo "$tool is required" >&2; exit 2; }
done

read -r api_port ui_port <<<"$("$PYTHON" "$HELPER" free-ports 2)"
project="memforge-smoke-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/memforge-smoke.XXXXXX")
state_file="$work_dir/state.json"
detail_file="$work_dir/detail.txt"
docs_dir="$work_dir/docs"
env_file="$work_dir/smoke.env"
override_file="$work_dir/compose.smoke.yaml"
api_url="http://127.0.0.1:$api_port"
ui_url="http://127.0.0.1:$ui_port"
https_proxy_value=${HTTPS_PROXY:-${https_proxy:-}}
http_proxy_value=${HTTP_PROXY:-${http_proxy:-}}
export MEMFORGE_API_HOST_PORT=$api_port MEMFORGE_ADMIN_UI_HOST_PORT=$ui_port

step_names=()
step_results=()
step_times=()
step_details=()
failed=0
interrupted=0
suspect_service=api
current_step=
run_started=$(date +%s)

log() { printf '\n== %s\n' "$*" >&2; }

compose() {
  docker compose \
    --project-name "$project" \
    --project-directory "$REPO_ROOT" \
    --env-file "$env_file" \
    -f "$REPO_ROOT/compose.yaml" \
    -f "$override_file" \
    "$@"
}

helper() {
  "$PYTHON" "$HELPER" "$@" --api "$api_url" --state "$state_file"
}

say() { printf '%s\n' "$*" >"$detail_file"; }

record() {
  step_names+=("$1")
  step_results+=("$2")
  step_times+=("$3")
  step_details+=("$4")
}

# run_step NAME FUNCTION: runs the step unless an earlier one failed. The
# function returns 0 for pass, $SKIPPED for skipped, anything else for fail,
# and leaves its one-line detail in $detail_file.
run_step() {
  local name=$1 started status result
  shift
  if (( failed || interrupted )); then
    record "$name" skipped 0 "not run: an earlier step failed"
    return
  fi
  log "$name"
  current_step=$name
  : >"$detail_file"
  started=$(date +%s)
  "$@"
  status=$?
  case $status in
    0) result=pass ;;
    "$SKIPPED") result=skipped ;;
    *) result=fail; failed=1 ;;
  esac
  current_step=
  record "$name" "$result" $(( $(date +%s) - started )) "$(tail -n 1 "$detail_file")"
  printf '%s: %s\n' "$result" "$(tail -n 1 "$detail_file")" >&2
}

# -- Setup ------------------------------------------------------------------------

write_inputs() {
  # A newcomer's .env is a copy of .env.example; only the host ports differ.
  sed -e "s/^MEMFORGE_API_HOST_PORT=.*/MEMFORGE_API_HOST_PORT=$api_port/" \
      -e "s/^MEMFORGE_ADMIN_UI_HOST_PORT=.*/MEMFORGE_ADMIN_UI_HOST_PORT=$ui_port/" \
      "$REPO_ROOT/.env.example" >"$env_file"

  mkdir -p "$docs_dir"
  cat >"$docs_dir/job-queue-storage.md" <<'EOF'
# Decision: the Kestrel job queue is stored in SQLite

The Kestrel job queue is stored in SQLite in WAL mode, not in Redis. The team
chose SQLite so that a self-hosted Kestrel install runs as a single container
with no separate queue service.

Requirement: every queue worker opens the database with a busy timeout of five
seconds, because WAL mode allows only one writer at a time.
EOF
  cat >"$docs_dir/release-tags.md" <<'EOF'
# Convention: Kestrel release tags

Kestrel releases use calendar versions such as 2026.10.1. A release tag is
always cut from the main branch, never from a feature branch, so that every
release contains all reviewed changes.
EOF
  if (( run_quality )); then
    cp "$REPO_ROOT"/evals/quality/quick/docs/*.md "$docs_dir"/
  fi
  chmod 755 "$docs_dir"
  chmod 644 "$docs_dir"/*.md

  # With --quality the API replays the source-type snapshots, mounted read-only;
  # only this Compose project turns evaluation snapshots on.
  {
    echo "services:"
    echo "  api:"
    echo "    env_file: !override"
    echo "      - path: \"$env_file\""
    if [ -n "$https_proxy_value$http_proxy_value" ] || (( run_quality )); then
      echo "    environment:"
      [ -n "$https_proxy_value" ] && echo "      HTTPS_PROXY: \${MEMFORGE_SMOKE_HTTPS_PROXY}"
      [ -n "$http_proxy_value" ] && echo "      HTTP_PROXY: \${MEMFORGE_SMOKE_HTTP_PROXY}"
      [ -n "$https_proxy_value$http_proxy_value" ] &&
        echo "      NO_PROXY: localhost,127.0.0.1,api,admin-ui,host.docker.internal"
      (( run_quality )) && echo "      MEMFORGE_EVALUATION_SNAPSHOT_ROOT: $SNAPSHOTS_IN_CONTAINER"
    fi
    echo "    volumes:"
    echo "      - type: bind"
    echo "        source: \"$docs_dir\""
    echo "        target: $DOCS_IN_CONTAINER"
    echo "        read_only: true"
    if (( run_quality )); then
      echo "      - type: bind"
      echo "        source: \"$SNAPSHOTS_DIR\""
      echo "        target: $SNAPSHOTS_IN_CONTAINER"
      echo "        read_only: true"
    fi
  } >"$override_file"
  export MEMFORGE_SMOKE_HTTPS_PROXY=$https_proxy_value MEMFORGE_SMOKE_HTTP_PROXY=$http_proxy_value
}

# -- Steps -------------------------------------------------------------------------

build_images() {
  suspect_service=
  local build_args=()
  [ -n "$https_proxy_value" ] && build_args+=(--build-arg "HTTPS_PROXY=$https_proxy_value")
  [ -n "$http_proxy_value" ] && build_args+=(--build-arg "HTTP_PROXY=$http_proxy_value")
  if ! compose build ${build_args[@]+"${build_args[@]}"} >&2; then
    say "docker compose build failed; its output is above"
    return 1
  fi
  say "built $(compose config --images | tr '\n' ' ')from $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'this checkout')"
}

start_stack() {
  suspect_service=api
  if ! compose up -d --no-build --wait --wait-timeout 300 >&2; then
    say "the stack did not become healthy within 300 s"
    return 1
  fi
  say "project $project healthy; API $api_url, admin UI $ui_url"
}

check_ports() {
  local containers published
  containers=$(compose ps -q)
  published=$(compose port api 8765)
  if [ "$published" != "127.0.0.1:$api_port" ]; then
    say "compose port api 8765 is '$published', not 127.0.0.1:$api_port"
    return 1
  fi
  # shellcheck disable=SC2086 # one container id per word
  docker inspect $containers | "$PYTHON" "$HELPER" bindings \
    --expect "api=$api_port" --expect "admin-ui=$ui_port" >"$detail_file"
}

probe_stack() {
  helper probe --ui "$ui_url" >"$detail_file"
}

mcp() {
  helper mcp --plugin "$PLUGIN_DIR" --home "$work_dir/agent-home" --phase "$1"
}

mcp_tools() {
  mcp tools >"$detail_file"
}

configure_models() {
  local endpoint
  if [ -z "$models_label" ]; then
    say "no model provider: pass --ollama, export OPENROUTER_API_KEY, or name each kind's --KIND-provider"
    return "$SKIPPED"
  fi
  for endpoint in $ollama_endpoints; do
    if ! compose exec -T api python -c '
import sys, urllib.request
urllib.request.urlopen(sys.argv[1] + "/api/version", timeout=10).read()
' "$endpoint" >&2; then
      say "the API container cannot reach Ollama at $endpoint;" \
        "start Ollama and let it accept connections from containers"
      return 1
    fi
  done
  helper models ${model_args[@]+"${model_args[@]}"} >"$detail_file"
}

add_source() {
  helper source --root "$DOCS_IN_CONTAINER" >"$detail_file"
}

sync_source() {
  local requested
  if [ -z "$models_label" ]; then
    say "needs a generation model"
    return "$SKIPPED"
  fi
  requested=$(helper sync) || { say "$requested"; return 1; }
  # The local daemon collects the folder: the CLI in the API image runs one
  # pass of it against the API, reading the folder mounted into the container.
  if ! compose exec -T -e MEMFORGE_API_URL=http://127.0.0.1:8765 api \
      memforge adapter daemon once >&2; then
    say "the local daemon pass failed; its report is above"
    return 1
  fi
  helper wait-sync --timeout "$sync_timeout" "$@" >"$detail_file" || return 1
  say "$requested, collected by one daemon pass; $(tail -n 1 "$detail_file")"
}

# The quality cohort's revision pairs: replace each revised document with its
# second version and sync again, so each is an applied update with recorded
# Support work for the Support cases to pin.
revise_source() {
  if [ -z "$models_label" ]; then
    say "needs a synced source"
    return "$SKIPPED"
  fi
  cp "$REPO_ROOT"/evals/quality/quick/revisions/*.md "$docs_dir"/
  chmod 644 "$docs_dir"/*.md
  sync_source --new-run || return 1
  say "$(ls "$REPO_ROOT"/evals/quality/quick/revisions/*.md | wc -l | tr -d ' ') documents revised; $(cat "$detail_file")"
}

# The source-type snapshots: one Source per snapshot set, each replaying its
# revision 1 and then its revision 2, so every type's update has recorded
# Support work for the Support cases to pin. A document Source syncs on the
# server; a conversation Source is collected by the device executor, which one
# daemon pass runs with the snapshot provider.
add_snapshot_sources() {
  if [ -z "$models_label" ]; then
    say "needs a generation model"
    return "$SKIPPED"
  fi
  helper snapshot-sources --snapshots "$SNAPSHOTS_DIR" >"$detail_file"
}

sync_snapshot_sources() {
  local revision=$1 requested
  if [ -z "$models_label" ]; then
    say "needs a generation model"
    return "$SKIPPED"
  fi
  requested=$(helper snapshot-sync --snapshots "$SNAPSHOTS_DIR" --revision "$revision") ||
    { say "$requested"; return 1; }
  if ! compose exec -T -e MEMFORGE_API_URL=http://127.0.0.1:8765 api \
      memforge adapter daemon once >&2; then
    say "the local daemon pass failed; its report is above"
    return 1
  fi
  helper snapshot-wait --timeout "$sync_timeout" >"$detail_file" || return 1
  say "$requested; $(tail -n 1 "$detail_file")"
}

retrieve_memories() {
  local searched
  if [ -z "$models_label" ]; then
    say "needs a synced source"
    return "$SKIPPED"
  fi
  searched=$(helper search) || { say "$searched"; return 1; }
  mcp memories >"$detail_file" || return 1
  say "API: $searched; MCP: $(tail -n 1 "$detail_file")"
}

# The quick quality evaluation runs through the CLI in the API image: for each
# case kind it pins the cohort's cases, then submits the run and waits for the
# worker's verdict. The step passes only when every kind's verdict is pass.
quality_checks() {
  local cohort="$REPO_ROOT/evals/quality/quick/cohort.json" kind command seeded run_spec summary status
  local verdicts=() failed=0
  if [ -z "$models_label" ]; then
    say "needs a synced source"
    return "$SKIPPED"
  fi
  quality_summary=
  for kind in derivation support reconciliation; do
    command=seed-$kind-cases
    run_spec="$work_dir/quality-run-$kind.json"
    if ! seeded=$(helper quality-seed-spec --cohort "$cohort" --kind "$kind" |
        compose exec -T -e MEMFORGE_API_URL=http://127.0.0.1:8765 api \
          memforge eval "$command" --spec /dev/stdin); then
      say "pinning the $kind cases failed: $(echo "$seeded" | tr '\n' ' ')"
      return 1
    fi
    if ! printf '%s' "$seeded" | helper quality-run-spec --cohort "$cohort" >"$run_spec"; then
      say "$kind: $(cat "$run_spec")"
      return 1
    fi
    summary=$(compose exec -T -e MEMFORGE_API_URL=http://127.0.0.1:8765 api \
      memforge eval offline-run --spec /dev/stdin --timeout "$sync_timeout" --poll 5 <"$run_spec")
    status=$?
    [ -n "$quality_summary" ] || quality_summary="- Models: $models_label"$'\n\n'
    quality_summary+="### $kind cases"$'\n\n'"$summary"$'\n\n'
    [ -n "$outputs_dir" ] && save_quality_outputs "$kind" "$summary"
    case $status in
      0) verdicts+=("$kind pass") ;;
      2) verdicts+=("$kind timed out"); failed=1 ;;
      *) verdicts+=("$kind $(echo "$summary" | sed -n 's/^- Verdict: \*\*\(.*\)\*\*$/\1/p' | head -n 1)"); failed=1 ;;
    esac
  done
  say "verdicts: $(IFS=,; echo "${verdicts[*]}" | sed 's/,/, /g'); summaries below the table"
  return "$failed"
}

# The case outputs of one case kind's run, named by the run its summary reports.
save_quality_outputs() {
  local kind=$1 summary=$2 run_id
  run_id=$(echo "$summary" | sed -n 's/^- Run: `\(aer-[^`]*\)`.*/\1/p' | head -n 1)
  [ -n "$run_id" ] || return 0
  mkdir -p "$outputs_dir" &&
    helper quality-outputs --run "$run_id" >"$outputs_dir/$kind.json" ||
    echo "the $kind case outputs were not saved: $(tail -n 1 "$outputs_dir/$kind.json" 2>/dev/null)" >&2
}

# The generation model calls of the run so far and their tokens, for the
# report: read from the API's log before the restart replaces its container,
# whether or not an earlier step failed.
read_model_usage() {
  [ -z "$models_label" ] && return
  model_usage=$(compose logs --no-color api 2>/dev/null |
    "$PYTHON" "$HELPER" model-usage --model "$generation_label") ||
    model_usage="not read: $model_usage"
}

restart_stack() {
  suspect_service=api
  if ! compose down --remove-orphans >&2 || ! compose up -d --no-build --wait --wait-timeout 300 >&2; then
    say "the stack did not come back healthy on the same volume"
    return 1
  fi
  say "down without removing volumes, then up: new containers on volume ${project}_memforge-data"
}

check_persisted() {
  local persisted
  persisted=$(helper persisted) || { say "$persisted"; return 1; }
  mcp verify >"$detail_file" || return 1
  say "$persisted; MCP: $(tail -n 1 "$detail_file")"
}

# -- Teardown and report -------------------------------------------------------

dump_logs() {
  local services service dumped=""
  services=$(compose ps -a --format '{{.Service}} {{.State}} {{.Health}}' 2>/dev/null |
    awk '$2 != "running" || ($3 != "" && $3 != "healthy") { print $1 }')
  for service in $services $suspect_service; do
    case " $dumped " in *" $service "*) continue ;; esac
    dumped="$dumped $service"
    log "last 80 log lines of $service"
    compose logs --no-color --tail 80 "$service" >&2
  done
}

leftovers() {
  local label="label=com.docker.compose.project=$project"
  {
    docker ps -aq --filter "$label"
    docker volume ls -q --filter "$label"
    docker network ls -q --filter "$label"
    docker image ls -q --filter "reference=$project-*"
  } 2>/dev/null
}

clean_up() {
  local started remaining
  started=$(date +%s)
  log "Clean up"
  compose down -v --remove-orphans --rmi local >&2
  remaining=$(leftovers)
  if [ -z "$remaining" ]; then
    record "Clean up" pass $(( $(date +%s) - started )) \
      "removed the containers, volume, network and images of $project; build cache kept"
  else
    failed=1
    record "Clean up" fail $(( $(date +%s) - started )) \
      "left behind: $(echo "$remaining" | tr '\n' ' ')"
  fi
}

format_seconds() {
  local seconds=$1
  if (( seconds >= 60 )); then
    printf '%dm %02ds' $(( seconds / 60 )) $(( seconds % 60 ))
  else
    printf '%ds' "$seconds"
  fi
}

print_report() {
  local commit models i detail
  commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
  if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
    commit="$commit with local changes"
  fi
  models=${models_label:-none (model, sync and retrieval steps skipped)}
  echo "### Compose fresh-install smoke"
  echo
  echo "- Commit: $commit"
  echo "- Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- Docker: $(docker version --format '{{.Server.Version}}' 2>/dev/null), Compose $(docker compose version --short 2>/dev/null)"
  echo "- Models: $models"
  [ -n "$model_usage" ] && echo "- Model usage: $model_usage"
  echo "- Wall time: $(format_seconds $(( $(date +%s) - run_started )))"
  echo
  echo "| Step | Result | Time | Detail |"
  echo "| --- | --- | --- | --- |"
  for i in "${!step_names[@]}"; do
    detail=${step_details[$i]//|/\\|}
    echo "| ${step_names[$i]} | ${step_results[$i]} | $(format_seconds "${step_times[$i]}") | $detail |"
  done
  if [ -n "$quality_summary" ]; then
    echo
    echo "$quality_summary"
  fi
}

finish() {
  local status=$?
  trap - EXIT INT TERM
  if (( failed || interrupted )); then
    dump_logs
  fi
  clean_up
  rm -rf "$work_dir"
  echo >&2
  if [ -n "$report_file" ]; then
    print_report | tee "$report_file"
  else
    print_report
  fi
  if (( interrupted )); then
    exit 130
  elif (( failed )); then
    exit 1
  fi
  exit "$status"
}

on_interrupt() {
  interrupted=1
  record "${current_step:-Interrupted}" fail 0 "interrupted"
  exit 130
}

trap finish EXIT
trap on_interrupt INT TERM

write_inputs
echo "Compose project $project, API $api_url, admin UI $ui_url" >&2

run_step "Build images" build_images
run_step "Start stack" start_stack
run_step "Ports bind to loopback only" check_ports
run_step "API health and admin UI" probe_stack
run_step "Packaged MCP proxy" mcp_tools
run_step "Model configuration" configure_models
run_step "Add local source" add_source
run_step "Local source sync" sync_source
run_step "Memory retrieval" retrieve_memories
if (( run_quality )); then
  run_step "Revised source sync" revise_source
  run_step "Add snapshot sources" add_snapshot_sources
  run_step "Snapshot sync, revision 1" sync_snapshot_sources 1
  run_step "Snapshot sync, revision 2" sync_snapshot_sources 2
  run_step "Quick quality evaluation" quality_checks
fi
read_model_usage
run_step "Restart on the same volume" restart_stack
run_step "State survives restart" check_persisted
