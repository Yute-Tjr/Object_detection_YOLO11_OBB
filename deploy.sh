#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${DEPLOY_PROJECT_ROOT:-$SCRIPT_DIR}
ENV_FILE=${DEPLOY_ENV_FILE:-$PROJECT_ROOT/.env}
POSTGRES_VOLUME=${DEPLOY_VOLUME_NAME:-terminal-inspection_postgres-data}
LOG_DIR=${DEPLOY_LOG_DIR:-$PROJECT_ROOT/var/deploy-logs}

DRY_RUN=false
ASSUME_YES=false
PULL_MODE=ask
CONFIG_CHANGED=false
DEPLOYMENT_MODE=""
LOG_FILE=""
COMPOSE_ARGS=()

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    BLUE=$'\033[34m'
    GREEN=$'\033[32m'
    YELLOW=$'\033[33m'
    RED=$'\033[31m'
    BOLD=$'\033[1m'
    RESET=$'\033[0m'
else
    BLUE=""
    GREEN=""
    YELLOW=""
    RED=""
    BOLD=""
    RESET=""
fi


usage() {
    cat <<'EOF'
用法：./deploy.sh [选项]

自动识别首次部署或更新部署，生成安全的 .env，并通过 Docker Compose
启动 PostgreSQL、API、Worker 和前端。

选项：
  --dry-run   显示配置和执行计划，不写入配置、不启动容器
  --yes       非交互确认；首次部署时必须通过环境变量提供数据库密码
  --pull      更新部署时直接执行 git pull --ff-only
  --no-pull   更新部署时不拉取远端代码
  -h, --help  显示帮助
EOF
}


info() {
    printf '%s%sℹ%s %s\n' "$BLUE" "$BOLD" "$RESET" "$*"
}


success() {
    printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$*"
}


warning() {
    printf '  %s!%s %s\n' "$YELLOW" "$RESET" "$*"
}


fatal() {
    printf '  %s✗%s %s\n' "$RED" "$RESET" "$*" >&2
    exit 1
}


section() {
    printf '\n%s%s[%s]%s %s\n' "$BLUE" "$BOLD" "$1" "$RESET" "$2"
}


banner() {
    local title=$1
    printf '\n%s%s╭────────────────────────────────────────────╮%s\n' "$BLUE" "$BOLD" "$RESET"
    printf '%s%s│          端子智能检测系统部署向导          │%s\n' "$BLUE" "$BOLD" "$RESET"
    printf '%s%s│          %-34s│%s\n' "$BLUE" "$BOLD" "$title" "$RESET"
    printf '%s%s╰────────────────────────────────────────────╯%s\n' "$BLUE" "$BOLD" "$RESET"
}


detect_deployment_mode() {
    local env_file=$1
    local volume_exists=$2

    if [[ -f "$env_file" && "$volume_exists" == "true" ]]; then
        printf '%s\n' "update"
    elif [[ -f "$env_file" ]]; then
        printf '%s\n' "configured_install"
    elif [[ "$volume_exists" == "true" ]]; then
        printf '%s\n' "orphaned_volume"
    else
        printf '%s\n' "first_install"
    fi
}


validate_db_password() {
    local password=$1
    [[ "$password" != "replace_with_a_strong_password" ]] \
        && [[ ${#password} -ge 12 ]] \
        && validate_db_password_charset "$password"
}


validate_db_password_charset() {
    [[ "$1" =~ ^[A-Za-z0-9._~-]+$ ]]
}


validate_identifier() {
    [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
}


validate_port() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( 1 <= 10#$1 && 10#$1 <= 65535 ))
}


validate_positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 > 0 ))
}


validate_device() {
    [[ "$1" == "cpu" || "$1" =~ ^[0-9]+$ ]]
}


write_env_file() {
    local destination=$1
    local temporary="${destination}.tmp.$$"
    local line
    local had_database_url=false

    umask 077
    {
        if [[ -f "$destination" ]]; then
            while IFS= read -r line || [[ -n "$line" ]]; do
                case "$line" in
                    "# Docker Compose deployment configuration"|POSTGRES_DB=*|POSTGRES_USER=*|POSTGRES_PASSWORD=*|POSTGRES_BIND_ADDRESS=*|POSTGRES_PORT=*|WEB_PORT=*|DETECTION_DEVICE=*|CLASSIFICATION_DEVICE=*|DETECTION_IMGSZ=*|CLASSIFICATION_IMGSZ=*|MAX_IMAGES_PER_TASK=*)
                        ;;
                    DATABASE_URL=*)
                        had_database_url=true
                        ;;
                    *)
                        printf '%s\n' "$line"
                        ;;
                esac
            done <"$destination"
            if [[ "$had_database_url" == "true" ]]; then
                printf 'DATABASE_URL=postgresql+psycopg://%s:%s@127.0.0.1:%s/%s\n' \
                    "$POSTGRES_USER" "$POSTGRES_PASSWORD" "$POSTGRES_PORT" "$POSTGRES_DB"
            fi
            printf '\n'
        fi
        printf '# Docker Compose deployment configuration\n'
        printf 'POSTGRES_DB=%s\n' "$POSTGRES_DB"
        printf 'POSTGRES_USER=%s\n' "$POSTGRES_USER"
        printf 'POSTGRES_PASSWORD=%s\n' "$POSTGRES_PASSWORD"
        printf 'POSTGRES_BIND_ADDRESS=%s\n' "$POSTGRES_BIND_ADDRESS"
        printf 'POSTGRES_PORT=%s\n' "$POSTGRES_PORT"
        printf 'WEB_PORT=%s\n' "$WEB_PORT"
        printf 'DETECTION_DEVICE=%s\n' "$DETECTION_DEVICE"
        printf 'CLASSIFICATION_DEVICE=%s\n' "$CLASSIFICATION_DEVICE"
        printf 'DETECTION_IMGSZ=%s\n' "$DETECTION_IMGSZ"
        printf 'CLASSIFICATION_IMGSZ=%s\n' "$CLASSIFICATION_IMGSZ"
        printf 'MAX_IMAGES_PER_TASK=%s\n' "$MAX_IMAGES_PER_TASK"
    } >"$temporary"
    chmod 600 "$temporary"
    mv "$temporary" "$destination"
}


load_env_file() {
    local source_file=$1
    local line key value

    while IFS= read -r line || [[ -n "$line" ]]; do
        line=${line%$'\r'}
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" == *=* ]] || continue
        key=${line%%=*}
        value=${line#*=}
        case "$key" in
            POSTGRES_DB|POSTGRES_USER|POSTGRES_PASSWORD|POSTGRES_BIND_ADDRESS|POSTGRES_PORT|WEB_PORT|DETECTION_DEVICE|CLASSIFICATION_DEVICE|DETECTION_IMGSZ|CLASSIFICATION_IMGSZ|MAX_IMAGES_PER_TASK)
                printf -v "$key" '%s' "$value"
                export "$key"
                ;;
            DATABASE_URL)
                LEGACY_DATABASE_URL=$value
                export LEGACY_DATABASE_URL
                ;;
        esac
    done <"$source_file"
}


secure_env_file() {
    local source_file=$1
    [[ ! -L "$source_file" ]] || fatal ".env 不能是符号链接：$source_file"
    [[ -f "$source_file" ]] || fatal ".env 必须是普通文件：$source_file"
    [[ -O "$source_file" ]] || fatal ".env 不属于当前用户，拒绝读取：$source_file"
    if [[ "$DRY_RUN" == "true" ]]; then
        success "演练模式仅检查 .env，不修改文件权限"
    else
        chmod 600 "$source_file" || fatal "无法将 .env 权限设置为 600。"
        success ".env 文件权限已确认为 600"
    fi
}


migrate_legacy_database_url() {
    local database_url=${LEGACY_DATABASE_URL:-}
    if [[ -n "${POSTGRES_DB:-}" && -n "${POSTGRES_USER:-}" && -n "${POSTGRES_PASSWORD:-}" && -n "${POSTGRES_PORT:-}" ]]; then
        return 0
    fi
    [[ -n "$database_url" ]] || return 0
    if [[ "$database_url" =~ ^postgresql(\+psycopg)?://([^:]+):([^@]+)@([^:/]+):([0-9]+)/([^?]+) ]]; then
        POSTGRES_USER=${POSTGRES_USER:-${BASH_REMATCH[2]}}
        POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-${BASH_REMATCH[3]}}
        POSTGRES_PORT=${POSTGRES_PORT:-${BASH_REMATCH[5]}}
        POSTGRES_DB=${POSTGRES_DB:-${BASH_REMATCH[6]}}
        export POSTGRES_DB POSTGRES_USER POSTGRES_PASSWORD POSTGRES_PORT
        CONFIG_CHANGED=true
    fi
}


detect_default_device() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        printf '%s\n' "0"
    else
        printf '%s\n' "cpu"
    fi
}


set_config_defaults() {
    POSTGRES_DB=${POSTGRES_DB:-terminal_inspection}
    POSTGRES_USER=${POSTGRES_USER:-terminal}
    POSTGRES_BIND_ADDRESS=${POSTGRES_BIND_ADDRESS:-127.0.0.1}
    POSTGRES_PORT=${POSTGRES_PORT:-5432}
    WEB_PORT=${WEB_PORT:-8080}
    DETECTION_DEVICE=${DETECTION_DEVICE:-$(detect_default_device)}
    CLASSIFICATION_DEVICE=${CLASSIFICATION_DEVICE:-$DETECTION_DEVICE}
    DETECTION_IMGSZ=${DETECTION_IMGSZ:-1280}
    CLASSIFICATION_IMGSZ=${CLASSIFICATION_IMGSZ:-224}
    MAX_IMAGES_PER_TASK=${MAX_IMAGES_PER_TASK:-100}
    export POSTGRES_DB POSTGRES_USER POSTGRES_PASSWORD POSTGRES_BIND_ADDRESS
    export POSTGRES_PORT WEB_PORT DETECTION_DEVICE CLASSIFICATION_DEVICE
    export DETECTION_IMGSZ CLASSIFICATION_IMGSZ MAX_IMAGES_PER_TASK
}


prompt_default() {
    local variable=$1
    local label=$2
    local fallback=$3
    local current=${!variable:-$fallback}
    local answer=""

    if [[ "$ASSUME_YES" == "true" ]]; then
        printf -v "$variable" '%s' "$current"
    else
        printf '  %s [%s]: ' "$label" "$current"
        IFS= read -r answer
        printf -v "$variable" '%s' "${answer:-$current}"
    fi
    export "$variable"
}


prompt_new_password() {
    local first=""
    local second=""

    if [[ "$ASSUME_YES" == "true" ]]; then
        [[ -n "${POSTGRES_PASSWORD:-}" ]] || fatal "非交互首次部署必须通过 POSTGRES_PASSWORD 提供数据库密码。"
        return
    fi
    while true; do
        printf '  数据库密码（至少 12 位，仅限字母、数字、._~-）: '
        IFS= read -rs first
        printf '\n  再次输入数据库密码: '
        IFS= read -rs second
        printf '\n'
        if [[ "$first" != "$second" ]]; then
            warning "两次密码不一致，请重新输入。"
        elif ! validate_db_password "$first"; then
            warning "密码至少 12 位，且只能包含字母、数字、点、下划线、波浪线和连字符。"
        else
            POSTGRES_PASSWORD=$first
            export POSTGRES_PASSWORD
            success "两次密码一致"
            return
        fi
    done
}


prompt_yes_no() {
    local label=$1
    local default=${2:-yes}
    local answer=""

    if [[ "$ASSUME_YES" == "true" ]]; then
        [[ "$default" == "yes" ]]
        return
    fi
    if [[ "$default" == "yes" ]]; then
        printf '  %s [Y/n]: ' "$label"
    else
        printf '  %s [y/N]: ' "$label"
    fi
    IFS= read -r answer
    answer=${answer:-$default}
    [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ || "$answer" == "yes" ]]
}


configure_initial_install() {
    section "02/10" "配置数据库"
    prompt_default POSTGRES_DB "数据库名称" "terminal_inspection"
    prompt_default POSTGRES_USER "数据库用户" "terminal"
    prompt_new_password
    prompt_default POSTGRES_BIND_ADDRESS "数据库监听地址" "127.0.0.1"
    prompt_default POSTGRES_PORT "数据库端口" "5432"

    section "03/10" "配置推理设备"
    prompt_default DETECTION_DEVICE "检测模型设备" "$(detect_default_device)"
    prompt_default CLASSIFICATION_DEVICE "分类模型设备" "$DETECTION_DEVICE"

    section "04/10" "其他配置"
    prompt_default WEB_PORT "网页端口" "8080"
    prompt_default DETECTION_IMGSZ "检测输入尺寸" "1280"
    prompt_default CLASSIFICATION_IMGSZ "分类输入尺寸" "224"
    prompt_default MAX_IMAGES_PER_TASK "单任务最大图片数" "100"
    CONFIG_CHANGED=true
}


configure_update() {
    success "复用现有数据库凭据"
    if prompt_yes_no "是否修改设备、端口或图片尺寸？" "no"; then
        section "03/10" "修改运行配置"
        prompt_default DETECTION_DEVICE "检测模型设备" "$DETECTION_DEVICE"
        prompt_default CLASSIFICATION_DEVICE "分类模型设备" "$CLASSIFICATION_DEVICE"
        prompt_default WEB_PORT "网页端口" "$WEB_PORT"
        prompt_default POSTGRES_PORT "数据库端口" "$POSTGRES_PORT"
        prompt_default DETECTION_IMGSZ "检测输入尺寸" "$DETECTION_IMGSZ"
        prompt_default CLASSIFICATION_IMGSZ "分类输入尺寸" "$CLASSIFICATION_IMGSZ"
        prompt_default MAX_IMAGES_PER_TASK "单任务最大图片数" "$MAX_IMAGES_PER_TASK"
        CONFIG_CHANGED=true
    fi
}


validate_configuration() {
    validate_identifier "$POSTGRES_DB" || fatal "数据库名称只能包含字母、数字和下划线，且不能以数字开头。"
    validate_identifier "$POSTGRES_USER" || fatal "数据库用户只能包含字母、数字和下划线，且不能以数字开头。"
    validate_db_password_charset "${POSTGRES_PASSWORD:-}" || fatal "数据库密码包含 URL 非安全字符；请仅使用字母、数字、._~-。"
    if [[ "$DEPLOYMENT_MODE" == "update" ]]; then
        if [[ "$POSTGRES_PASSWORD" == "replace_with_a_strong_password" ]]; then
            warning "当前数据库仍在使用示例占位密码；本次更新不会自动改库，请尽快安排密码轮换。"
        fi
        if (( ${#POSTGRES_PASSWORD} < 12 )); then
            warning "旧数据库密码少于 12 位，本次更新继续复用；请另行安排数据库密码轮换。"
        fi
    elif ! validate_db_password "$POSTGRES_PASSWORD"; then
        fatal "数据库密码至少需要 12 位，且不能使用示例占位密码。"
    fi
    [[ "$POSTGRES_BIND_ADDRESS" == "127.0.0.1" || "$POSTGRES_BIND_ADDRESS" == "localhost" ]] || fatal "数据库仅允许绑定到 127.0.0.1 或 localhost。"
    validate_port "$POSTGRES_PORT" || fatal "PostgreSQL 端口无效。"
    validate_port "$WEB_PORT" || fatal "网页端口无效。"
    validate_device "$DETECTION_DEVICE" || fatal "检测设备必须是 cpu 或非负 GPU 编号。"
    validate_device "$CLASSIFICATION_DEVICE" || fatal "分类设备必须是 cpu 或非负 GPU 编号。"
    validate_positive_integer "$DETECTION_IMGSZ" || fatal "检测输入尺寸必须是正整数。"
    validate_positive_integer "$CLASSIFICATION_IMGSZ" || fatal "分类输入尺寸必须是正整数。"
    validate_positive_integer "$MAX_IMAGES_PER_TASK" || fatal "单任务图片数必须是正整数。"
    (( MAX_IMAGES_PER_TASK <= 100 )) || fatal "单任务最大图片数不能超过 100。"
}


print_config_summary() {
    printf '\n%s配置摘要%s\n' "$BOLD" "$RESET"
    printf '%s\n' '──────────────────────────────────────────────'
    printf '  %-18s %s\n' "数据库名称" "$POSTGRES_DB"
    printf '  %-18s %s\n' "数据库用户" "$POSTGRES_USER"
    printf '  %-18s %s\n' "数据库密码" "************"
    printf '  %-18s %s\n' "检测设备" "$DETECTION_DEVICE"
    printf '  %-18s %s\n' "分类设备" "$CLASSIFICATION_DEVICE"
    printf '  %-18s %s\n' "网页端口" "$WEB_PORT"
    printf '  %-18s %s\n' "单任务图片数" "$MAX_IMAGES_PER_TASK"
    printf '%s\n' '──────────────────────────────────────────────'
}


check_required_commands() {
    command -v docker >/dev/null 2>&1 || fatal "未找到 Docker，请先安装 Docker Engine。"
    success "Docker 客户端可用"
    docker compose version >/dev/null 2>&1 || fatal "未找到 Docker Compose v2。"
    success "Docker Compose v2 可用"
    docker info >/dev/null 2>&1 || fatal "无法连接 Docker Engine，请先启动 Docker 服务。"
    success "Docker Engine 连接正常"
}


postgres_volume_exists() {
    local output=""
    if output=$(docker volume inspect "$POSTGRES_VOLUME" 2>&1); then
        return 0
    fi
    case "$output" in
        *"no such volume"*|*"No such volume"*)
            return 1
            ;;
        *)
            fatal "无法检查 PostgreSQL 数据卷：$output"
            ;;
    esac
}


check_weights() {
    local missing=()
    local path
    for path in \
        "$PROJECT_ROOT/weights/detector/yolo11l_obb_best.pt" \
        "$PROJECT_ROOT/weights/classifiers/label3/resnet18_best.pt" \
        "$PROJECT_ROOT/weights/classifiers/label5/resnet18_best.pt"; do
        [[ -s "$path" ]] || missing+=("${path#"$PROJECT_ROOT/"}")
    done
    if (( ${#missing[@]} > 0 )); then
        fatal "模型权重不存在或为空：${missing[*]}"
    fi
    success "三个模型权重完整"
}


check_gpu() {
    local indexes=""
    local requested=""
    local index=""
    local found=false
    if [[ "$DETECTION_DEVICE" == "cpu" && "$CLASSIFICATION_DEVICE" == "cpu" ]]; then
        success "使用 CPU 推理，不启用 GPU 覆盖配置"
        return
    fi
    command -v nvidia-smi >/dev/null 2>&1 || fatal "配置了 GPU，但系统中未找到 nvidia-smi。"
    nvidia-smi >/dev/null 2>&1 || fatal "nvidia-smi 执行失败，请检查 NVIDIA 驱动。"
    indexes=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null) \
        || fatal "无法读取 NVIDIA GPU 编号。"
    for requested in "$DETECTION_DEVICE" "$CLASSIFICATION_DEVICE"; do
        [[ "$requested" == "cpu" ]] && continue
        found=false
        while IFS= read -r index; do
            index=${index//[[:space:]]/}
            if [[ "$index" == "$requested" ]]; then
                found=true
                break
            fi
        done <<<"$indexes"
        [[ "$found" == "true" ]] || fatal "GPU 设备 $requested 不存在；可用编号：${indexes//$'\n'/, }"
    done
    success "NVIDIA GPU、驱动及所选设备编号可用"
}


build_compose_args() {
    COMPOSE_ARGS=(compose)
    if [[ -f "$ENV_FILE" ]]; then
        COMPOSE_ARGS+=(--env-file "$ENV_FILE")
    fi
    COMPOSE_ARGS+=(-f "$PROJECT_ROOT/compose.yaml")
    if [[ "$DETECTION_DEVICE" != "cpu" || "$CLASSIFICATION_DEVICE" != "cpu" ]]; then
        COMPOSE_ARGS+=(-f "$PROJECT_ROOT/compose.gpu.yaml")
    fi
}


compose() {
    docker "${COMPOSE_ARGS[@]}" "$@"
}


create_log_file() {
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/deploy-$(date '+%Y%m%d-%H%M%S').log"
    : >"$LOG_FILE"
    chmod 600 "$LOG_FILE"
}


run_logged_step() {
    local number=$1
    local label=$2
    shift 2
    section "$number" "$label"
    info "正在执行，详细输出写入 $LOG_FILE"
    if "$@" >>"$LOG_FILE" 2>&1; then
        success "$label 完成"
    else
        local status=$?
        warning "详细错误日志：$LOG_FILE"
        tail -n 60 "$LOG_FILE" >&2 || true
        fatal "$label 失败（退出码 $status）"
    fi
}


wait_for_service_health() {
    local service=$1
    local timeout=${2:-180}
    local started=$SECONDS
    local container status

    while (( SECONDS - started < timeout )); do
        container=$(compose ps -q "$service" 2>/dev/null || true)
        if [[ -n "$container" ]]; then
            status=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container" 2>/dev/null || true)
            [[ "$status" == "healthy" || "$status" == "running" ]] && return 0
            [[ "$status" == "unhealthy" || "$status" == "exited" || "$status" == "dead" ]] && return 1
        fi
        sleep 2
    done
    return 1
}


wait_for_worker_ready() {
    local timeout=${1:-300}
    local started=$SECONDS
    while (( SECONDS - started < timeout )); do
        if compose exec -T api python -c 'import json, sys, urllib.request; data=json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/v1/health", timeout=3)); sys.exit(0 if data.get("workerReady") and data.get("modelsReady") else 1)' >/dev/null 2>&1; then
            return 0
        fi
        sleep 3
    done
    return 1
}


backup_database() {
    local backup_dir="$PROJECT_ROOT/backups"
    local stamp
    local temporary
    local backup_file
    mkdir -p "$backup_dir"
    stamp=$(date '+%Y%m%d-%H%M%S')
    temporary=$(mktemp "$backup_dir/.backup.XXXXXX") || fatal "无法创建数据库备份临时文件。"
    backup_file="$backup_dir/${POSTGRES_DB}_${stamp}.${temporary##*.}.dump"
    section "05/10" "备份 PostgreSQL"
    if compose exec -T postgres pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc >"$temporary" 2>>"$LOG_FILE"; then
        chmod 600 "$temporary"
        mv "$temporary" "$backup_file"
        success "数据库已备份到 ${backup_file#"$PROJECT_ROOT/"}"
    else
        rm -f "$temporary"
        fatal "数据库备份失败，已停止更新。"
    fi
}


git_pull_requested() {
    case "$PULL_MODE" in
        always) return 0 ;;
        never) return 1 ;;
    esac
    prompt_yes_no "是否从当前 Git 上游拉取最新代码？" "yes"
}


maybe_pull_updates() {
    local restart_args=(--no-pull)
    [[ "$DEPLOYMENT_MODE" == "update" ]] || return 0
    git_pull_requested || return 0
    if [[ "$DRY_RUN" == "true" ]]; then
        info "演练：将使用 git pull --ff-only 更新当前上游。"
        return
    fi
    command -v git >/dev/null 2>&1 || fatal "更新模式需要 Git。"
    git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fatal "项目目录不是 Git 仓库。"
    [[ -z "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=no)" ]] || fatal "存在未提交的已跟踪文件，拒绝自动拉取。"
    git -C "$PROJECT_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' >/dev/null 2>&1 || fatal "当前分支没有配置 Git 上游。"
    create_log_file
    run_logged_step "02/10" "拉取远端代码" git -C "$PROJECT_ROOT" pull --ff-only
    [[ -f "$PROJECT_ROOT/deploy.sh" ]] || fatal "更新后找不到 deploy.sh。"
    [[ "$ASSUME_YES" == "true" ]] && restart_args+=(--yes)
    info "使用更新后的部署脚本重新检查环境与配置。"
    exec bash "$PROJECT_ROOT/deploy.sh" "${restart_args[@]}"
}


show_dry_run_plan() {
    section "05/10" "演练模式"
    success "不会写入 .env"
    success "不会构建镜像或启动容器"
    printf '\n预计执行步骤：\n'
    printf '  1. 验证 Compose 配置和模型权重\n'
    [[ "$DEPLOYMENT_MODE" == "update" ]] && printf '  2. 备份 PostgreSQL 数据库\n'
    printf '  3. 构建 API、Worker 和前端镜像\n'
    printf '  4. 启动 PostgreSQL 并执行 Alembic 迁移\n'
    printf '  5. 启动 API、Worker 和前端并等待健康检查\n'
}


run_deployment() {
    [[ -n "$LOG_FILE" ]] || create_log_file
    run_logged_step "05/10" "验证 Compose 配置" compose config --quiet

    if [[ "$DEPLOYMENT_MODE" == "update" ]]; then
        run_logged_step "06/10" "启动 PostgreSQL" compose up -d postgres
        section "06/10" "等待 PostgreSQL 健康"
        wait_for_service_health postgres 180 || fatal "PostgreSQL 未能进入健康状态。"
        success "PostgreSQL 已就绪"
        backup_database
    fi

    run_logged_step "07/10" "构建服务镜像" compose build --pull api worker frontend

    if [[ "$DETECTION_DEVICE" != "cpu" || "$CLASSIFICATION_DEVICE" != "cpu" ]]; then
        run_logged_step "07/10" "验证容器 GPU 运行时" \
            compose run --rm --no-deps worker python -c \
            'import os, torch; requested={os.environ["DETECTION_DEVICE"], os.environ["CLASSIFICATION_DEVICE"]}-{"cpu"}; assert torch.cuda.is_available(), "CUDA is unavailable inside worker container"; count=torch.cuda.device_count(); missing=sorted(d for d in requested if int(d) >= count); assert not missing, f"CUDA devices unavailable in container: {missing}; count={count}"'
    fi

    if [[ "$DEPLOYMENT_MODE" == "update" ]]; then
        run_logged_step "08/10" "暂停旧版 API 与 Worker" compose stop worker api
    fi

    if [[ "$DEPLOYMENT_MODE" != "update" ]]; then
        run_logged_step "08/10" "启动 PostgreSQL" compose up -d postgres
        section "08/10" "等待 PostgreSQL 健康"
        wait_for_service_health postgres 180 || fatal "PostgreSQL 未能进入健康状态。"
        success "PostgreSQL 已就绪"
    fi

    run_logged_step "08/10" "执行数据库迁移" compose run --rm api alembic upgrade head
    run_logged_step "09/10" "重置 Worker 就绪状态" \
        compose run --rm --no-deps api python -c \
        'from pathlib import Path; Path("/data/terminal-inspection/worker-readiness.json").unlink(missing_ok=True)'
    run_logged_step "09/10" "启动检测服务" compose up -d --remove-orphans api worker frontend

    section "10/10" "健康检查"
    if ! wait_for_service_health api 180; then
        compose logs --tail=100 api >>"$LOG_FILE" 2>&1 || true
        fatal "API 未能进入健康状态，查看 $LOG_FILE。"
    fi
    success "API：healthy"
    if ! wait_for_worker_ready 300; then
        compose logs --tail=100 worker >>"$LOG_FILE" 2>&1 || true
        fatal "Worker 未在规定时间内完成模型加载，查看 $LOG_FILE。"
    fi
    success "Worker：ready，三个模型已加载"
    if compose exec -T frontend wget -q -O /dev/null http://127.0.0.1/ >/dev/null 2>&1; then
        success "前端：HTTP 200"
    else
        fatal "前端健康检查失败。"
    fi

    printf '\n%s%s部署完成%s\n' "$GREEN" "$BOLD" "$RESET"
    printf '网页地址：http://服务器地址:%s\n' "$WEB_PORT"
    printf '详细日志：%s\n' "$LOG_FILE"
    printf '实时日志：docker compose logs -f api worker frontend\n'
    printf '停止服务：docker compose down\n'
}


parse_args() {
    while (( $# > 0 )); do
        case "$1" in
            --dry-run) DRY_RUN=true ;;
            --yes) ASSUME_YES=true ;;
            --pull) PULL_MODE=always ;;
            --no-pull) PULL_MODE=never ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                usage >&2
                fatal "未知参数：$1"
                ;;
        esac
        shift
    done
}


main() {
    parse_args "$@"
    cd "$PROJECT_ROOT"

    section "01/10" "检查运行环境"
    check_required_commands

    local volume_exists=false
    postgres_volume_exists && volume_exists=true
    DEPLOYMENT_MODE=$(detect_deployment_mode "$ENV_FILE" "$volume_exists")

    case "$DEPLOYMENT_MODE" in
        first_install) banner "首次部署 · Docker Compose" ;;
        configured_install) banner "首次部署 · 复用已有配置" ;;
        update) banner "更新部署 · Docker Compose" ;;
        orphaned_volume)
            banner "安全检查未通过"
            fatal "PostgreSQL 数据卷已经存在，但 .env 不存在。请恢复原 .env，脚本不会使用默认凭据覆盖现有数据库。"
            ;;
    esac

    maybe_pull_updates

    if [[ -f "$ENV_FILE" ]]; then
        secure_env_file "$ENV_FILE"
        load_env_file "$ENV_FILE"
        migrate_legacy_database_url
    fi
    set_config_defaults

    if [[ "$DEPLOYMENT_MODE" == "update" ]]; then
        configure_update
    else
        configure_initial_install
    fi
    validate_configuration
    print_config_summary

    if [[ "$ASSUME_YES" != "true" ]] && ! prompt_yes_no "确认开始部署？" "yes"; then
        info "用户取消部署。"
        exit 0
    fi

    check_weights
    check_gpu

    if [[ "$DRY_RUN" == "true" ]]; then
        show_dry_run_plan
        exit 0
    fi

    if [[ "$CONFIG_CHANGED" == "true" ]]; then
        if [[ -f "$ENV_FILE" ]]; then
            local env_backup="${ENV_FILE}.backup.$(date '+%Y%m%d-%H%M%S')"
            cp "$ENV_FILE" "$env_backup"
            chmod 600 "$env_backup"
        fi
        write_env_file "$ENV_FILE"
        success ".env 已写入且权限为 600"
    fi

    build_compose_args
    run_deployment
}


if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
