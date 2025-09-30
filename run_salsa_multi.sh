#!/usr/bin/env bash
# set -e 不开，防止单个任务失败中断全部；但保留 -u 和 pipefail 更安全
set -u
set -o pipefail

PY=python
MAIN=main_SALSA_multi.py

BACKBONE="dinov3"
BASE_OUT="/mnt/sda/sijiali/GlaucomaCode/Results_SALSA_multi"
MODALITY="rnflt+slab"
TASK="tds"
TRANSFORM="imagenet"
TRAIN_BATCH=32
VALID_BATCH=32
EPOCHS=500

# 搜索空间
SCOPES=("all")                        # "head" "all"
POOLS=("cls" "mean_patch")            # "cls" "mean_patch"
FUSIONS=("concat" "sum" "gated-sum" "attn") # "gated-sum" "attn"
LRS=(5e-5 1e-4 1e-3)
# LRS=(5e-5 1e-4 1e-3)

# 并发分组（每组一条 --gpus）
PAIRS=("0,1" "2,3,4" "5,6,7")               # 两组，每组3张卡
NUM_GROUPS=${#PAIRS[@]}

# 生成作业列表：每个元素 "pool|scope|lr|fusion"
JOBS=()
for p in "${POOLS[@]}"; do
  for s in "${SCOPES[@]}"; do
    for lr in "${LRS[@]}"; do
      for fu in "${FUSIONS[@]}"; do
        JOBS+=("${p}|${s}|${lr}|${fu}")
      done
    done
  done
done

# 记录每组当前 PID
declare -a GROUP_PIDS
for ((g=0; g<NUM_GROUPS; g++)); do GROUP_PIDS[$g]=""; done

launch_job () {
  local group_idx="$1"
  local job="$2"
  local pool scope lr fu

  IFS="|" read -r pool scope lr fu <<< "${job}"
  local pair="${PAIRS[$group_idx]}"

  local RUN_DIR="${BASE_OUT}/${BACKBONE}/${BACKBONE}_${TRANSFORM}_${scope}_${pool}_${fu}_lr${lr}"
  mkdir -p "${RUN_DIR}/ckpts" "${RUN_DIR}/logs"

  echo "[LAUNCH] G${group_idx}  --gpus=${pair}  POOL=${pool}  SCOPE=${scope}  FUSION=${fu}  LR=${lr}"
  echo "  OUTDIR: ${RUN_DIR}"

  # 开跑
  nohup "${PY}" "${MAIN}" \
    --backbone "${BACKBONE}" \
    --modality_type "${MODALITY}" \
    --task "${TASK}" \
    --transform "${TRANSFORM}" \
    --train_batch "${TRAIN_BATCH}" \
    --valid_batch "${VALID_BATCH}" \
    --epochs "${EPOCHS}" \
    --gpus "${pair}" \
    --lr "${lr}" \
    --vit_pool "${pool}" \
    --train_scope "${scope}" \
    --fusion "${fu}" \
    > "${RUN_DIR}/train.log" 2>&1 &

  GROUP_PIDS[$group_idx]=$!
}

# 先占满所有组
idx=0
total=${#JOBS[@]}
for ((g=0; g<NUM_GROUPS; g++)); do
  (( idx < total )) || break
  launch_job "$g" "${JOBS[$idx]}"
  ((idx++))
done

# 轮询：哪个组先结束，就在该组继续发下一个
while (( idx < total )); do
  for ((g=0; g<NUM_GROUPS; g++)); do
    pid="${GROUP_PIDS[$g]}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      launch_job "$g" "${JOBS[$idx]}"
      ((idx++))
      (( idx < total )) || break
    fi
  done
  sleep 3
done

# 等所有组结束
for ((g=0; g<NUM_GROUPS; g++)); do
  pid="${GROUP_PIDS[$g]}"
  if [[ -n "${pid}" ]]; then
    wait "${pid}" || true
  fi
done

echo "All ${total} runs completed (concurrency=${NUM_GROUPS}, GPUs per run as in PAIRS)."
