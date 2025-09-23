#!/usr/bin/env bash
set -u  # 不用 set -e，防止单个任务失败把全局中断

PY=python
MAIN=main_SALSA.py

BACKBONE="dinov3"                 # vit, dinov3
BASE_OUT="/mnt/sda/sijiali/GlaucomaCode/Results_SALSA"
MODALITY="rnflt"
TASK="tds"
TRANSFORM="imagenet"              # vit | imagenet | albumentations | none
TRAIN_BATCH=32
VALID_BATCH=32
EPOCHS=500

# Change
POOLS=("mean_patch") #"cls" "mean_patch"
SCOPES=("all") # "head" "all"
LRS=(5e-5 1e-4 1e-3 5e-3) 
# LRS=(1e-4 1e-3 5e-3)    

# 并发=2：两组，每组3张卡
# PAIRS=("0,2,3" "4,5,7")
PAIRS=("0,2" "5,7")

# 生成作业列表：每个元素形如 "cls|head|1e-4"
JOBS=()
for p in "${POOLS[@]}"; do
  for s in "${SCOPES[@]}"; do
    for lr in "${LRS[@]}"; do
      JOBS+=("${p}|${s}|${lr}")
    done
  done
done
# JOBS=()
# for p in "${POOLS[@]}"; do
#   for s in "${SCOPES[@]}"; do
#     for lr in "${LRS[@]}"; do
#       # 规则：当 lr=1e-4 时，只跑 mean_patch
#       if [[ "$lr" == "1e-4" && "$p" != "mean_patch" ]]; then
#         continue
#       fi
#       JOBS+=("${p}|${s}|${lr}")
#     done
#   done
# done

GROUP_PIDS=("" "")  # 记录每组当前 PID

launch_job () {
  local group_idx="$1"
  local job="$2"
  IFS="|" read -r pool scope lr <<< "${job}"
  local pair="${PAIRS[$group_idx]}"

  # 路径名里带上组合信息
  local RUN_DIR="${BASE_OUT}/${BACKBONE}/${BACKBONE}_${TRANSFORM}_${pool}_${scope}_lr${lr}"
  mkdir -p "${RUN_DIR}/ckpts" "${RUN_DIR}/logs"

  echo "[LAUNCH] GROUP:${group_idx}  --gpus ${pair}   POOL:${pool}  SCOPE:${scope}  LR:${lr}"
  nohup ${PY} ${MAIN} \
    --backbone "${BACKBONE}" \
    --modality_type "${MODALITY}" \
    --task "${TASK}" \
    --transform "${TRANSFORM}" \
    --train_batch ${TRAIN_BATCH} \
    --valid_batch ${VALID_BATCH} \
    --epochs ${EPOCHS} \
    --gpus "${pair}" \
    --lr "${lr}" \
    --vit_pool "${pool}" \
    --train_scope "${scope}" \
    > "${RUN_DIR}/train.log" 2>&1 &

  GROUP_PIDS[$group_idx]=$!
}

# 先占满两组
idx=0
total=${#JOBS[@]}
for g in 0 1; do
  if (( idx < total )); then
    launch_job "$g" "${JOBS[$idx]}"
    ((idx++))
  fi
done

# 谁先结束，在该组继续发下一个
while (( idx < total )); do
  for g in 0 1; do
    pid="${GROUP_PIDS[$g]}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      launch_job "$g" "${JOBS[$idx]}"
      ((idx++))
      [[ $idx -ge $total ]] && break
    fi
  done
  sleep 3
done

# 等最后两组结束
for g in 0 1; do
  pid="${GROUP_PIDS[$g]}"
  [[ -n "$pid" ]] && wait "$pid" || true
done

echo "All ${total} runs completed (concurrency=2, 3 GPUs per run)."
