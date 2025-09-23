#!/usr/bin/env bash
set -u  # 别用 set -e，避免某个任务失败导致脚本提前退出

PY=python
MAIN=main_HGF.py
BACKBONE="dinov3"  # vit, dinov3
BASE_OUT="/mnt/sda/sijiali/GlaucomaCode/Results_HGF"
MODALITY="rnflt"
TASK="tds"
DEPTH=3
TRANSFORM="vit"  # vit, cnn, albumentations, none, imagenet
TRAIN_BATCH=128
VALID_BATCH=64
EPOCHS=500

# 并发=2：两组，每组3张卡
PAIRS=("0,1,2" "3,4,5")

# 6 个学习率（按需改）
LRS=(5e-5 5e-4 1e-3 5e-3 1e-2)

# 记录每组当前 PID（空字符串=空闲）
GROUP_PIDS=("" "")

launch_job () {
  local group_idx="$1"
  local lr="$2"
  local pair="${PAIRS[$group_idx]}"

  local RUN_DIR="${BASE_OUT}/${BACKBONE}/${BACKBONE}_${TRANSFORM}_lr${lr}"
  mkdir -p "${RUN_DIR}/ckpts" "${RUN_DIR}/logs"

  echo "[LAUNCH] GROUP:${group_idx}  --gpus ${pair}  LR:${lr}"
  nohup ${PY} ${MAIN} \
    --backbone "${BACKBONE}" \
    --modality_type "${MODALITY}" \
    --task "${TASK}" \
    --depth ${DEPTH} \
    --transform "${TRANSFORM}" \
    --train_batch ${TRAIN_BATCH} \
    --valid_batch ${VALID_BATCH} \
    --epochs ${EPOCHS} \
    --gpus "${pair}" \
    --lr "${lr}" \
    > "${RUN_DIR}/train.log" 2>&1 &

  GROUP_PIDS[$group_idx]=$!
}

# 先占满两组（并发=2）
idx=0
for g in 0 1; do
  if (( idx < ${#LRS[@]} )); then
    launch_job "$g" "${LRS[$idx]}"
    ((idx++))
  fi
done

# 轮询：谁先结束，就在那一组启动下一个
while (( idx < ${#LRS[@]} )); do
  for g in 0 1; do
    pid="${GROUP_PIDS[$g]}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      launch_job "$g" "${LRS[$idx]}"
      ((idx++))
      [[ $idx -ge ${#LRS[@]} ]] && break
    fi
  done
  sleep 3
done

# 等最后两个结束
for g in 0 1; do
  pid="${GROUP_PIDS[$g]}"
  [[ -n "$pid" ]] && wait "$pid" || true
done

echo "All ${#LRS[@]} runs completed (concurrency=2, 3 GPUs per run)."
