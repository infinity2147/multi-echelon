#!/bin/zsh
# Runs the full 40-run experiment grid, 6 processes at a time.
cd "/Users/anantasati/Desktop/multi-echleon" || exit 1
mkdir -p logs results

run_stage() {
  printf '%s\n' "$@" | xargs -P 6 -n 2 sh -c '
    echo "[$(date +%H:%M:%S)] start $0 seed $1"
    ./.venv/bin/python train.py --case "$0" --seed "$1" \
      > "logs/${0}_seed${1}.log" 2>&1
    echo "[$(date +%H:%M:%S)] done  $0 seed $1"
  '
}

stage1=(); stage2=()
for s in 0 1 2 3 4 5 6 7 8 9; do
  stage1+=("linear" "$s" "divergent" "$s")
  stage2+=("general_nolimit" "$s" "general_limit" "$s")
done

echo "=== stage 1: linear + divergent ($(date)) ==="
run_stage "${stage1[@]}"
echo "=== stage 2: general ($(date)) ==="
run_stage "${stage2[@]}"
echo "=== all done ($(date)) ==="
