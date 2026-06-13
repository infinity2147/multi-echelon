#!/bin/zsh
# Re-run of the general case with the corrected transition-limit interpretation
# (observation clipping) and 1-thread-per-process pinning for parallel speed.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "/Users/anantasati/Desktop/multi-echleon" || exit 1
mkdir -p logs results

jobs=()
for s in 0 1 2 3 4 5 6 7 8 9; do
  jobs+=("general_nolimit $s" "general_limit $s")
done

echo "=== general re-run start $(date) ==="
printf '%s\n' "${jobs[@]}" | xargs -P 7 -n 2 sh -c '
  echo "[$(date +%H:%M:%S)] start $0 seed $1"
  ./.venv/bin/python train.py --case "$0" --seed "$1" > "logs/${0}_seed${1}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $0 seed $1"
'
echo "=== general re-run done $(date) ==="
