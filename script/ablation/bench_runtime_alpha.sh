#!/bin/bash
# Bench runtime per stage + alpha sensitivity for IJAT reviewer response.
set -u
ROOT=/home/mas/proj/study/foels
INPUT=$ROOT/data/todaiura2
NFRAMES=30
WORK=$ROOT/result/ijat_bench
mkdir -p $WORK

# 1. Stage a small copy of the input (first NFRAMES).
SMALL=$WORK/input_small
mkdir -p $SMALL
ls $INPUT/*.png 2>/dev/null | sort | head -n $NFRAMES | xargs -I{} cp -u {} $SMALL/

# 2. Hardware info.
{
  echo "===== HARDWARE ====="
  echo "Host: $(hostname)"
  echo "Date: $(date -Iseconds)"
  echo
  echo "----- CPU -----"
  lscpu | grep -E "Model name|^CPU\(s\):|Thread\(s\) per core|MHz"
  echo
  echo "----- RAM -----"
  free -h | head -2
  echo
  echo "----- GPU -----"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo
  echo "----- INPUT -----"
  echo "Input dir: $SMALL"
  echo "Frames staged: $(ls $SMALL | wc -l)"
} > $WORK/hardware.txt
echo "[INFO] hardware info written"

cd $ROOT
source .venv/bin/activate

# 3. Compute optical flow once with UniMatch on the small input (timed).
FLOW_RES=$WORK/flow
mkdir -p $FLOW_RES

FIRST_BASE=$(ls $SMALL | head -1 | sed 's/\.png//')
if [ ! -f "$FLOW_RES/${FIRST_BASE}_pred.flo" ]; then
  echo "[INFO] computing optical flow (UniMatch)..."
  T_FLOW_START=$(date +%s.%N)
  python foels/ext/unimatch/main_flow.py \
    --inference_dir $SMALL \
    --output_path $FLOW_RES \
    --resume foels/ext/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
    --inference_size 768 1024 \
    --padding_factor 32 --upsample_factor 4 --num_scales 2 \
    --attn_splits_list 2 8 --corr_radius_list -1 4 --prop_radius_list -1 1 \
    --reg_refine --num_reg_refine 6 --save_flo_flow \
    > $WORK/flow.log 2>&1
  T_FLOW_END=$(date +%s.%N)
  echo "[INFO] flow elapsed: $(echo "$T_FLOW_END - $T_FLOW_START" | bc) seconds for $NFRAMES frames"
  echo "$T_FLOW_END - $T_FLOW_START seconds for $NFRAMES frames" > $WORK/flow_elapsed.txt
else
  echo "[INFO] flow already computed, skipping."
fi

# 4. For each alpha, run instrumented FoELS and dump per-frame CSV.
for ALPHA in 0.0 0.25 0.5; do
  ARES=$WORK/alpha_${ALPHA}
  mkdir -p $ARES
  ASEG=$WORK/alpha_${ALPHA}_seg
  mkdir -p $ASEG
  CSV=$WORK/timing_alpha_${ALPHA}.csv
  echo "===== ALPHA=$ALPHA ====="
  python $ROOT/script/bench_foels.py \
    --config $ROOT/script/foels_param.yaml \
    --timing_csv $CSV \
    --input_dir $SMALL \
    --flow_result_dir $FLOW_RES \
    --segment_result_dir $ASEG \
    --result_dir $ARES \
    --movprob_lengthfactor_coeff $ALPHA \
    > $WORK/run_alpha_${ALPHA}.log 2>&1
  echo "[INFO] alpha=$ALPHA done. csv=$CSV"
  # Count moving pixels per frame as alpha-sensitivity proxy (no DAVIS GT available).
  python -c "
import os, sys, glob, numpy as np, cv2
masks = sorted(glob.glob('$ARES/*_mask.png'))
if not masks:
    print('  no masks produced')
else:
    counts = []
    for m in masks:
        img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        counts.append(int((img > 127).sum()))
    print(f'  moving-pixel count per frame: mean={np.mean(counts):.1f} std={np.std(counts):.1f} min={min(counts)} max={max(counts)} n={len(counts)}')
" >> $WORK/alpha_summary.txt
done

echo "[DONE] $(date -Iseconds)"
echo "Results under: $WORK"
