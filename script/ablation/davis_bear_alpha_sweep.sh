#!/bin/bash
# Download DAVIS2016, run FoELS on the bear sequence for alpha in {0,0.25,0.5},
# and compute IoU vs DAVIS Annotations.
set -u
ROOT=/home/mas/proj/study/foels
DAVIS_BASE=$ROOT/foels/ext/unsupervised_detection/download/DAVIS2016
DAVIS_ROOT=$DAVIS_BASE/DAVIS
ZIP_URL=https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
WORK=$ROOT/result/ijat_davis_bear
mkdir -p $WORK

cd $ROOT
source .venv/bin/activate

# 1. Download + extract DAVIS-data.zip if not already extracted.
if [ ! -d "$DAVIS_ROOT/JPEGImages/480p/bear" ]; then
  echo "[INFO] DAVIS bear sequence not found; downloading DAVIS-data.zip..."
  mkdir -p $DAVIS_BASE
  ZIP=$DAVIS_BASE/DAVIS-data.zip
  if [ ! -f "$ZIP" ]; then
    wget -c -O $ZIP $ZIP_URL 2>&1 | tail -5
  fi
  echo "[INFO] extracting..."
  unzip -q -o $ZIP -d $DAVIS_BASE
fi

BEAR_IMG=$DAVIS_ROOT/JPEGImages/480p/bear
BEAR_ANNO=$DAVIS_ROOT/Annotations/480p/bear

if [ ! -d "$BEAR_IMG" ]; then
  echo "[ERROR] bear images not found at $BEAR_IMG"; exit 1
fi
NFRAMES=$(ls $BEAR_IMG/*.jpg 2>/dev/null | wc -l)
echo "[INFO] bear sequence: $NFRAMES frames at $BEAR_IMG"

# 2. Hardware info.
{
  echo "===== HARDWARE ====="
  echo "Host: $(hostname)"
  echo "Date: $(date -Iseconds)"
  lscpu | grep -E "Model name"
  free -h | head -2
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "Bear frames: $NFRAMES"
} > $WORK/hardware.txt

# 3. Run UniMatch optical flow once on bear (shared across all alpha).
FLOW_RES=$WORK/flow
mkdir -p $FLOW_RES
if [ -z "$(ls $FLOW_RES/*_pred.flo 2>/dev/null)" ]; then
  echo "[INFO] computing optical flow on bear..."
  python foels/ext/unimatch/main_flow.py \
    --inference_dir $BEAR_IMG \
    --output_path $FLOW_RES \
    --resume foels/ext/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
    --inference_size 768 1024 \
    --padding_factor 32 --upsample_factor 4 --num_scales 2 \
    --attn_splits_list 2 8 --corr_radius_list -1 4 --prop_radius_list -1 1 \
    --reg_refine --num_reg_refine 6 --save_flo_flow \
    > $WORK/flow.log 2>&1
  echo "[INFO] flow done."
fi

# 4. For each alpha, run FoELS and compute IoU vs DAVIS bear Annotations.
for ALPHA in 0.0 0.25 0.5; do
  ARES=$WORK/alpha_${ALPHA}
  ASEG=$WORK/alpha_${ALPHA}_seg
  mkdir -p $ARES $ASEG
  echo "===== ALPHA=$ALPHA ====="
  python $ROOT/script/bench_foels.py \
    --config $ROOT/script/foels_param.yaml \
    --timing_csv $WORK/timing_alpha_${ALPHA}.csv \
    --input_dir $BEAR_IMG \
    --flow_result_dir $FLOW_RES \
    --segment_result_dir $ASEG \
    --result_dir $ARES \
    --movprob_lengthfactor_coeff $ALPHA \
    > $WORK/run_alpha_${ALPHA}.log 2>&1
  echo "[INFO] alpha=$ALPHA run done; computing IoU..."
done

# 5. IoU computation in numpy.
python $ROOT/script/compute_iou.py \
  --pred_root $WORK \
  --gt_dir $BEAR_ANNO \
  --alphas 0.0 0.25 0.5 \
  --out $WORK/iou_summary.txt

echo "[DONE] $(date -Iseconds)"
echo "Summary at $WORK/iou_summary.txt"
cat $WORK/iou_summary.txt
