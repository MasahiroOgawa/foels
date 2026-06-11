#!/bin/bash
# Sweep the length-factor formula F_l across the four variants
#   diff_log  (current code):  |log10(1 + |Δlength|)|
#   diff_lin              :    |Δlength|
#   ratio_log (paper claim):   |log10(d_l)|,   d_l = ||v_P|| / mean(||v_P,static||)
#   ratio_lin             :    |d_l - 1|
# on every sequence of DAVIS 2016 trainval_movobj.
#
# Reuses UniMatch flow, OneFormer segmentation, and the FoE state across the
# four runs per sequence so RANSAC runs once per sequence — the only thing that
# changes between runs is the per-pixel length-factor formula in
# comp_movpixprob().
set -u
ROOT=/home/mas/proj/study/foels
DAVIS_ROOT=$ROOT/foels/ext/unsupervised_detection/download/DAVIS2016/DAVIS
IMG_ROOT=$DAVIS_ROOT/JPEGImages/480p
ANNO_ROOT=$DAVIS_ROOT/Annotations/480p
SEQ_LIST=$ROOT/foels/ext/unsupervised_detection/data/trainval_movobj.txt
WORK=$ROOT/result/ijat_davis_fl_formula_sweep
mkdir -p $WORK

cd $ROOT
source .venv/bin/activate

SEQUENCES=$(awk -F/ '{print $4}' $SEQ_LIST | sort -u)
NSEQ=$(echo "$SEQUENCES" | wc -l)
{
  echo "[INFO] $NSEQ sequences to sweep:"
  echo "$SEQUENCES" | nl
  echo "[INFO] Started: $(date -Iseconds)"
} | tee $WORK/header.log

FORMULAS="diff_log diff_lin ratio_log ratio_lin"

for SEQ in $SEQUENCES; do
  SEQ_IMG=$IMG_ROOT/$SEQ
  SEQ_ANNO=$ANNO_ROOT/$SEQ
  SEQ_WORK=$WORK/$SEQ
  mkdir -p $SEQ_WORK
  FLOW=$SEQ_WORK/flow
  SEG=$SEQ_WORK/seg
  FOE_CACHE=$SEQ_WORK/foe_cache
  mkdir -p $FLOW $SEG $FOE_CACHE

  if [ ! -d "$SEQ_IMG" ] || [ ! -d "$SEQ_ANNO" ]; then
    echo "[WARN] missing imgs or annos for $SEQ, skipping."
    continue
  fi
  NFR=$(ls $SEQ_IMG/*.jpg 2>/dev/null | wc -l)
  echo "===== $SEQ ($NFR frames) at $(date +%H:%M:%S) ====="

  # UniMatch flow once per sequence.
  if [ -z "$(ls $FLOW/*_pred.flo 2>/dev/null)" ]; then
    python foels/ext/unimatch/main_flow.py \
      --inference_dir $SEQ_IMG \
      --output_path $FLOW \
      --resume foels/ext/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
      --inference_size 768 1024 \
      --padding_factor 32 --upsample_factor 4 --num_scales 2 \
      --attn_splits_list 2 8 --corr_radius_list -1 4 --prop_radius_list -1 1 \
      --reg_refine --num_reg_refine 6 --save_flo_flow \
      > $SEQ_WORK/flow.log 2>&1
  fi

  for F in $FORMULAS; do
    FRES=$SEQ_WORK/alpha_${F}
    mkdir -p $FRES
    python $ROOT/script/bench_foels.py \
      --config $ROOT/script/foels_param.yaml \
      --timing_csv $SEQ_WORK/timing_${F}.csv \
      --input_dir $SEQ_IMG \
      --flow_result_dir $FLOW \
      --segment_result_dir $SEG \
      --foe_cache_dir $FOE_CACHE \
      --result_dir $FRES \
      --fl_formula $F \
      > $SEQ_WORK/run_${F}.log 2>&1
  done

  python $ROOT/script/compute_iou.py \
    --pred_root $SEQ_WORK \
    --gt_dir $SEQ_ANNO \
    --alphas $FORMULAS \
    --out $SEQ_WORK/iou_summary.txt
  echo "----- $SEQ IoU -----"
  cat $SEQ_WORK/iou_summary.txt
done

# Aggregate macro mean IoU per formula across sequences.
python - <<'PY'
import csv, os, statistics
root = "/home/mas/proj/study/foels/result/ijat_davis_fl_formula_sweep"
formulas = ["diff_log", "diff_lin", "ratio_log", "ratio_lin"]
agg = {f: [] for f in formulas}
per_seq = {}
for sd in sorted(os.listdir(root)):
    f = os.path.join(root, sd, "iou_summary.txt")
    if not os.path.isfile(f):
        continue
    per_seq.setdefault(sd, {})
    for row in csv.DictReader(open(f)):
        key = row.get("alpha") or row.get("formula")
        if key in agg and row["IoU_mean"] not in ("nan", ""):
            v = float(row["IoU_mean"])
            agg[key].append(v)
            per_seq[sd][key] = v
print("===== Per-sequence mean IoU =====")
hdr = f"{'sequence':20s} " + " ".join(f"{f:>10s}" for f in formulas)
print(hdr)
for s in sorted(per_seq):
    row = per_seq[s]
    cells = " ".join(
        f"{row[f]:>10.4f}" if f in row else f"{'nan':>10s}" for f in formulas
    )
    print(f"{s:20s} {cells}")
print()
print("===== Macro-average across sequences =====")
for f in formulas:
    v = agg[f]
    if v:
        std = statistics.stdev(v) if len(v) > 1 else 0.0
        print(
            f"  {f:>10s}: mean IoU={statistics.mean(v):.4f}  std={std:.4f}  "
            f"n_sequences={len(v)}"
        )
PY

echo "[DONE] $(date -Iseconds)"
