#!/bin/bash
# Sweep alpha in {0, 0.25, 0.5} across ALL DAVIS 2016 trainval_movobj sequences.
# Per-sequence: UniMatch flow once, then bench_foels.py three times, then
# compute_iou.py against DAVIS Annotations. Final macro average across sequences.
set -u
ROOT=/home/mas/proj/study/foels
DAVIS_ROOT=$ROOT/foels/ext/unsupervised_detection/download/DAVIS2016/DAVIS
IMG_ROOT=$DAVIS_ROOT/JPEGImages/480p
ANNO_ROOT=$DAVIS_ROOT/Annotations/480p
SEQ_LIST=$ROOT/foels/ext/unsupervised_detection/data/trainval_movobj.txt
WORK=$ROOT/result/ijat_davis_all
mkdir -p $WORK

cd $ROOT
source .venv/bin/activate

# Unique sequence names from trainval_movobj.txt.
SEQUENCES=$(awk -F/ '{print $4}' $SEQ_LIST | sort -u)
NSEQ=$(echo "$SEQUENCES" | wc -l)
{
  echo "[INFO] $NSEQ sequences to sweep:"
  echo "$SEQUENCES" | nl
  echo "[INFO] Started: $(date -Iseconds)"
} | tee $WORK/header.log

for SEQ in $SEQUENCES; do
  SEQ_IMG=$IMG_ROOT/$SEQ
  SEQ_ANNO=$ANNO_ROOT/$SEQ
  SEQ_WORK=$WORK/$SEQ
  mkdir -p $SEQ_WORK
  FLOW=$SEQ_WORK/flow
  mkdir -p $FLOW

  if [ ! -d "$SEQ_IMG" ] || [ ! -d "$SEQ_ANNO" ]; then
    echo "[WARN] missing imgs or annos for $SEQ, skipping."
    continue
  fi
  NFR=$(ls $SEQ_IMG/*.jpg 2>/dev/null | wc -l)
  echo "===== $SEQ ($NFR frames) at $(date +%H:%M:%S) ====="

  # UniMatch flow once per sequence; alpha doesn't affect flow.
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

  for ALPHA in 0.0 0.25 0.5; do
    ARES=$SEQ_WORK/alpha_${ALPHA}
    ASEG=$SEQ_WORK/alpha_${ALPHA}_seg
    mkdir -p $ARES $ASEG
    python $ROOT/script/bench_foels.py \
      --config $ROOT/script/foels_param.yaml \
      --timing_csv $SEQ_WORK/timing_alpha_${ALPHA}.csv \
      --input_dir $SEQ_IMG \
      --flow_result_dir $FLOW \
      --segment_result_dir $ASEG \
      --result_dir $ARES \
      --movprob_lengthfactor_coeff $ALPHA \
      > $SEQ_WORK/run_alpha_${ALPHA}.log 2>&1
  done

  python $ROOT/script/compute_iou.py \
    --pred_root $SEQ_WORK \
    --gt_dir $SEQ_ANNO \
    --alphas 0.0 0.25 0.5 \
    --out $SEQ_WORK/iou_summary.txt
  echo "----- $SEQ IoU -----"
  cat $SEQ_WORK/iou_summary.txt
done

# Aggregate macro mean IoU per alpha across sequences.
python - <<'PY'
import csv, os, statistics
root = "/home/mas/proj/study/foels/result/ijat_davis_all"
agg = {"0.0": [], "0.25": [], "0.5": []}
per_seq = {}
for sd in sorted(os.listdir(root)):
    f = os.path.join(root, sd, "iou_summary.txt")
    if not os.path.isfile(f):
        continue
    per_seq.setdefault(sd, {})
    for row in csv.DictReader(open(f)):
        a = row["alpha"]
        if a in agg and row["IoU_mean"] not in ("nan", ""):
            v = float(row["IoU_mean"])
            agg[a].append(v)
            per_seq[sd][a] = v
print("===== Per-sequence mean IoU =====")
print(f"{'sequence':20s} {'α=0.00':>8s} {'α=0.25':>8s} {'α=0.50':>8s}")
for s in sorted(per_seq):
    row = per_seq[s]
    a0 = f"{row.get('0.0', float('nan')):.4f}" if '0.0' in row else "  nan "
    a1 = f"{row.get('0.25', float('nan')):.4f}" if '0.25' in row else "  nan "
    a2 = f"{row.get('0.5', float('nan')):.4f}" if '0.5' in row else "  nan "
    print(f"{s:20s} {a0:>8s} {a1:>8s} {a2:>8s}")
print()
print("===== Macro-average across sequences =====")
for a in ["0.0", "0.25", "0.5"]:
    v = agg[a]
    if v:
        std = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"  α={a:>4s}: mean IoU={statistics.mean(v):.4f}  std={std:.4f}  n_sequences={len(v)}")
PY

echo "[DONE] $(date -Iseconds)"
