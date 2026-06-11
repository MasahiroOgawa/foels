"""Compute mean IoU per alpha between FoELS predicted masks and DAVIS ground truth.

Predicted masks (from extract_moving_objects.py) are written as
  <pred_root>/alpha_<ALPHA>/<NAME>_mask.png
where NAME matches the input frame stem (e.g. 00000).  Values are 0 or 255.

Ground-truth masks (DAVIS 2016) live at
  <gt_dir>/<NAME>.png
where moving-object pixels have value > 0 (typically 255 for a single
object, or per-instance IDs).  We binarise GT with > 0 to get the
moving-vs-static mask.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np


def iou(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0  # both empty -> perfect
    return inter / union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--alphas", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt_paths = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(args.gt_dir, "*.png"))
    }
    if not gt_paths:
        print(f"[ERROR] no GT png in {args.gt_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] {len(gt_paths)} GT frames in {args.gt_dir}")

    lines = []
    lines.append(
        "alpha,N_matched,IoU_mean,IoU_std,IoU_min,IoU_max,J_mean(==IoU)"
    )
    for alpha in args.alphas:
        pred_dir = os.path.join(args.pred_root, f"alpha_{alpha}")
        pred_paths = {
            os.path.basename(p)[: -len("_mask.png")]: p
            for p in glob.glob(os.path.join(pred_dir, "*_mask.png"))
        }
        ious = []
        for stem, gtp in sorted(gt_paths.items()):
            if stem not in pred_paths:
                continue  # FoELS skips the last frame (needs t+1 flow)
            gt = cv2.imread(gtp, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_paths[stem], cv2.IMREAD_GRAYSCALE)
            if gt is None or pred is None:
                continue
            if gt.shape != pred.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            ious.append(iou(pred > 127, gt > 0))
        if not ious:
            lines.append(f"{alpha},0,nan,nan,nan,nan,nan")
            continue
        arr = np.asarray(ious)
        lines.append(
            f"{alpha},{len(arr)},{arr.mean():.4f},{arr.std():.4f},{arr.min():.4f},{arr.max():.4f},{arr.mean():.4f}"
        )

    text = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
