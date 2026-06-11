"""Instrumented FoELS runner for IJAT reviewer response.

Monkey-patches process_image() to record per-stage timing and writes a
per-frame CSV. Invoked exactly like extract_moving_objects.py — extra arg:
  --timing_csv PATH   (where to dump the per-frame timing)

Stages timed:
  read_img:   cv2.imread of the current frame
  optflow:    optflow.compute (reads precomputed .flo file)
  seg:        seg.compute (OneFormer panoptic segmentation forward pass)
  foe:        foe.compute (RANSAC FoE estimation + length-factor likelihood)
  fuse:       moving_prob multiplication + thresholding
  obj:        _compute_moving_obj_mask (object-level refinement)
"""
import argparse
import csv
import os
import sys
import time
import yaml

# Make project root importable so 'import extract_moving_objects' resolves.
ROOT = "/home/mas/proj/study/foels"
sys.path.insert(0, os.path.join(ROOT, "foels"))

import cv2  # noqa: E402

import extract_moving_objects as emo  # noqa: E402
from focus_of_expansion import FoE  # noqa: E402


def _now():
    return time.perf_counter()


_TIMING_ROWS = []


def patched_process_image(self):
    t0 = _now()
    self.cur_img = cv2.imread(os.path.join(self.args.input_dir, self.cur_imgname))
    t_read = _now()

    self.optflow.compute(self.cur_imgname)
    t_flow = _now()

    self.seg.compute(self.cur_imgname)
    t_seg = _now()

    cached_state = None
    cache_path = None
    if self.foe_cache_dir:
        base = os.path.splitext(self.cur_imgname)[0]
        cache_path = os.path.join(self.foe_cache_dir, f"{base}.foe.json")
        cached_state = FoE.load_state_file(cache_path)
    self.foe.compute(
        self.optflow.flow,
        self.seg.sky_mask,
        self.seg.static_mask,
        cached_state=cached_state,
    )
    if cache_path is not None and cached_state is None:
        FoE.save_state_file(cache_path, self.foe.get_state())
    t_foe = _now()

    self.posterior_movpix_prob = self.seg.moving_prob * self.foe.moving_prob
    self.posterior_movpix_mask = (
        self.posterior_movpix_prob > self.args.thre_moving_prob
    )
    t_fuse = _now()

    self._compute_moving_obj_mask()
    t_obj = _now()

    _TIMING_ROWS.append(
        {
            "frame": self.cur_imgname,
            "read_img_ms": (t_read - t0) * 1000.0,
            "optflow_ms": (t_flow - t_read) * 1000.0,
            "seg_ms": (t_seg - t_flow) * 1000.0,
            "foe_ms": (t_foe - t_seg) * 1000.0,
            "fuse_ms": (t_fuse - t_foe) * 1000.0,
            "obj_ms": (t_obj - t_fuse) * 1000.0,
            "total_process_ms": (t_obj - t0) * 1000.0,
        }
    )


emo.MovingObjectExtractor.process_image = patched_process_image


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", required=True)
    pre.add_argument("--timing_csv", required=True)
    known, rest = pre.parse_known_args()
    with open(known.config) as f:
        cfg = yaml.safe_load(f).get("MovingObjectExtractor", {})

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--input_dir", default=cfg.get("input_dir"))
    p.add_argument("--flow_result_dir")
    p.add_argument("--segment_result_dir")
    p.add_argument("--result_dir", default=cfg.get("result_dir"))
    p.add_argument(
        "--segment_model_type", default=cfg.get("segment_model_type")
    )
    p.add_argument(
        "--segment_model_name", default=cfg.get("segment_model_name")
    )
    p.add_argument(
        "--segment_task_type", default=cfg.get("segment_task_type")
    )
    p.add_argument(
        "--result_img_suffix", default=cfg.get("result_img_suffix")
    )
    p.add_argument("--loglevel", type=int, default=cfg.get("loglevel", 1))
    p.add_argument("--resultimg_width", type=int, default=cfg.get("resultimg_width"))
    p.add_argument("--skip_frames", type=int, default=cfg.get("skip_frames", 0))
    p.add_argument(
        "--ransac_all_inlier_estimation",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=cfg.get("ransac_all_inlier_estimation"),
    )
    p.add_argument(
        "--foe_search_step", type=int, default=cfg.get("foe_search_step")
    )
    p.add_argument(
        "--num_ransac", type=int, default=cfg.get("num_ransac")
    )
    p.add_argument(
        "--thre_inlier_angle_deg",
        type=float,
        default=cfg.get("thre_inlier_angle_deg"),
    )
    p.add_argument(
        "--thre_inlier_rate", type=float, default=cfg.get("thre_inlier_rate")
    )
    p.add_argument(
        "--thre_flow_existing_rate",
        type=float,
        default=cfg.get("thre_flow_existing_rate"),
    )
    p.add_argument(
        "--thre_flowlength", type=float, default=cfg.get("thre_flowlength")
    )
    p.add_argument(
        "--thre_moving_fraction_in_obj",
        type=float,
        default=cfg.get("thre_moving_fraction_in_obj"),
    )
    p.add_argument(
        "--movprob_lengthfactor_coeff",
        type=float,
        default=cfg.get("movprob_lengthfactor_coeff"),
    )
    p.add_argument(
        "--middle_theta_deg", type=float, default=cfg.get("middle_theta_deg")
    )
    p.add_argument(
        "--thre_moving_prob", type=float, default=cfg.get("thre_moving_prob")
    )
    p.add_argument(
        "--thre_static_prob", type=float, default=cfg.get("thre_static_prob")
    )
    p.add_argument(
        "--flowarrow_step_forvis",
        type=int,
        default=cfg.get("flowarrow_step_forvis"),
    )
    p.add_argument(
        "--flowlength_factor_forvis",
        type=int,
        default=cfg.get("flowlength_factor_forvis"),
    )
    p.add_argument(
        "--fl_formula",
        type=str,
        default=cfg.get("fl_formula", "diff_log"),
        choices=("diff_log", "diff_lin", "ratio_log", "ratio_lin"),
    )
    p.add_argument(
        "--foe_cache_dir",
        type=str,
        default=cfg.get("foe_cache_dir"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    moe = emo.MovingObjectExtractor(args)
    moe.compute()

    keys = [
        "frame",
        "read_img_ms",
        "optflow_ms",
        "seg_ms",
        "foe_ms",
        "fuse_ms",
        "obj_ms",
        "total_process_ms",
    ]
    with open(args.timing_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in _TIMING_ROWS:
            w.writerow(row)
    print(f"[BENCH] wrote {len(_TIMING_ROWS)} rows to {args.timing_csv}")

    # Also print a summary on stdout.
    if _TIMING_ROWS:
        n = len(_TIMING_ROWS)
        print(f"\n[BENCH] per-stage mean ms over {n} frames:")
        for k in keys[1:]:
            mean = sum(r[k] for r in _TIMING_ROWS) / n
            print(f"  {k}: {mean:.2f}")


if __name__ == "__main__":
    main()
