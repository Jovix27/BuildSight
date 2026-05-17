#!/usr/bin/env python3
"""
BuildSight Ensemble Pipeline
=============================
Primary layer  : YOLOv11 + YOLOv26 inference + WBF fusion
                 (ported from ensemble_inference.py.bak — 2026-04-10)
Post-processing: Scene-aware preprocessing, condition-tuned WBF profiles,
                 crowded-worker recovery, temporal PPE tracker
                 (ported from site_aware_ensemble.py.bak — 2026-04-11)

Architecture
------------
  Frame
    │
    ├─► detect_condition()          — classify S1/S2/S3/S4 from image stats
    │
    ├─► preprocess_frame()          — CLAHE / gamma / sharpening per condition
    │
    ├─► model_v11.predict()  ─┐
    ├─► model_v26.predict()  ─┴─► wbf_fuse()
    │                                │
    │       Geometric guardrails ◄───┘   (aspect ratio, size, portrait check)
    │
    ├─► wbf_fuse_condition()        — condition-tuned second-pass fusion
    │
    ├─► recover_crowded_workers()   — S4 recovery with spatial cooldown
    │
    └─► TemporalPPEFilter.update()  — ghost suppression, static filter, PPE inference

Usage
-----
  from scripts.buildsight_ensemble import EnsemblePipeline

  pipeline = EnsemblePipeline()
  result   = pipeline.run(frame_bgr, condition='auto')
  # result → list of dicts: {class, confidence, box, has_helmet, has_vest}
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Model weight paths ─────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
WEIGHTS_DIR = _HERE / "weights"

MODEL_V11_PATH = WEIGHTS_DIR / "yolov11_buildsight_best.pt"
MODEL_V26_PATH = WEIGHTS_DIR / "yolov26_buildsight_best.pt"


# SASTRA 3-class schema
CLS_HELMET = 0
CLS_VEST   = 1
CLS_WORKER = 2
CLS_NAMES  = {CLS_HELMET: "helmet", CLS_VEST: "safety_vest", CLS_WORKER: "worker"}

# Model ensemble weights  (v11 slightly stronger on Indian CCTV)
MODEL_WEIGHTS = [0.55, 0.45]   # [v11_weight, v26_weight]

# ── Primary layer: inference config ───────────────────────────────────────────
PRE_CONF = 0.45      # pre-NMS confidence sent to each model
WBF_IOU  = 0.50      # WBF box-matching IoU threshold (primary pass)

# Post-WBF per-class confidence gates (original Toni/Jovi values)
CLS_THR = {
    CLS_HELMET: 0.45,
    CLS_VEST:   0.50,
    CLS_WORKER: 0.72,   # High threshold — forces v11+v26 consensus on workers
}

# ── Post-processing: condition profiles ───────────────────────────────────────
_PROFILES: Dict[str, Dict] = {
    "S1_normal": {
        "pre_conf":       0.10,
        "wbf_iou":        {CLS_HELMET: 0.45, CLS_VEST: 0.50, CLS_WORKER: 0.55},
        "post_gate":      {CLS_HELMET: 0.22, CLS_VEST: 0.24, CLS_WORKER: 0.28},
        "recover_worker": 0.0,
    },
    "S2_dusty": {
        "pre_conf":       0.08,
        "wbf_iou":        {CLS_HELMET: 0.42, CLS_VEST: 0.48, CLS_WORKER: 0.52},
        "post_gate":      {CLS_HELMET: 0.18, CLS_VEST: 0.20, CLS_WORKER: 0.20},
        "recover_worker": 0.0,
    },
    "S3_low_light": {
        "pre_conf":       0.08,
        "wbf_iou":        {CLS_HELMET: 0.40, CLS_VEST: 0.46, CLS_WORKER: 0.50},
        "post_gate":      {CLS_HELMET: 0.16, CLS_VEST: 0.18, CLS_WORKER: 0.18},
        "recover_worker": 0.0,
    },
    "S4_crowded": {
        # RECALL FIX 2026-04-11:
        # wbf_iou[WORKER] 0.40→0.28  — lower IoU merge threshold so two adjacent
        #   workers close in frame space are NOT fused into one box.
        # post_gate[WORKER] 0.20→0.14 — allow weaker fused worker detections to
        #   survive; both models weakly agreeing at 0.15-0.16 gives fused ~0.15+bonus.
        # post_gate[HELMET] 0.18→0.13 — small/distant helmet recall.
        # post_gate[VEST]   0.22→0.14 — partially-occluded vest recall.
        # recover_worker 0.32→0.20    — candidates from raw model output only need
        #   0.20 to be considered; tighter survival conditions still gate quality.
        "pre_conf":       0.07,
        "wbf_iou":        {CLS_HELMET: 0.35, CLS_VEST: 0.42, CLS_WORKER: 0.28},
        "post_gate":      {CLS_HELMET: 0.13, CLS_VEST: 0.14, CLS_WORKER: 0.14},
        "recover_worker": 0.20,
    },
}

# ── Spatial cooldown for crowded-worker recovery ───────────────────────────────
_recovery_cooldown: Dict[str, int] = {}
_recovery_frame: int = 0
# Lowered 4→2: allow re-triggering in the same grid cell every 2 frames so
# newly-arrived workers (elevated wall, newly visible) get recovered promptly.
RECOVERY_COOLDOWN_FRAMES = 2


# ══════════════════════════════════════════════════════════════════════════════
#  Scene classification
# ══════════════════════════════════════════════════════════════════════════════

def detect_condition(frame: np.ndarray, rough_worker_boxes: Optional[List[List[float]]] = None) -> str:
    """Classify frame as S1_normal / S2_dusty / S3_low_light / S4_crowded."""
    if rough_worker_boxes is None:
        rough_worker_boxes = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    brightness   = float(np.mean(gray))
    contrast     = float(np.std(gray))
    saturation   = float(np.mean(hsv[:, :, 1]))
    worker_count = len(rough_worker_boxes)
    crowd_overlap = _mean_pairwise_iou(rough_worker_boxes)

    if brightness < 72:
        return "S3_low_light"
    # brightness < 160 gate: dusty/hazy scenes are DIM. Without this gate,
    # overexposed frames (bright blown-out background) match the same low-contrast
    # + low-saturation profile and get CLAHE applied, amplifying the overexposure.
    if contrast < 52 and saturation < 78 and brightness < 160:
        return "S2_dusty"
    if worker_count >= 3 or crowd_overlap >= 0.05:
        return "S4_crowded"
    return "S1_normal"


def preprocess_frame(frame: np.ndarray, condition: str) -> np.ndarray:
    """Apply CLAHE / gamma correction / unsharp mask per site condition."""
    out = frame.copy()

    if condition in {"S2_dusty", "S3_low_light"}:
        clip = 3.0 if condition == "S2_dusty" else 4.0
        lab  = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l   = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if condition == "S3_low_light":
        gamma = 1.7
        lut   = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        out   = cv2.LUT(out, lut)
    elif condition == "S2_dusty":
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        out  = cv2.addWeighted(out, 1.35, blur, -0.35, 0)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  IoU helpers
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a: List[float], b: List[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1e-6)


def _mean_pairwise_iou(boxes: List[List[float]]) -> float:
    if len(boxes) < 2:
        return 0.0
    pairs = [_iou(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i+1, len(boxes))]
    return float(np.mean(pairs))


# ══════════════════════════════════════════════════════════════════════════════
#  Primary layer — WBF fusion with geometric guardrails
# ══════════════════════════════════════════════════════════════════════════════

def wbf_fuse(
    preds: List[Tuple[List, List, List]],
    img_w: int,
    img_h: int,
    weights: List[float] = MODEL_WEIGHTS,
    iou_thr: float = WBF_IOU,
) -> List[Dict]:
    """
    Primary WBF pass.
    preds: list of (boxes_xyxy, scores, labels) — one entry per model.
           Boxes are in pixel coords relative to img_w / img_h.
    Returns list of dicts: {box, score, cls, n_models}
    """
    # Normalise to [0,1]
    flat: List[Tuple[List, float, int, int]] = []
    for mi, (boxes, scores, labels) in enumerate(preds):
        for box, score, label in zip(boxes, scores, labels):
            norm = [box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h]
            flat.append((norm, float(score), int(label), mi))

    flat.sort(key=lambda x: -x[1])
    used   = [False] * len(flat)
    fused: List[Dict] = []

    for i in range(len(flat)):
        if used[i]:
            continue
        b_i, s_i, l_i, m_i = flat[i]
        cluster_b = [b_i]
        cluster_s = [s_i]
        cluster_m = [m_i]
        used[i]   = True

        for j in range(i + 1, len(flat)):
            if used[j] or flat[j][2] != l_i:
                continue
            if any(_iou(flat[j][0], b) > iou_thr for b in cluster_b):
                cluster_b.append(flat[j][0])
                cluster_s.append(flat[j][1])
                cluster_m.append(flat[j][3])
                used[j] = True

        rw        = np.array([MODEL_WEIGHTS[m] * s for m, s in zip(cluster_m, cluster_s)], dtype=np.float32)
        ws        = rw / max(rw.sum(), 1e-6)
        fused_s   = float(np.average(cluster_s, weights=ws))
        n_models  = len(set(cluster_m))

        if fused_s < CLS_THR.get(l_i, 0.30):
            continue

        box = np.average(cluster_b, axis=0, weights=ws).tolist()

        # ── Geometric guardrails (original Jovi / Toni calibration) ─────────
        bx_w   = box[2] - box[0]
        bx_h   = box[3] - box[1]
        aspect = bx_w / max(bx_h, 1e-6)

        if l_i == CLS_WORKER:
            # Workers are tall: discard landscape boxes (excavators, vehicles)
            if aspect > 1.0:
                continue
            # Webcam close-ups: a person can fill 30-70% of the frame (area ~0.35).
            # Only discard whole-frame false positives (>80%) and sub-pixel noise.
            if bx_w * bx_h < 0.003 or bx_w * bx_h > 0.80:
                continue

        fused.append({
            "box":      [box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h],
            "score":    min(0.99, fused_s),
            "cls":      l_i,
            "n_models": n_models,
        })

    return fused


# ══════════════════════════════════════════════════════════════════════════════
#  Post-processing layer — condition-aware second-pass WBF + worker recovery
# ══════════════════════════════════════════════════════════════════════════════

def wbf_fuse_condition(
    primary_dets: List[Dict],
    all_preds: List[Tuple[List, List, List]],
    img_w: int,
    img_h: int,
    condition: str,
) -> List[Dict]:
    """
    Second-pass WBF using condition-specific IoU and post-gate thresholds.
    Runs on top of primary_dets, not raw model outputs — refines the fusion.
    """
    profile = _PROFILES[condition]
    flat: List[Tuple[List, float, int, int]] = []

    for mi, (boxes, scores, labels) in enumerate(all_preds):
        for box, score, label in zip(boxes, scores, labels):
            norm = [box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h]
            flat.append((norm, float(score), int(label), mi))

    refined: List[Dict] = []
    for cls_id in (CLS_HELMET, CLS_VEST, CLS_WORKER):
        cls_flat = [(b, s, mi) for b, s, l, mi in flat if l == cls_id]
        if not cls_flat:
            continue
        cls_flat.sort(key=lambda x: -x[1])
        used      = [False] * len(cls_flat)
        iou_thresh = profile["wbf_iou"][cls_id]

        for i in range(len(cls_flat)):
            if used[i]:
                continue
            cluster_b = [cls_flat[i][0]]
            cluster_s = [cls_flat[i][1]]
            cluster_m = [cls_flat[i][2]]
            used[i]   = True

            for j in range(i + 1, len(cls_flat)):
                if used[j]:
                    continue
                if any(_iou(cls_flat[j][0], b) >= iou_thresh for b in cluster_b):
                    cluster_b.append(cls_flat[j][0])
                    cluster_s.append(cls_flat[j][1])
                    cluster_m.append(cls_flat[j][2])
                    used[j] = True

            rw       = np.array([MODEL_WEIGHTS[mi] * s for mi, s in zip(cluster_m, cluster_s)], dtype=np.float32)
            nw       = rw / max(rw.sum(), 1e-6)
            bonus    = 0.03 * max(0, len(set(cluster_m)) - 1)
            fused_s  = float(np.average(cluster_s, weights=nw) + bonus)

            if fused_s < profile["post_gate"][cls_id]:
                continue

            fb = np.average(cluster_b, axis=0, weights=nw)
            # Apply same geometric guardrails as primary WBF pass (fix: primary_dets
            # was passed in but never used, so guardrails were silently bypassed)
            if cls_id == CLS_WORKER:
                bx_w = fb[2] - fb[0]
                bx_h = fb[3] - fb[1]
                if bx_h > 0 and (bx_w / bx_h) > 1.0:
                    continue
                if bx_w * bx_h < 0.003 or bx_w * bx_h > 0.80:
                    continue

            refined.append({
                "box":      [fb[0]*img_w, fb[1]*img_h, fb[2]*img_w, fb[3]*img_h],
                "score":    min(0.99, fused_s),
                "cls":      cls_id,
                "n_models": len(set(cluster_m)),
            })

    if condition == "S4_crowded":
        _advance_recovery_frame()
        refined = _recover_crowded_workers(
            refined, all_preds, profile["recover_worker"], img_h=img_h
        )

    return refined


def _advance_recovery_frame() -> None:
    global _recovery_frame
    _recovery_frame += 1
    # Trim stale entries every 100 frames to prevent unbounded dict growth
    if _recovery_frame % 100 == 0:
        cutoff = _recovery_frame - RECOVERY_COOLDOWN_FRAMES * 10
        stale = [k for k, v in _recovery_cooldown.items() if v < cutoff]
        for k in stale:
            del _recovery_cooldown[k]


def _grid_cell(box: List[float], cell_size: int = 80) -> str:
    cx = int((box[0] + box[2]) / 2.0 / cell_size)
    cy = int((box[1] + box[3]) / 2.0 / cell_size)
    return f"{cx},{cy}"


def _recover_crowded_workers(
    fused: List[Dict],
    all_preds: List[Tuple[List, List, List]],
    min_score: float,
    img_h: int = 0,
) -> List[Dict]:
    """
    Recover workers that WBF missed in crowded scenes.

    RECALL FIX 2026-04-11 changes:
    - min_score floor lowered to 0.20 (profile driven)
    - Duplicate IoU lowered 0.65→0.50 — genuinely different nearby workers
      have IoU 0.10-0.45; re-merging at 0.65 was too aggressive.
    - Height floor lowered 24→14px — elevated wall workers seen from below
      appear as very small boxes at camera distance.
    - Aspect filter widened 0.90→1.30 — workers on walls can appear slightly
      wider than tall from below-camera angles.
    - Survival conditions widened:
        score ≥ 0.56  →  score ≥ 0.36
        neighbor + score ≥ 0.52  →  neighbor + score ≥ 0.28
    - New path: elevated-zone recovery — workers in the top 40% of frame
      with score ≥ 0.25 are recovered even without PPE/neighbors (they are
      almost never construction materials at that height).
    - PPE search expanded: also checks ±25% horizontal margin around worker
      box so PPE that slightly overlaps but isn't fully inside still counts.
    """
    if min_score <= 0:
        return fused

    workers   = [d for d in fused if d["cls"] == CLS_WORKER]
    ppe       = [d for d in fused if d["cls"] in (CLS_HELMET, CLS_VEST)]
    recovered: List[Dict] = []

    candidates: List[Dict] = []
    for boxes, scores, labels in all_preds:
        for box, score, label in zip(boxes, scores, labels):
            if int(label) == CLS_WORKER and float(score) >= min_score:
                candidates.append({
                    "box": list(box), "score": float(score),
                    "cls": CLS_WORKER, "n_models": 1,
                })

    for cand in sorted(candidates, key=lambda x: -x["score"]):
        # Skip if this box substantially overlaps an already-kept worker
        if any(_iou(cand["box"], e["box"]) >= 0.50 for e in workers + recovered):
            continue

        bw  = cand["box"][2] - cand["box"][0]
        bh  = cand["box"][3] - cand["box"][1]
        asp = bw / max(bh, 1.0)

        # Geometry gates — reject clearly non-human shapes
        if bh < 14:          # too tiny to be a real worker
            continue
        if asp > 1.30:       # wider-than-tall boxes are scaffolding/planks
            continue

        # ── PPE support: check interior AND a 25% horizontal margin ─────────
        cx1_exp = cand["box"][0] - bw * 0.25
        cx2_exp = cand["box"][2] + bw * 0.25
        ppe_support = sum(
            1 for d in ppe
            if cx1_exp <= (d["box"][0]+d["box"][2])/2 <= cx2_exp
            and cand["box"][1] <= (d["box"][1]+d["box"][3])/2 <= cand["box"][3]
        )

        neighbor_support = sum(
            1 for w in workers if _iou(cand["box"], w["box"]) >= 0.02
        )

        # ── Elevated-zone path — top 40% of frame ───────────────────────────
        # Workers standing on elevated walls/platforms appear in the upper
        # portion of frame. Construction materials (bags, bricks) are almost
        # never placed at height. Lower threshold significantly.
        is_elevated = (img_h > 0 and cand["box"][1] < img_h * 0.40)

        # Survival decision
        survives = (
            ppe_support > 0                                    # has nearby PPE
            or (neighbor_support > 0 and cand["score"] >= 0.28)  # near workers, moderate conf
            or cand["score"] >= 0.36                           # strong standalone detection
            or (is_elevated and cand["score"] >= 0.25)        # elevated zone — very permissive
        )

        if survives:
            cell       = _grid_cell(cand["box"])
            last_fired = _recovery_cooldown.get(cell, -999)
            if (_recovery_frame - last_fired) < RECOVERY_COOLDOWN_FRAMES:
                continue
            _recovery_cooldown[cell] = _recovery_frame
            recovered.append(cand)

    return fused + recovered


# ══════════════════════════════════════════════════════════════════════════════
#  Temporal PPE tracker
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _WorkerTrack:
    track_id:        int
    bbox:            List[float]
    hits:            int   = 0
    misses:          int   = 0
    human_score_ema: float = 0.0
    helmet_streak:   int   = 0
    vest_streak:     int   = 0
    last_frame:      int   = -1
    position_history: List[List[float]] = field(default_factory=list)

    def centre(self) -> List[float]:
        return [(self.bbox[0]+self.bbox[2])/2.0, (self.bbox[1]+self.bbox[3])/2.0]

    def is_static(self, min_hits: int = 6, max_var_px: float = 3.0) -> bool:
        if self.hits < min_hits or len(self.position_history) < min_hits:
            return False
        recent  = self.position_history[-min_hits:]
        xs, ys  = [p[0] for p in recent], [p[1] for p in recent]
        return float(np.var(xs) + np.var(ys)) < (max_var_px ** 2)


def _worker_has_ppe(worker_box: List[float], ppe: List[Dict], cls_id: int) -> bool:
    for d in ppe:
        if d["cls"] != cls_id:
            continue
        cx = (d["box"][0] + d["box"][2]) / 2.0
        cy = (d["box"][1] + d["box"][3]) / 2.0
        if worker_box[0] <= cx <= worker_box[2] and worker_box[1] <= cy <= worker_box[3]:
            return True
    return False


def _synthetic_ppe(worker_box: List[float], cls_id: int, score: float) -> Optional[Dict]:
    x1, y1, x2, y2 = worker_box
    w, h = x2 - x1, y2 - y1
    if cls_id == CLS_HELMET:
        return {"box": [x1+0.28*w, max(0.0, y1-0.04*h), x1+0.72*w, y1+0.22*h], "score": score, "cls": cls_id, "synthetic": True, "n_models": 1}
    if cls_id == CLS_VEST:
        return {"box": [x1+0.18*w, y1+0.18*h, x1+0.82*w, y1+0.62*h], "score": score, "cls": cls_id, "synthetic": True, "n_models": 1}
    return None


class TemporalPPEFilter:
    """Ghost suppression + static-object filter + PPE inference from streaks."""

    def __init__(self, max_misses: int = 6):
        self.max_misses   = max_misses
        self._next_id     = 1
        self.tracks: Dict[int, _WorkerTrack] = {}

    def update(self, dets: List[Dict], frame_index: int) -> List[Dict]:
        workers = [d for d in dets if d["cls"] == CLS_WORKER]
        ppe     = [d for d in dets if d["cls"] in (CLS_HELMET, CLS_VEST)]

        # ── Track assignment ─────────────────────────────────────────────────
        assignments: Dict[int, int] = {}
        unmatched   = set(self.tracks)

        for wi, w in enumerate(workers):
            best_iou, best_id = 0.0, None
            for tid, t in self.tracks.items():
                ov = _iou(w["box"], t.bbox)
                if ov > 0.30 and ov > best_iou:
                    best_iou, best_id = ov, tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.tracks[best_id] = _WorkerTrack(track_id=best_id, bbox=list(w["box"]))
            assignments[wi] = best_id
            unmatched.discard(best_id)

        for tid in unmatched:
            self.tracks[tid].misses += 1

        # ── Update matched tracks ────────────────────────────────────────────
        for wi, w in enumerate(workers):
            t = self.tracks[assignments[wi]]
            t.hits      += 1
            t.misses     = 0
            t.last_frame = frame_index
            t.bbox       = list(w["box"])

            hs = float(w.get("human_score", w["score"]))
            t.human_score_ema = (0.6 * t.human_score_ema + 0.4 * hs) if t.hits > 1 else hs
            w["track_id"] = t.track_id

            cx = (t.bbox[0] + t.bbox[2]) / 2.0
            cy = (t.bbox[1] + t.bbox[3]) / 2.0
            t.position_history.append([cx, cy])
            if len(t.position_history) > 20:
                t.position_history.pop(0)

            has_h = _worker_has_ppe(w["box"], ppe, CLS_HELMET)
            has_v = _worker_has_ppe(w["box"], ppe, CLS_VEST)
            t.helmet_streak = min(6, t.helmet_streak + 1) if has_h else max(0, t.helmet_streak - 1)
            t.vest_streak   = min(6, t.vest_streak   + 1) if has_v else max(0, t.vest_streak   - 1)

            # Infer PPE from streak — prevents flicker on occluded helmets/vests.
            # Fix: set flags directly on worker instead of generating synthetic
            # boxes that contaminate the shared PPE pool and cause false compliance
            # on adjacent workers.
            if not has_h and t.helmet_streak >= 2:
                w["_inferred_helmet"] = True
            if not has_v and t.vest_streak >= 2:
                w["_inferred_vest"] = True

        # ── Prune stale tracks ───────────────────────────────────────────────
        for tid in [tid for tid, t in self.tracks.items() if t.misses > self.max_misses]:
            del self.tracks[tid]

        # ── Filter: ghost / static / low-human-score suppression ────────────
        valid_workers: List[Dict] = []
        for w in workers:
            t = self.tracks.get(w.get("track_id"))
            if t is None:
                valid_workers.append(w)
                continue

            # Ghost suppression: require 2 consecutive frames before emitting.
            # RECALL FIX: lowered score bypass 0.72→0.45 so workers that both
            # models agree on (fused score ~0.35-0.44) aren't invisible on frame 1.
            has_ppe = t.helmet_streak > 0 or t.vest_streak > 0
            # Ghost suppression: bypass if both models agree (n_models >= 2) —
            # dual-model consensus is strong evidence even on the first frame.
            if t.hits < 2 and not has_ppe and w["score"] < 0.45 and w.get("n_models", 1) < 2:
                continue

            # Human-score gate: relaxed EMA floor 0.40→0.28 for S4 —
            # distant/elevated workers have genuinely lower human-like scores.
            if (t.hits >= 3 and t.human_score_ema < 0.28
                    and t.helmet_streak == 0 and t.vest_streak == 0):
                continue

            # Static-object gate (cement bags, scaffolding, machinery).
            # RECALL FIX: raised score bypass 0.80→0.60 so static workers
            # (workers standing still while working) are not suppressed.
            if (t.is_static() and w["score"] < 0.60
                    and t.helmet_streak == 0 and t.vest_streak == 0):
                continue

            valid_workers.append(w)

        out = valid_workers + ppe
        out.sort(key=lambda d: (d["cls"], -d["score"]))
        return out

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1


# ══════════════════════════════════════════════════════════════════════════════
#  PPE association helper
# ══════════════════════════════════════════════════════════════════════════════

def associate_ppe(dets: List[Dict]) -> List[Dict]:
    """Tag each worker with has_helmet / has_vest based on centroid containment
    and streak-inferred flags set by TemporalPPEFilter."""
    workers = [d for d in dets if d["cls"] == CLS_WORKER]
    ppe     = [d for d in dets if d["cls"] in (CLS_HELMET, CLS_VEST)]

    for w in workers:
        w["has_helmet"] = w.pop("_inferred_helmet", False) or _worker_has_ppe(w["box"], ppe, CLS_HELMET)
        w["has_vest"]   = w.pop("_inferred_vest",   False) or _worker_has_ppe(w["box"], ppe, CLS_VEST)

    return dets


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline class
# ══════════════════════════════════════════════════════════════════════════════

class EnsemblePipeline:
    """
    Full two-layer ensemble pipeline.

    Parameters
    ----------
    device      : 'cuda:0' | 'cpu' | 'auto'
    use_half    : FP16 inference (GPU only)
    condition   : site condition key, or 'auto' (default) to classify per frame
    """

    def __init__(
        self,
        device:   str  = "auto",
        use_half: bool = True,
        condition: str = "auto",
        model_v11=None,
        model_v26=None,
    ) -> None:
        from ultralytics import YOLO
        import torch

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device    = device
        self.use_half  = use_half and device.startswith("cuda")
        self.condition = condition

        if model_v11 is not None:
            self.model_v11 = model_v11
        else:
            print(f"[EnsemblePipeline] Loading YOLOv11 from {MODEL_V11_PATH} ...")
            self.model_v11 = YOLO(str(MODEL_V11_PATH))

        if model_v26 is not None:
            self.model_v26 = model_v26
        else:
            print(f"[EnsemblePipeline] Loading YOLOv26 from {MODEL_V26_PATH} ...")
            self.model_v26 = YOLO(str(MODEL_V26_PATH))

        self.temporal  = TemporalPPEFilter(max_misses=6)
        self._frame_idx = 0

        # Warm-up pass to avoid first-frame latency spike
        if device.startswith("cuda") and model_v11 is None:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            for m in (self.model_v11, self.model_v26):
                m.predict(dummy, device=device, verbose=False, conf=PRE_CONF,
                          half=self.use_half)
            print("[EnsemblePipeline] GPU warm-up done.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        frame_bgr:  np.ndarray,
        condition:  str = "auto",
        reset:      bool = False,
    ) -> Dict:
        """
        Run the full pipeline on one BGR frame.

        Returns
        -------
        {
            detections     : list of {class, confidence, box, has_helmet, has_vest},
            class_counts   : {class_name: count},
            condition      : S1_normal | S2_dusty | S3_low_light | S4_crowded,
            scene_reason   : human-readable reason string,
            elapsed_ms     : float,
        }
        """
        import time
        t0 = time.perf_counter()

        if reset:
            global _recovery_cooldown, _recovery_frame
            self.temporal.reset()
            self._frame_idx = 0
            _recovery_cooldown.clear()
            _recovery_frame = 0

        self._frame_idx += 1

        # ── Step 1: scene classification ──────────────────────────────────────
        use_condition = condition if condition != "auto" else self.condition
        if use_condition == "auto":
            # Use previous-frame worker bboxes for crowd detection (1-frame lag,
            # far better than always passing [] which made S4_crowded unreachable)
            prev_workers = [list(t.bbox) for t in self.temporal.tracks.values() if t.misses == 0]
            use_condition = detect_condition(frame_bgr, prev_workers)

        # ── Step 2: condition-aware preprocessing ─────────────────────────────
        processed = preprocess_frame(frame_bgr, use_condition)
        h, w      = processed.shape[:2]

        # ── Step 3: run both models ────────────────────────────────────────────
        # Use condition-profile pre_conf (0.07-0.10), NOT the module-level
        # PRE_CONF=0.45. High pre_conf silently kills 40-50% confidence
        # detections before WBF fusion ever runs.
        profile      = _PROFILES.get(use_condition, _PROFILES["S1_normal"])
        pre_conf_eff = profile.get("pre_conf", PRE_CONF)
        raw_preds = []
        for model in (self.model_v11, self.model_v26):
            r = model.predict(
                processed,
                device=self.device,
                verbose=False,
                conf=pre_conf_eff,
                half=self.use_half,
            )[0]
            boxes, scores, labels = [], [], []
            for x in (r.boxes or []):
                xy = x.xyxy[0].cpu().numpy().tolist()
                boxes.append(xy)
                scores.append(float(x.conf[0]))
                labels.append(int(x.cls[0]))
            raw_preds.append((boxes, scores, labels))

        # ── Step 4: primary WBF + geometric guardrails ────────────────────────
        primary = wbf_fuse(raw_preds, w, h)

        # ── Step 5: condition-tuned second-pass WBF + worker recovery ─────────
        refined = wbf_fuse_condition(primary, raw_preds, w, h, use_condition)

        # ── Step 5.5: head-to-body expansion ─────────────────────────────────
        # On a close-up webcam the YOLO model often outputs a near-square face/
        # head box (aspect w/h > 0.60). Expand it to a full-body box before the
        # temporal tracker so tracking is stable on the correct region.
        _HEAD_BODY_EXT = 3.5   # extend 3.5× head height downward
        _SHOULDER_MULT = 0.65  # shoulder width as fraction of head width
        for d in refined:
            if d.get("cls") != CLS_WORKER:
                continue
            bx = d["box"]
            bw = bx[2] - bx[0]
            bh = bx[3] - bx[1]
            if bh <= 0:
                continue
            if (bw / bh) > 0.60:  # near-square → head/face detection
                bx[3] = min(float(h), bx[3] + bh * _HEAD_BODY_EXT)
                cx    = (bx[0] + bx[2]) / 2.0
                hw    = bw * _SHOULDER_MULT
                bx[0] = max(0.0,      cx - hw)
                bx[2] = min(float(w), cx + hw)

        # ── Step 6: temporal PPE filter ───────────────────────────────────────
        filtered = self.temporal.update(refined, self._frame_idx)

        # ── Step 7: PPE association ───────────────────────────────────────────
        final = associate_ppe(filtered)

        # ── Format output ─────────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        detections = []
        counts: Dict[str, int] = {}
        for d in final:
            cls_name = CLS_NAMES.get(d["cls"], str(d["cls"]))
            det: Dict = {
                "class":      cls_name,
                "confidence": round(d["score"], 4),
                "box":        [round(v, 1) for v in d["box"]],
            }
            if "track_id" in d:
                det["track_id"] = d["track_id"]
            if d["cls"] == CLS_WORKER:
                det["has_helmet"] = d.get("has_helmet", False)
                det["has_vest"]   = d.get("has_vest",   False)
            detections.append(det)
            counts[cls_name] = counts.get(cls_name, 0) + 1

        reason = _condition_reason(use_condition)
        return {
            "detections":   detections,
            "class_counts": counts,
            "total":        len(detections),
            "condition":    use_condition,
            "scene_reason": reason,
            "elapsed_ms":   round(elapsed_ms, 1),
        }


def _condition_reason(condition: str) -> str:
    return {
        "S1_normal":    "Normal lighting, clear visibility, standard density.",
        "S2_dusty":     "Low contrast and saturation indicate dust or haze; CLAHE applied.",
        "S3_low_light": "Low brightness detected; gamma correction and CLAHE applied.",
        "S4_crowded":   "High worker density or occlusion; crowded recovery mode active.",
    }.get(condition, "")


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python buildsight_ensemble.py <image_path> [condition]")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Cannot read image: {sys.argv[1]}")
        sys.exit(1)

    cond     = sys.argv[2] if len(sys.argv) > 2 else "auto"
    pipeline = EnsemblePipeline()
    result   = pipeline.run(img, condition=cond)

    print(f"\nCondition : {result['condition']}  ({result['scene_reason']})")
    print(f"Detections: {result['total']}  |  {result['class_counts']}")
    print(f"Elapsed   : {result['elapsed_ms']} ms\n")
    for d in result["detections"]:
        ppe = ""
        if "has_helmet" in d:
            ppe = f"  helmet={'Y' if d['has_helmet'] else 'N'}  vest={'Y' if d['has_vest'] else 'N'}"
        print(f"  {d['class']:12s}  {d['confidence']:.2f}  {d['box']}{ppe}")
