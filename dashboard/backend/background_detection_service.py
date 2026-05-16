"""
BuildSight Background Detection Service
========================================
Runs video inference in a daemon thread independently of frontend
tab state. Results are published to all WebSocket clients via a
broadcast callback injected at construction time (avoids circular
imports with server.py).

Usage (from server.py):
    from background_detection_service import BackgroundDetectionService
    bg_service = BackgroundDetectionService(
        broadcast_fn   = ws_manager.broadcast_from_thread,
        inference_fn   = run_inference,        # server.run_inference
        classify_fn    = _classify_scene_wrap, # lambda frame: classify_scene_fast(...)
    )
"""

import logging
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from coordinate_projector import PixelToWorldProjector
from camera_stream import get_camera

logger = logging.getLogger("buildsight.bg_detection")


class BackgroundDetectionService:
    """Singleton service that runs continuous video inference in a daemon thread."""

    def __init__(
        self,
        broadcast_fn:  Callable[[dict], None],
        inference_fn:  Callable,
        classify_fn:   Optional[Callable] = None,
    ):
        self._broadcast   = broadcast_fn
        self._inference   = inference_fn
        self._classify    = classify_fn  # (frame) -> scene_str; optional

        self.is_running   = False
        self.is_paused    = False
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Video state
        self.video_path   : Optional[str]  = None
        self._cap         : Optional[cv2.VideoCapture] = None
        self.total_frames : int   = 0
        self.current_frame: int   = 0
        self.fps          : float = 25.0

        # Detection state — shared across all tabs via WebSocket broadcast
        self.latest_scene        : str   = "S1_normal"
        self.latest_worker_count : int   = 0
        self.detection_fps       : float = 0.0
        self.latency_ms          : float = 0.0
        self.frame_count         : int   = 0

        # Spatial state — consumed by GeoAI map
        self.worker_positions: list = []
        self.zone_occupancy  : dict = {}
        self.active_violations: list = []
        self.detections      : list = []

        self._lock = threading.Lock()
        self._projector = PixelToWorldProjector()

        # Throttle: emit every 150 ms max
        self._emit_interval = 0.15

    # ── Public control API ────────────────────────────────────────────────────

    def load_video(self, video_path: str, camera_config: Optional[dict] = None) -> None:
        """Load a video file and (optionally) a camera calibration config."""
        self.stop()
        if camera_config:
            self._projector = PixelToWorldProjector(camera_config)

        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self._cap          = cap
        self.total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps           = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.current_frame = 0
        logger.info("[BG] Loaded: %s | frames=%d fps=%.1f", video_path, self.total_frames, self.fps)

    def start(self) -> None:
        """Start background detection in a daemon thread."""
        if self.is_running:
            logger.warning("[BG] Already running — ignoring start()")
            return
        if not self._cap and not (get_camera() and get_camera().is_running):
            raise RuntimeError("Call load_video() or init_camera() before start()")

        self._stop_event.clear()
        self.is_running = True
        self.is_paused  = False
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="BuildSight-BG-Detection",
        )
        self._thread.start()
        logger.info("[BG] Detection thread started")
        self._broadcast({"type": "detection_started", "total_frames": self.total_frames, "fps": self.fps})

    def pause(self) -> None:
        self.is_paused = True
        self._broadcast({"type": "detection_paused"})

    def resume(self) -> None:
        self.is_paused = False
        self._broadcast({"type": "detection_resumed"})

    def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=4.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("[BG] Detection stopped")
        self._broadcast({"type": "detection_stopped"})

    def get_current_state(self) -> dict:
        """Thread-safe snapshot used by REST and WebSocket snapshot endpoints."""
        with self._lock:
            return {
                "is_running":       self.is_running,
                "is_paused":        self.is_paused,
                "current_frame":    self.current_frame,
                "total_frames":     self.total_frames,
                "worker_count":     self.latest_worker_count,
                "scene_condition":  self.latest_scene,
                "fps":              round(self.detection_fps, 1),
                "latency_ms":       round(self.latency_ms, 1),
                "worker_positions": self.worker_positions,
                "zone_occupancy":   self.zone_occupancy,
                "violations":       self.active_violations,
                "detections":       self.detections,
            }

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        last_emit = 0.0
        conf_threshold = 0.25   # reasonable default; operator can tune via API

        # GPU memory management
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None

        _frame_skip_n = 2        # process 1 in every N frames (balanced GPU load)
        _frame_skip_counter = 0
        _cache_clear_every = 50  # call empty_cache every N processed frames
        _processed_count = 0

        while not self._stop_event.is_set():
            if self.is_paused:
                time.sleep(0.05)
                continue

            cm = get_camera()
            if cm and cm.is_running:
                # Get frame from live stream, block up to 1 second
                fid, ts, frame = cm.get_latest_frame(block=True, timeout=1.0)
                if frame is None:
                    continue
                # For live stream, use the frame_id from the camera manager
                with self._lock:
                    self.current_frame = fid
                    self.frame_count += 1
            else:
                # Fallback to video file
                if self._cap is None:
                    time.sleep(1)
                    continue
                ret, frame = self._cap.read()
                if not ret:
                    # Loop video
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    self._broadcast({"type": "video_looped"})
                    continue
                with self._lock:
                    self.current_frame += 1
                    self.frame_count += 1



            # ── Frame skipping — reduces GPU pressure on heavy scenes ─────
            _frame_skip_counter += 1
            if _frame_skip_counter % _frame_skip_n != 0:
                continue

            t0 = time.time()

            try:
                # ── Scene classification ──────────────────────────────────
                scene = self.latest_scene
                if self._classify is not None:
                    try:
                        scene = self._classify(frame)
                    except Exception:
                        pass  # keep previous scene on error

                # ── Resize frame to max 640px before inference ────────────
                h, w = frame.shape[:2]
                if max(h, w) > 640:
                    scale = 640 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                      interpolation=cv2.INTER_LINEAR)
                
                # Track inference dimensions for frontend coordinate mapping
                infer_h, infer_w = frame.shape[:2]

                # ── Inference ─────────────────────────────────────────────
                detections, _, _perf = self._inference(frame, scene, conf_threshold)

                # ── Periodic GPU cache flush ──────────────────────────────
                _processed_count += 1
                if _torch is not None and _torch.cuda.is_available() and _processed_count % _cache_clear_every == 0:
                    _torch.cuda.empty_cache()

                workers = [d for d in detections if d.get("cls") == 0]

                # ── Project pixel → world ─────────────────────────────────
                worker_positions = []
                for i, w in enumerate(workers):
                    box = w.get("box", [0, 0, 0, 0])
                    cx  = (box[0] + box[2]) / 2  # horizontal center
                    cy  = box[3]                   # foot point: bottom of bbox
                    pos = self._projector.pixel_to_world(cx, cy, frame.shape)

                    worker_positions.append({
                        "worker_id":    w.get("track_id", f"W{i+1}"),
                        "confidence":   round(w.get("score", w.get("confidence", 0)), 3),
                        "lat":          pos["lat"],
                        "lng":          pos["lng"],
                        "utm_e":        pos["utm_e"],
                        "utm_n":        pos["utm_n"],
                        "has_helmet":   w.get("has_helmet"),
                        "has_vest":     w.get("has_vest"),
                        "ppe_compliant": bool(w.get("has_helmet") and w.get("has_vest")),
                        "violation_type": self._violation_types(w),
                        "pixel_x":      round(cx, 1),
                        "pixel_y":      round(cy, 1),
                        "zone_id":      pos["zone_id"],
                        "zone_name":    pos["zone_name"],
                    })

                # ── Zone occupancy ────────────────────────────────────────
                zone_occupancy: dict = {}
                for wp in worker_positions:
                    zid = wp.get("zone_id")
                    if zid:
                        zone_occupancy.setdefault(zid, []).append(wp["worker_id"])

                # ── Violation detection ───────────────────────────────────
                violations = self._detect_violations(worker_positions, zone_occupancy)

                latency = (time.time() - t0) * 1000

                with self._lock:
                    self.latest_scene        = scene
                    self.latest_worker_count = len(workers)
                    self.detection_fps       = 1.0 / max(time.time() - t0, 0.001)
                    self.latency_ms          = latency
                    self.worker_positions    = worker_positions
                    self.zone_occupancy      = zone_occupancy
                    self.active_violations   = violations
                    self.detections          = detections

                # ── Throttled broadcast ───────────────────────────────────
                now = time.time()
                if now - last_emit >= self._emit_interval:
                    self._broadcast({
                        "type":             "detection_update",
                        "worker_count":     len(workers),
                        "scene_condition":  scene,
                        "fps":              round(self.detection_fps, 1),
                        "latency_ms":       round(latency, 1),
                        "frame_number":     self.current_frame,
                        "total_frames":     self.total_frames,
                        "worker_positions": worker_positions,
                        "zone_occupancy":   zone_occupancy,
                        "violations":       violations,
                        "detections":       detections,
                        "frame_width":      infer_w,
                        "frame_height":     infer_h,
                        "timestamp":        now,
                    })
                    last_emit = now

            except Exception as exc:
                logger.error("[BG] Frame %d error: %s", self.current_frame, exc, exc_info=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _violation_types(worker: dict) -> list:
        vt = []
        if worker.get("has_helmet") is False:
            vt.append("NO_HELMET")
        if worker.get("has_vest") is False:
            vt.append("NO_VEST")
        return vt

    _CRITICAL_ZONES = {"zone_steel_erection_01"}

    def _detect_violations(self, worker_positions: list, zone_occupancy: dict) -> list:
        violations = []
        now = time.time()

        for wp in worker_positions:
            zid  = wp.get("zone_id")
            zname = wp.get("zone_name", "Unknown Zone")

            if not zid:
                continue

            if not wp.get("ppe_compliant", True):
                violations.append({
                    "type":      "PPE_VIOLATION_IN_ZONE",
                    "severity":  "CRITICAL",
                    "worker_id": wp["worker_id"],
                    "zone_id":   zid,
                    "zone_name": zname,
                    "details":   f"Missing: {', '.join(wp['violation_type'])}",
                    "lat":       wp["lat"],
                    "lng":       wp["lng"],
                    "timestamp": now,
                })

            if zid in self._CRITICAL_ZONES:
                violations.append({
                    "type":      "CRITICAL_ZONE_ENTRY",
                    "severity":  "CRITICAL",
                    "worker_id": wp["worker_id"],
                    "zone_id":   zid,
                    "zone_name": zname,
                    "details":   "Worker inside critical restricted zone",
                    "lat":       wp["lat"],
                    "lng":       wp["lng"],
                    "timestamp": now,
                })

        return violations
