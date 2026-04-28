#!/usr/bin/env python3
"""
BuildSight GeoAI WebSocket Broadcast Server
=============================================
Standalone WebSocket server on port 8765 that bridges the
IntelligenceEngine output to the frontend useGeoAIWebSocket hook.

Protocol:
  Every 2 seconds, broadcasts a JSON payload of type 'heatmap_update'
  containing: workers, cells, alerts, events, trails, kpi, backend_health.

  The frontend connects to ws://localhost:8765 and reads these payloads.
  If the detection pipeline is active, live data is used; otherwise,
  the engine generates its own demo telemetry from the last known state.

Author: BuildSight / Green Build AI
"""

import asyncio
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import os
from dotenv import load_dotenv

import websockets
from websockets.asyncio.server import serve, ServerConnection

# Add project root to path so we can import the intelligence engine
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from buildsight_intelligence import (
    IntelligenceEngine,
    SpatialWorker,
    SpatialEvent,
    SITE_CONFIG,
    EventPriority,
)

from heatmap_engine import HeatmapEngine

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("BuildSight.WS")

# ── Constants ──────────────────────────────────────────────────────────────────
WS_PORT = 8765
BROADCAST_INTERVAL_S = 0.2  # 5 FPS telemetry updates

SW_LAT = SITE_CONFIG["sw_lat"]
SW_LON = SITE_CONFIG["sw_lon"]

# H matrix calibrated at 848×478 — must match SITE_CONFIG calib_frame_w/h
_H_PATH = PROJECT_ROOT / "spatial" / "calibration" / "camera_cam01_H.npy"
_CALIB_W = SITE_CONFIG.get("calib_frame_w", 848)
_CALIB_H = SITE_CONFIG.get("calib_frame_h", 478)

# Load environment variables
load_dotenv(PROJECT_ROOT / "dashboard" / "backend" / ".env")

_DB_CONFIG = None
if os.getenv("POSTGRES_DB"):
    _DB_CONFIG = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "buildsight_geoai"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "password")
    }

# ── GeoAI State ────────────────────────────────────────────────────────────────

class GeoAIBroadcaster:
    """
    Manages the intelligence engine state and broadcasts
    telemetry payloads to all connected WebSocket clients.
    """

    # If the pipeline goes silent for this many seconds, clear stale workers
    _PIPELINE_TIMEOUT_S = 30.0

    def __init__(self):
        self.engine = IntelligenceEngine(
            homography_path=str(_H_PATH) if _H_PATH.exists() else None,
            db_config=_DB_CONFIG,
            frame_width=_CALIB_W,
            frame_height=_CALIB_H,
        )
        self.heatmap_engine = HeatmapEngine()
        self.clients: Set[ServerConnection] = set()
        self.cycle = 0
        self._last_workers: List[SpatialWorker] = []
        self._last_fps = 0
        self._last_latency = 0
        self._last_condition = "S1_normal"
        self._last_update_time: float = 0.0
        self._running = False
        log.info("GeoAIBroadcaster initialized")

    def update_from_detections(
        self,
        detections: list,
        fps: float = 0,
        latency_ms: float = 0,
        scene_condition: str = "S1_normal",
        frame_w: int = 0,
        frame_h: int = 0,
    ):
        """
        Called by the FastAPI server or GeoAI pipeline via WS
        when new detection results are available.

        Args:
            detections: List of detection dicts from the pipeline
            fps: Current pipeline FPS
            latency_ms: Processing latency
            scene_condition: Current estimated site condition
            frame_w: Width of the source frame in pixels (0 = keep current)
            frame_h: Height of the source frame in pixels (0 = keep current)
        """
        self._last_fps = fps
        self._last_latency = latency_ms
        self._last_condition = scene_condition
        self._last_update_time = time.time()

        # Sync mapper frame dimensions so pixel→world mapping is accurate
        if frame_w > 0 and frame_h > 0:
            self.engine.mapper.frame_w = frame_w
            self.engine.mapper.frame_h = frame_h
        
        worker_profiles = []
        for d in detections:
            cls = d.get("class", "")
            if cls not in ("worker", "person"):
                continue
            
            box = d.get("box", [0, 0, 0, 0])
            
            # Create a simple profile object compatible with IntelligenceEngine
            class WorkerProfile:
                pass
            
            wp = WorkerProfile()
            wp.worker_id = d.get("track_id", id(d) % 10000)
            wp.worker_box = box
            wp.has_helmet = d.get("has_helmet", False)
            wp.has_vest = d.get("has_vest", False)
            wp.helmet = None
            wp.vest = None
            worker_profiles.append(wp)
        
        self._last_workers = self.engine.process_frame(worker_profiles) if worker_profiles else []
        if worker_profiles:
            log.info("GeoAI pipeline connected to WS server")
            log.info("Pushing live detections to dashboard")
        # Always sync heatmap (clears stale workers when detections go empty)
        self.heatmap_engine.update(self._last_workers)
    
    def _build_risk_cells(self) -> List[Dict]:
        """Generate risk grid cells from the site configuration."""
        cells = []
        bw = SITE_CONFIG["width_m"]
        bd = SITE_CONFIG["depth_m"]
        # Increased from 2.0 → 8.0 to cut grid points by ~16x and prevent
        # the event loop from blocking during payload construction.
        cell_size = 8.0
        
        cols = int(bw / cell_size)
        rows = int(bd / cell_size)
        
        for ci in range(cols):
            for ri in range(rows):
                wx = ci * cell_size + cell_size / 2
                wy = ri * cell_size + cell_size / 2
                
                m_lat = 110574.0
                m_lon = 111319.0 * math.cos(math.radians(SW_LAT))
                angle_rad = math.radians(SITE_CONFIG.get("rotation_deg", 0))
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)
                rx = wx * cos_a - wy * sin_a
                ry = wx * sin_a + wy * cos_a
                
                lat = SW_LAT + ry / m_lat
                lon = SW_LON + rx / m_lon
                
                # Compute cell risk from nearby worker density
                risk_score = 0.0
                worker_count = 0
                ppe_violations = 0
                
                for sw in self._last_workers:
                    dist = math.sqrt((sw.world_x - wx) ** 2 + (sw.world_y - wy) ** 2)
                    if dist < cell_size * 1.5:
                        worker_count += 1
                        if not sw.ppe_compliant:
                            ppe_violations += 1
                            risk_score = max(risk_score, 0.7 + 0.1 * ppe_violations)
                        else:
                            risk_score = max(risk_score, 0.2)
                
                # Edge risk bonus
                edge = min(wx, bw - wx, wy, bd - wy)
                edge_risk = max(0, 0.15 - edge * 0.02)
                risk_score = min(1.0, risk_score + edge_risk)
                
                if risk_score > 0.7:
                    risk_level = "CRITICAL"
                elif risk_score > 0.5:
                    risk_level = "HIGH"
                elif risk_score > 0.3:
                    risk_level = "MODERATE"
                else:
                    risk_level = "LOW"
                
                cells.append({
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "risk_score": round(risk_score, 4),
                    "risk_level": risk_level,
                    "worker_count": worker_count,
                    "ppe_violations": ppe_violations,
                })
        
        # Safety cap: never send more than 1000 cells to the frontend
        if len(cells) > 1000:
            cells = cells[:1000]
        return cells
    
    def _build_alerts(self) -> List[Dict]:
        """Generate active alerts from the event dispatcher."""
        alerts = []
        for ev in self.engine.dispatcher.events:
            if ev.priority in (EventPriority.CRITICAL, EventPriority.WARNING):
                if ev.state in ("ACTIVE", "ESCALATED"):
                    alerts.append({
                        "alert_level": ev.priority,
                        "alert_type": ev.event_type,
                        "alert_message": ev.message,
                        "bocw_reference": "BOCW Act 1996 / NBC 2016",
                        "worker_count": 1,
                        "ppe_violations": 1 if "PPE" in ev.event_type else 0,
                        "risk_score": 0.85 if ev.priority == "CRITICAL" else 0.55,
                        "timestamp": ev.timestamp,
                    })
                    if len(alerts) >= 5:
                        break
        return alerts

    def build_payload(self) -> Dict:
        """Build the full HeatmapUpdatePayload for broadcast."""
        self.cycle += 1

        # Auto-clear stale workers if pipeline has gone silent
        if (
            self._last_workers
            and self._last_update_time > 0
            and time.time() - self._last_update_time > self._PIPELINE_TIMEOUT_S
        ):
            log.info("Pipeline silent for %.0fs — clearing stale workers", self._PIPELINE_TIMEOUT_S)
            self._last_workers = []
            self.heatmap_engine.update([])

        workers = []
        for sw in self._last_workers:
            workers.append({
                "lat": sw.lat,
                "lon": sw.lon,
                "ppe_ok": sw.ppe_compliant,
                "has_helmet": sw.has_helmet,
                "has_vest": sw.has_vest,
                "zone": sw.zone,
                "risk": sw.risk,
                "height_m": sw.height_m,
                "worker_id": sw.worker_id,
                "status": sw.status,
                "dwell_time_s": sw.dwell_time_s,
                "dwell_severity": sw.dwell_severity,
                "camera_id": sw.camera_id,
            })
        
        cells = self._build_risk_cells()
        alerts = self._build_alerts()
        events = self.engine.dispatcher.get_timeline(limit=20)
        trails = self.engine.get_worker_trails(max_age_s=600)
        kpi = self.engine.get_kpi_summary()
        health = self.engine.get_backend_health()
        
        compliant = sum(1 for w in workers if w["ppe_ok"])
        
        # Only send the heavy heatmap payload every 2nd broadcast to save bandwidth (~2.5 updates/sec)
        include_heatmap = (self.cycle % 2 == 0)
        
        payload = {
            "type": "heatmap_update",
            "cycle": self.cycle,
            "workers": workers,
            "cells": cells,
            "alerts": alerts,
            "site_stats": {
                "total_workers": len(workers),
                "ppe_compliant": compliant,
                "max_risk_score": max((c["risk_score"] for c in cells), default=0.0),
                "critical_zones": sum(1 for c in cells if c["risk_level"] == "CRITICAL"),
                "fps": self._last_fps,
                "latency_ms": self._last_latency,
                "scene_condition": self._last_condition,
            },
            "events": events,
            "trails": trails,
            "kpi": kpi,
            "backend_health": health,
        }
        
        if include_heatmap:
            payload["heatmap"] = self.heatmap_engine.get_frontend_payload()
            
        return payload

    async def register(self, ws: ServerConnection):
        """Register a new WebSocket client."""
        self.clients.add(ws)
        log.info(f"Client connected ({len(self.clients)} total)")
        # Send initial payload immediately
        try:
            payload = self.build_payload()
            await ws.send(json.dumps(payload, default=str))
        except Exception as e:
            log.warning(f"Failed to send initial payload: {e}")

    async def unregister(self, ws: ServerConnection):
        """Unregister a WebSocket client."""
        self.clients.discard(ws)
        log.info(f"Client disconnected ({len(self.clients)} total)")

    async def broadcast(self):
        """Send the current payload to all connected clients."""
        if not self.clients:
            return
        
        payload = self.build_payload()
        message = json.dumps(payload, default=str)
        
        disconnected = set()
        for ws in list(self.clients):
            try:
                await ws.send(message)
            except websockets.ConnectionClosed:
                disconnected.add(ws)
            except Exception as e:
                log.warning(f"Broadcast error: {e}")
                disconnected.add(ws)
        
        for ws in disconnected:
            await self.unregister(ws)

    async def broadcast_loop(self):
        """Run the periodic broadcast loop."""
        self._running = True
        log.info(f"Broadcast loop started (interval={BROADCAST_INTERVAL_S}s)")
        while self._running:
            try:
                # Hard timeout: if broadcast takes longer than 150ms the event
                # loop is saturated. Skip this cycle rather than block clients.
                await asyncio.wait_for(self.broadcast(), timeout=0.15)
            except asyncio.TimeoutError:
                log.warning("Broadcast cycle exceeded safe interval (>150ms) — skipping")
            except Exception as e:
                log.error(f"Unexpected broadcast error: {e}")
            await asyncio.sleep(BROADCAST_INTERVAL_S)

    def stop(self):
        """Stop the broadcast loop."""
        self._running = False


# ── Global broadcaster instance ───────────────────────────────────────────────
broadcaster = GeoAIBroadcaster()


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def ws_handler(websocket: ServerConnection):
    """Handle an individual WebSocket connection."""
    await broadcaster.register(websocket)
    try:
        async for message in websocket:
            # Handle incoming commands from the frontend
            try:
                cmd = json.loads(message)
                cmd_type = cmd.get("type", "")
                
                if cmd_type == "acknowledge_event":
                    event_id = cmd.get("event_id", "")
                    if event_id:
                        broadcaster.engine.dispatcher.acknowledge(event_id)
                        log.info(f"Event {event_id} acknowledged")
                
                elif cmd_type == "resolve_event":
                    event_id = cmd.get("event_id", "")
                    if event_id:
                        broadcaster.engine.dispatcher.resolve(event_id)
                        log.info(f"Event {event_id} resolved")
                
                elif cmd_type == "update_detections":
                    detections = cmd.get("detections", [])
                    broadcaster.update_from_detections(
                        detections,
                        fps=cmd.get("fps", 0),
                        latency_ms=cmd.get("latency_ms", 0),
                        scene_condition=cmd.get("scene_condition", "S1_normal"),
                        frame_w=cmd.get("frame_w", 0),
                        frame_h=cmd.get("frame_h", 0),
                    )
                
            except json.JSONDecodeError:
                pass
    except websockets.ConnectionClosed:
        pass
    finally:
        await broadcaster.unregister(websocket)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    """Start the WebSocket server and broadcast loop."""
    log.info(f"Starting GeoAI WebSocket Server on port {WS_PORT}")
    
    async with serve(
        ws_handler,
        "0.0.0.0",
        WS_PORT,
        max_size=8 * 1024 * 1024,   # 8 MB — prevents 1009 "message too big" crash
        ping_interval=20,
        ping_timeout=30,
    ):
        log.info(f"GeoAI WS Server ready on ws://localhost:{WS_PORT}")
        await broadcaster.broadcast_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("WebSocket server stopped")
        broadcaster.stop()
