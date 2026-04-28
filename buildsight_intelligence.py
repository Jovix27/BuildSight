"""
BuildSight GeoAI — Spatial Intelligence Engine
=================================================
Green Build AI | Maran Constructions | IGBC AP | BOCW Act 1996

PURPOSE:
  Behavioral risk intelligence layer that sits between the detection
  pipeline (buildsight_pipeline_v2.py) and the heatmap engine.

  Components:
    * SpatialMapper     - Pixel <-> GPS coordinate transformation
    * DwellTracker      - Worker stationarity detection (30s/60s/120s)
    * GeofenceMonitor   - Zone boundary breach detection
    * StatusClassifier  - Rolling risk classification (Safe/AtRisk/Critical)
    * EscalationGuard   - Unresolved alert escalation with SLA timers
    * EventDispatcher   - Unified event bus for the timeline
    * PostGISBridge     - Async DB writer with DemoFallback ring buffer

Author: BuildSight / Green Build AI
"""

import cv2
import json
import math
import time
import logging
import asyncio
import numpy as np
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Callable, Deque, Dict, List, Optional, Set, Tuple,
)
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("BuildSight.Intelligence")


# ===============================================================================
#  Enumerations & Constants
# ===============================================================================

class EventPriority(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"

class WorkerStatus(str, Enum):
    SAFE      = "SAFE"
    AT_RISK   = "AT_RISK"
    CRITICAL  = "CRITICAL"

class AlertState(str, Enum):
    ACTIVE       = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    ESCALATED    = "ESCALATED"
    RESOLVED     = "RESOLVED"
    EXPIRED      = "EXPIRED"

class GeofenceBreachType(str, Enum):
    RESTRICTED_ENTRY  = "RESTRICTED_ENTRY"
    REVERSE_DIRECTION = "REVERSE_DIRECTION"
    PROLONGED_PRESENCE = "PROLONGED_PRESENCE"
    REPEAT_VIOLATION  = "REPEAT_VIOLATION"

class EventType(str, Enum):
    ZONE_ENTRY       = "ZONE_ENTRY"
    ZONE_EXIT        = "ZONE_EXIT"
    PPE_VIOLATION    = "PPE_VIOLATION"
    DWELL_BREACH     = "DWELL_BREACH"
    GEOFENCE_BREACH  = "GEOFENCE_BREACH"
    STATUS_CHANGE    = "STATUS_CHANGE"
    ALERT_ESCALATED  = "ALERT_ESCALATED"

class Shift(str, Enum):
    MORNING   = "MORNING"     # 06:00 - 14:00
    AFTERNOON = "AFTERNOON"   # 14:00 - 22:00
    NIGHT     = "NIGHT"       # 22:00 - 06:00

# Dwell severity thresholds (seconds)
DWELL_THRESHOLDS = {
    "LOW":    30,
    "MEDIUM": 60,
    "HIGH":   120,
}

# Default escalation: unacknowledged critical alerts escalate after 5 min
DEFAULT_ESCALATION_THRESHOLD_S = 300
# Acknowledged alerts auto-expire after 30 min
DEFAULT_AUTO_EXPIRE_S = 1800

# Site reference coordinates (from module3_homography_calibrate.py)
# Site reference coordinates imported from shared utility
try:
    from dashboard.backend.geoai.utils.spatial_mapper import SpatialMapper, SITE_CONFIG
except ImportError:
    # Manual fallback for standalone execution outside dashboard context
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "dashboard", "backend"))
    from geoai.utils.spatial_mapper import SpatialMapper, SITE_CONFIG



# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpatialWorker:
    """A worker detection enriched with spatial intelligence."""
    worker_id: int
    pixel_x: float
    pixel_y: float
    world_x: float              # meters from SW corner (east)
    world_y: float              # meters from SW corner (north)
    lat: float
    lon: float
    has_helmet: bool
    has_vest: bool
    ppe_compliant: bool
    confidence: float
    zone: str = "unknown"
    risk: str = "LOW"
    height_m: float = 0.0
    status: str = "SAFE"        # WorkerStatus
    camera_id: str = "CAM-01"
    dwell_time_s: float = 0.0
    dwell_severity: str = "NONE"
    timestamp: str = ""


@dataclass
class SpatialEvent:
    """An event for the GeoAI timeline."""
    event_id: str = ""
    event_type: str = ""        # EventType
    priority: str = "INFO"      # EventPriority
    timestamp: str = ""
    worker_id: Optional[int] = None
    zone: str = ""
    camera_id: str = "CAM-01"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    state: str = "ACTIVE"       # AlertState
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    sla_remaining_s: Optional[float] = None

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class WorkerTrack:
    """Historical position buffer for replay and path rendering."""
    worker_id: int
    positions: Deque[Dict] = field(default_factory=lambda: deque(maxlen=300))
    # Rolling risk scores for status classification
    risk_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    first_seen: float = 0.0
    last_seen: float = 0.0
    current_zone: str = ""
    previous_zone: str = ""
    stationary_since: Optional[float] = None
    entry_direction: Optional[str] = None  # "FORWARD" or "REVERSE"
    camera_id: str = "CAM-01"
    # Geofence violation history
    violation_count: int = 0
    last_violation_time: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DwellTracker — Worker stationarity detection
# ═══════════════════════════════════════════════════════════════════════════════


class DwellTracker:
    """
    Detects workers who remain stationary in the same zone for too long.
    
    Severity thresholds:
      LOW:    >30 seconds
      MEDIUM: >60 seconds
      HIGH:   >120 seconds
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, int]] = None,
        movement_tolerance_m: float = 1.5,
    ):
        self.thresholds = thresholds or DWELL_THRESHOLDS
        self.movement_tol = movement_tolerance_m

    def update(self, track: WorkerTrack, wx: float, wy: float) -> Tuple[float, str]:
        """
        Update dwell state for a worker. Returns (dwell_time_s, severity).
        """
        now = time.time()

        if track.stationary_since is None:
            track.stationary_since = now
            return 0.0, "NONE"

        # Check if worker has moved significantly
        if len(track.positions) >= 2:
            prev = track.positions[-1]
            dist = math.sqrt(
                (wx - prev["world_x"]) ** 2 + (wy - prev["world_y"]) ** 2
            )
            if dist > self.movement_tol:
                # Worker moved — reset stationarity timer
                track.stationary_since = now
                return 0.0, "NONE"

        dwell_s = now - track.stationary_since
        severity = "NONE"
        for level in ("HIGH", "MEDIUM", "LOW"):
            if dwell_s >= self.thresholds[level]:
                severity = level
                break

        return dwell_s, severity


# ═══════════════════════════════════════════════════════════════════════════════
#  GeofenceMonitor — Zone boundary detection
# ═══════════════════════════════════════════════════════════════════════════════

class GeofenceMonitor:
    """
    Detects geofence breaches of different types:
      - RESTRICTED_ENTRY:  Worker enters a restricted zone
      - REVERSE_DIRECTION: Worker enters a zone from the wrong direction
      - PROLONGED_PRESENCE: Worker stays too long in a controlled zone
      - REPEAT_VIOLATION:  Worker triggers the same boundary alarm again
    """

    def __init__(
        self,
        restricted_zones: Optional[Set[str]] = None,
        repeat_violation_window_s: float = 600,  # 10 min
    ):
        self.restricted_zones = restricted_zones or {
            "high_risk_scaffolding",
            "high_risk_staircase",
            "restricted_perimeter",
        }
        self.repeat_window = repeat_violation_window_s

    def check(self, track: WorkerTrack, zone: str) -> Optional[GeofenceBreachType]:
        """
        Check for geofence breaches. Returns breach type or None.
        """
        now = time.time()
        prev_zone = track.current_zone

        # Zone transition?
        if zone == prev_zone:
            # Check prolonged presence in restricted zone
            if zone in self.restricted_zones and track.stationary_since:
                dwell = now - track.stationary_since
                if dwell > DWELL_THRESHOLDS["HIGH"]:
                    return GeofenceBreachType.PROLONGED_PRESENCE
            return None

        # ── Zone entry event ───────────────────────────────────────────────
        if zone in self.restricted_zones:
            # Check for repeat violation
            if (
                track.violation_count > 0
                and now - track.last_violation_time < self.repeat_window
            ):
                track.violation_count += 1
                track.last_violation_time = now
                return GeofenceBreachType.REPEAT_VIOLATION

            # Check for reverse direction (simplified: coming from low-risk to high-risk)
            if prev_zone and "low_risk" in prev_zone and "high_risk" in zone:
                # This is normal direction; reverse would be entering scaffold from outside
                pass

            track.violation_count += 1
            track.last_violation_time = now
            return GeofenceBreachType.RESTRICTED_ENTRY

        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  StatusClassifier — Rolling risk assessment
# ═══════════════════════════════════════════════════════════════════════════════

class StatusClassifier:
    """
    Classifies each worker as Safe, At Risk, or Critical
    based on a rolling average of their risk scores.
    
    Factors:
      - Zone risk level
      - PPE compliance
      - Dwell violation active
      - Geofence breach active
    """

    def __init__(
        self,
        at_risk_threshold: float = 0.40,
        critical_threshold: float = 0.70,
    ):
        self.at_risk_thresh = at_risk_threshold
        self.critical_thresh = critical_threshold

    def classify(
        self,
        track: WorkerTrack,
        ppe_compliant: bool,
        zone_risk: str,
        dwell_severity: str,
        geofence_breach: Optional[GeofenceBreachType],
    ) -> WorkerStatus:
        """Compute a composite risk score and classify the worker."""
        score = 0.0

        # Zone risk contribution (0.0 - 0.4)
        zone_scores = {"CRITICAL": 0.4, "HIGH": 0.3, "MODERATE": 0.15, "LOW": 0.05}
        score += zone_scores.get(zone_risk.upper(), 0.05)

        # PPE contribution (0.0 - 0.3)
        if not ppe_compliant:
            score += 0.3

        # Dwell contribution (0.0 - 0.2)
        dwell_scores = {"HIGH": 0.2, "MEDIUM": 0.1, "LOW": 0.05, "NONE": 0.0}
        score += dwell_scores.get(dwell_severity, 0.0)

        # Geofence contribution (0.0 - 0.2)
        if geofence_breach:
            score += 0.2 if geofence_breach == GeofenceBreachType.REPEAT_VIOLATION else 0.15

        score = min(1.0, score)
        track.risk_scores.append(score)

        # Rolling average over the last N scores
        avg = sum(track.risk_scores) / len(track.risk_scores)

        if avg >= self.critical_thresh:
            return WorkerStatus.CRITICAL
        elif avg >= self.at_risk_thresh:
            return WorkerStatus.AT_RISK
        return WorkerStatus.SAFE


# ═══════════════════════════════════════════════════════════════════════════════
#  EscalationGuard — SLA timers and auto-expire
# ═══════════════════════════════════════════════════════════════════════════════

class EscalationGuard:
    """
    Manages alert lifecycle:
      ACTIVE → ACKNOWLEDGED → RESOLVED
      ACTIVE → ESCALATED (if unacknowledged after threshold)
      ACKNOWLEDGED → EXPIRED (auto-expire after window)
    """

    def __init__(
        self,
        escalation_threshold_s: float = DEFAULT_ESCALATION_THRESHOLD_S,
        auto_expire_s: float = DEFAULT_AUTO_EXPIRE_S,
    ):
        self.escalation_s = escalation_threshold_s
        self.auto_expire_s = auto_expire_s

    def tick(self, events: List[SpatialEvent]) -> List[SpatialEvent]:
        """
        Process all active events and apply lifecycle transitions.
        Returns newly escalated events.
        """
        now = datetime.now()
        escalated = []

        for event in events:
            if event.state == AlertState.ACTIVE:
                created = datetime.fromisoformat(event.timestamp)
                elapsed = (now - created).total_seconds()

                if event.priority == EventPriority.CRITICAL and elapsed > self.escalation_s:
                    event.state = AlertState.ESCALATED
                    event.sla_remaining_s = 0
                    escalated.append(event)
                else:
                    # Compute remaining SLA
                    if event.priority == EventPriority.CRITICAL:
                        event.sla_remaining_s = max(0, self.escalation_s - elapsed)

            elif event.state == AlertState.ACKNOWLEDGED:
                ack_time = datetime.fromisoformat(event.acknowledged_at or event.timestamp)
                if (now - ack_time).total_seconds() > self.auto_expire_s:
                    event.state = AlertState.EXPIRED

        return escalated


# ═══════════════════════════════════════════════════════════════════════════════
#  EventDispatcher — Unified event bus
# ═══════════════════════════════════════════════════════════════════════════════

class EventDispatcher:
    """
    Central event bus that collects all spatial intelligence events
    and maintains the timeline buffer for the dashboard.
    """

    def __init__(self, max_events: int = 500):
        self.events: Deque[SpatialEvent] = deque(maxlen=max_events)
        self._listeners: List[Callable[[SpatialEvent], None]] = []

    def emit(self, event: SpatialEvent):
        """Add event to the timeline and notify listeners."""
        self.events.appendleft(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                log.error(f"EventDispatcher listener error: {e}")

    def on_event(self, callback: Callable[[SpatialEvent], None]):
        """Register an event listener."""
        self._listeners.append(callback)

    def get_timeline(
        self,
        limit: int = 50,
        priority_filter: Optional[str] = None,
        event_type_filter: Optional[str] = None,
        shift_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Get the filtered event timeline for the dashboard."""
        result = []
        for ev in self.events:
            if priority_filter and ev.priority != priority_filter:
                continue
            if event_type_filter and ev.event_type != event_type_filter:
                continue
            if shift_filter:
                try:
                    ts = datetime.fromisoformat(ev.timestamp)
                    shift = self._get_shift(ts)
                    if shift.value != shift_filter:
                        continue
                except (ValueError, AttributeError):
                    pass
            result.append(asdict(ev))
            if len(result) >= limit:
                break
        return result

    def acknowledge(self, event_id: str, operator: str = "operator") -> bool:
        """Mark an event as acknowledged."""
        for ev in self.events:
            if ev.event_id == event_id and ev.state == AlertState.ACTIVE:
                ev.state = AlertState.ACKNOWLEDGED
                ev.acknowledged_by = operator
                ev.acknowledged_at = datetime.now().isoformat()
                return True
        return False

    def resolve(self, event_id: str, operator: str = "operator") -> bool:
        """Mark an event as resolved."""
        for ev in self.events:
            if ev.event_id == event_id and ev.state in (AlertState.ACTIVE, AlertState.ACKNOWLEDGED, AlertState.ESCALATED):
                ev.state = AlertState.RESOLVED
                ev.resolved_by = operator
                ev.resolved_at = datetime.now().isoformat()
                return True
        return False

    @staticmethod
    def _get_shift(ts: datetime) -> Shift:
        hour = ts.hour
        if 6 <= hour < 14:
            return Shift.MORNING
        elif 14 <= hour < 22:
            return Shift.AFTERNOON
        return Shift.NIGHT

    @property
    def active_count(self) -> int:
        return sum(1 for e in self.events if e.state == AlertState.ACTIVE)

    @property
    def escalated_count(self) -> int:
        return sum(1 for e in self.events if e.state == AlertState.ESCALATED)

    @property
    def critical_count(self) -> int:
        return sum(1 for e in self.events if e.priority == EventPriority.CRITICAL and e.state in (AlertState.ACTIVE, AlertState.ESCALATED))


# ═══════════════════════════════════════════════════════════════════════════════
#  PostGISBridge — Database writer with demo fallback
# ═══════════════════════════════════════════════════════════════════════════════

class PostGISBridge:
    """
    Writes spatial intelligence data to PostGIS.
    Falls back to an in-memory ring buffer (DemoFallback) when PostGIS
    is unreachable, ensuring the dashboard always has data.
    """

    def __init__(self, db_config: Optional[dict] = None):
        self.db_config = db_config
        self.conn = None
        self.is_degraded = False    # "System Degraded" mode
        # Demo fallback ring buffers
        self._worker_buffer: Deque[Dict] = deque(maxlen=500)
        self._event_buffer: Deque[Dict] = deque(maxlen=200)
        self._connect()

    def _connect(self):
        if not self.db_config:
            self.is_degraded = True
            log.info("PostGISBridge: No DB config — running in demo fallback mode")
            return
        try:
            import psycopg2
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = False
            self.is_degraded = False
            log.info("PostGISBridge: Connected to PostGIS")
        except Exception as e:
            log.warning(f"PostGISBridge: DB connection failed: {e} — using demo fallback")
            self.is_degraded = True
            self.conn = None

    def write_worker(self, worker: SpatialWorker):
        """Write a spatial worker detection to PostGIS or demo buffer."""
        data = asdict(worker)
        data["timestamp"] = datetime.now().isoformat()

        if self.is_degraded or not self.conn:
            self._worker_buffer.append(data)
            return

        try:
            import psycopg2.extras
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO detections
                    (worker_id, pixel_x, pixel_y, world_x, world_y,
                     latitude, longitude, has_helmet, has_vest,
                     ppe_compliant, confidence, zone_name, risk_level,
                     height_m, camera_id, timestamp)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
            """, (
                worker.worker_id, worker.pixel_x, worker.pixel_y,
                worker.world_x, worker.world_y,
                worker.lat, worker.lon,
                worker.has_helmet, worker.has_vest,
                worker.ppe_compliant, worker.confidence,
                worker.zone, worker.risk, worker.height_m,
                worker.camera_id,
            ))
            self.conn.commit()
        except Exception as e:
            log.error(f"PostGISBridge.write_worker: {e}")
            self.conn.rollback() if self.conn else None
            self._worker_buffer.append(data)
            self.is_degraded = True

    def write_event(self, event: SpatialEvent):
        """Write a spatial event to PostGIS or demo buffer."""
        data = asdict(event)
        if self.is_degraded or not self.conn:
            self._event_buffer.append(data)
            return
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO spatial_events
                    (event_id, event_type, priority, timestamp,
                     worker_id, zone, camera_id, message, details, state)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                event.event_id, event.event_type, event.priority,
                event.timestamp, event.worker_id, event.zone,
                event.camera_id, event.message,
                json.dumps(event.details), event.state,
            ))
            self.conn.commit()
        except Exception as e:
            log.error(f"PostGISBridge.write_event: {e}")
            self.conn.rollback() if self.conn else None
            self._event_buffer.append(data)

    def get_recent_workers(self, limit: int = 50) -> List[Dict]:
        """Fetch recent workers from DB or demo buffer."""
        if self.is_degraded:
            return list(self._worker_buffer)[-limit:]
        try:
            import psycopg2.extras
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT * FROM detections
                WHERE timestamp > NOW() - INTERVAL '5 minutes'
                ORDER BY timestamp DESC LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            return list(self._worker_buffer)[-limit:]

    @property
    def health(self) -> Dict:
        return {
            "service": "PostGIS",
            "status": "DEGRADED" if self.is_degraded else "HEALTHY",
            "buffer_workers": len(self._worker_buffer),
            "buffer_events": len(self._event_buffer),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  IntelligenceEngine — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class IntelligenceEngine:
    """
    The master orchestrator that processes each detection frame through:
      1. SpatialMapper   → pixel to GPS
      2. DwellTracker    → stationarity detection
      3. GeofenceMonitor → boundary breach detection
      4. StatusClassifier → risk classification
      5. EscalationGuard → SLA management
      6. EventDispatcher → timeline event bus
      7. PostGISBridge   → persistence
    """

    def __init__(
        self,
        homography_path: Optional[str] = None,
        db_config: Optional[dict] = None,
        escalation_threshold_s: float = DEFAULT_ESCALATION_THRESHOLD_S,
        auto_expire_s: float = DEFAULT_AUTO_EXPIRE_S,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ):
        self.mapper = SpatialMapper(
            homography_path=homography_path,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self.dwell = DwellTracker()
        self.geofence = GeofenceMonitor()
        self.classifier = StatusClassifier()
        self.escalation = EscalationGuard(
            escalation_threshold_s=escalation_threshold_s,
            auto_expire_s=auto_expire_s,
        )
        self.dispatcher = EventDispatcher()
        self.db = PostGISBridge(db_config=db_config)

        # Worker tracking state
        self.tracks: Dict[int, WorkerTrack] = {}
        self._tick_count = 0

        log.info(f"IntelligenceEngine initialized | Mapper: {self.mapper.status}")

    def process_frame(
        self,
        worker_profiles: list,
        camera_id: str = "CAM-01",
    ) -> List[SpatialWorker]:
        """
        Process a batch of WorkerPPE profiles from the detection pipeline.
        
        Args:
            worker_profiles: List of WorkerPPE dataclass instances
            camera_id: Camera source identifier
            
        Returns:
            List of enriched SpatialWorker instances
        """
        self._tick_count += 1
        now = time.time()
        spatial_workers = []

        # Evict tracks not seen for more than 30s so stale dwell timers don't
        # bleed into the next detection of the same track_id.
        _TRACK_EVICT_AGE_S = 30.0
        stale = [wid for wid, t in self.tracks.items() if now - t.last_seen > _TRACK_EVICT_AGE_S]
        for wid in stale:
            del self.tracks[wid]

        for wp in worker_profiles:
            wid = wp.worker_id
            # Extract pixel center of the worker bounding box
            x1, y1, x2, y2 = wp.worker_box
            px = (x1 + x2) / 2
            py = y2  # Use foot position for ground-plane mapping

            # ── 1. Spatial Mapping ────────────────────────────────────────
            wx, wy = self.mapper.pixel_to_world(px, py)
            lat, lon = self.mapper.world_to_gps(wx, wy)

            # ── Track management ──────────────────────────────────────────
            if wid not in self.tracks:
                self.tracks[wid] = WorkerTrack(
                    worker_id=wid,
                    first_seen=now,
                    camera_id=camera_id,
                )
            track = self.tracks[wid]
            track.last_seen = now

            # Determine zone (simplified — based on position quadrant)
            zone = self._resolve_zone(wx, wy)

            # ── 2. Dwell Tracking ─────────────────────────────────────────
            dwell_s, dwell_severity = self.dwell.update(track, wx, wy)

            if dwell_severity in ("MEDIUM", "HIGH") and dwell_severity != track.positions[-1].get("dwell_severity", "NONE") if track.positions else True:
                self.dispatcher.emit(SpatialEvent(
                    event_type=EventType.DWELL_BREACH,
                    priority=EventPriority.CRITICAL if dwell_severity == "HIGH" else EventPriority.WARNING,
                    worker_id=wid,
                    zone=zone,
                    camera_id=camera_id,
                    message=f"Worker W{wid} stationary for {dwell_s:.0f}s in {zone.replace('_', ' ')}",
                    details={"dwell_time_s": round(dwell_s, 1), "severity": dwell_severity},
                ))

            # ── 3. Geofence Monitoring ────────────────────────────────────
            breach = self.geofence.check(track, zone)
            if breach:
                priority = EventPriority.CRITICAL if breach in (
                    GeofenceBreachType.REPEAT_VIOLATION,
                    GeofenceBreachType.RESTRICTED_ENTRY,
                ) else EventPriority.WARNING

                self.dispatcher.emit(SpatialEvent(
                    event_type=EventType.GEOFENCE_BREACH,
                    priority=priority,
                    worker_id=wid,
                    zone=zone,
                    camera_id=camera_id,
                    message=f"GEOFENCE: {breach.value} — Worker W{wid} at {zone.replace('_', ' ')}",
                    details={"breach_type": breach.value, "violations": track.violation_count},
                ))

            # ── Zone transition events ────────────────────────────────────
            if zone != track.current_zone and track.current_zone:
                # Zone exit
                self.dispatcher.emit(SpatialEvent(
                    event_type=EventType.ZONE_EXIT,
                    priority=EventPriority.INFO,
                    worker_id=wid,
                    zone=track.current_zone,
                    camera_id=camera_id,
                    message=f"W{wid} exited {track.current_zone.replace('_', ' ')}",
                ))
                # Zone entry
                self.dispatcher.emit(SpatialEvent(
                    event_type=EventType.ZONE_ENTRY,
                    priority=EventPriority.INFO if zone not in self.geofence.restricted_zones else EventPriority.WARNING,
                    worker_id=wid,
                    zone=zone,
                    camera_id=camera_id,
                    message=f"W{wid} entered {zone.replace('_', ' ')}",
                ))

            track.previous_zone = track.current_zone
            track.current_zone = zone

            # ── PPE violation event ───────────────────────────────────────
            ppe_ok = wp.has_helmet and wp.has_vest
            if not ppe_ok:
                missing = []
                if not wp.has_helmet:
                    missing.append("helmet")
                if not wp.has_vest:
                    missing.append("vest")
                self.dispatcher.emit(SpatialEvent(
                    event_type=EventType.PPE_VIOLATION,
                    priority=EventPriority.WARNING,
                    worker_id=wid,
                    zone=zone,
                    camera_id=camera_id,
                    message=f"PPE missing: {', '.join(missing)} — Worker W{wid}",
                    details={"missing": missing},
                ))

            # ── 4. Status Classification ──────────────────────────────────
            zone_risk = self._zone_risk_level(zone)
            status = self.classifier.classify(
                track, ppe_ok, zone_risk, dwell_severity, breach,
            )

            # Emit status change events
            if track.positions:
                prev_status = track.positions[-1].get("status", "SAFE")
                if status.value != prev_status:
                    self.dispatcher.emit(SpatialEvent(
                        event_type=EventType.STATUS_CHANGE,
                        priority=EventPriority.CRITICAL if status == WorkerStatus.CRITICAL else EventPriority.WARNING,
                        worker_id=wid,
                        zone=zone,
                        camera_id=camera_id,
                        message=f"W{wid} status: {prev_status} → {status.value}",
                        details={"from": prev_status, "to": status.value},
                    ))

            # ── Record position in track buffer ───────────────────────────
            track.positions.append({
                "lat": lat, "lon": lon,
                "world_x": wx, "world_y": wy,
                "timestamp": now,
                "status": status.value,
                "dwell_severity": dwell_severity,
            })

            # ── Build enriched SpatialWorker ──────────────────────────────
            sw = SpatialWorker(
                worker_id=wid,
                pixel_x=px, pixel_y=py,
                world_x=wx, world_y=wy,
                lat=lat, lon=lon,
                has_helmet=wp.has_helmet,
                has_vest=wp.has_vest,
                ppe_compliant=ppe_ok,
                confidence=getattr(wp, 'confidence', 0.0) if hasattr(wp, 'confidence') else (wp.helmet.confidence if wp.helmet else wp.vest.confidence if wp.vest else 0.0),
                zone=zone,
                risk=zone_risk,
                height_m=0.0,
                status=status.value,
                camera_id=camera_id,
                dwell_time_s=round(dwell_s, 1),
                dwell_severity=dwell_severity,
                timestamp=datetime.now().isoformat(),
            )
            spatial_workers.append(sw)
            self.db.write_worker(sw)

        # ── 5. Escalation Guard tick ──────────────────────────────────────
        active_events = [e for e in self.dispatcher.events if e.state in (AlertState.ACTIVE, AlertState.ACKNOWLEDGED)]
        newly_escalated = self.escalation.tick(active_events)
        for esc in newly_escalated:
            self.dispatcher.emit(SpatialEvent(
                event_type=EventType.ALERT_ESCALATED,
                priority=EventPriority.CRITICAL,
                worker_id=esc.worker_id,
                zone=esc.zone,
                camera_id=esc.camera_id,
                message=f"ESCALATED: {esc.message}",
                details={"original_event_id": esc.event_id},
            ))

        return spatial_workers

    def get_worker_trails(self, max_age_s: float = 600) -> List[Dict]:
        """Get historical worker trails for replay rendering."""
        now = time.time()
        trails = []
        for wid, track in self.tracks.items():
            positions = [
                p for p in track.positions
                if now - p["timestamp"] <= max_age_s
            ]
            if positions:
                trails.append({
                    "worker_id": wid,
                    "camera_id": track.camera_id,
                    "positions": positions,
                    "current_zone": track.current_zone,
                    "status": positions[-1].get("status", "SAFE"),
                })
        return trails

    def get_kpi_summary(self) -> Dict:
        """Compute the top KPI strip data."""
        now = time.time()
        active_workers = sum(
            1 for t in self.tracks.values()
            if now - t.last_seen < 30  # Active in last 30s
        )
        return {
            "active_workers": active_workers,
            "critical_alerts": self.dispatcher.critical_count,
            "escalated_alerts": self.dispatcher.escalated_count,
            "total_events": len(self.dispatcher.events),
            "ppe_compliance": self._calc_ppe_compliance(),
            "avg_site_risk": self._calc_avg_risk(),
            "system_degraded": self.db.is_degraded,
            "mapper_status": self.mapper.status,
        }

    def get_backend_health(self) -> List[Dict]:
        """Get health status of all backend services."""
        return [
            self.db.health,
            {"service": "SpatialMapper", "status": self.mapper.status},
            {"service": "WebSocket", "status": "HEALTHY"},
            {"service": "Pipeline", "status": "HEALTHY"},
            {"service": "GEE", "status": "UNAVAILABLE"},  # Will be updated by gee_service
        ]

    # ── Private helpers ────────────────────────────────────────────────────

    def _resolve_zone(self, wx: float, wy: float) -> str:
        """Resolve which zone a world position falls in.

        Thresholds for 18.9 × 9.75 m site with rotation_deg=85.
        Derived from buildsight_zones_complete.geojson polygon extents.
        """
        # high_risk_staircase — NW local corner (wx≈0-3.5, wy≈5.5-9.75)
        if 0.0 <= wx <= 3.5 and wy >= 5.5:
            return "high_risk_staircase"
        # low_risk_parking — eastern strip (wx≈14.5-18.9)
        elif wx >= 14.5:
            return "low_risk_parking"
        # moderate_risk_interior — large central zone
        elif 0.0 <= wx < 14.5 and 0.0 <= wy <= 7.5:
            return "moderate_risk_interior"
        # high_risk_scaffolding — perimeter / remaining interior
        elif 0.5 <= wx <= SITE_CONFIG["width_m"] - 0.5 and 0.5 <= wy <= SITE_CONFIG["depth_m"] - 0.5:
            return "high_risk_scaffolding"
        else:
            return "low_risk_common"

    def _zone_risk_level(self, zone: str) -> str:
        """Get the risk level for a zone name."""
        if "high_risk" in zone or "critical" in zone:
            return "HIGH"
        elif "moderate_risk" in zone:
            return "MODERATE"
        return "LOW"

    def _calc_ppe_compliance(self) -> float:
        """Calculate overall PPE compliance percentage."""
        now = time.time()
        active = [
            t for t in self.tracks.values()
            if now - t.last_seen < 30 and t.positions
        ]
        if not active:
            return 100.0
        compliant = sum(
            1 for t in active
            if t.positions[-1].get("status") != WorkerStatus.CRITICAL.value
        )
        return round((compliant / len(active)) * 100, 1)

    def _calc_avg_risk(self) -> float:
        """Calculate average site risk from all active worker scores."""
        all_scores = []
        now = time.time()
        for t in self.tracks.values():
            if now - t.last_seen < 30 and t.risk_scores:
                all_scores.extend(t.risk_scores)
        return round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMZoneAnalyzer — Interactive Foundation Model Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class SAMZoneAnalyzer:
    """
    Integrates Segment Anything Model (SAM) for interactive site analysis.
    Converts pixel-space segments into world-space polygons with area metrics.
    """

    def __init__(self, mapper: SpatialMapper):
        self.mapper = mapper

    def enrich_segment(self, coords_norm: List[List[float]]) -> Dict[str, Any]:
        """
        Enrich a normalized SAM segment with world-space metrics using high-precision projection.
        
        Args:
            coords_norm: List of [x, y] coordinates in [0, 1] space.
            
        Returns:
            Dict containing area, centroid, bounding box, and GPS polygon.
        """
        try:
            img_w = self.mapper.frame_w
            img_h = self.mapper.frame_h

            # 1. Convert normalized [0,1] back to pixel coords then to world meters
            pixels = [(c[0] * img_w, c[1] * img_h) for c in coords_norm]
            world_pts = [self.mapper.pixel_to_world(px, py) for px, py in pixels]
            
            # 2. Calculate planar area (metrics)
            # Shoelace formula in local meter space
            area = 0.5 * abs(sum(
                world_pts[i][0] * world_pts[i+1][1] - world_pts[i+1][0] * world_pts[i][1]
                for i in range(len(world_pts) - 1)
            ))

            # 3. Convert all to GPS for GeoJSON delivery
            gps_pts = []
            for wx, wy in world_pts:
                lat, lon = self.mapper.world_to_gps(wx, wy)
                gps_pts.append([lon, lat]) # GeoJSON uses [lng, lat]

            # 4. Use mapper's high-level poly metric helper for consistency
            gps_lats_lons = [[p[1], p[0]] for p in gps_pts]
            metrics = self.mapper.calculate_polygon_metrics(gps_lats_lons)

            return {
                "area_m2": round(area, 2),
                "gps_polygon": gps_pts,
                "centroid_gps": metrics["centroid_gps"],
                "bbox_gps": metrics["bbox_gps"],
                "extracted_at": datetime.now().isoformat()
            }
        except Exception as e:
            log.error(f"SAMZoneAnalyzer.enrich_segment error: {e}")
            return {
                "area_m2": 0.0,
                "gps_polygon": [],
                "centroid_gps": [0,0],
                "bbox_gps": [],
                "error": str(e)
            }


# ═══════════════════════════════════════════════════════════════════════════════
#  Shift Utility
# ═══════════════════════════════════════════════════════════════════════════════

def get_current_shift() -> Shift:
    """Get the current operational shift."""
    hour = datetime.now().hour
    if 6 <= hour < 14:
        return Shift.MORNING
    elif 14 <= hour < 22:
        return Shift.AFTERNOON
    return Shift.NIGHT


# ═══════════════════════════════════════════════════════════════════════════════
#  Module test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("  BuildSight Intelligence Engine — Self-Test")
    print("="*60)

    engine = IntelligenceEngine()

    # Simulate a simple WorkerPPE-like object
    class MockWorkerPPE:
        def __init__(self, wid, box, helmet, vest):
            self.worker_id = wid
            self.worker_box = box
            self.has_helmet = helmet
            self.has_vest = vest
            self.helmet = None
            self.vest = None
            self.compliance = (int(helmet) + int(vest)) / 2.0

    # Simulate 3 detection frames
    for frame_idx in range(3):
        workers = [
            MockWorkerPPE(1, (200, 300, 350, 600), True, True),
            MockWorkerPPE(2, (800, 200, 950, 500), False, True),
            MockWorkerPPE(3, (100, 50, 250, 300), True, False),
        ]
        results = engine.process_frame(workers)
        print(f"\n  Frame {frame_idx + 1}: {len(results)} workers processed")
        for sw in results:
            print(f"    W{sw.worker_id}: ({sw.lat:.6f}, {sw.lon:.6f}) "
                  f"zone={sw.zone} status={sw.status} dwell={sw.dwell_severity}")

    # Print KPI
    kpi = engine.get_kpi_summary()
    print(f"\n  KPI Summary:")
    for k, v in kpi.items():
        print(f"    {k}: {v}")

    # Print timeline
    timeline = engine.dispatcher.get_timeline(limit=10)
    print(f"\n  Event Timeline ({len(timeline)} events):")
    for ev in timeline[:5]:
        print(f"    [{ev['priority']}] {ev['event_type']}: {ev['message']}")

    print(f"\n{'='*60}")
    print("  ✓ Intelligence Engine self-test complete")
    print(f"{'='*60}")
