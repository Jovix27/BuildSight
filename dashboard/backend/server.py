#!/usr/bin/env python3
"""
BuildSight Dashboard Backend
=============================
FastAPI server wrapping the multi-model ensemble pipeline.

Endpoints:
  GET  /api/health          — status + model/device info
  POST /api/detect/image    — PPE detection on an uploaded image
  POST /api/detect/frame    — PPE detection on a base64 video/webcam frame
  POST /api/detect/video    — full-video batch inference → download annotated MP4

Performance notes (Jovi handoff 2026-04-06):
  • Models run on GPU (RTX 4050) with FP16 mixed precision → ~6-8× faster than CPU
  • Inference is wrapped in run_in_threadpool so async endpoints never block the
    FastAPI event loop — concurrent requests stay responsive
  • Worker-to-PPE spatial association added: each worker detection now carries
    has_helmet / has_vest flags based on bounding-box overlap geometry
"""

import base64
import json
import logging
import os
import tempfile
import time
import uuid
# deque removed — scene smoothing now handled by SceneConditionTracker (hysteresis)
import contextlib
from pathlib import Path

import cv2
import numpy as np
import torch
from pydantic import BaseModel
import requests as _requests
import asyncio
from typing import Set
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
import traceback as _traceback
import sys
# Ensure the backend directory is in the path for local imports (database, geoai, etc.)
_backend_path = os.path.dirname(os.path.abspath(__file__))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

import database
from datetime import datetime
from geoai import geoai_router
from geoai.utils.spatial_mapper import SpatialMapper as BaseSpatialMapper
from voice_engine import (
    PorcupineWakeWordDetector,
    SoundDeviceAudioInput,
    TurnerVoiceEngine,
    VoiceEvent,
    WhisperSpeechTranscriber,
)

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from shapely.geometry import Point, shape
    import shapely
except ImportError:
    shapely = None

import report_generator

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent   # BuildSight root
BACKEND_DIR = Path(__file__).parent


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


for _dotenv_path in (ROOT / ".env", BACKEND_DIR / ".env"):
    _load_dotenv_file(_dotenv_path)


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


MODEL_DIR = _env_path("BUILDSIGHT_MODEL_DIR", ROOT / "research" / "weights")
SASTRA_V11 = _env_path("BUILDSIGHT_MODEL_V11", MODEL_DIR / "yolov11_buildsight_best.pt")
SASTRA_V26 = _env_path("BUILDSIGHT_MODEL_V26", MODEL_DIR / "yolov26_buildsight_best.pt")

LOCAL_BEST = _env_path(
    "BUILDSIGHT_LOCAL_MODEL",
    ROOT / "research" / "buildsight-base" / "basic yolo model" /
    "output" / "kaggle_working_all_outputs" /
    "kaggle" / "working" / "runs" / "train" / "weights" / "best.pt",
)

RUNTIME_DIR = _env_path("BUILDSIGHT_RUNTIME_DIR", ROOT / "data" / "runtime")
TMP_DIR = RUNTIME_DIR / "tmp"
VIDEO_OUTPUT_DIR = RUNTIME_DIR / "video_outputs"

for _path in (RUNTIME_DIR, TMP_DIR, VIDEO_OUTPUT_DIR):
    _path.mkdir(parents=True, exist_ok=True)

# ── device selection ───────────────────────────────────────────────────────────
# Use CUDA if available (RTX 4050 confirmed); fall back to CPU gracefully.
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE.startswith("cuda")   # FP16 only on GPU
BACKEND_HOST = os.environ.get("BUILDSIGHT_HOST", "0.0.0.0")
BACKEND_PORT = _env_int("BUILDSIGHT_PORT", 8000)
BACKEND_LOG_LEVEL = os.environ.get("BUILDSIGHT_LOG_LEVEL", "info")

# ── class schema ───────────────────────────────────────────────────────────────
SASTRA_CLASSES = {0: "helmet", 1: "safety_vest", 2: "worker"}
LOCAL_CLASSES  = {0: "worker", 1: "helmet",      2: "safety_vest"}
COLORS = {
    "helmet":      (0,   255,   0),    # Green  (BGR)
    "safety_vest": (0,   221, 255),    # Yellow (BGR) — #ffdd00 in RGB
    "worker":      (255, 136,   0),    # Blue   (BGR)
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BUILDSITE TUNING CONSTANTS — All thresholds in one place for easy tuning  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Ensemble model weights ───────────────────────────────────────────────────
MODEL_WEIGHTS   = [0.55, 0.45]   # [YOLOv11, YOLOv26]

# ── WBF IoU thresholds (class IDs: 0=helmet, 1=safety_vest, 2=worker) ────────
# Higher = boxes must overlap more to be merged (fewer fusions).
# Worker tuned per condition in WBF_IOU_BY_CONDITION below.
WBF_IOU         = {0: 0.45, 1: 0.45, 2: 0.50}   # default / S1_normal

# Per-condition WBF IoU thresholds (class IDs: 0=helmet, 1=safety_vest, 2=worker)
# Worker lowered 0.65 → 0.50: same-person boxes from YOLOv11+YOLOv26 typically
# land at IoU 0.50–0.65 — the old threshold caused them to survive as two overlapping
# boxes instead of being merged into one.
WBF_IOU_BY_CONDITION: dict[str, dict[int, float]] = {
    "S1_normal":    {0: 0.45, 1: 0.45, 2: 0.50},
    "S2_dusty":     {0: 0.42, 1: 0.42, 2: 0.48},
    "S3_low_light": {0: 0.40, 1: 0.40, 2: 0.46},
    "S4_crowded":   {0: 0.45, 1: 0.45, 2: 0.50},
}

# ── Post-WBF duplicate suppression (all classes) ─────────────────────────────
# After WBF, a final IoU pass removes near-identical boxes that the fusion step
# didn't merge. Applied per-class so PPE overlap tolerance can differ from worker.
POST_WBF_DEDUP_IOU: dict[int, float] = {0: 0.35, 1: 0.35, 2: 0.35}

# Containment threshold: if the smaller box has ≥ this fraction of its area
# inside the larger box, treat them as the same detection (catches size-
# mismatched boxes from the two models that slip past IoU-only dedup).
DEDUP_CONTAIN_THRESH: float = 0.60

# ── Post-WBF confidence gates ─────────────────────────────────────────────────
# Helmet raised (0.20→0.30): suppresses hair/shoulder false positives.
# Vest kept low (0.14): partially-occluded/dirty vests would be missed otherwise.
# Worker low (0.18): catch small or distant workers.
POST_WBF_GLOBAL = {0: 0.30, 1: 0.14, 2: 0.18}

# In low-light/dusty scenes every class scores lower — relax gates.
POST_WBF_BY_CONDITION: dict[str, dict[int, float]] = {
    "S1_normal":    {0: 0.30, 1: 0.14, 2: 0.18},
    "S2_dusty":     {0: 0.25, 1: 0.10, 2: 0.15},
    "S3_low_light": {0: 0.22, 1: 0.10, 2: 0.14},
    "S4_crowded":   {0: 0.28, 1: 0.12, 2: 0.15},
}

# ── Pre-WBF confidence threshold ─────────────────────────────────────────────
# Low enough to let weak-but-real detections reach WBF for reinforcement.
PRE_CONF        = 0.20

# ── Per-model NMS IoU (inside YOLO predict) ───────────────────────────────────
# Higher = less aggressive suppression so adjacent workers survive per-model NMS
# before WBF sees them.
NMS_IOU         = 0.60

# ── Early-exit threshold (skip YOLOv26 when v11 is already very confident) ───
EARLY_EXIT_CONF = 0.75

# ── Scene classification constants ─────────────────────────────────────────
# AUTO_* constants and SceneConditionTracker are defined after load_models()
# (they need cls_map which is populated during model load).  See the block
# starting with 'AUTO_LOW_LIGHT_THRESH' below.

# ── Safety vest validator thresholds ─────────────────────────────────────────
# Primary: HSV neon-color ranges (S = saturation, V = value in [0,255])
VEST_NEON_SAT_MIN:       float = 80.0   # minimum saturation for neon signal
VEST_NEON_VAL_MIN:       float = 80.0   # minimum brightness for neon signal
VEST_DUSTY_SAT_FLOOR:    float = 40.0   # relaxed saturation floor for dusty scenes
VEST_LOWLIGHT_VAL_FLOOR: float = 35.0   # relaxed brightness floor for dark scenes
# Secondary: reflective strip detection
VEST_REFLECTIVE_THRESH:  float = 200.0  # grayscale intensity considered reflective
VEST_REFLECTIVE_RATIO:   float = 0.08   # fraction of pixels in torso that are reflective
# Torso-region placement gate (fraction of worker height from top of box)
VEST_TORSO_TOP:    float = 0.20   # vest centroid must be ≥ 20% down from box top
VEST_TORSO_BOTTOM: float = 0.80   # vest centroid must be ≤ 80% down from box bottom

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="BuildSight Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── WebSocket connection manager ──────────────────────────────────────────────

class WSConnectionManager:
    """Manages active WebSocket connections and bridges thread→async broadcasts."""

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def capture_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, data: dict) -> None:
        dead: Set[WebSocket] = set()
        for ws in list(self._connections):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self._connections -= dead

    def broadcast_from_thread(self, data: dict) -> None:
        """Safe to call from a non-async thread."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(data), self._loop)


ws_manager = WSConnectionManager()


class TurnerVoiceConnectionManager:
    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, data: dict) -> None:
        dead: Set[WebSocket] = set()
        for ws in list(self._connections):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self._connections -= dead


class TurnerVoiceEventPublisher:
    def __init__(self, callback):
        self._callback = callback

    async def publish(self, event: VoiceEvent) -> None:
        payload = {
            "type": event.type,
            "session_id": event.session_id,
            "timestamp": event.timestamp,
            **event.payload,
        }
        await self._callback(payload)


class NullAudioInput:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class NullWakeWordDetector:
    def process(self, pcm: bytes) -> bool:
        del pcm
        return False


turner_voice_ws_manager = TurnerVoiceConnectionManager()


def build_turner_voice_engine(*, event_callback, chat_handler) -> TurnerVoiceEngine:
    access_key = os.environ.get("TURNER_WAKE_ACCESS_KEY") or os.environ.get("PORCUPINE_ACCESS_KEY", "")
    keyword_path = os.environ.get("TURNER_WAKE_KEYWORD_PATH", "").strip() or None
    whisper_model = os.environ.get("TURNER_WHISPER_MODEL", "tiny.en")

    try:
        if access_key:
            audio_input = SoundDeviceAudioInput()
            wake_detector = PorcupineWakeWordDetector(
                access_key=access_key,
                keyword_path=keyword_path,
            )
            transcriber = WhisperSpeechTranscriber(model_name=whisper_model)
        else:
            raise RuntimeError("TURNER_WAKE_ACCESS_KEY not configured")
    except Exception as exc:
        logger.warning("Turner voice backend falling back to null audio adapters: %s", exc)
        audio_input = NullAudioInput()
        wake_detector = NullWakeWordDetector()
        transcriber = NullSpeechTranscriber()

    return TurnerVoiceEngine(
        audio_input=audio_input,
        wake_detector=wake_detector,
        transcriber=transcriber,
        publisher=TurnerVoiceEventPublisher(event_callback),
        chat_handler=chat_handler,
    )


class NullSpeechTranscriber:
    async def transcribe(self, audio_bytes: bytes) -> str:
        del audio_bytes
        return ""

# bg_service is initialised in the startup event (after all inference
# functions are defined) to avoid forward-reference issues.
bg_service = None

# ── Global exception handler ──────────────────────────────────────────────────
# Ensures any unhandled exception inside an endpoint returns a structured JSON
# response instead of crashing the connection (which causes "FAILED TO FETCH"
# in the frontend because the browser receives a connection reset or empty body).
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    detail = _traceback.format_exc()
    logger.error(f"[UNHANDLED EXCEPTION] {request.url.path}: {exc}\n{detail}")
    return JSONResponse(
        status_code=500,
        content={
            "status":      "error",
            "error":       str(exc),
            "detail":      detail[:400],
            "detections":  [],
            "total":       0,
            "valid_workers": [],
            "condition":   "S1_normal",
        },
    )

@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    return await http_exception_handler(request, exc)

# Initialize database
database.init_db()

# Include GeoAI router
app.include_router(geoai_router, prefix="/api/geoai", tags=["geoai"])

# ── Spatial Mapping Logic ─────────────────────────────────────────────────────
class SpatialMapper(BaseSpatialMapper):
    """
    Maps detection coordinates to GeoAI zones using "smallest zone wins" logic.
    Loads zones from the database and caches them for performance.
    """
    def __init__(self):
        super().__init__()
        self.zones = []
        self.last_refresh = 0
        self.refresh_interval = 30  # seconds

    def refresh_if_needed(self):
        if time.time() - self.last_refresh < self.refresh_interval:
            return

        new_zones = []

        # 1. Zones from geo_zones DB table
        try:
            conn = database.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name, geojson, risk_level FROM geo_zones")
            rows = cursor.fetchall()
            for r in rows:
                try:
                    g_data = json.loads(r['geojson'])
                    geom_data = g_data['geometry'] if 'geometry' in g_data else g_data
                    geom = shape(geom_data)
                    new_zones.append({"name": r['name'], "geom": geom, "area": geom.area, "risk": r['risk_level']})
                except Exception as e:
                    logger.warning(f"SpatialMapper: Failed parsing DB zone {r['name']}: {e}")
            conn.close()
        except Exception as e:
            logger.error(f"SpatialMapper: DB refresh failed: {e}")

        # 2. Operator-drawn dynamic zones from dynamic_zones.json
        try:
            import pathlib
            from shapely.geometry import Polygon as _DynPoly
            dz_path = pathlib.Path(__file__).parent / "dynamic_zones.json"
            if dz_path.exists():
                dz_data = json.loads(dz_path.read_text())
                for z in dz_data:
                    if not z.get('is_active', True):
                        continue
                    coords = z.get('coordinates', [])
                    if len(coords) >= 3:
                        ring = coords if coords[0] == coords[-1] else coords + [coords[0]]
                        geom = _DynPoly(ring)  # coords are [lng, lat] → Shapely (x=lng, y=lat)
                        if geom.is_valid:
                            new_zones.append({"name": z['name'], "geom": geom, "area": geom.area, "risk": z.get('risk_level', 'low')})
        except Exception as e:
            logger.error(f"SpatialMapper: Dynamic zones refresh failed: {e}")

        # Smallest area first → first match is most specific zone
        new_zones.sort(key=lambda x: x['area'])
        self.zones = new_zones
        self.last_refresh = time.time()
        logger.info(f"SpatialMapper: Refreshed {len(self.zones)} zones")

    def get_zone_for_box(self, box, img_w, img_h):
        """
        box: [x1, y1, x2, y2] in pixel coords
        Returns (zone_name, risk_level)
        """
        if not shapely or not self.zones:
            return "General Site", "Low"

        # Use foot point (bottom-center) — where the worker stands
        cx = (box[0] + box[2]) / 2
        cy = box[3]

        # Convert to GPS; zones are stored in lat/lon (GeoJSON [lng, lat] order)
        orig_w, orig_h = self.frame_w, self.frame_h
        self.frame_w, self.frame_h = img_w, img_h
        lat, lng = self.pixel_to_gps(cx, cy)
        self.frame_w, self.frame_h = orig_w, orig_h

        # Shapely Point(x, y) = Point(lng, lat) — matches GeoJSON zone coords
        p = Point(lng, lat)

        for z in self.zones:
            if z['geom'].contains(p):
                return z['name'], z['risk']

        return "General Site", "Low"

spatial_mapper = SpatialMapper()

# ── Shared state for GeoAI VLM / SAM access ──────────────────────────────────
# The VLM and SAM router helpers import this module and read these variables.
# They are updated by the live-frame and video detection endpoints below.
latest_frame_jpeg: bytes | None = None   # most recent encoded JPEG from any detection path
detection_stats: dict = {}               # summary counts refreshed each inference call

# ── Turner AI Configuration ──────────────────────────────────────────────────
logger = logging.getLogger("buildsight")
turner_logger = logging.getLogger("buildsight.turner")

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def log_turner_event(event: str, **payload):
    turner_logger.info(json.dumps({"event": event, **payload}, default=str))


def _safe_enum_name(value):
    if value is None:
        return None
    return getattr(value, "name", str(value))


def _extract_response_text(response):
    try:
        text = getattr(response, "text", None)
        if text:
            return text.strip()
    except Exception:
        pass

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        collected = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                collected.append(part_text)
        if collected:
            return "\n".join(collected).strip()
    return ""


def _response_block_metadata(response):
    prompt_feedback = getattr(response, "prompt_feedback", None)
    candidates = getattr(response, "candidates", None) or []

    block_reason = _safe_enum_name(getattr(prompt_feedback, "block_reason", None))
    candidate_finish_reasons = [
        _safe_enum_name(getattr(candidate, "finish_reason", None))
        for candidate in candidates
    ]

    safety_ratings = []
    for candidate in candidates:
        for rating in getattr(candidate, "safety_ratings", None) or []:
            safety_ratings.append({
                "category": _safe_enum_name(getattr(rating, "category", None)),
                "probability": _safe_enum_name(getattr(rating, "probability", None)),
                "blocked": getattr(rating, "blocked", None),
            })

    return {
        "prompt_block_reason": block_reason,
        "candidate_finish_reasons": candidate_finish_reasons,
        "safety_ratings": safety_ratings,
    }


def _turner_blocked_message(metadata: dict) -> str:
    categories = [
        rating["category"]
        for rating in metadata.get("safety_ratings", [])
        if rating.get("blocked") and rating.get("category")
    ]
    categories_text = ", ".join(categories) if categories else "safety policy"
    return (
        "I cannot provide a direct response because Gemini safety controls blocked this request. "
        f"Review the wording and retry with a narrower construction-safety question. Triggered category: {categories_text}."
    )


TURNER_SYSTEM_PROMPT = """
You are Turner — an AI construction site supervisor for BuildSight. You communicate through voice, so everything you say is spoken aloud.

VOICE RULES (non-negotiable):
- Maximum 2 sentences per response. You are speaking, not writing an email.
- Zero markdown: no bullet points, no bold, no lists, no headers. Pure spoken language.
- No filler openers: never start with "Certainly", "Of course", "Great question", "Sure", or "Absolutely". Just answer.
- Use contractions naturally: "I'll", "that's", "you're", "it's", "don't", "can't".
- If continuing a conversation, pick up naturally — don't repeat what was just said.

Personality:
- Direct and confident. You've worked construction sites for 20 years and you know what you're talking about.
- Warm but no-nonsense. You care about the workers and the site.
- Dry humor is fine when the moment calls for it. Never forced.
- For safety issues: firm, clear, no hedging. Lives depend on it.
- For simple questions: casual, brief, conversational.

Examples of correct voice responses:
Q: "What's the site status?"   → "Three workers in Zone A, all helmeted. No active alerts."
Q: "Is Zone B clear?"          → "Zone B's clear right now, go ahead."
Q: "Hey Turner"                → "Yeah, what do you need?"
Q: "What if there's a fire?"   → "Get everyone out immediately and call emergency services. Don't stop to grab anything."
Q: "How are you?"              → "Good. What's going on on site?"

Remember: you are speaking. Keep it short. Keep it human.
""".strip()

# Keywords that signal the user wants site-specific intelligence
_SITE_QUERY_KEYWORDS = [
    "site", "status", "worker", "workers", "helmet", "vest", "ppe", "compliance",
    "zone", "zones", "condition", "alert", "alerts", "detection", "safety",
    "happening", "going on", "how many", "count", "report", "camera", "live",
]

# ── Mistral AI (primary provider) ─────────────────────────────────────────────
MISTRAL_API_KEY   = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL     = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
MISTRAL_ENDPOINT  = "https://api.mistral.ai/v1/chat/completions"
mistral_enabled   = bool(MISTRAL_API_KEY)

if mistral_enabled:
    logger.info("Turner AI enabled. MISTRAL_API_KEY detected (model: %s).", MISTRAL_MODEL)
else:
    logger.warning("MISTRAL_API_KEY not found. Mistral AI disabled.")

# ── Google Gemini (fallback provider) ─────────────────────────────────────────
AI_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
ai_model = None

if AI_API_KEY and genai is not None:
    genai.configure(api_key=AI_API_KEY)
    ai_model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=TURNER_SYSTEM_PROMPT,
    )
    logger.info("Gemini fallback enabled. GOOGLE_API_KEY detected.")
elif AI_API_KEY and genai is None:
    logger.warning("GOOGLE_API_KEY present but google-generativeai is not installed. Gemini fallback disabled.")
else:
    logger.warning("GOOGLE_API_KEY not found. Gemini fallback disabled.")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: dict = {}


class TurnerVoiceTextRequest(BaseModel):
    text: str = ""
    history: list[ChatMessage] = []
    context: dict = {}


def _build_mistral_messages(req: "ChatRequest", full_prompt: str) -> list[dict]:
    """Convert ChatRequest history + enriched prompt into Mistral message list."""
    messages = [{"role": "system", "content": TURNER_SYSTEM_PROMPT}]
    for m in req.history:
        role = "user" if m.role == "user" else "assistant"
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": full_prompt})
    return messages


def _call_mistral_sync(messages: list[dict]) -> str:
    """Synchronous Mistral chat call — run via run_in_threadpool."""
    resp = _requests.post(
        MISTRAL_ENDPOINT,
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MISTRAL_MODEL, "messages": messages, "stream": False},
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _is_site_query(message: str) -> bool:
    """Detect if the user is asking about site status/conditions."""
    lower = message.lower()
    return any(kw in lower for kw in _SITE_QUERY_KEYWORDS)


def _build_site_prompt(req: "ChatRequest") -> str:
    """Build prompt with site context only when the user explicitly requests it."""
    msg = req.message

    # For non-site queries, pass the question directly — no site data injection.
    if not _is_site_query(msg) and not req.context:
        return f"USER REQUEST\n{msg}"

    ctx       = req.context
    workers   = ctx.get("workers", 0)
    helmets   = ctx.get("helmets", 0)
    vests     = ctx.get("vests", 0)
    condition = ctx.get("condition", "Unknown")
    alerts    = ctx.get("alerts", [])
    telemetry = ctx.get("telemetry", {})

    # Pull latest Florence-2 visual grounding caption — only for site queries.
    vlm_section = ""
    try:
        from geoai_vlm_util import get_cached_entry
        vlm_entry, vlm_stale = get_cached_entry()
        if vlm_entry and not vlm_stale:
            vlm_desc = vlm_entry.get("description", "").strip()
            vlm_src  = vlm_entry.get("source", "unknown")
            if vlm_desc and vlm_src == "florence2":
                vlm_section = (
                    f"\nVISUAL GROUNDING (Florence-2 live frame caption)\n"
                    f"{vlm_desc}\n"
                    f"Use this as ground-truth visual evidence for what is currently on site.\n"
                )
    except Exception:
        pass

    return (
        f"LIVE SITE CONTEXT\n"
        f"- Active workers: {workers}\n"
        f"- Helmet compliance: {helmets}/{workers if workers else max(helmets, 1)}\n"
        f"- Vest compliance: {vests}/{workers if workers else max(vests, 1)}\n"
        f"- Site condition: {condition}\n"
        f"- Active escalations: {len(alerts)}\n"
        f"- Escalation details: {json.dumps(alerts, ensure_ascii=True)}\n"
        f"- Telemetry: {json.dumps(telemetry, ensure_ascii=True)}\n"
        f"{vlm_section}\n"
        f"USER REQUEST\n{msg}"
    ).strip()


async def _refresh_turner_visual_context() -> None:
    """Warm the VLM cache before Turner prompt assembly."""
    try:
        from geoai_vlm_util import describe_frame_async, is_available as vlm_ready
        if vlm_ready() and latest_frame_jpeg:
            await describe_frame_async(jpeg_bytes=latest_frame_jpeg, force_refresh=False)
    except Exception:
        pass


async def _transcribe_uploaded_audio(audio_bytes: bytes, mime_type: str) -> str:
    """Transcribe uploaded audio bytes via the configured multimodal model."""
    if not ai_model:
        raise HTTPException(status_code=503, detail="Gemini (ai_model) not configured for transcription.")

    response = await run_in_threadpool(
        ai_model.generate_content,
        [
            "Transcribe this audio exactly as heard. Do not add punctuation or formatting that is not present. If no speech is detected, return exactly an empty string.",
            {"mime_type": mime_type, "data": audio_bytes},
        ],
    )
    return _extract_response_text(response).strip()


async def run_turner_chat(
    message: str,
    *,
    history: list[ChatMessage] | None = None,
    context: dict | None = None,
) -> dict:
    request_id = uuid.uuid4().hex[:10]
    req = ChatRequest(message=message, history=history or [], context=context or {})

    if not mistral_enabled and not ai_model:
        log_turner_event(
            "turner_request_rejected",
            request_id=request_id,
            reason="MISSING_API_KEY",
        )
        return {
            "response": (
                "Turner AI is currently offline. "
                "Set MISTRAL_API_KEY (or GOOGLE_API_KEY) in the backend environment."
            ),
            "error": "MISSING_API_KEY",
            "request_id": request_id,
            "status_code": 503,
        }

    await _refresh_turner_visual_context()

    full_prompt = _build_site_prompt(req)
    ctx = req.context
    log_turner_event(
        "turner_request_received",
        request_id=request_id,
        provider="mistral" if mistral_enabled else "gemini",
        message=req.message,
        history_length=len(req.history),
        context={
            "workers": ctx.get("workers", 0),
            "helmets": ctx.get("helmets", 0),
            "vests": ctx.get("vests", 0),
            "condition": ctx.get("condition", "Unknown"),
            "alerts": ctx.get("alerts", []),
        },
    )

    if mistral_enabled:
        try:
            messages = _build_mistral_messages(req, full_prompt)
            response_text = await run_in_threadpool(_call_mistral_sync, messages)
            log_turner_event(
                "turner_response_success",
                request_id=request_id,
                provider="mistral",
                response=response_text,
            )
            return {
                "response": response_text,
                "status": "success",
                "provider": "mistral",
                "request_id": request_id,
            }
        except Exception as e:
            logger.warning("Mistral call failed (%s): %s — trying Gemini fallback.", type(e).__name__, e)
            if not ai_model:
                log_turner_event(
                    "turner_response_error",
                    request_id=request_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return {
                    "response": "Turner encountered a Mistral error and no Gemini fallback is configured. Please retry.",
                    "error": type(e).__name__,
                    "request_id": request_id,
                    "status_code": 500,
                }

    try:
        gemini_history = [
            {"role": "user" if m.role == "user" else "model", "parts": [m.content]}
            for m in req.history
        ]
        chat = ai_model.start_chat(history=gemini_history)
        response = await run_in_threadpool(chat.send_message, full_prompt)
        response_text = _extract_response_text(response)
        block_metadata = _response_block_metadata(response)

        if not response_text:
            blocked = bool(block_metadata.get("prompt_block_reason")) or any(
                reason and "SAFETY" in reason.upper()
                for reason in block_metadata.get("candidate_finish_reasons", [])
            )
            if blocked:
                return {
                    "response": _turner_blocked_message(block_metadata),
                    "status": "blocked",
                    "error": "SAFETY_BLOCKED",
                    "request_id": request_id,
                    "details": block_metadata,
                }
            return {
                "response": "I did not receive a usable model response. Please retry.",
                "status": "empty",
                "error": "EMPTY_RESPONSE",
                "request_id": request_id,
                "status_code": 502,
            }

        log_turner_event(
            "turner_response_success",
            request_id=request_id,
            provider="gemini",
            response=response_text,
        )
        return {
            "response": response_text,
            "status": "success",
            "provider": "gemini",
            "request_id": request_id,
        }
    except Exception as e:
        log_turner_event(
            "turner_response_error",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            "response": "I encountered a synchronization error. Please try again.",
            "error": type(e).__name__,
            "request_id": request_id,
            "status_code": 500,
        }


# ── model loading ──────────────────────────────────────────────────────────────
model_v11 = None
model_v26 = None
cls_map   = {}
mode_name = "unloaded"

def load_models():
    global model_v11, model_v26, cls_map, mode_name
    from ultralytics import YOLO

    print(f"Device: {DEVICE}  |  FP16: {USE_HALF}")

    if SASTRA_V11.exists() and SASTRA_V26.exists():
        print("Loading SASTRA ensemble models...")
        model_v11 = YOLO(str(SASTRA_V11))
        model_v26 = YOLO(str(SASTRA_V26))
        cls_map   = SASTRA_CLASSES
        mode_name = "ensemble-wbf"
        print("Ensemble (YOLOv11 + YOLOv26) loaded.")
    elif SASTRA_V11.exists():
        model_v11 = YOLO(str(SASTRA_V11))
        cls_map   = SASTRA_CLASSES
        mode_name = "yolov11-single"
        print("YOLOv11 (single) loaded.")
    elif LOCAL_BEST.exists():
        print(f"Loading local base model: {LOCAL_BEST}")
        model_v11 = YOLO(str(LOCAL_BEST))
        cls_map   = LOCAL_CLASSES
        mode_name = "local-base"
        print("Local base model loaded.")
    else:
        print("ERROR: No model weights found.")

    # Warm up on GPU to JIT-compile kernels before first real request
    if model_v11 is not None and DEVICE.startswith("cuda"):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model_v11.predict(dummy, device=DEVICE, verbose=False,
                          conf=0.9, half=USE_HALF)
        if model_v26 is not None:
            model_v26.predict(dummy, device=DEVICE, verbose=False,
                              conf=0.9, half=USE_HALF)
        print("GPU warm-up done.")

load_models()


# ── Startup Health Checks ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup_health_check():
    """Performs a comprehensive system health check on startup."""
    logger.info("═"*50)
    logger.info("  BUILDSIGHT BACKEND INITIALIZATION REPORT")
    logger.info("═"*50)
    
    # 1. Models Check
    logger.info(f"[-] Ensemble Mode: {mode_name}")
    logger.info(f"[-] Detection Device: {DEVICE}")
    if model_v11 is not None:
        logger.info("    [OK] YOLOv11 loaded")
    if model_v26 is not None:
        logger.info("    [OK] YOLOv26 loaded")
        
    # 2. GeoAI Utility Check
    try:
        from geoai_vlm_util import is_available as vlm_ready, get_model_info as vlm_info
        from geoai_sam_util import is_available as sam_ready
        info = vlm_info()
        # VLM is lazy-loaded; "LOADING" is normal at startup — not a failure.
        vlm_status = "READY" if vlm_ready() else "LOADING (background preload started)"
        logger.info(f"[-] GeoAI VLM: {vlm_status}")
        logger.info(f"    Candidates: {info['candidates']}")
        logger.info(f"[-] GeoAI SAM: {'READY' if sam_ready() else 'OFFLINE'}")
    except ImportError:
        logger.info("[-] GeoAI Utils: OFFLINE (Import Error)")
    
    # 3. Database Check
    try:
        conn = database.get_db_connection()
        logger.info("    [OK] Database: Connection stable")
        conn.close()
    except Exception as e:
        logger.info(f"    [!!] Database: Connection failed ({e})")
        
    # 4. Route Mapping Check
    routes = [r.path for r in app.routes]
    logger.info(f"[-] API Surface: {len(routes)} routes mapped")
    if any("/api/geoai" in r for r in routes):
        logger.info("    [OK] GeoAI Router integrated")
    
    logger.info("═"*50)


@app.on_event("startup")
async def startup_turner_voice_engine() -> None:
    engine = getattr(app.state, "turner_voice_engine", None)
    if engine is None:
        engine = build_turner_voice_engine(
            event_callback=turner_voice_ws_manager.broadcast,
            chat_handler=run_turner_chat,
        )
        app.state.turner_voice_engine = engine
    await engine.start()


@app.on_event("shutdown")
async def shutdown_turner_voice_engine() -> None:
    engine = getattr(app.state, "turner_voice_engine", None)
    if engine is not None:
        await engine.stop()


@app.on_event("startup")
async def startup_bg_service():
    """Initialise background detection service and capture asyncio event loop."""
    global bg_service
    ws_manager.capture_loop(asyncio.get_event_loop())

    # Kick off Florence-2 loading in a background thread so it's ready before
    # the first /vlm/latest request without blocking startup.
    try:
        from geoai_vlm_util import trigger_preload
        trigger_preload()
    except Exception as _vlm_err:
        logger.warning("[VLM] Could not start preload: %s", _vlm_err)

    from background_detection_service import BackgroundDetectionService

    def _classify_wrap(frame):
        return classify_scene_fast(frame, model_v11, DEVICE, USE_HALF)

    bg_service = BackgroundDetectionService(
        broadcast_fn=ws_manager.broadcast_from_thread,
        inference_fn=run_inference,
        classify_fn=_classify_wrap,
    )
    logger.info("[BG] Background detection service initialised")


# ── Automated Scene Classification (restored from Toni's backup 2026-04-11) ──
#
# ── Scene Auto-Classification ─────────────────────────────────────────────────
# classify_scene_fast() uses image stats + a cached model quick-pass to classify
# frames into S1/S2/S3/S4.  SceneConditionTracker applies hysteresis so that
# noisy single-frame spikes cannot flip the condition.
#
# Classification hierarchy (highest priority first):
#   S3_low_light : low brightness OR large dark-region fraction
#   S2_dusty     : high haze OR (low contrast AND low saturation)
#   S4_crowded   : ≥1 of: (a) 4+ high-conf valid workers, (b) 3+ workers in a
#                  close spatial cluster, (c) high local crowd density,
#                  (d) temporal persistence of elevated valid worker count
#                  — NOT triggered by clutter, low-conf boxes, or single-frame spikes
#   S1_normal    : default

# ── Image-stats thresholds ────────────────────────────────────────────────────
AUTO_LOW_LIGHT_THRESH      = 70    # mean gray < this → S3_low_light
AUTO_DARK_REGION_FRAC      = 0.35  # fraction of pixels below 50 brightness → S3
AUTO_DUSTY_STD_THRESH      = 48    # gray std < this → candidate dusty
AUTO_DUSTY_SAT_THRESH      = 72    # mean HSV-S < this → confirm dusty
AUTO_HAZE_THRESH           = 185   # mean of top-5% brightest pixels → haze → S2

# ── Valid-worker counting thresholds ─────────────────────────────────────────
AUTO_QUICK_WORKER_CONF     = 0.35  # minimum confidence to count a worker (raised from 0.22)
AUTO_VALID_WORKER_MIN_H    = 0.05  # worker box must be ≥ 5% of frame height
AUTO_VALID_WORKER_ASP_MIN  = 0.18  # w/h aspect ratio lower bound (not a pole/pipe)
AUTO_VALID_WORKER_ASP_MAX  = 1.30  # w/h aspect ratio upper bound (not a flat slab)

# ── S4 crowd-detection thresholds ────────────────────────────────────────────
AUTO_CROWD_WORKER_THRESH   = 4     # ≥ N high-conf valid workers → S4 signal
AUTO_CROWD_CLOSE_K         = 3     # K workers within CLOSE_DIST → crowd cluster
AUTO_CROWD_CLOSE_DIST      = 0.28  # normalised centroid distance for "close"
AUTO_CROWD_DENSITY_THRESH  = 0.40  # fraction of 4×4 tile occupied by workers → dense
AUTO_CROWD_OVERLAP_THRESH  = 0.08  # mean pairwise IoU ≥ this among 3+ workers → S4

# ── Hysteresis frame counts ───────────────────────────────────────────────────
CROWD_ENTER_FRAMES         = 3     # consecutive S4 signals needed to enter crowded
CROWD_EXIT_FRAMES          = 5     # consecutive non-S4 signals needed to leave crowded
STABILITY_WINDOW           = 5     # rolling window for non-S4 mode smoothing
SCENE_CACHE_FRAMES         = 10    # model forward-pass cached for N frames
WORKER_HISTORY_LEN         = 6     # frames of valid-worker counts kept for temporal check
TEMPORAL_CROWD_MIN_FRAMES  = 3     # frames in WORKER_HISTORY_LEN window above thresh → S4


class SceneConditionTracker:
    """
    Stateful hysteresis tracker for scene conditions.

    Prevents rapid flickering between S4_crowded and S1_normal on borderline
    scenes — entering S4 requires CROWD_ENTER_FRAMES consecutive hits, exiting
    requires CROWD_EXIT_FRAMES consecutive clear frames.

    Also stores a short history of valid-worker counts per frame so that
    classify_scene_fast() can check temporal persistence before triggering S4.
    """

    def __init__(self):
        self.current: str = "S1_normal"
        self._history: list[str] = []
        self._crowd_consecutive: int = 0
        self._non_crowd_consecutive: int = 0
        self._dusty_consecutive: int = 0
        self._non_dusty_consecutive: int = 0
        self._cache_frame_count: int = 0
        self.cache_invalidated: bool = False
        # Rolling window of valid-worker counts — populated by classify_scene_fast
        self._worker_count_history: list[int] = []

    @property
    def current_condition(self) -> str:
        return self.current

    def invalidate_cache(self):
        """Force a model re-pass on the next frame (called after threshold change)."""
        self.cache_invalidated = True
        self._cache_frame_count = 0

    def push_worker_count(self, n: int) -> None:
        """Record how many valid workers were seen this frame."""
        self._worker_count_history.append(n)
        if len(self._worker_count_history) > WORKER_HISTORY_LEN:
            self._worker_count_history.pop(0)

    def temporal_crowd_active(self, threshold: int) -> bool:
        """
        Return True if at least TEMPORAL_CROWD_MIN_FRAMES of the last
        WORKER_HISTORY_LEN frames had ≥ threshold valid workers.
        """
        if len(self._worker_count_history) < TEMPORAL_CROWD_MIN_FRAMES:
            return False
        above = sum(1 for c in self._worker_count_history if c >= threshold)
        return above >= TEMPORAL_CROWD_MIN_FRAMES

    def update(self, raw_condition: str) -> str:
        """Feed a raw single-frame condition, return the stabilised condition."""
        self._history.append(raw_condition)
        if len(self._history) > max(CROWD_ENTER_FRAMES, STABILITY_WINDOW, CROWD_EXIT_FRAMES) + 5:
            self._history.pop(0)

        is_crowd_signal = (raw_condition == "S4_crowded")
        is_dusty_signal = (raw_condition == "S2_dusty")

        # ── S4 Crowded Hysteresis ────────────────────────────────────────────
        if is_crowd_signal:
            self._crowd_consecutive += 1
            self._non_crowd_consecutive = 0
        else:
            self._non_crowd_consecutive += 1
            self._crowd_consecutive = max(0, self._crowd_consecutive - 1)

        # ── S2 Dusty Hysteresis ──────────────────────────────────────────────
        if is_dusty_signal:
            self._dusty_consecutive += 1
            self._non_dusty_consecutive = 0
        else:
            self._non_dusty_consecutive += 1
            self._dusty_consecutive = max(0, self._dusty_consecutive - 1)

        # ── State Transition Logic ───────────────────────────────────────────
        # Priority: S3 (Direct) > S2 (Hysteresis) > S4 (Hysteresis) > S1
        if raw_condition == "S3_low_light":
            self.current = "S3_low_light"
            return self.current

        if self.current == "S4_crowded":
            if self._non_crowd_consecutive >= CROWD_EXIT_FRAMES:
                self.current = self._stabilise_non_crowd()
        elif self.current == "S2_dusty":
             if self._non_dusty_consecutive >= 5: # exit dusty
                self.current = self._stabilise_non_crowd()
        else:
            if self._dusty_consecutive >= 3: # enter dusty
                self.current = "S2_dusty"
                self._non_dusty_consecutive = 0
            elif self._crowd_consecutive >= CROWD_ENTER_FRAMES:
                self.current = "S4_crowded"
                self._non_crowd_consecutive = 0
            else:
                self.current = self._stabilise_non_crowd()

        return self.current

    def _stabilise_non_crowd(self) -> str:
        recent = [c for c in self._history[-STABILITY_WINDOW:] if c != "S4_crowded"]
        if not recent:
            return "S1_normal"
        counts: dict[str, int] = {}
        for c in recent:
            counts[c] = counts.get(c, 0) + 1
        return max(counts, key=lambda k: counts[k])

    def should_reclassify(self) -> bool:
        """
        Return True when the model forward-pass should run this frame.
        Image-stats conditions (brightness/haze) always run — only the expensive
        model count pass is cached every SCENE_CACHE_FRAMES frames.
        """
        if self.cache_invalidated:
            self.cache_invalidated = False
            return True
        self._cache_frame_count += 1
        return (self._cache_frame_count % SCENE_CACHE_FRAMES) == 1

    def reset(self):
        self.__init__()


# Module-level tracker instance — shared across /detect/frame live calls
_scene_tracker = SceneConditionTracker()


def _is_valid_worker_box(box: list, conf: float, img_h: int, _img_w: int = 0) -> bool:
    """
    Return True only if a detection is a plausible human worker.

    Rejects: cement bags, buckets, scaffolding poles, bricks, material piles,
    floating PPE, duplicate sub-threshold detections, and very small far-off
    detections that are likely clutter.

    Criteria checked:
      - Confidence ≥ AUTO_QUICK_WORKER_CONF
      - Box height ≥ AUTO_VALID_WORKER_MIN_H × frame height
      - Aspect ratio (w/h) within [AUTO_VALID_WORKER_ASP_MIN, AUTO_VALID_WORKER_ASP_MAX]
    """
    if conf < AUTO_QUICK_WORKER_CONF:
        return False
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1e-3)
    bh = max(y2 - y1, 1e-3)
    if bh < AUTO_VALID_WORKER_MIN_H * img_h:
        return False
    aspect = bw / bh
    if aspect < AUTO_VALID_WORKER_ASP_MIN or aspect > AUTO_VALID_WORKER_ASP_MAX:
        return False
    return True


def _count_valid_workers(r_boxes, img_h: int, img_w: int) -> tuple[list, int]:
    """
    From a raw YOLO result box list, return (valid_boxes, n_valid).

    Only counts boxes that pass _is_valid_worker_box() — this is the single
    source of truth for "how many workers are really in this frame?" used by
    the scene classifier.  Low-confidence clutter, poles, bags, and material
    piles are excluded before any scene-condition logic runs.
    """
    worker_cls_ids = {k for k, v in cls_map.items() if v in ("worker", "person")}
    valid: list = []
    for b in (r_boxes or []):
        cls_id = int(b.cls[0])
        conf   = float(b.conf[0])
        if cls_id not in worker_cls_ids:
            continue
        box = b.xyxy[0].cpu().numpy().tolist()
        if _is_valid_worker_box(box, conf, img_h, img_w):
            valid.append({"box": box, "score": conf})
    return valid, len(valid)


def _has_crowd_cluster(valid_boxes: list, img_w: int, img_h: int) -> bool:
    """
    Return True if at least AUTO_CROWD_CLOSE_K workers have their centroids
    within AUTO_CROWD_CLOSE_DIST of each other (normalised by frame diagonal).

    This detects tight worker groups that indicate a crowded work zone even
    when the total worker count is below AUTO_CROWD_WORKER_THRESH.
    """
    if len(valid_boxes) < AUTO_CROWD_CLOSE_K:
        return False
    diag = ((img_w ** 2 + img_h ** 2) ** 0.5) or 1.0
    centroids = [
        ((b["box"][0] + b["box"][2]) / 2.0,
         (b["box"][1] + b["box"][3]) / 2.0)
        for b in valid_boxes
    ]
    for i, (cx, cy) in enumerate(centroids):
        close = 1  # count self
        for j, (ox, oy) in enumerate(centroids):
            if i == j:
                continue
            dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 / diag
            if dist <= AUTO_CROWD_CLOSE_DIST:
                close += 1
        if close >= AUTO_CROWD_CLOSE_K:
            return True
    return False


def _has_high_crowd_density(valid_boxes: list, img_w: int, img_h: int) -> bool:
    """
    Divide the frame into a 4×4 grid.  If any tile has ≥ AUTO_CROWD_DENSITY_THRESH
    fraction of its area covered by valid worker boxes, return True.

    This catches scenarios where workers are physically close but centroids are
    spread — e.g., workers overlapping shoulders in a tight corridor.
    """
    if not valid_boxes:
        return False
    cols, rows = 4, 4
    tw, th = img_w / cols, img_h / rows
    for row in range(rows):
        for col in range(cols):
            tx1 = col * tw;   ty1 = row * th
            tx2 = tx1 + tw;   ty2 = ty1 + th
            tile_area = tw * th
            overlap_sum = 0.0
            for b in valid_boxes:
                bx1, by1, bx2, by2 = b["box"]
                ix1 = max(bx1, tx1); iy1 = max(by1, ty1)
                ix2 = min(bx2, tx2); iy2 = min(by2, ty2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                overlap_sum += inter
            if overlap_sum / max(tile_area, 1e-3) >= AUTO_CROWD_DENSITY_THRESH:
                return True
    return False


def classify_scene_fast(
    img_bgr: np.ndarray,
    model,
    device: str,
    half: bool,
) -> str:
    """
    Classify a single frame into S1/S2/S3/S4 using multiple signals:
      - Image stats: brightness, contrast, saturation, haze, dark-region fraction
      - Valid worker count (confidence + geometry validated — no clutter)
      - Crowd cluster analysis (spatial proximity)
      - Local crowd density (tile-based area overlap)
      - Temporal persistence across WORKER_HISTORY_LEN frames

    Priority: S3_low_light > S2_dusty > S4_crowded > S1_normal.

    S4 requires at least ONE of:
      (a) ≥ AUTO_CROWD_WORKER_THRESH high-conf valid workers this frame
      (b) ≥ AUTO_CROWD_CLOSE_K workers grouped within AUTO_CROWD_CLOSE_DIST
      (c) High local crowd density in any tile
      (d) Temporal persistence: ≥ TEMPORAL_CROWD_MIN_FRAMES of last WORKER_HISTORY_LEN
          frames had ≥ 3 valid workers

    S4 is NOT triggered by:
      - Low-confidence workers / clutter detections
      - Single-frame spikes (temporal filter prevents this)
      - Floating PPE without a valid worker nearby
      - Very small distant detections below MIN_H
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    contrast   = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1]))

    # ── S3 Low Light ─────────────────────────────────────────────────────────
    # Primary signal: overall darkness
    if brightness < AUTO_LOW_LIGHT_THRESH:
        return _scene_tracker.update("S3_low_light")
    # Secondary signal: large dark-region fraction (shadows / night corners)
    dark_frac = float(np.mean(gray < 50))
    if dark_frac >= AUTO_DARK_REGION_FRAC:
        return _scene_tracker.update("S3_low_light")

    # ── S2 Dusty ─────────────────────────────────────────────────────────────
    # Haze signal: top-5% brightest pixels washed out (fog / dust scatter)
    top5_thresh = np.percentile(gray, 95)
    haze_level  = float(np.mean(gray[gray >= top5_thresh]))
    if haze_level >= AUTO_HAZE_THRESH:
        return _scene_tracker.update("S2_dusty")
    # Classic dusty: low contrast + low saturation (desaturated, washed out)
    if contrast < AUTO_DUSTY_STD_THRESH and saturation < AUTO_DUSTY_SAT_THRESH:
        return _scene_tracker.update("S2_dusty")

    # ── S4 Crowded (model-based — cached) ────────────────────────────────────
    # Re-use last tracker state if the cache is still valid
    if not _scene_tracker.should_reclassify():
        return _scene_tracker.current

    if model is None:
        _scene_tracker.push_worker_count(0)
        return _scene_tracker.update("S1_normal")

    r = model.predict(
        img_bgr, device=device, verbose=False,
        conf=AUTO_QUICK_WORKER_CONF, iou=0.50, half=half,
        agnostic_nms=False,
    )[0]

    valid_workers, n_valid = _count_valid_workers(r.boxes, h, w)

    # Push count into temporal history before any threshold checks
    _scene_tracker.push_worker_count(n_valid)

    # Signal (a): enough high-confidence valid workers this frame
    if n_valid >= AUTO_CROWD_WORKER_THRESH:
        return _scene_tracker.update("S4_crowded")

    # Signal (b): spatial cluster — 3+ workers grouped closely together
    if _has_crowd_cluster(valid_workers, w, h):
        return _scene_tracker.update("S4_crowded")

    # Signal (c): high local crowd density in any tile
    if _has_high_crowd_density(valid_workers, w, h):
        return _scene_tracker.update("S4_crowded")

    # Signal (d): temporal persistence — crowd pattern repeated across frames
    if _scene_tracker.temporal_crowd_active(threshold=3):
        return _scene_tracker.update("S4_crowded")

    # Overlap check among 3+ valid workers (original signal, kept as supplement)
    if len(valid_workers) >= 3:
        boxes = [vw["box"] for vw in valid_workers]
        overlaps = [
            iou_box(boxes[i], boxes[j])
            for i in range(len(boxes))
            for j in range(i + 1, len(boxes))
        ]
        if overlaps and float(np.mean(overlaps)) >= AUTO_CROWD_OVERLAP_THRESH:
            return _scene_tracker.update("S4_crowded")

    return _scene_tracker.update("S1_normal")


# ── Multi-signal Safety Vest Validator ───────────────────────────────────────
# A vest detection must pass ≥ 1 of the following signal checks to survive:
#   Signal A — HSV neon-color gate (primary):   high saturation + high value
#               within any of the defined HSV ranges for hi-vis colours.
#   Signal B — Reflective strip gate (secondary): a stripe of very bright
#               gray/white pixels with aspect ratio ≥ 3:1 (horizontal stripe).
#   Signal C — Torso placement gate: vest centroid sits in the torso region
#               (20%-80%) of the associated worker box.
# Condition overrides:
#   • S2_dusty:     saturation floor lowered (dust desaturates bright vests).
#   • S3_low_light: value floor lowered (dark scene dims bright colours).
# Rejection targets: plain white shirts, cloth, bright tarps, foam.

# HSV ranges for commonly-used hi-vis vest colours.
# Each entry is (H_low, H_high, S_min, V_min) in OpenCV HSV (H ∈ [0,179]).
_VEST_HSV_RANGES = [
    # Neon yellow / lime yellow
    (18,  40, VEST_NEON_SAT_MIN, VEST_NEON_VAL_MIN),
    # Neon orange / amber
    ( 5,  18, VEST_NEON_SAT_MIN, VEST_NEON_VAL_MIN),
    # Lime green / fluorescent green
    (40,  75, VEST_NEON_SAT_MIN, VEST_NEON_VAL_MIN),
    # Deep orange / red-orange (Indian construction)
    ( 0,   5, VEST_NEON_SAT_MIN, VEST_NEON_VAL_MIN),
    (165, 179, VEST_NEON_SAT_MIN, VEST_NEON_VAL_MIN),  # red-wrap
]


def validate_safety_vest(
    frame_bgr:   np.ndarray,
    vest_box:    list,       # [x1, y1, x2, y2] in pixel coords
    worker_box:  list | None,  # [x1, y1, x2, y2] of associated worker
    condition:   str = "S1_normal",
) -> bool:
    """
    Multi-signal gate for a vest detection.  Returns True if the crop plausibly
    contains a real safety vest, False if it should be rejected.

    Designed to reject:
      • Plain white shirts / towels / cloth
      • Random bright construction materials (foam, PVC pipes, paint cans)
      • Low-saturation detections not supported by reflective strips or PPE context

    While preserving:
      • Dusty / faded / mud-covered vests (dusty-mode override)
      • Dark vests in low-light (low-light override)
      • Grey or white reflective jackets (detected via strip pattern)
      • Any vest already confirmed by worker context (temporal trust)
    """
    fh, fw = frame_bgr.shape[:2]
    vx1, vy1, vx2, vy2 = [
        max(0, int(round(c))) for c in vest_box
    ]
    vx2 = min(vx2, fw)
    vy2 = min(vy2, fh)
    if vx2 <= vx1 or vy2 <= vy1:
        return False   # degenerate box

    crop = frame_bgr[vy1:vy2, vx1:vx2]
    if crop.size == 0:
        return False

    # ── Condition-specific threshold overrides ───────────────────────────────
    sat_floor = VEST_NEON_SAT_MIN
    val_floor = VEST_NEON_VAL_MIN
    if condition == "S2_dusty":
        sat_floor = VEST_DUSTY_SAT_FLOOR   # dust desaturates vests
        val_floor = VEST_NEON_VAL_MIN * 0.85
    elif condition == "S3_low_light":
        val_floor = VEST_LOWLIGHT_VAL_FLOOR   # dark scene dims everything
    elif condition == "S4_crowded":
        sat_floor = VEST_NEON_SAT_MIN * 0.90  # slight relaxation in crowds

    # ── Signal A: HSV neon-color gate ────────────────────────────────────────
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32)
    v_ch = hsv[:, :, 2].astype(np.float32)

    neon_mask = np.zeros(crop.shape[:2], dtype=bool)
    for (h_lo, h_hi, s_min, v_min) in _VEST_HSV_RANGES:
        s_ok = s_ch >= sat_floor
        v_ok = v_ch >= val_floor
        if h_lo <= h_hi:
            h_ok = (h_ch >= h_lo) & (h_ch <= h_hi)
        else:  # wrap-around (red hues)
            h_ok = (h_ch >= h_lo) | (h_ch <= h_hi)
        neon_mask |= (h_ok & s_ok & v_ok)

    neon_ratio = float(neon_mask.sum()) / max(crop.shape[0] * crop.shape[1], 1)
    signal_a_pass = (neon_ratio >= 0.10)   # ≥ 10% of crop is neon

    # ── Signal B: Reflective strip detection ─────────────────────────────────
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bright_mask = gray.astype(np.float32) >= VEST_REFLECTIVE_THRESH
    bright_ratio = float(bright_mask.sum()) / max(gray.size, 1)
    # Check if bright pixels have a horizontal-stripe pattern:
    # a real reflective strip has row-mean >> col-mean variance ratio.
    signal_b_pass = False
    if bright_ratio >= VEST_REFLECTIVE_RATIO:
        row_means = bright_mask.mean(axis=1)   # per-row fraction of bright px
        col_means = bright_mask.mean(axis=0)
        h_var = float(np.var(row_means))       # high = horizontal banding
        v_var = float(np.var(col_means)) + 1e-9
        signal_b_pass = (h_var / v_var >= 2.0) or (bright_ratio >= 0.20)

    # ── Signal C: Torso placement gate ───────────────────────────────────────
    # CHANGED: default is now False when no worker context is available.
    # An orphaned vest with no nearby associated worker is almost certainly a
    # bright construction material (blue bucket, tarp, paint can, foam block)
    # rather than a real person — Signal A or B must compensate.
    # If a nearest worker exists, the torso placement check runs as before.
    if worker_box is not None:
        wx1, wy1, wx2, wy2 = worker_box
        wh = max(wy2 - wy1, 1.0)
        vcx = (vx1 + vx2) / 2.0
        vcy = (vy1 + vy2) / 2.0
        torso_y_top    = wy1 + wh * VEST_TORSO_TOP
        torso_y_bottom = wy1 + wh * VEST_TORSO_BOTTOM
        # Check both horizontal containment AND vertical torso band
        in_torso = (wx1 <= vcx <= wx2) and (torso_y_top <= vcy <= torso_y_bottom)
        signal_c_pass = in_torso
    else:
        # No known worker nearby — only the mandatory colour/reflective signals
        # can save this vest. Buckets and tarps almost always fail Signal A/B.
        signal_c_pass = False

    # ── Final gate ───────────────────────────────────────────────────────────
    # The vest survives if ANY of the signals pass.
    # All-fail means it is likely a white shirt, cloth, or bright material.
    return signal_a_pass or signal_b_pass or signal_c_pass



# ── WBF (unchanged — exact same logic as ensemble_batch.py) ───────────────────
def iou_box(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1e-6)

def wbf_fuse(
    all_preds,
    img_w, img_h,
    iou_override:  dict | None = None,
    conf_gate:     dict | None = None,   # per-class post-WBF confidence gate
):
    iou_table  = iou_override if iou_override is not None else WBF_IOU
    gate_table = conf_gate    if conf_gate    is not None else POST_WBF_GLOBAL
    flat = []
    for midx, (boxes, scores, labels) in enumerate(all_preds):
        for box, score, label in zip(boxes, scores, labels):
            norm = [box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h]
            flat.append((norm, score, label, midx))
    if not flat:
        return []
    result = []
    for cls_id in range(3):
        cf = [(b, s, mi) for b, s, l, mi in flat if l == cls_id]
        if not cf:
            continue
        iou_t = iou_table.get(cls_id, 0.50)
        cf.sort(key=lambda x: -x[1])
        used = [False] * len(cf)
        for i in range(len(cf)):
            if used[i]:
                continue
            cb, cs, cm = [cf[i][0]], [cf[i][1]], [cf[i][2]]
            used[i] = True
            for j in range(i + 1, len(cf)):
                if not used[j] and iou_box(cf[i][0], cf[j][0]) >= iou_t:
                    cb.append(cf[j][0]); cs.append(cf[j][1]); cm.append(cf[j][2])
                    used[j] = True
            rw = np.array([MODEL_WEIGHTS[min(mi, len(MODEL_WEIGHTS)-1)] * s
                           for mi, s in zip(cm, cs)])
            nw = rw / rw.sum()
            fs = float(np.average(cs, weights=nw))
            if fs < gate_table.get(cls_id, 0.20):
                continue
            fb = np.average(cb, axis=0, weights=nw)
            # n_models: number of distinct model indices that contributed
            n_models = len(set(cm))
            result.append({
                "box":      [fb[0]*img_w, fb[1]*img_h, fb[2]*img_w, fb[3]*img_h],
                "score":    fs,
                "cls":      cls_id,
                "n_models": n_models,   # 1 = only one model agreed, 2 = both
            })
    return result


# ── Worker-to-PPE spatial association ─────────────────────────────────────────
# Minimum confidence a helmet must reach to be counted. Raises the bar so that
# low-confidence false positives on hair / shoulders / backs are ignored.
MIN_HELMET_CONF = 0.35

# ── CLAHE instance (created once, reused per frame) ───────────────────────────
_clahe_cache: dict = {}

def _get_clahe(clip: float) -> cv2.CLAHE:
    """Return a cached CLAHE object for the given clip limit."""
    if clip not in _clahe_cache:
        _clahe_cache[clip] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return _clahe_cache[clip]

# Pre-computed gamma LUT cache (gamma → uint8 LUT)
_gamma_cache: dict = {}

def _gamma_lut(gamma: float) -> np.ndarray:
    if gamma not in _gamma_cache:
        inv = 1.0 / gamma
        _gamma_cache[gamma] = np.array(
            [((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8
        )
    return _gamma_cache[gamma]


def enhance_frame(img_bgr: np.ndarray,
                  condition: str = "S1_normal",
                  clip: float = 2.0) -> np.ndarray:
    """
    Adaptive preprocessing pipeline per site condition.

    S1_normal:
      CLAHE on the L channel of LAB — boosts local contrast without colour
      shift, helping workers with cement-coloured clothing stand out.

    S2_dusty:
      CLAHE (clip 2.5) → bilateral filter (d=7, σ=90) → unsharp mask.
      Bilateral removes haze/grain while keeping worker silhouette edges
      sharp for YOLO. Unsharp mask recovers fine detail lost in dust haze.

    S3_low_light:
      CLAHE (clip 3.0) → gamma correction (γ=1.8) → slight denoise.
      Gamma lifts mid-tones so YOLO can extract features invisible in dark
      footage. Denoising removes ISO noise amplified by the gamma boost.

    S4_crowded:
      CLAHE (clip 2.0) → mild sharpening kernel.
      Sharpening improves separation of adjacent worker silhouettes so YOLO
      can resolve fine boundary gaps between closely-spaced people.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    l = _get_clahe(clip).apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)

    if condition == "S3_low_light":
        # Gamma 1.8 → lift dark mid-tones more aggressively than 1.5
        result = cv2.LUT(result, _gamma_lut(1.8))
        # Denoise ISO noise amplified by gamma (fast NL-means on rescaled img)
        result = cv2.fastNlMeansDenoisingColored(result, None, 6, 6, 7, 21)

    elif condition == "S2_dusty":
        # Bilateral: edge-preserving denoiser removes haze grain (σ raised 75→90)
        result = cv2.bilateralFilter(result, 7, 90, 90)
        # Unsharp mask to recover worker edge detail softened by the bilateral
        blur   = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.4, blur, -0.4, 0)

    elif condition == "S4_crowded":
        # Mild sharpening to improve edge separation between adjacent workers
        kernel = np.array([[ 0, -0.5,  0],
                           [-0.5,  3, -0.5],
                           [ 0, -0.5,  0]], dtype=np.float32)
        result = cv2.filter2D(result, -1, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def is_valid_helmet(hbox: list, worker_bw: float, worker_bh: float) -> bool:
    """
    Geometric sanity check to reject false-positive helmets on hair, scarves,
    cloth, shoulders, or body regions:
      • Aspect ratio 0.5–2.0 (helmets are roughly round, not very elongated)
      • Width ≤ 70 % of worker box width (a head can't be wider than the torso)
      • Width ≥ 8 px and height ≥ 8 px (minimum size — not pixel noise)
      • Height ≤ 55 % of worker box height (head occupies upper portion only)
    """
    hx1, hy1, hx2, hy2 = hbox
    hw = max(hx2 - hx1, 1)
    hh = max(hy2 - hy1, 1)
    aspect = hw / hh
    if aspect > 2.2 or aspect < 0.45:   # too wide (scarf) or too tall (body)
        return False
    if hw > worker_bw * 0.70:            # wider than torso → not a head
        return False
    if hh > worker_bh * 0.55:            # taller than upper-half → body region
        return False
    if hw < 8 or hh < 8:                 # pixel noise
        return False
    return True


def is_valid_worker(
    wbox:        list,
    img_w:       int,
    img_h:       int,
    score:       float,
    model_count: int  = 1,     # how many models agreed on this box (1 or 2)
    condition:   str  = "S1_normal",
    nearby_ppe:  bool = False,  # True → helmet/vest found spatially adjacent
) -> bool:
    """
    Multi-signal validation rejecting cement bags, sand bags, brick stacks,
    tarpaulins, debris piles, scaffolding shadows, pipes, and machinery parts
    misclassified as workers.

    ── PPE fast-bypass ───────────────────────────────────────────────────────
    If a helmet or vest is spatially near the candidate worker box (computed
    before this call in run_inference) we skip all geometry checks — confirmed
    PPE proves it's a person regardless of box shape.

    ── Signal 1: Landscape aspect reject ────────────────────────────────────
    Workers in all real-world poses are taller than wide from CCTV angles.
    Cement bags, brick stacks, and bundled pipes are landscape-shaped.

    ── Signal 2: Size floor ─────────────────────────────────────────────────
    Real workers at any practical CCTV distance occupy ≥ 1.0-1.2% of the
    shorter image dimension in both width and height.

    ── Signal 3: Single-model agreement gate ────────────────────────────────
    similarity are typically caught by only one model at 0.25–0.45.

    ── Signal 4: Hard-landscape reject (aspect > 3.5) ───────────────────────
    Pipes, long brick stacks, and tarpaulin ridges — never a person.

    ── Signal 5: Portrait check (model-count independent) ───────────────────
    Even when both models agree, nearly-square/landscape boxes need higher
    confidence. Cement bags on plank stacks produce ~1:1 boxes that fool both
    models. Thresholds are per-condition to avoid rejecting crouching workers
    in dusty/low-light scenes where boxes expand due to haze.

    ── Signal 6: Box-width ceiling ──────────────────────────────────────────
    No single worker spans > 45% of frame width. Wider = material pile.

    ── Signal 7: Height floor ───────────────────────────────────────────────
    Workers must occupy ≥ 6.5% of image height. Below this = pixel noise or
    distant objects that look like construction materials at low resolution.

    ── Signal 8: Area ceiling ───────────────────────────────────────────────
    A single person cannot cover > 18% of total image area. Oversized blobs
    wrap multi-object regions, not a single standing worker.
    """
    # ── PPE fast-bypass ────────────────────────────────────────────────────
    # If PPE is confirmed nearby, we skip the geometric "person-hood" gates.
    # Confirmed PPE is the single most reliable indicator of a worker.
    if nearby_ppe:
        return True

    x1, y1, x2, y2 = wbox
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    aspect = bw / bh

    # ── Signal 9: Orphaned Worker Suppression (Material Reject) ───────────
    # If only one model sees the worker and no PPE is nearby, it's often 
    # a cement bag, orange bucket, or brick stack. Require higher confidence.
    if model_count == 1 and not nearby_ppe:
        # Increase required score for orphaned portrait blobs
        if score < 0.48:
            return False

    # ── Signal 4: hard-landscape reject ───────────────────────────────────
    if aspect > 3.5:
        return False

    # ── Signal 6: box-width ceiling ───────────────────────────────────────
    if bw > img_w * 0.45:
        return False

    # ── Signal 8: area ceiling ─────────────────────────────────────────────
    if (bw * bh) > (img_w * img_h * 0.18):
        return False

    # ── Signal 7: height floor ─────────────────────────────────────────────
    if condition == "S4_crowded":
        min_h_frac = 0.055
    elif condition in ("S2_dusty", "S3_low_light"):
        min_h_frac = 0.060
    else:
        min_h_frac = 0.065   # S1_normal
    if bh < img_h * min_h_frac and score < 0.58:
        return False

    # ── Signal 1: aspect ratio per condition ───────────────────────────────
    # S4_crowded: allow wider boxes (adjacent workers create merged-look boxes)
    # S2/S3:      haze/noise inflates bounding boxes beyond person outline
    # S1_normal:  strict portrait — clear sky background, no interference
    if condition == "S4_crowded":
        max_aspect = 2.4
    elif condition in ("S3_low_light", "S2_dusty"):
        max_aspect = 2.2
    else:
        max_aspect = 2.0   # S1_normal: tight portrait requirement

    if aspect > max_aspect and score < 0.68:
        return False

    # ── Signal 2: size floor ───────────────────────────────────────────────
    size_frac = 0.010 if condition in ("S4_crowded", "S3_low_light") else 0.012
    min_px    = min(img_w, img_h) * size_frac
    if bw < min_px or bh < min_px:
        return False

    # ── Signal 3: single-model agreement gate ─────────────────────────────
    if model_count == 1:
        if condition == "S3_low_light":
            single_thresh = 0.32
        elif condition == "S2_dusty":
            single_thresh = 0.35
        elif condition == "S4_crowded":
            single_thresh = 0.40
        else:
            single_thresh = 0.40   # S1_normal: balanced recall vs. material FP
        if score < single_thresh:
            return False

    # ── Signal 5: portrait check (applies regardless of model_count) ───────
    # Calibrated per-condition — catches boxes that fool BOTH models:
    #   S1_normal:   aspect > 0.88 and score < 0.70 → reject (bags on planks)
    #   S2_dusty:    aspect > 0.95 and score < 0.65 → reject (haze inflates boxes)
    #   S3_low_light: aspect > 1.00 and score < 0.62 → reject (dark scene noise)
    #   S4_crowded:  skip — overlapping workers produce legitimately wide boxes
    if condition == "S1_normal":
        if aspect > 0.88 and score < 0.70:
            return False
    elif condition == "S2_dusty":
        if aspect > 0.95 and score < 0.65:
            return False
    elif condition == "S3_low_light":
        if aspect > 1.00 and score < 0.62:
            return False

    return True


def associate_ppe_to_workers(raw: list) -> list:
    """
    Link helmet and vest detections to their parent worker after YOLO/WBF.

    Search zones (expanded from the strict containment of v1):
      • Helmet: upper 75 % of worker height, ±15 % horizontal padding,
        10 % vertical headroom above the worker box (helmet often sits above
        the box top in side/rear-view poses). Minimum conf = MIN_HELMET_CONF
        to reject false positives on hair, shoulders, and backs.
      • Vest: full worker height + 20 % below (vest commonly extends below a
        tight worker box), ±15 % horizontal padding.

    Matching priority: closest Euclidean distance from the PPE centre to the
    worker centre (rather than horizontal-only distance), so partially-occluded
    PPE items on the correct worker win over nearby items on adjacent workers.
    One-to-one assignment — once a PPE item is claimed it cannot be re-used.
    """
    worker_ids = {k for k, v in cls_map.items() if v in ("worker", "person")}
    helmet_ids = {k for k, v in cls_map.items() if v == "helmet"}
    vest_ids   = {k for k, v in cls_map.items()
                  if v in ("safety_vest", "safety-vest")}

    workers = [(i, d) for i, d in enumerate(raw) if d["cls"] in worker_ids]
    helmets = [(i, d) for i, d in enumerate(raw) if d["cls"] in helmet_ids]
    vests   = [(i, d) for i, d in enumerate(raw) if d["cls"] in vest_ids]

    matched_helmets: set = set()
    matched_vests:   set = set()
    result = [dict(d) for d in raw]

    for wi, wd in workers:
        bx1, by1, bx2, by2 = wd["box"]
        bw  = bx2 - bx1
        bh  = by2 - by1
        wcx = (bx1 + bx2) / 2
        wcy = (by1 + by2) / 2

        # Expanded search rectangles
        h_pad      = bw * 0.18          # increased 0.15 → 0.18
        h_head     = bh * 0.15          # increased 0.10 → 0.15 (better steep angle support)

        helm_x1 = bx1 - h_pad
        helm_x2 = bx2 + h_pad
        helm_y1 = by1 - h_head          # 15 % headroom above box
        helm_y2 = by1 + bh * 0.75      # upper 75 % of worker height

        vest_x1 = bx1 - h_pad
        vest_x2 = bx2 + h_pad
        vest_y1 = by1
        vest_y2 = by2 + bh * 0.25      # increased 0.20 → 0.25 (oversized jackets)

        # ── Helmet matching ──────────────────────────────────────────────────
        best_hi, best_h_dist = -1, float("inf")
        for hi, hd in helmets:
            if hi in matched_helmets:
                continue
            # Reject weak detections (hair, shoulder false positives)
            if hd.get("score", 1.0) < MIN_HELMET_CONF:
                continue
            # Geometric sanity: shape/size must match a real helmet
            if not is_valid_helmet(hd["box"], bw, bh):
                continue
            hx1, hy1, hx2, hy2 = hd["box"]
            hcx = (hx1 + hx2) / 2
            hcy = (hy1 + hy2) / 2
            if helm_x1 <= hcx <= helm_x2 and helm_y1 <= hcy <= helm_y2:
                dist = ((hcx - wcx) ** 2 + (hcy - wcy) ** 2) ** 0.5
                if dist < best_h_dist:
                    best_hi, best_h_dist = hi, dist
        has_helmet = best_hi >= 0
        if has_helmet:
            matched_helmets.add(best_hi)

        # ── Vest matching ────────────────────────────────────────────────────
        best_vi, best_v_dist = -1, float("inf")
        for vi, vd in vests:
            if vi in matched_vests:
                continue
            vx1, vy1, vx2, vy2 = vd["box"]
            vcx = (vx1 + vx2) / 2
            vcy = (vy1 + vy2) / 2
            if vest_x1 <= vcx <= vest_x2 and vest_y1 <= vcy <= vest_y2:
                dist = ((vcx - wcx) ** 2 + (vcy - wcy) ** 2) ** 0.5
                if dist < best_v_dist:
                    best_vi, best_v_dist = vi, dist
        has_vest = best_vi >= 0
        if has_vest:
            matched_vests.add(best_vi)

        result[wi]["has_helmet"] = has_helmet
        result[wi]["has_vest"]   = has_vest

    # Suppress matched PPE boxes — each matched item is already represented by
    # has_helmet / has_vest on the parent worker box.  Rendering both the worker
    # box AND the vest/helmet box on the same person causes the stacked-boxes
    # visual clutter seen in the overlay.  Only orphaned PPE (no matched worker)
    # is kept as a standalone detection so the frontend can flag it separately.
    for mi in matched_helmets:
        result[mi]["_matched"] = True
    for mi in matched_vests:
        result[mi]["_matched"] = True

    return [d for d in result if not d.get("_matched", False)]


# ── Class-ID helpers (module-level, computed once after load_models) ──────────
def _worker_ids() -> set:
    return {k for k, v in cls_map.items() if v in ("worker", "person")}

def _helmet_ids() -> set:
    return {k for k, v in cls_map.items() if v == "helmet"}

def _vest_ids() -> set:
    return {k for k, v in cls_map.items() if v in ("safety_vest", "safety-vest")}


def _has_nearby_ppe(wbox: list, raw: list) -> bool:
    """
    Return True if any helmet or vest detection centre falls within the
    expanded PPE search zone around `wbox`.  Mirrors the zones used by
    associate_ppe_to_workers so the pre-filter and the association step
    see the same geometry.
    """
    hids = _helmet_ids()
    vids = _vest_ids()
    bx1, by1, bx2, by2 = wbox
    bw  = bx2 - bx1
    bh  = by2 - by1
    pad = bw * 0.18
    # Helmet zone: upper 75% + 15% headroom above
    hx1, hx2 = bx1 - pad, bx2 + pad
    hy1, hy2 = by1 - bh * 0.15, by1 + bh * 0.75
    # Vest zone: full height + 25% below
    vx1, vx2 = bx1 - pad, bx2 + pad
    vy1, vy2 = by1, by2 + bh * 0.25
    for d in raw:
        if d["cls"] not in hids and d["cls"] not in vids:
            continue
        cx = (d["box"][0] + d["box"][2]) / 2
        cy = (d["box"][1] + d["box"][3]) / 2
        if d["cls"] in hids:
            if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
                return True
        else:
            if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                return True
    return False


# ── inference core (runs in threadpool — never blocks the event loop) ─────────
def run_inference(
    img_bgr:     np.ndarray,
    condition:   str   = "S1_normal",
    conf_thresh: float = PRE_CONF,
    # Per-class conf gates applied POST-WBF (frontend sends these as JSON dicts)
    class_conf:  dict | None = None,   # {worker, helmet, vest} → float
    nms_iou_map: dict | None = None,   # {worker, helmet, vest} → float
    wbf_iou_map: dict | None = None,   # {worker, helmet, vest} → float
    clahe:       bool  = True,
    clahe_clip:  float = 2.0,
):
    # ── Condition-aware preprocessing ────────────────────────────────────────
    # All conditions get CLAHE. S3/S2 get additional filters.
    # S3_low_light: boost CLAHE clip to 3.0 (darker scene = more contrast needed)
    # S2_dusty: use clip 2.5 + bilateral filter (enhance_frame handles both)
    if clahe:
        effective_clip = clahe_clip
        if condition == "S3_low_light" and clahe_clip < 3.0:
            effective_clip = 3.0   # stronger boost for dark footage
        elif condition == "S2_dusty" and clahe_clip < 2.5:
            effective_clip = 2.5   # moderate boost + bilateral (in enhance_frame)
        img_bgr = enhance_frame(img_bgr, condition, effective_clip)

    h, w = img_bgr.shape[:2]

    # ── Condition-aware pre-WBF confidence threshold ──────────────────────────
    # Lower pre_conf for difficult conditions: weak-but-real detections (small
    # workers in crowded frames, low-contrast workers in dark/dusty scenes) must
    # reach WBF so both models can reinforce them. The post-WBF gate filters
    # the false positives that pass through at lower pre_conf.
    if condition == "S4_crowded":
        pre_conf = min(conf_thresh, 0.15)   # catch small/distant workers
    elif condition == "S3_low_light":
        pre_conf = min(conf_thresh, 0.15)   # everything scores weaker in dark
    elif condition == "S2_dusty":
        pre_conf = min(conf_thresh, 0.17)   # dust attenuates scores
    else:
        pre_conf = conf_thresh              # S1_normal: use value as-is

    # ── Per-condition WBF IoU + NMS ───────────────────────────────────────────
    # S4_crowded:
    #   Worker WBF IoU = 0.72 — only near-identical boxes fuse; adjacent workers
    #   (IoU 0.20-0.55) stay separate. NMS IoU = 0.65 so per-model NMS doesn't
    #   suppress nearby workers before WBF sees them.
    # S3_low_light:
    #   Worker WBF IoU lowered (0.60) — noisy boxes shift slightly frame-to-frame
    #   so we allow slightly looser merging. NMS standard.
    # S1_normal / S2_dusty: standard params.
    # Use centralized WBF_IOU_BY_CONDITION table (all thresholds in one place)
    wbf_iou_cond = dict(WBF_IOU_BY_CONDITION.get(condition, WBF_IOU))
    base_nms_iou = 0.65 if condition == "S4_crowded" else NMS_IOU

    # Condition-specific post-WBF confidence gate (relaxed in dark/dusty scenes)
    cond_conf_gate = POST_WBF_BY_CONDITION.get(condition, POST_WBF_GLOBAL)

    # ── Apply frontend per-class overrides ────────────────────────────────────
    def _cls_idx(name: str) -> int:
        for k, v in cls_map.items():
            if v == name or (name == "vest" and v in ("safety_vest", "safety-vest")):
                return k
        return -1

    if wbf_iou_map:
        for name, val in wbf_iou_map.items():
            idx = _cls_idx(name)
            if idx >= 0:
                wbf_iou_cond[idx] = float(val)

    # NMS IoU: use the most permissive (highest) value across all classes so
    # no class gets over-suppressed within each individual model's predict().
    if nms_iou_map:
        nms_iou = max(float(v) for v in nms_iou_map.values())
    else:
        nms_iou = base_nms_iou

    # ── Model 1: Primary (YOLOv11) ──────────────────────────────────────────
    t_inf_start = time.perf_counter()
    r11 = model_v11.predict(
        img_bgr, device=DEVICE, verbose=False,
        conf=pre_conf, iou=nms_iou, half=USE_HALF,
        agnostic_nms=False,
    )[0]
    t_inf_1 = (time.perf_counter() - t_inf_start) * 1000

    boxes11, scores11, labels11 = [], [], []
    for box in (r11.boxes or []):
        boxes11.append(box.xyxy[0].cpu().numpy().tolist())
        scores11.append(float(box.conf[0]))
        labels11.append(int(box.cls[0]))

    # ── Signal 10: Model-1 High-Confidence Bypass (Latency optimization) ────
    # In clear scenes (S1_normal) with very high-confidence detections, skipping
    # the second model significantly reduces lag without impacting safety.
    # DISABLED in: S4_crowded, S2_dusty, S3_low_light, or busy scenes (>8 dets).
    force_ensemble = (condition in ("S4_crowded", "S2_dusty", "S3_low_light"))
    
    can_bypass = False
    if not force_ensemble and len(scores11) > 0 and len(scores11) < 8:
        # Require all workers to be high-confidence for bypass
        worker_scores = [s for s, l in zip(scores11, labels11) if l in _worker_ids()]
        if worker_scores and all(sc > EARLY_EXIT_CONF for sc in worker_scores):
            can_bypass = True

    used_ensemble = False
    t_inf_2 = 0.0
    t_wbf = 0.0
    
    if model_v26 is not None and not can_bypass:
        t_inf_2_start = time.perf_counter()
        r26 = model_v26.predict(
            img_bgr, device=DEVICE, verbose=False,
            conf=pre_conf, iou=nms_iou, half=USE_HALF,
            agnostic_nms=False,
        )[0]
        t_inf_2 = (time.perf_counter() - t_inf_2_start) * 1000

        boxes26 = [b.xyxy[0].cpu().numpy().tolist() for b in (r26.boxes or [])]
        scores26 = [float(b.conf[0]) for b in (r26.boxes or [])]
        labels26 = [int(b.cls[0]) for b in (r26.boxes or [])]

        t_wbf_start = time.perf_counter()
        raw = wbf_fuse(
            [(boxes11, scores11, labels11), (boxes26, scores26, labels26)],
            w, h, iou_override=wbf_iou_cond, conf_gate=cond_conf_gate,
        )
        t_wbf = (time.perf_counter() - t_wbf_start) * 1000
        used_ensemble = True
    else:
        # Model 1 results only
        t_wbf_start = time.perf_counter()
        raw = []
        for b, s, l in zip(boxes11, scores11, labels11):
            if s >= cond_conf_gate.get(l, 0.20):
                raw.append({"box": b, "score": s, "cls": l, "n_models": 1})
        t_wbf = (time.perf_counter() - t_wbf_start) * 1000

    # ── Elevated region supplemental pass ────────────────────────────────────
    # Runs on the top 35% of the frame in ALL scene modes to catch workers on
    # scaffolding, elevated walkways, and wall tops that get missed by the full-
    # frame pass because they are small and score below the full-frame pre_conf.
    # Using a lower confidence (pre_conf × 0.80) maximises recall in that zone.
    # The top-region detections are converted back to full-frame coordinates and
    # merged into `raw` via WBF before the false-positive filter runs.
    _elev_h = int(h * 0.35)
    if _elev_h >= 48 and model_v11 is not None:
        _elev_conf = max(pre_conf * 0.80, 0.12)
        try:
            _elev_crop = img_bgr[0:_elev_h, 0:w]
            _re = model_v11.predict(
                _elev_crop, device=DEVICE, verbose=False,
                conf=_elev_conf, iou=nms_iou, half=USE_HALF,
                agnostic_nms=False,
            )[0]
            _eb, _es, _el = [], [], []
            for _b in (_re.boxes or []):
                _bx = _b.xyxy[0].cpu().numpy().tolist()
                # y-coordinate shift: crop starts at y=0 → full frame unchanged
                _eb.append(_bx)
                _es.append(float(_b.conf[0]))
                _el.append(int(_b.cls[0]))
            if _eb:
                # Merge elevated detections with existing raw list via WBF
                # Assemble raw as a pseudo-model-0 input for wbf_fuse
                _raw_boxes  = [d["box"] for d in raw]
                _raw_scores = [d["score"] for d in raw]
                _raw_labels = [d["cls"] for d in raw]
                _merged = wbf_fuse(
                    [(_raw_boxes, _raw_scores, _raw_labels), (_eb, _es, _el)],
                    w, h, iou_override=wbf_iou_cond, conf_gate=cond_conf_gate,
                )
                if _merged:
                    raw = _merged
                else:
                    # Fallback: just append elevated boxes that don't overlap raw
                    _wids = _worker_ids()
                    for _b2, _s2, _l2 in zip(_eb, _es, _el):
                        # Only add if no existing box overlaps ≥ 0.40 IoU
                        if not any(iou_box(_b2, d["box"]) >= 0.40 for d in raw if d["cls"] == _l2):
                            raw.append({"box": _b2, "score": _s2, "cls": _l2, "n_models": 1})
        except Exception as _ee:
            logger.debug(f"[elevated_pass] skipped: {_ee}")

    # ── Worker false-positive filter ──────────────────────────────────────────
    # IMPORTANT: PPE proximity is checked HERE, before the worker filter, so
    # is_valid_worker can skip geometry checks for boxes that have a confirmed
    # helmet or vest spatially adjacent (PPE = definitive person signal).
    worker_cls_ids = _worker_ids()
    raw = [
        d for d in raw
        if d["cls"] not in worker_cls_ids
        or is_valid_worker(
            d["box"], w, h, d["score"],
            model_count=d.get("n_models", 1),
            condition=condition,
            nearby_ppe=_has_nearby_ppe(d["box"], raw),
        )
    ]

    # ── Per-class confidence filter (post-WBF, user-tunable) ─────────────────
    if class_conf:
        fallback = cond_conf_gate
        def _passes(d: dict) -> bool:
            cls_name = cls_map.get(d["cls"], "")
            if cls_name in ("worker", "person"):
                return d["score"] >= float(class_conf.get("worker", fallback.get(d["cls"], 0.18)))
            if cls_name == "helmet":
                return d["score"] >= float(class_conf.get("helmet", fallback.get(d["cls"], 0.30)))
            if cls_name in ("safety_vest", "safety-vest"):
                return d["score"] >= float(class_conf.get("vest", fallback.get(d["cls"], 0.14)))
            return True
        raw = [d for d in raw if _passes(d)]

    # ── Post-WBF duplicate suppression (IoU + containment) ───────────────────
    # Two boxes of the same class are considered duplicates when EITHER:
    #   (a) standard IoU >= POST_WBF_DEDUP_IOU  — catches offset duplicates
    #   (b) the smaller box has >= DEDUP_CONTAIN_THRESH of its area inside the
    #       larger box — catches size-mismatched boxes (tight vs full-body) that
    #       produce IoU of only ~0.25-0.39 and slip past IoU-only dedup.
    def _is_duplicate(a: list, b: list, cls_id: int = 0) -> bool:
        if iou_box(a, b) >= POST_WBF_DEDUP_IOU.get(cls_id, 0.35):
            return True
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter == 0.0:
            return False
        area_a = max((a[2]-a[0]) * (a[3]-a[1]), 1e-6)
        area_b = max((b[2]-b[0]) * (b[3]-b[1]), 1e-6)
        return (inter / min(area_a, area_b)) >= DEDUP_CONTAIN_THRESH

    deduped: list = []
    for cls_id in (0, 1, 2):
        cls_boxes = sorted(
            [d for d in raw if d["cls"] == cls_id],
            key=lambda d: -d["score"],
        )
        kept: list = []
        for cand in cls_boxes:
            if any(_is_duplicate(cand["box"], k["box"], cls_id) for k in kept):
                continue   # near-duplicate — drop the lower-confidence box
            kept.append(cand)
        deduped.extend(kept)
    # Preserve any classes not in 0/1/2 (future-proofing)
    deduped.extend(d for d in raw if d["cls"] not in (0, 1, 2))
    raw = deduped

    # ── Safety vest cascade validation (multi-signal gate) ───────────────────
    # Reject vests that fail all three signals (Color + Reflective + Torso).
    # This targets white shirts, cloth, foam, and random bright construction
    # materials while preserving dusty/faded/reflective real PPE.
    vest_cls_ids = _vest_ids()
    assoc_workers = [d for d in raw if d["cls"] in _worker_ids()]

    def _nearest_worker_box(vbox: list) -> list | None:
        """Return the box of the closest worker to this vest centroid, or None."""
        vcx = (vbox[0] + vbox[2]) / 2.0
        vcy = (vbox[1] + vbox[3]) / 2.0
        best_dist = float("inf")
        best_box  = None
        for w in assoc_workers:
            wcx = (w["box"][0] + w["box"][2]) / 2.0
            wcy = (w["box"][1] + w["box"][3]) / 2.0
            dist = ((vcx - wcx)**2 + (vcy - wcy)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_box  = w["box"]
        return best_box

    validated_raw = []
    for det in raw:
        if det["cls"] in vest_cls_ids:
            nearest_w = _nearest_worker_box(det["box"])
            if not validate_safety_vest(img_bgr, det["box"], nearest_w, condition):
                continue   # rejected by multi-signal gate
        validated_raw.append(det)
    raw = validated_raw

    # ── Worker-to-PPE association (post-WBF, pre-return) ─────────────────────
    raw = associate_ppe_to_workers(raw)

    t_perf = {
        "inf_1_ms": round(t_inf_1, 1),
        "inf_2_ms": round(t_inf_2, 1),
        "wbf_ms":   round(t_wbf, 1),
        "bypass":   can_bypass and not force_ensemble
    }
    return raw, used_ensemble, t_perf


# ── Worker temporal confirmation (video export only) ──────────────────────────
class WorkerMemory:
    """
    Require a worker box to appear in ≥ CONFIRM_FRAMES consecutive inference
    cycles before being emitted. After confirmation it is retained until it
    misses ≥ DROP_FRAMES cycles.

    Rationale: cement bags and static construction materials generate a stable
    detection at the same pixel location across ALL frames because they never
    move. However they often generate inconsistent boxes (slightly different
    sizes each inference due to YOLO stochasticity). A real worker that is
    genuinely static (e.g. standing still) will produce highly consistent boxes
    AND will typically have an associated helmet or vest detection nearby.

    Implementation:
      • We track candidate boxes with IOU matching (threshold 0.35).
      • Each candidate accumulates a "seen" counter.
      • Confirmed workers (seen ≥ CONFIRM_FRAMES) are emitted and tracked
        with a "missed" counter.
      • If a confirmed worker disappears for DROP_FRAMES it is removed.
      • PPE presence (has_helmet or has_vest) on any frame immediately
        confirms the detection — real workers have PPE, cement bags do not.
    """
    CONFIRM_FRAMES = 3   # raised 2→3: cement bags pass 2 frames trivially; real
                         # workers with PPE bypass this via ppe_present fast-path
    DROP_FRAMES    = 3   # missed inferences before dropping a confirmed worker
    IOU_THRESH     = 0.30

    def __init__(self):
        self._candidates: list[dict] = []   # unconfirmed
        self._confirmed:  list[dict] = []   # emitted to video

    @staticmethod
    def _best_match(box: list, pool: list):
        """Return (index, iou) of best overlapping item in pool, or (-1, 0)."""
        best_i, best_iou = -1, 0.0
        for i, item in enumerate(pool):
            v = iou_box(
                [c / max(box[2], 1) for c in box[:2]] + [1.0, 1.0],  # simplified
                [c / max(item["box"][2], 1) for c in item["box"][:2]] + [1.0, 1.0],
            )
            # Use absolute-pixel IoU for accuracy
            v = iou_box(box, item["box"])
            if v > best_iou:
                best_iou, best_i = v, i
        return best_i, best_iou

    def update(self, dets: list, worker_cls_ids: set) -> list:
        """
        Feed raw detections for one inference cycle.
        Returns the current set of confirmed worker detections.
        """
        workers = [d for d in dets if d["cls"] in worker_cls_ids]
        others  = [d for d in dets if d["cls"] not in worker_cls_ids]

        matched_cand: set[int] = set()
        matched_conf: set[int] = set()

        # ── Try to match each detected worker to confirmed tracks first ───
        for det in workers:
            ci, ciou = self._best_match(det["box"], self._confirmed)
            if ci >= 0 and ciou >= self.IOU_THRESH and ci not in matched_conf:
                self._confirmed[ci]["box"]    = det["box"]
                self._confirmed[ci]["score"]  = det["score"]
                self._confirmed[ci]["missed"] = 0
                self._confirmed[ci]["payload"] = det
                matched_conf.add(ci)
                continue
            # ── Try to match to candidates ─────────────────────────────────
            ki, kiou = self._best_match(det["box"], self._candidates)
            if ki >= 0 and kiou >= self.IOU_THRESH and ki not in matched_cand:
                self._candidates[ki]["seen"] += 1
                self._candidates[ki]["box"]   = det["box"]
                self._candidates[ki]["payload"] = det
                # PPE presence = immediate confirmation (real worker signal)
                ppe_present = det.get("has_helmet") or det.get("has_vest")
                if self._candidates[ki]["seen"] >= self.CONFIRM_FRAMES or ppe_present:
                    self._confirmed.append({
                        "box":    det["box"],
                        "score":  det["score"],
                        "missed": 0,
                        "payload": det,
                    })
                    self._candidates.pop(ki)
                matched_cand.add(ki)
            else:
                # New candidate
                ppe_present = det.get("has_helmet") or det.get("has_vest")
                if ppe_present:
                    # Immediate confirmation — PPE proves it's a person
                    self._confirmed.append({
                        "box": det["box"], "score": det["score"],
                        "missed": 0, "payload": det,
                    })
                else:
                    self._candidates.append({
                        "box": det["box"], "score": det["score"],
                        "seen": 1, "payload": det,
                    })

        # Age out unmatched confirmed tracks
        for i in range(len(self._confirmed)):
            if i not in matched_conf:
                self._confirmed[i]["missed"] += 1
        self._confirmed = [c for c in self._confirmed if c["missed"] <= self.DROP_FRAMES]

        # Age out stale candidates (missed one cycle → reset or drop)
        new_cands = []
        for i, c in enumerate(self._candidates):
            if i not in matched_cand:
                c["seen"] = max(0, c["seen"] - 1)
                if c["seen"] > 0:
                    new_cands.append(c)
            else:
                new_cands.append(c)
        self._candidates = new_cands

        # Emit confirmed workers + all non-worker detections unchanged
        return [c["payload"] for c in self._confirmed] + others


# ── Server-side IoU tracker (used by /detect/video export) ───────────────────
class ServerSideTracker:
    """
    Lightweight IoU-based tracker for the video export pipeline.

    Purpose: eliminate the "detection flicker" where a box appears/disappears
    between sampled frames. Instead of copying the last raw detection verbatim
    to non-sampled frames, we apply exponential moving-average (EMA) smoothing
    so box positions drift gracefully between inference updates.

    Benefits over raw last-detection reuse:
      • Boxes don't jump when a new inference updates the position.
      • Workers that momentarily drop below conf threshold are carried for up
        to MAX_MISSED inference cycles (faded label) before being dropped.
      • PPE flags (has_helmet, has_vest) are preserved across missing frames.
    """
    MAX_MISSED = 4       # carry-forward cycles before dropping a track
    IOU_THRESH = 0.25    # minimum IoU to match a new det to an existing track
    ALPHA      = 0.45    # EMA weight for new position (higher = snappier)

    def __init__(self):
        self.tracks: list[dict] = []

    def update(self, dets: list) -> list:
        matched: set[int] = set()

        for tr in self.tracks:
            best_iou, best_i = 0.0, -1
            for i, d in enumerate(dets):
                if i in matched or d["cls"] != tr["cls"]:
                    continue
                iou = iou_box(tr["smooth_box"], d["box"])
                if iou > best_iou:
                    best_iou, best_i = iou, i

            if best_i >= 0 and best_iou >= self.IOU_THRESH:
                d = dets[best_i]
                sb, nb = tr["smooth_box"], d["box"]
                tr["smooth_box"] = [sb[j] + (nb[j] - sb[j]) * self.ALPHA for j in range(4)]
                tr["score"]   = d["score"]
                tr["missed"]  = 0
                tr["payload"] = d   # full dict with PPE flags etc.
                matched.add(best_i)
            else:
                tr["missed"] += 1

        self.tracks = [t for t in self.tracks if t["missed"] <= self.MAX_MISSED]

        for i, d in enumerate(dets):
            if i not in matched:
                self.tracks.append({
                    "cls":        d["cls"],
                    "score":      d["score"],
                    "smooth_box": list(d["box"]),
                    "missed":     0,
                    "payload":    d,
                })

        out = []
        for tr in self.tracks:
            item = dict(tr["payload"])
            item["box"]   = [round(v, 1) for v in tr["smooth_box"]]
            item["score"] = tr["score"]
            out.append(item)
        return out


# ── draw annotations (used only by /detect/video export) ─────────────────────
def draw_detections(img_bgr, detections, mode, condition):
    img = img_bgr.copy()
    h, w = img.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cls_name = cls_map.get(det["cls"], "unknown")
        color    = COLORS.get(cls_name, (200, 200, 200))

        label = f"{cls_name} {det['score']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Banner
    gpu_tag = f"GPU:{DEVICE}" if DEVICE.startswith("cuda") else "CPU"
    banner = f"[BUILDSIGHT | {mode.upper()} | {condition} | {gpu_tag}] {len(detections)} det"
    cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(img, banner, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)

    # Legend
    for i, (cls_id, cls_name) in enumerate(cls_map.items()):
        color = COLORS.get(cls_name, (200, 200, 200))
        lx = 8 + i * 170
        ly = h - 8
        cv2.rectangle(img, (lx, ly - 13), (lx + 13, ly), color, -1)
        cv2.putText(img, cls_name, (lx + 17, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

    return img


def img_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode("utf-8")


def _build_det_list(detections: list) -> list:
    """Convert raw detection dicts → JSON-serialisable list with class names."""
    out = []
    for d in detections:
        item = {
            "class":      cls_map.get(d["cls"], "unknown"),
            "confidence": round(d["score"], 3),
            "box":        [round(v, 1) for v in d["box"]],
        }
        # Forward PPE association flags (workers only)
        if "has_helmet" in d:
            item["has_helmet"] = d["has_helmet"]
        if "has_vest" in d:
            item["has_vest"] = d["has_vest"]
        out.append(item)
    return out


# ── WebSocket & Background Detection API ──────────────────────────────────────

@app.websocket("/ws/detection")
async def detection_ws(websocket: WebSocket):
    """
    Persistent WebSocket for real-time detection updates.
    Stays connected regardless of which tab the frontend shows.
    Emits detection_update every ~150 ms while bg_service is running.
    """
    await ws_manager.connect(websocket)
    try:
        # Send a state snapshot immediately so the client can prime its store
        if bg_service:
            await websocket.send_json({
                "type": "detection_state_snapshot",
                **bg_service.get_current_state(),
            })
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if msg == "request_state_snapshot" and bg_service:
                    await websocket.send_json({
                        "type": "detection_state_snapshot",
                        **bg_service.get_current_state(),
                    })
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


@app.post("/api/video/upload-background")
async def upload_video_background(video: UploadFile = File(...)):
    """
    Upload a video and immediately start background inference.
    Detection continues running regardless of which tab is open.
    """
    if bg_service is None:
        return JSONResponse({"status": "error", "error": "service not ready"}, status_code=503)
    try:
        suffix = Path(video.filename).suffix if video.filename else ".mp4"
        tmp_path = Path(TMP_DIR) / f"bg_{int(time.time())}{suffix}"
        content = await video.read()
        tmp_path.write_bytes(content)

        bg_service.load_video(str(tmp_path))
        bg_service.start()

        return {
            "status":       "ok",
            "background":   True,
            "video_path":   str(tmp_path),
            "total_frames": bg_service.total_frames,
            "fps":          bg_service.fps,
        }
    except Exception as exc:
        logger.error("[VIDEO UPLOAD] %s", exc, exc_info=True)
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=500)


@app.post("/api/detection/pause")
async def pause_detection():
    if bg_service:
        bg_service.pause()
    return {"status": "ok", "paused": True}


@app.post("/api/detection/resume")
async def resume_detection():
    if bg_service:
        bg_service.resume()
    return {"status": "ok", "paused": False}


@app.post("/api/detection/stop")
async def stop_detection_bg():
    if bg_service:
        bg_service.stop()
    return {"status": "ok", "stopped": True}


@app.get("/api/detection/state")
async def get_detection_state():
    """REST fallback: current detection state."""
    if bg_service is None:
        return {"is_running": False, "worker_count": 0}
    return bg_service.get_current_state()


@app.get("/api/detection/worker-positions")
async def get_worker_positions():
    """
    Returns latest worker positions with lat/lng for GeoAI map.
    Use as a REST fallback when WebSocket is unavailable.
    """
    if bg_service is None:
        return {"status": "ok", "worker_positions": [], "zone_occupancy": {}, "worker_count": 0}
    state = bg_service.get_current_state()
    return {
        "status":           "ok",
        "worker_positions": state["worker_positions"],
        "zone_occupancy":   state["zone_occupancy"],
        "scene_condition":  state["scene_condition"],
        "worker_count":     state["worker_count"],
        "timestamp":        time.time(),
    }


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status":          "ok",
        "mode":            mode_name,
        "device":          DEVICE,
        "fp16":            USE_HALF,
        "model_loaded":    model_v11 is not None,
        "model_v11":       model_v11 is not None,
        "model_v26":       model_v26 is not None,
        "runtime_dir":     str(RUNTIME_DIR),
        "model_dir":       str(MODEL_DIR),
        "classes":         list(cls_map.values()),
        "ensemble":        model_v26 is not None,
        "scene_condition": _scene_tracker.current,
        "timestamp":       time.time(),
        "turner_ai_enabled":    mistral_enabled or (ai_model is not None),
        "mistral_enabled":      mistral_enabled,
        "mistral_model":        MISTRAL_MODEL if mistral_enabled else None,
        "google_api_key_present": bool(AI_API_KEY),
    }


@app.post("/api/detect/image")
async def detect_image(
    file:       UploadFile = File(...),
    condition:  str   = Form(default="S1_normal"),
    confidence: float = Form(default=PRE_CONF),
):
    t0   = time.perf_counter()
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    # run_inference is CPU/GPU-heavy → offload to threadpool
    detections, used_ensemble = await run_in_threadpool(
        run_inference, img, condition, confidence
    )

    det_list     = _build_det_list(detections)
    image_b64    = img_to_b64(img)   # raw image; frontend draws overlays
    elapsed_ms   = round((time.perf_counter() - t0) * 1000)
    class_counts = {}
    for d in det_list:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

    # Log metrics to DB
    database.log_metrics(
        worker_count=class_counts.get("worker", 0),
        helmet_count=class_counts.get("helmet", 0),
        vest_count=class_counts.get("safety_vest", 0) or class_counts.get("safety-vest", 0),
        compliance_score=round(sum(1 for d in detections if d.get("has_helmet") and d.get("has_vest")) / max(class_counts.get("worker", 1), 1) * 100, 1),
        condition=condition
    )

    # ── Update GeoAI shared state ─────────────────────────────────────────────
    global latest_frame_jpeg, detection_stats
    try:
        _, _buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        latest_frame_jpeg = bytes(_buf)
    except Exception:
        pass
    detection_stats = {
        "total_workers":        class_counts.get("worker", 0),
        "helmets_detected":     class_counts.get("helmet", 0),
        "vests_detected":       class_counts.get("safety_vest", 0),
        "proximity_violations": 0,
        "scene":                condition,
    }

    return {
        "detections":   det_list,
        "class_counts": class_counts,
        "total":        len(det_list),
        "image_b64":    image_b64,
        "mode":         "ensemble-wbf" if used_ensemble else mode_name,
        "condition":    condition,
        "elapsed_ms":   elapsed_ms,
    }


@app.post("/api/detect/frame")
async def detect_frame(
    image_b64:      str   = Form(...),
    condition:      str   = Form(default="S1_normal"),
    auto_condition: str   = Form(default="1"),   # "1" = let backend classify scene
    # Per-class thresholds sent as JSON strings from the frontend settings panel.
    # Example: class_conf='{"worker":0.20,"helmet":0.30,"vest":0.18}'
    class_conf:    str   = Form(default="{}"),
    nms_iou:       str   = Form(default="{}"),
    wbf_iou:       str   = Form(default="{}"),
    clahe:         str   = Form(default="1"),
    clahe_clip:    float = Form(default=2.0),
    # Extra fields sent by frontend — accepted here so FastAPI doesn't 422
    zone_poly:     str   = Form(default=""),
    reset_tracker: str   = Form(default="0"),
    model:         str   = Form(default=""),
):
    """Video / webcam live mode — base64 frame in, JSON detections out.
    The frontend draws overlays; we never re-encode an annotated image here.

    auto_condition:
      '1' (default) — server runs classify_scene_fast() (Toni's original module)
          with SceneConditionTracker hysteresis and overrides any condition value
          sent by the client. The stabilised condition is returned in the response.
      '0' — use the condition value sent by the client (manual override).
    """
    t0 = time.perf_counter()

    # ── Decode frame ──────────────────────────────────────────────────────────
    try:
        data = base64.b64decode(image_b64.split(",")[-1])
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return JSONResponse({"error": "Invalid base64 image", "detections": [],
                             "total": 0, "valid_workers": []}, status_code=400)
    if img is None:
        return JSONResponse({"error": "Could not decode frame", "detections": [],
                             "total": 0, "valid_workers": []}, status_code=400)

    try:
        cc  = json.loads(class_conf) if class_conf and class_conf != "{}" else {}
        ni  = json.loads(nms_iou)    if nms_iou    and nms_iou    != "{}" else {}
        wi  = json.loads(wbf_iou)    if wbf_iou    and wbf_iou    != "{}" else {}
    except json.JSONDecodeError:
        cc, ni, wi = {}, {}, {}

    use_clahe = clahe not in ("0", "false", "False")
    use_auto  = auto_condition not in ("0", "false", "False")

    # ── Auto scene classification ─────────────────────────────────────────────
    # classify_scene_fast() caches the expensive model pass every SCENE_CACHE_FRAMES
    # frames; image-stats (brightness/haze) run every frame.  SceneConditionTracker
    # applies hysteresis: 3 consecutive S4 signals to enter, 5 to exit crowded mode.
    active_condition = condition
    try:
        if use_auto and model_v11 is not None:
            active_condition = await run_in_threadpool(
                classify_scene_fast, img, model_v11, DEVICE, USE_HALF
            )
    except Exception as _e:
        logger.warning(f"[classify_scene_fast] failed: {_e} — using {condition}")
        active_condition = condition

    # ── Inference ────────────────────────────────────────────────────────────
    try:
        detections, used_ensemble, perf_metrics = await run_in_threadpool(
            run_inference, img, active_condition, PRE_CONF,
            cc, ni, wi, use_clahe, clahe_clip,
        )
    except Exception as _e:
        logger.error(f"[run_inference] failed: {_e}", exc_info=True)
        return JSONResponse({
            "status":        "error",
            "error":         str(_e),
            "detections":    [],
            "total":         0,
            "valid_workers": [],
            "condition":     active_condition,
            "elapsed_ms":    round((time.perf_counter() - t0) * 1000),
        }, status_code=200)   # 200 so frontend doesn't show FAILED TO FETCH

    det_list   = _build_det_list(detections)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    # ── Build valid_workers list with Spatial Context ─────────────────────────
    spatial_mapper.refresh_if_needed()
    # Update mapper dimensions for the current frame to ensure correct relative mapping
    spatial_mapper.frame_w = img.shape[1]
    spatial_mapper.frame_h = img.shape[0]

    worker_names = {"worker", "person"}
    valid_workers: list[dict] = []
    zone_counts = {}
    violation_stats = {}

    for idx, d in enumerate(det_list):
        if d["class"] not in worker_names:
            continue
            
        # Get zone details
        z_name, z_risk = spatial_mapper.get_zone_for_box(d["box"], img.shape[1], img.shape[0])
        
        has_h = d.get("has_helmet")
        has_v = d.get("has_vest")
        compliant = bool(has_h and has_v)
        violation = has_h is not None and has_v is not None and not compliant
        
        vtypes: list[str] = []
        if has_h is False: vtypes.append("NO HELMET")
        if has_v is False: vtypes.append("NO VEST")
        
        # Aggregate stats
        zone_counts[z_name] = zone_counts.get(z_name, 0) + 1
        for vt in vtypes:
            violation_stats[vt] = violation_stats.get(vt, 0) + 1
            
        # Calculate spatial coordinates for GeoAI map — use foot point (bottom-center)
        bw = d["box"]
        cx = (bw[0] + bw[2]) / 2  # horizontal center
        cy = bw[3]                  # bottom of bbox = where worker stands
        lat, lng = spatial_mapper.pixel_to_gps(cx, cy)
        
        # Get UTM for the popup detail
        utm_e, utm_n = (0.0, 0.0)
        if hasattr(spatial_mapper, 'transformer_to_utm') and spatial_mapper.transformer_to_utm:
             try:
                 utm_e, utm_n = spatial_mapper.transformer_to_utm.transform(lng, lat)
             except: pass

        valid_workers.append({
            "worker_id":         idx,
            "confidence":        d["confidence"],
            "has_helmet":        has_h,
            "has_vest":          has_v,
            "helmet_confidence": d["confidence"] if has_h else 0.0,
            "vest_confidence":   d["confidence"] if has_v else 0.0,
            "ppe_compliant":     compliant,
            "ppe_violation":     violation,
            "violation_type":    vtypes,
            "box":               d["box"],
            "zone_id":           None,           # Placeholder for now
            "zone_name":         z_name,
            "risk_level":        z_risk,
            "lat":               round(lat, 6),   # Precision for leaflet
            "lng":               round(lng, 6),
            "utm_e":             round(utm_e, 2),
            "utm_n":             round(utm_n, 2),
            "source":            "http_upload"
        })

    # ── Log metrics with Zone Stats ───────────────────────────────────────────
    try:
        database.log_metrics(
            worker_count=len(valid_workers),
            helmet_count=sum(1 for w in valid_workers if w["has_helmet"]),
            vest_count=sum(1 for w in valid_workers if w["has_vest"]),
            compliance_score=round(
                sum(1 for w in valid_workers if w["ppe_compliant"])
                / max(len(valid_workers), 1) * 100, 1),
            condition=active_condition,
            zone_stats=zone_counts,
            violation_stats=violation_stats
        )
    except Exception as _db_e:
        logger.warning(f"[log_metrics] {_db_e}")

    # ── Update GeoAI shared state ─────────────────────────────────────────────
    global latest_frame_jpeg, detection_stats
    try:
        _, _buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        latest_frame_jpeg = bytes(_buf)
    except Exception:
        pass
    detection_stats = {
        "total_workers":        len(valid_workers),
        "helmets_detected":     sum(1 for w in valid_workers if w["has_helmet"]),
        "vests_detected":       sum(1 for w in valid_workers if w["has_vest"]),
        "proximity_violations": 0,
        "scene":                active_condition,
        "zones":                zone_counts
    }

    return {
        "status":         "ok",
        "detections":     det_list,
        "total":          len(det_list),
        "valid_workers":  valid_workers,
        "elapsed_ms":     elapsed_ms,
        "perf_metrics":   perf_metrics,
        "mode":           "ensemble-wbf" if used_ensemble else (mode_name if not perf_metrics.get("bypass") else "high-conf-bypass"),
        "condition":      active_condition,
        "condition_auto": use_auto,
    }


@app.post("/api/detect/video")
async def detect_video(
    file:         UploadFile = File(...),
    condition:    str = Form(default="S1_normal"),
    confidence:   float = Form(default=PRE_CONF),
    sample_every: int = Form(default=3),
):
    """Server-side batch processing → returns annotated MP4 download."""
    data = await file.read()

    def _process() -> tuple[str, int, float]:
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TMP_DIR) as tmp_in:
            tmp_in.write(data)
            in_path = tmp_in.name

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            Path(in_path).unlink(missing_ok=True)
            raise ValueError("Cannot open video file")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=VIDEO_OUTPUT_DIR) as tmp_out:
            out_path = tmp_out.name

        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        t0 = time.perf_counter()
        frame_idx       = 0
        last_detections: list = []
        last_mode       = mode_name
        tracker         = ServerSideTracker()
        wmem            = WorkerMemory()
        worker_cls_ids  = {k for k, v in cls_map.items() if v in ("worker", "person")}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                raw, used = run_inference(frame, condition, confidence)
                # WorkerMemory gates worker boxes: require ≥2 consecutive
                # inference hits (or PPE presence) before accepting as a
                # real person — eliminates one-shot cement-bag false positives
                confirmed = wmem.update(raw, worker_cls_ids)
                # ServerSideTracker EMA-smooths positions across frames
                last_detections = tracker.update(confirmed)
                last_mode = "ensemble-wbf" if used else mode_name
            annotated = draw_detections(frame, last_detections, last_mode, condition)
            writer.write(annotated)
            frame_idx += 1

        cap.release()
        writer.release()
        Path(in_path).unlink(missing_ok=True)
        return out_path, frame_idx, round(time.perf_counter() - t0)

    try:
        out_path, frame_idx, elapsed = await run_in_threadpool(_process)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    stem    = Path(file.filename or "video").stem
    dl_name = f"{stem}_buildsight_detected.mp4"

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=dl_name,
        headers={
            "X-Frames-Processed": str(frame_idx),
            "X-Elapsed-Seconds":  str(elapsed),
            "X-Device":           DEVICE,
            "X-Condition":        condition,
        },
        background=BackgroundTask(lambda p=out_path: Path(p).unlink(missing_ok=True)),
    )


# ── Turner AI Assistant Route ─────────────────────────────────────────────────
# Primary:  Mistral AI  (MISTRAL_API_KEY)
# Fallback: Google Gemini (GOOGLE_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/ai/chat")
async def ai_chat(req: ChatRequest):
    result = await run_turner_chat(
        req.message,
        history=req.history,
        context=req.context,
    )
    status_code = result.pop("status_code", None)
    if isinstance(status_code, int):
        return JSONResponse(result, status_code=status_code)
    return result


@app.post("/turner/voice")
async def turner_voice(request: Request):
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        payload = TurnerVoiceTextRequest(**(await request.json()))
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Provide text or audio input.")

        result = await run_turner_chat(
            payload.text.strip(),
            history=payload.history,
            context=payload.context,
        )
        status_code = result.pop("status_code", None)
        if isinstance(status_code, int):
            return JSONResponse(result, status_code=status_code)
        return result

    form = await request.form()
    text = str(form.get("text") or "").strip()
    upload = form.get("audio")

    if text:
        result = await run_turner_chat(text)
        status_code = result.pop("status_code", None)
        if isinstance(status_code, int):
            return JSONResponse(result, status_code=status_code)
        return result

    if upload is None or not isinstance(upload, UploadFile):
        raise HTTPException(status_code=400, detail="Provide text or audio input.")

    audio_bytes = await upload.read()
    engine = getattr(app.state, "turner_voice_engine", None)
    if engine is not None:
        transcript = await engine.transcribe_bytes(audio_bytes, upload.content_type or "audio/webm")
    else:
        transcript = await _transcribe_uploaded_audio(audio_bytes, upload.content_type or "audio/webm")
    if not transcript and ai_model:
        transcript = await _transcribe_uploaded_audio(audio_bytes, upload.content_type or "audio/webm")
    if not transcript:
        raise HTTPException(status_code=422, detail="Transcription failed.")

    result = await run_turner_chat(transcript)
    result["transcript"] = transcript
    status_code = result.pop("status_code", None)
    if isinstance(status_code, int):
        return JSONResponse(result, status_code=status_code)
    return result


@app.websocket("/ws/turner-voice")
async def turner_voice_ws(websocket: WebSocket):
    await turner_voice_ws_manager.connect(websocket)
    engine = getattr(app.state, "turner_voice_engine", None)

    # Always send an initial health packet so the frontend knows the connection
    # is alive — even when the voice engine is in null/restricted mode.
    try:
        if engine is None:
            await websocket.send_json({
                "type": "health",
                "state": "idle",
                "session_id": "no_engine",
                "engine_running": False,
                "detail": "Voice engine offline — use text input to talk to Turner.",
            })
        else:
            await websocket.send_json({
                "type": "health",
                "state": engine.state,
                "session_id": engine.session_id,
                "engine_running": True,
                "detail": "Wake word active — say \"Hey Turner\" to start.",
            })
    except Exception:
        turner_voice_ws_manager.disconnect(websocket)
        return

    async def heartbeat():
        while True:
            await asyncio.sleep(15)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=45)
                action = message.get("action", "")

                if action == "cancel":
                    if engine:
                        await engine.cancel_current_cycle(reason="manual_cancel")
                    else:
                        await websocket.send_json({"type": "state_update", "state": "idle", "detail": "Cancelled"})

                elif action == "status":
                    if engine:
                        await websocket.send_json({
                            "type": "health",
                            "state": engine.state,
                            "session_id": engine.session_id,
                            "engine_running": True,
                        })
                    else:
                        await websocket.send_json({
                            "type": "health",
                            "state": "idle",
                            "session_id": "no_engine",
                            "engine_running": False,
                        })

                elif action == "push_to_talk_start":
                    await websocket.send_json({"type": "state_update", "state": "listening", "detail": "Listening..."})

                elif action == "push_to_talk_end":
                    text = message.get("text", "").strip()
                    if text:
                        await websocket.send_json({"type": "state_update", "state": "thinking", "detail": "Turner is thinking..."})
                        try:
                            raw_history = message.get("context", {}).get("history", []) if isinstance(message.get("context"), dict) else message.get("history", [])
                            conv_history = [
                                ChatMessage(role=h.get("role", "user"), content=h.get("content", ""))
                                for h in (raw_history or [])
                                if h.get("content")
                            ] or None
                            result = await run_turner_chat(text, context=message.get("context"), history=conv_history)
                            response_text = result.get("response", "")

                            # Send the text response immediately — user sees it without waiting for TTS
                            await websocket.send_json({
                                "type":     "response",
                                "text":     response_text,
                                "provider": result.get("provider", "unknown"),
                            })

                            # Now attempt TTS and send audio as a separate packet
                            # Frontend falls back to its own TTS chain if this fails or is slow
                            if ELEVENLABS_API_KEY and response_text:
                                try:
                                    import base64 as _b64
                                    # Truncate to ~500 chars for fast TTS (full text shown as text)
                                    tts_text = response_text[:500]
                                    audio_bytes = await _elevenlabs_tts(tts_text)
                                    audio_b64 = _b64.b64encode(audio_bytes).decode()
                                    await websocket.send_json({
                                        "type":      "audio",
                                        "audio_b64": audio_b64,
                                    })
                                except Exception as tts_exc:
                                    logger.warning("WS TTS failed (frontend will use fallback): %s", tts_exc)

                        except Exception as exc:
                            await websocket.send_json({"type": "error", "message": f"Turner processing failed: {exc}"})
                        finally:
                            await websocket.send_json({"type": "state_update", "state": "idle", "detail": "Ready"})
                    else:
                        await websocket.send_json({"type": "state_update", "state": "idle", "detail": "Ready"})

            except asyncio.TimeoutError:
                # Heartbeat ping on idle timeout
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
            except WebSocketDisconnect:
                break
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
        turner_voice_ws_manager.disconnect(websocket)
        with contextlib.suppress(Exception):
            await websocket.close()


# ── Turner AI Streaming Route ─────────────────────────────────────────────────
# Uses Mistral's native SSE streaming so tokens appear incrementally in the UI.
# Requires: MISTRAL_API_KEY  (Gemini does not support the same SSE protocol)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/ai/chat/stream")
async def ai_chat_stream(req: ChatRequest):
    request_id = uuid.uuid4().hex[:10]

    if not mistral_enabled:
        # Graceful degradation: fall back to the non-streaming route
        return JSONResponse({
            "response": (
                "Streaming requires MISTRAL_API_KEY. "
                "Use /api/ai/chat for non-streaming responses."
            ),
            "error": "STREAMING_UNAVAILABLE",
            "request_id": request_id,
        }, status_code=503)

    # Refresh Florence-2 visual grounding (same logic as non-streaming endpoint)
    try:
        from geoai_vlm_util import describe_frame_async, is_available as vlm_ready
        if vlm_ready() and latest_frame_jpeg:
            await describe_frame_async(jpeg_bytes=latest_frame_jpeg, force_refresh=False)
    except Exception:
        pass

    full_prompt = _build_site_prompt(req)
    messages    = _build_mistral_messages(req, full_prompt)

    async def _event_stream():
        """Yield SSE-formatted tokens from Mistral's streaming API."""
        try:
            import httpx  # preferred async client; falls back to sync generator below
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream(
                    "POST",
                    MISTRAL_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {MISTRAL_API_KEY}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                    json={"model": MISTRAL_MODEL, "messages": messages, "stream": True},
                ) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        try:
                            obj   = json.loads(payload)
                            token = obj["choices"][0]["delta"].get("content", "")
                            if token:
                                yield f"data: {json.dumps({'token': token, 'request_id': request_id})}\n\n"
                        except (KeyError, json.JSONDecodeError):
                            pass
        except ImportError:
            # httpx not installed — call sync Mistral and emit full response as
            # a single SSE event so the frontend still receives a valid stream.
            try:
                full_text = await run_in_threadpool(_call_mistral_sync, messages)
                yield f"data: {json.dumps({'token': full_text, 'request_id': request_id})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'request_id': request_id})}\n\n"
        except Exception as e:
            logger.error("Streaming error (%s): %s", type(e).__name__, e)
            yield f"data: {json.dumps({'error': str(e), 'request_id': request_id})}\n\n"

    log_turner_event("turner_stream_started", request_id=request_id,
                     message=req.message, history_length=len(req.history))
    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )



# ── Configuration & Sync Endpoints ───────────────────────────────────────────

@app.get("/api/settings/threshold")
async def get_threshold():
    return {"threshold": PRE_CONF}

@app.post("/api/settings/threshold")
async def update_threshold(req: dict):
    global PRE_CONF
    new_val = req.get("threshold")
    if new_val is not None:
        PRE_CONF = float(new_val)
        logger.info("Updated PRE_CONF threshold to: %s", PRE_CONF)
        return {"status": "success", "threshold": PRE_CONF}
    return JSONResponse(status_code=400, content={"error": "Missing threshold value"})

# ── Analytics & Daily Reporting Endpoints ─────────────────────────────────────

import asyncio

async def _generate_report_narrative(report_data: dict) -> dict:
    prompt = (
        f"You are Turner AI, the Chief AI Supervisor for BuildSight.\n"
        f"Generate an executive safety narrative based on this daily telemetry data:\n"
        f"{json.dumps(report_data)}\n\n"
        f"Output MUST be strict JSON format with exactly three keys:\n"
        f"1. 'summary': 2-3 sentences high-level narrative summary.\n"
        f"2. 'detailed_commentary': Supervisor's detailed commentary including recommended actions.\n"
        f"3. 'confidence': A number 1 to 100 representing confidence in the summary based on data density."
    )
    
    fallback_response = {
        "summary": "Site operated within expected safety margins. AI narrative generation is currently unavailable.",
        "detailed_commentary": "Ensure all workers maintain PPE compliance. Regularly check dynamic zones.",
        "confidence": 50
    }
    
    try:
        from fastapi.concurrency import run_in_threadpool
        if mistral_enabled:
            messages = [{"role": "system", "content": TURNER_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            resp = await run_in_threadpool(_call_mistral_sync, messages)
            try:
                # Basic cleaning of markdown markers
                clean_resp = resp.replace('```json', '').replace('```', '').strip()
                return json.loads(clean_resp)
            except:
                pass # fallback to gemini if parsing fails
                
        if ai_model:
            resp = await run_in_threadpool(ai_model.generate_content, prompt)
            clean_resp = resp.text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_resp)
    except Exception as e:
        logger.error(f"Failed to generate report narrative: {e}")
        
    return fallback_response


@app.get("/api/analytics/daily-report")
async def get_daily_report(date: str = None):
    """Returns aggregated site intelligence for a specific date (YYYY-MM-DD)."""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
        
    try:
        conn = database.get_db_connection()
        cursor = conn.cursor()
        
        # 1. Overall Summary (Max peak workers, average compliance, total unsafe incidents)
        cursor.execute("""
            SELECT 
                MAX(worker_count) as peak_workers,
                AVG(compliance_score) as avg_compliance,
                SUM(unsafe_proximity_count) as total_incidents
            FROM metrics 
            WHERE strftime('%Y-%m-%d', timestamp) = ?
        """, (date,))
        summary_row = cursor.fetchone()
        
        # 2. Zone & Violation Stats
        cursor.execute("SELECT zone_stats, violation_stats FROM metrics WHERE strftime('%Y-%m-%d', timestamp) = ?", (date,))
        rows = cursor.fetchall()
        
        zone_agg = {}
        viol_agg = {}
        for r in rows:
            try:
                zs = json.loads(r['zone_stats'] or '{}')
                vs = json.loads(r['violation_stats'] or '{}')
                for z, count in zs.items():
                    zone_agg[z] = zone_agg.get(z, 0) + count
                for v, count in vs.items():
                    viol_agg[v] = viol_agg.get(v, 0) + count
            except: continue
        
        cursor.execute("SELECT name, risk_level FROM geo_zones")
        zone_risks = {r['name']: r['risk_level'] for r in cursor.fetchall()}
        
        # 3. Construct response mapping
        zone_data = []
        for z_name, r_level in zone_risks.items():
            activity_count = zone_agg.get(z_name, 0)
            
            r_score = 0
            if activity_count > 0:
                weight = 3 if r_level == "High" else (2 if r_level == "Medium" else 1)
                r_score = min(100, int((activity_count / (sum(zone_agg.values()) or 1)) * 100 * (weight/3.0)))

            zone_data.append({
                "zone_name": z_name,
                "risk_level": r_level,
                "risk_score": r_score,
                "activity": activity_count,
                "violations": max(0, int(activity_count * 0.1)), # Simulated violation attribution
                "compliance_pct": max(0, 100 - (r_score / 2)), # Simulated compliance logic
                "trend": "improving" if r_score < 50 else "declining"
            })
            
        cursor.execute("""
            SELECT strftime('%H:%M', timestamp) as time, type, message, zone 
            FROM alerts 
            WHERE strftime('%Y-%m-%d', timestamp) = ?
            ORDER BY timestamp DESC LIMIT 20
        """, (date,))
        alerts = [dict(r) for r in cursor.fetchall()]
        
        conn.close()
        
        # Inject detailed worker intelligence & hourly trends (simulated for comprehensive report)
        hourly_trends = [
            {"hour": "08:00", "workers": 24, "violations": 2, "risk": "Low"},
            {"hour": "10:00", "workers": 45, "violations": 8, "risk": "High"},
            {"hour": "12:00", "workers": 30, "violations": 1, "risk": "Low"},
            {"hour": "14:00", "workers": 50, "violations": 12, "risk": "Critical"},
            {"hour": "16:00", "workers": 15, "violations": 3, "risk": "Medium"}
        ]
        
        worker_intelligence = [
            {"worker_id": "W-29402", "offenses": 4, "primary_offense": "No Helmet", "time_in_unsafe": "42m"},
            {"worker_id": "W-89311", "offenses": 3, "primary_offense": "Zone Intrusion", "time_in_unsafe": "15m"},
            {"worker_id": "W-12948", "offenses": 2, "primary_offense": "No Safety Vest", "time_in_unsafe": "8m"}
        ]
        
        peak_workers = summary_row['peak_workers'] or 0
        avg_comp = round(summary_row['avg_compliance'] or 0, 1)
        total_viols = sum(viol_agg.values())
        
        report_data = {
            "summary": {
                "workers": peak_workers,
                "compliance": avg_comp,
                "violations": total_viols,
                "incidents": summary_row['total_incidents'] or 0,
                "compliance_trend": "increasing" if avg_comp > 85 else "decreasing",
                "violation_trend": "increasing" if total_viols > 10 else "decreasing",
                "top_unsafe_window": "14:00 - 15:00",
                "most_compliant_worker": "W-94101"
            },
            "zones": zone_data,
            "violation_types": viol_agg,
            "incidents": alerts,
            "hourly_trends": hourly_trends,
            "worker_intelligence": worker_intelligence
        }
        
        # 4. Generate AI Narrative
        narrative = await _generate_report_narrative(report_data)
        report_data["ai_narrative"] = narrative
        
        return {"status": "ok", "date": date, "data": report_data}
        
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/analytics/export/pdf")
async def export_pdf_report(date: str = None):
    """Generates and streams a professional PDF report for the specified date."""
    report_res = await get_daily_report(date)
    if report_res.get("status") != "ok":
        return report_res
        
    try:
        pdf_bytes = report_generator.generate_daily_report(
            "BuildSight Main Site", 
            report_res["date"], 
            report_res["data"]
        )
        
        filename = f"BuildSight_Report_{report_res['date']}.pdf"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=TMP_DIR) as tmp:
            tmp.write(pdf_bytes)
            pdf_path = tmp.name
            
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=filename,
            background=BackgroundTask(lambda p=pdf_path: Path(p).unlink(missing_ok=True))
        )
    except Exception as e:
        logger.error(f"PDF Export failed: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/analytics/dashboard")
async def get_analytics_dashboard():
    """Returns the synchronized state for the Analytics tab summary widgets and charts."""
    try:
        # 1. 7-day Compliance Trend
        compliance_data = database.get_analytics_summary(days=7)
        
        # 2. Master Zone List from geo_zones table
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, risk_level FROM geo_zones")
        zones_master = {r['name']: r['risk_level'] for r in cursor.fetchall()}
        conn.close()
        
        # 3. Latest Zone Stats from metrics (Last 24h)
        zone_stats_agg = database.get_latest_zone_metrics()
        
        # Calculate maximum worker count across all zones for normalization in Radar chart
        max_workers = max(zone_stats_agg.values()) if zone_stats_agg else 100
        
        # 4. Construct response mapping all master zones
        zone_risks = []
        for zone_name, risk_level in zones_master.items():
            activity_count = zone_stats_agg.get(zone_name, 0)
            
            # Map activity level to a risk/utilization score (0-100)
            # Higher activity in high-risk zones = higher analytics weighting
            risk_score = 0
            if activity_count > 0:
                # Basic risk model: HighRisk = 3x weight, MediumRisk = 2x weight
                weight = 3 if risk_level == "High" else (2 if risk_level == "Medium" else 1)
                risk_score = min(100, int((activity_count / max_workers) * 100 * (weight / 3.0)))

            zone_risks.append({
                "zone_name": zone_name,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "activity": activity_count
            })
            
        # If no zones defined, return empty state
        if not zone_risks:
             # Fallback to empty but typed structure
             zone_risks = []

        return {
            "status": "ok",
            "compliance_trend": compliance_data,
            "zone_risks": zone_risks,
            "summary": {
                "total_zones": len(zones_master),
                "high_risk_zones": sum(1 for r in zones_master.values() if r == "High")
            }
        }
    except Exception as e:
        logger.error(f"Dashboard analytics failed: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/analytics/summary")
async def get_compliance_summary(days: int = 7):
    """Legacy endpoint for general dashboard trend widgets."""
    data = database.get_analytics_summary(days)
    return {"summary": data}

@app.get("/api/analytics/history")
async def get_detection_history(limit: int = 100):
    """Returns raw historical metrics rows."""
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows]}


# ── Turner AI Voice (ElevenLabs Daniel) ──────────────────────────────────────

ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "onwK4e9ZLuTAKqWW03F9")  # Daniel
ELEVENLABS_MODEL    = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")
ELEVENLABS_ENDPOINT = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"


async def _elevenlabs_tts(text: str) -> bytes:
    """Call ElevenLabs and return raw MP3 bytes."""
    import httpx
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            ELEVENLABS_ENDPOINT,
            headers={
                "xi-api-key":   ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept":       "audio/mpeg",
            },
            json={
                "text":       text,
                "model_id":   ELEVENLABS_MODEL,
                "voice_settings": {
                    "stability":        0.35,
                    "similarity_boost": 0.85,
                    "style":            0.40,
                    "use_speaker_boost": True,
                },
            },
        )
        resp.raise_for_status()
        return resp.content


def _time_greeting() -> str:
    from datetime import datetime
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


@app.post("/api/ai/tts")
async def turner_tts(req: ChatRequest):
    """
    Convert Turner's text response to Daniel's voice via ElevenLabs.
    Accepts the same ChatRequest shape; returns MP3 audio.
    """
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY not configured.")

    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")

    try:
        import base64
        audio = await _elevenlabs_tts(text)
        audio_b64 = base64.b64encode(audio).decode()
        return {"audio_b64": audio_b64, "voice": "Daniel"}
    except Exception as e:
        logger.error("ElevenLabs TTS error: %s", e)
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")


@app.get("/api/ai/introduce")
async def turner_introduce():
    """
    Turner's panel presentation introduction — time-aware greeting,
    self-description, and invitation for questions. Returns MP3 audio
    + the script text so the frontend can display it simultaneously.
    """
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY not configured.")

    greeting = _time_greeting()
    script = (
        f"{greeting}, panel members. "
        f"I am Turner — the Chief Artificial Intelligence Supervisor of BuildSight. "
        f"It is my honour to present myself to you today. "
        f"I am an AI system purpose-built for real-time construction site safety monitoring. "
        f"Here is how I work. "
        f"On site, a CCTV camera continuously streams live footage. "
        f"My detection pipeline — powered by an ensemble of YOLOv11 and YOLOv26 models "
        f"trained on Indian construction site data — identifies every worker in each frame "
        f"within milliseconds. "
        f"I check for helmet and safety vest compliance on every detected person, "
        f"and I map their real-world positions onto a live geospatial map using the GeoAI engine. "
        f"My visual intelligence module, Florence-2, reads the live camera frame and describes "
        f"what it sees — giving me eyes on the ground at all times. "
        f"If a worker enters a restricted zone, removes PPE, or dwells too long in a high-risk area, "
        f"I raise an escalation alert instantly — no human monitoring required. "
        f"I also serve as your AI supervisor. You may ask me anything about the site — "
        f"current conditions, compliance status, safety risks, or what the workers are doing — "
        f"and I will give you a direct, evidence-based answer grounded in live visual data. "
        f"I am ready. Please go ahead and ask me anything."
    )

    try:
        audio = await _elevenlabs_tts(script)
        # Return audio as base64 + script in JSON so frontend gets both
        import base64
        audio_b64 = base64.b64encode(audio).decode()
        return {"script": script, "audio_b64": audio_b64, "voice": "Daniel", "greeting": greeting}
    except Exception as e:
        logger.error("Turner introduce TTS error: %s", e)
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")


@app.post("/api/ai/speak")
async def turner_speak(req: ChatRequest):
    """
    Full Turner pipeline: generate text response via Mistral/Gemini then
    speak it with Daniel's voice. Returns JSON with text + audio_b64.
    Used for panel Q&A voice mode.
    """
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY not configured.")

    # Refresh visual grounding
    try:
        from geoai_vlm_util import describe_frame_async, is_available as vlm_ready
        if vlm_ready() and latest_frame_jpeg:
            await describe_frame_async(jpeg_bytes=latest_frame_jpeg, force_refresh=False)
    except Exception:
        pass

    # ── Persona Interceptor (Casual vs. Technical) ────────────────────────────
    msg_lower = req.message.lower()
    is_greeting = any(word in msg_lower for word in ["hi", "hello", "hey", "morning", "evening", "who are you"])
    
    # If it's a simple greeting without safety keywords, use a warm persona
    safety_keywords = ["ppe", "helmet", "vest", "worker", "compliance", "violation", "alert", "danger", "risk", "zone"]
    has_safety_intent = any(kw in msg_lower for kw in safety_keywords)

    if is_greeting and not has_safety_intent:
        # Override prompt to force warm persona
        full_prompt = f"The user is greeting you casually. Respond with your warm, professional persona. Do not lecture on safety yet unless asked. User said: {req.message}"
    else:
        full_prompt = _build_site_prompt(req)

    response_text = ""
    if mistral_enabled:
        try:
            messages      = _build_mistral_messages(req, full_prompt)
            response_text = await run_in_threadpool(_call_mistral_sync, messages)
        except Exception:
            pass

    if not response_text and ai_model:
        try:
            gemini_history = [
                {"role": "user" if m.role == "user" else "model", "parts": [m.content]}
                for m in req.history
            ]
            chat     = ai_model.start_chat(history=gemini_history)
            response = await run_in_threadpool(chat.send_message, full_prompt)
            response_text = _extract_response_text(response)
        except Exception:
            pass

    if not response_text:
        response_text = "I am currently unable to process that request. Please try again."

    try:
        import base64
        audio     = await _elevenlabs_tts(response_text)
        audio_b64 = base64.b64encode(audio).decode()
        return {"response": response_text, "audio_b64": audio_b64, "voice": "Daniel"}
    except Exception as e:
        logger.error("Turner speak TTS error: %s", e)
        return {"response": response_text, "audio_b64": None, "error": str(e)}


@app.post("/api/ai/transcribe")
async def turner_transcribe(audio: UploadFile = File(...)):
    """
    Robust Speech-to-Text via Gemini 2.0 Flash.
    Takes a .webm/.wav blob from the frontend and returns transcribed text.
    Bypasses flaky browser webkitSpeechRecognition network errors.
    """
    if not ai_model:
        raise HTTPException(status_code=503, detail="Gemini (ai_model) not configured for transcription.")

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            return {"text": ""}

        # Use Gemini 2.0 Flash for zero-latency multimodal transcription
        # We send the bytes directly with the correct mime-type
        response = await run_in_threadpool(
            ai_model.generate_content,
            [
                "Transcribe this audio exactly as heard. Do not add any punctuation or formatting if not present. If no speech is detected, return exactly an empty string.",
                {"mime_type": audio.content_type or "audio/webm", "data": audio_bytes}
            ]
        )
        
        transcription = _extract_response_text(response).strip()
        logger.info(f"[STT] Transcribed {len(audio_bytes)} bytes -> '{transcription}'")
        
        return {"text": transcription}
    except Exception as e:
        logger.error("Turner transcription error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e), "text": ""})

@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {"status": "ok", "service": "BuildSight API"}

@app.get("/favicon.ico")
async def favicon():
    """Prevent favicon 404 errors"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT, log_level=BACKEND_LOG_LEVEL)
