# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BuildSight is an AI-powered construction site safety monitoring system. It detects PPE compliance (helmets, safety vests, workers) using a dual-model YOLO ensemble, maps detections to GPS coordinates via GeoAI, and serves a live React dashboard with real-time alerts, a spatial heatmap, and an AI voice assistant called Turner.

**Two git remotes:**
- `origin` → `https://github.com/greenbuildai/BuildSight.git`
- `jovix27` → `https://github.com/Jovix27/BuildSight.git` (primary working remote — Jovi's fork)

---

## Commands

### Frontend (React + Vite)
All frontend commands run from `dashboard/`:

```powershell
cd dashboard
npm run dev        # Dev server (Vite HMR) — http://localhost:5173
npm run build      # Type-check + production build
npm run lint       # ESLint
npm run preview    # Preview production build
```

### Backend (Python / FastAPI)
Run from `dashboard/backend/`:

```powershell
# Start everything (API + GeoAI WS + Symphony + Heatmap + Pipeline)
python start_backend.py

# Start only the FastAPI detection server (port 8000)
python start_backend.py --api

# Start only the GeoAI WebSocket broadcast (port 8765)
python start_backend.py --ws
```

Individual services can also be run directly:
```powershell
python dashboard/backend/server.py            # FastAPI, port 8000
python dashboard/backend/geoai_ws_server.py   # GeoAI WS, port 8765
python dashboard/backend/symphony_service.py  # Symphony/Socket.IO, port 3001
```

### GeoAI Pipeline (standalone inference-to-map)
```powershell
python geoai_pipeline.py --source <video.mp4> [--db] [--ws] [--model-mode ensemble]
```

### Python environment
The repo uses a `.venv` at the root. GPU requirements are in `requirements_gpu.txt`; the main Python stack targets CUDA 12.1 + PyTorch 2.5.1+cu121.

**GPU confirmed: RTX 4050, cuda:0, FP16 enabled.**

---

## Architecture

### Dual-layer structure
The project has two independent but integrated layers:

1. **`dashboard/`** — full-stack application (React frontend + FastAPI backend)
2. **Root-level scripts** — research, training, GeoAI pipeline, and data tooling

### Backend (`dashboard/backend/`)

- **`server.py`** — FastAPI app. Core endpoints: `POST /api/detect/image|frame|video`, `GET /api/health`, `WS /ws/detection` (streams detection state to the frontend). Wraps all YOLO inference in `run_in_threadpool` to keep the event loop unblocked. Also mounts `geoai_router`. Hosts live camera streaming endpoints (`POST /api/stream/start`, `GET /api/stream/live` MJPEG, `POST /api/stream/stop`). Also exposes `GET /api/stream/cameras` which probes camera indices 0–9 using the **default backend (NOT DSHOW)** to avoid locking hardware handles.
- **`buildsight_ensemble.py`** — Single source of truth for all detection logic. Runs YOLOv11 + YOLOv26 in parallel, fuses boxes using Weighted Box Fusion (WBF), applies per-scene-condition profiles (S1–S4), then associates PPE to workers by bounding-box overlap. Output boxes are in **absolute pixel coordinates** of the inference frame. See **Ensemble Pipeline Critical Notes** below.
- **`background_detection_service.py`** — Daemon thread service that reads frames from the live camera (or video file), resizes to max 640px, runs the ensemble, and broadcasts `detection_update` via WebSocket every ~150ms. Frame skipping: processes 1 in every 2 frames (`_frame_skip_n = 2`).
- **`camera_stream.py`** — `CameraStreamManager`. Opens a local webcam at **640×480 @ 30 FPS** (Windows uses DSHOW backend for the actual stream). Warm-up discards **5 frames** (not 20 — was mistakenly changed to 20, reverted). `get_camera()` returns the singleton instance.
- **`voice_engine.py`** — Turner AI voice engine. Protocol-based interfaces (`AudioInput`, `WakeWordDetector`, `SpeechTranscriber`) with a `SoundDeviceAudioInput` implementation. Emits `VoiceEvent`s consumed by `server.py`.
- **`geoai/`** — Sub-package: `router.py` (FastAPI routes), `models.py` (Pydantic schemas), `utils/spatial_mapper.py` (pixel → UTM/GPS via PnP homography), `utils/intelligence.py` (BOCW safety rule engine).
- **`geoai_ws_server.py`** — Standalone WebSocket broadcast server (port 8765) that pushes spatial detection events to the frontend GeoAI map.
- **`heatmap_engine.py`** — KDE-based worker density and risk-score heatmap generator.
- **`symphony_service.py`** — FastAPI + Socket.IO orchestration service (port 3001) for multi-agent task management.
- **`database.py`** — SQLite (`buildsight.db`). Tables: `metrics`, `alerts`, `geo_zones`. Schema migrations done inline via `PRAGMA table_info` checks.
- **`report_generator.py`** — PDF report generation from stored metrics.

Model weights are resolved via env vars (`BUILDSIGHT_MODEL_V11`, `BUILDSIGHT_MODEL_V26`) or default to `research/weights/`. The backend reads `.env` from both the project root and `dashboard/backend/`.

### Frontend (`dashboard/src/`)

- **`App.tsx`** — Root shell. Manages the active view (`dashboard | analytics | geoai | turner | brain | settings`) and `dashboardMode` (`LIVE | VIDEO | IMAGE`). LIVE mode renders `<LiveSurveillance />`, VIDEO and IMAGE modes render `<DetectionPanel mode={...} />`.
- **`store/detectionStore.ts`** — Zustand store. Holds a single `WebSocket` to `ws://localhost:8000/ws/detection` that opens at app startup and stays open across all tab changes. Key fields: `detections[]`, `frameWidth`, `frameHeight`, `fps`, `latencyMs`, `sceneCondition`, `workerPositions[]`, `violations[]`.
- **`components/LiveSurveillance.tsx`** — Live camera PPE detection. Uses MJPEG `<img>` stream from `/api/stream/live` for video display and WebSocket `detection_update` messages for bounding box data. Canvas overlay draws boxes directly from `store.detections`. See **Live Surveillance Critical Notes** below.
- **`components/DetectionPanel.tsx`** — Video upload and image inference modes.
- **`components/TurnerPage.tsx` / `TurnerAssistant.tsx` / `TurnerSidebar.tsx`** — Turner AI supervisor interface. Connects to `ws://localhost:8000/ws/turner-voice`.
- **`components/GeoAIPage.tsx` / `GeoAIMap.tsx` / `GeoAIHUD.tsx`** — GeoAI spatial view on a Leaflet map.
- **`hooks/useCameraDevices.ts`** — Browser `getUserMedia` + `enumerateDevices`. **NOT used in `LiveSurveillance` for the camera dropdown** — see Live Surveillance Critical Notes.

---

## Ensemble Pipeline Critical Notes

### PRE_CONF and condition profiles
`PRE_CONF = 0.45` is defined at module level but **must NOT be used directly** in `EnsemblePipeline.run()` model.predict() calls. Instead use:
```python
profile      = _PROFILES.get(use_condition, _PROFILES["S1_normal"])
pre_conf_eff = profile.get("pre_conf", PRE_CONF)   # 0.08–0.10 per condition
```
The profile pre_conf values (0.08 S2_dusty, 0.10 S1_normal) are much lower than 0.45 — this is intentional. Low pre_conf passes many candidates to WBF, which then filters via the post_gate per-class thresholds. Using PRE_CONF=0.45 directly would kill 40–44% confidence detections before WBF ever runs.

### Worker box area guardrails (`wbf_fuse` and `wbf_fuse_condition`)
```python
if bx_w * bx_h < 0.003:   # sub-pixel noise — too small to be a person
    continue
if bx_w * bx_h > 0.80:    # whole-frame false positive only
    continue
```
**Do NOT restore the old `> 0.15` upper limit.** That was calibrated for drone/CCTV footage where workers are tiny. On a webcam close-up a person fills 30–70% of the frame (area ≈ 0.35), which would be silently discarded by 0.15, leaving only the face region (~0.03 area) as the surviving detection.

### Head-to-body expansion (Step 5.5 in `EnsemblePipeline.run()`)
When the YOLO model outputs a near-square worker box (aspect w/h > 0.60), it is a face/head-only detection. The box is expanded before the temporal tracker:
```python
if (bw / bh) > 0.60:
    bx[3] = min(float(h), bx[3] + bh * 3.5)   # extend 3.5× head height down
    cx    = (bx[0] + bx[2]) / 2.0
    hw    = bw * 0.65                            # shoulder width from head width
    bx[0] = max(0.0, cx - hw)
    bx[2] = min(float(w), cx + hw)
```
This runs **before** `temporal.update()` so tracking is stable on the expanded box.

### Scene classifier — S2_DUSTY overexposure gate
```python
if contrast < 52 and saturation < 78 and brightness < 160:
    return "S2_dusty"
```
The `brightness < 160` gate is critical. Without it, overexposed scenes (bright blown-out background from a window) are misclassified as S2_DUSTY, applying CLAHE + unsharp mask preprocessing that amplifies overexposure. Dusty/hazy scenes are dim (brightness < 160); overexposed scenes are bright.

---

## Live Surveillance Critical Notes (`LiveSurveillance.tsx`)

### Camera selector — use backend list, NOT browser `getUserMedia`
`useCameraDevices` is **not imported** in `LiveSurveillance.tsx`. The camera dropdown is populated by fetching `GET /api/stream/cameras` (backend probes DSHOW indices 0–9). The selected value is the backend camera index (number), sent directly as `rtsp_url` to `POST /api/stream/start`.

**Why:** `navigator.mediaDevices.getUserMedia()` locks the physical camera hardware handle on Windows for 10–60 seconds even after `track.stop()`. During that window, the backend's DSHOW open fails and retries every 5 seconds — causing 1–3 minute detection delays.

**Browser device list order ≠ DSHOW index order on Windows.** If you use `findIndex()` on the browser device list as the camera index, the wrong camera opens (integrated cam opens when USB is selected and vice versa).

### Camera switch delay
When switching cameras while streaming, wait **1500ms** after `stopCamera()` before calling `startCamera()`:
```typescript
await stopCamera()
await new Promise<void>(r => setTimeout(r, 1500))
await startCamera(camIdx)
```
Windows DSHOW needs ~1–1.5 seconds to fully release the handle after stop.

### FPS/latency UI — separate effects
FPS and latency update on every backend frame (even 0 detections). Detection count updates only when workers are found. These are two separate `useEffect` hooks. Do not merge them back into one with `if (!dets?.length) return` at the top — that causes the UI to show 0 FPS/latency when the pipeline is running but no workers are in frame.

### `enumerate_cameras` in `server.py` — default backend only
The probe in `GET /api/stream/cameras` uses `cv2.VideoCapture(idx)` (default backend), **not** `cv2.VideoCapture(idx, cv2.CAP_DSHOW)`. DSHOW probes lock hardware handles for seconds after release, blocking the subsequent `stream/start` DSHOW open.

---

## Canvas Overlay Pattern (Critical)

Both `LiveSurveillance` and `DetectionPanel` draw bounding boxes on a `<canvas>` overlaid on a media element (`<img>` or `<video>`).

### Coordinate transform
```typescript
const fw = store.frameWidth  || 640   // authoritative — sent by backend
const fh = store.frameHeight || 480
const { rw, rh, ox, oy } = _letterbox(fw, fh, canvas.width, canvas.height)
const x1 = (box[0] / fw) * rw + ox
const y1 = (box[1] / fh) * rh + oy
```

### Canvas sizing rules
- Size canvas from the **media element's `clientWidth/clientHeight`**, NOT the canvas itself.
- Set `canvas.width = dw; canvas.height = dh`. Do NOT set `canvas.style.width`.
- Do NOT apply DPR (`devicePixelRatio`) scaling.

### Detection persistence
```typescript
const st = useDetectionStore.getState()   // always fresh — no stale closure
const dets = st.detections
if (dets && dets.length > 0) lastDetsRef.current = dets
const drawDets = lastDetsRef.current
```

### DO NOT use `img.naturalWidth` for coordinate mapping
Always use `store.frameWidth/frameHeight` instead.

---

## Detection Data Format

Every `detection_update` WebSocket message contains:
```json
{
  "type": "detection_update",
  "detections": [
    {
      "class": "worker",
      "confidence": 0.87,
      "box": [120.5, 45.2, 380.1, 420.8],
      "has_helmet": false,
      "has_vest": true,
      "track_id": 3
    }
  ],
  "frame_width": 640,
  "frame_height": 480,
  "fps": 14.2,
  "latency_ms": 68.4,
  "scene_condition": "S2_dusty",
  "worker_positions": [...],
  "zone_occupancy": {...},
  "violations": [...]
}
```

- `box` values are **absolute pixel coordinates** `[x1, y1, x2, y2]` in the inference frame.
- Class names: `"worker"`, `"helmet"`, `"safety_vest"`.
- `has_helmet` / `has_vest` only present on `"worker"` class detections.

---

## Live Camera Pipeline

```
CameraStreamManager (camera_stream.py)
  └─ Opens webcam at 640×480 via cv2.VideoCapture (DSHOW on Windows)
  └─ Warm-up: discards 5 frames (not 20)
  └─ Stores latest frame in condition variable

BackgroundDetectionService (background_detection_service.py)
  └─ Reads frames from CameraStreamManager
  └─ Resizes to max 640px if needed
  └─ Runs EnsemblePipeline.run(frame, scene)
  └─ Broadcasts detection_update every ~150ms

WSConnectionManager (server.py)
  └─ Thread-safe broadcast via asyncio.run_coroutine_threadsafe

Frontend (LiveSurveillance.tsx)
  └─ POST /api/stream/start → backend opens camera + starts BackgroundDetectionService
  └─ GET /api/stream/live  → MJPEG stream displayed in <img>
  └─ WS /ws/detection      → detections drawn on <canvas> overlay via RAF loop
```

---

## Scene Condition System (S1–S4)

- `S1_normal` — standard conditions, pre_conf=0.10, post_gate worker=0.28
- `S2_dusty` — dim hazy scenes (brightness < 160), pre_conf=0.08, CLAHE + unsharp mask. **NOT applied to overexposed/bright scenes.**
- `S3_low_light` — low brightness (<72), CLAHE + gamma correction
- `S4_crowded` — high worker density, lower IoU merge threshold

Classification thresholds in `detect_condition()`:
```python
if brightness < 72:                            → S3_low_light
if contrast < 52 and saturation < 78
   and brightness < 160:                       → S2_dusty
if worker_count >= 3 or crowd_overlap >= 0.05: → S4_crowded
else:                                          → S1_normal
```

---

## Known Pitfalls

- **Never use `img.naturalWidth` for coordinate mapping** — unreliable on MJPEG streams.
- **Never apply DPR scaling** to the canvas buffer.
- **Never set `canvas.style.width`** in the RAF loop.
- **Always read Zustand store via `useDetectionStore.getState()`** inside RAF callbacks.
- **`background_detection_service.py` skips every other frame** (`_frame_skip_n = 2`) — intentional for GPU load management.
- **Do NOT use `useCameraDevices` in `LiveSurveillance`** — `getUserMedia` locks the camera on Windows, causing 1–3 minute detection startup delays.
- **Do NOT use `cv2.CAP_DSHOW` in `enumerate_cameras`** — locks the hardware handle, blocking the subsequent stream/start open.
- **Do NOT restore `bx_w * bx_h > 0.15` area guardrail** — kills full-body webcam detections. Use `> 0.80`.
- **Do NOT use `PRE_CONF = 0.45` directly in model.predict()** — use `profile["pre_conf"]` instead.
- **Camera warm-up is 5 frames, not 20** — 20 was a mistake that was reverted.

---

## Data & Research

- `research/weights/` — YOLO model checkpoints (not tracked in git)
- `research/training/` — Dataset generation, augmentation, and categorization scripts
- `SASTRA/` — Indian construction site dataset (untracked)
- `data/runtime/` — Auto-created at startup for temp files and annotated video output

---

## Environment Variables (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `BUILDSIGHT_MODEL_V11` | `research/weights/yolov11_buildsight_best.pt` | YOLOv11 weights path |
| `BUILDSIGHT_MODEL_V26` | `research/weights/yolov26_buildsight_best.pt` | YOLOv26 weights path |
| `BUILDSIGHT_MODEL_DIR` | `research/weights/` | Weight directory override |
| `BUILDSIGHT_PORT` | `8000` | FastAPI port |
| `BUILDSIGHT_HOST` | `0.0.0.0` | FastAPI bind host |
| `BUILDSIGHT_LOG_LEVEL` | `info` | Uvicorn log level |
| `BUILDSIGHT_RUNTIME_DIR` | `data/runtime/` | Temp/output directory |

Eleven Labs (`ELEVENLABS_API_KEY`) and Gemini (`GOOGLE_API_KEY`) keys are also loaded from `.env` for Turner voice and GeoAI intelligence respectively.
