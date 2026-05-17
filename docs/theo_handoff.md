# Handoff.md — Task Execution Protocol

## Purpose
This file serves as the communication channel between **Jovi (Planner)** and the **Execution Team** (Toni + Leon + Theo).
Jovi defines goals. Toni leads execution. Leon is secondary executor. Theo handles optimization last.

---

## Agent Hierarchy

```
Jovi (Planner)
 └── Toni / Claude    → Main Executor (1st)         — leads all task execution
 └── Leon             → Secondary Executor (2nd)     — takes over when Toni is unavailable
 └── Theo / OpenCode  → Tertiary Executor (3rd)      — optimization, refactoring, validation
```

---

## Team Status

| Agent | Role | Priority | Status |
|-------|------|----------|--------|
| **Jovi** | Planner — defines goals, priorities, and architecture | — | ✅ Active |
| **Toni (Claude)** | Main executor — leads all task execution | 1st | ✅ Active |
| **Leon** | Secondary executor — takes over when Toni is unavailable | 2nd | ✅ Active |
| **Theo (OpenCode)** | Tertiary executor — optimization, refactoring, validation | 3rd | ✅ Active |

---

## Role Definitions

| Role | Responsibility |
|------|----------------|
| **Jovi** | Planner — defines goals, requirements, priorities. Does NOT write code. |
| **Toni (Claude)** | Main executor — owns the task pipeline, makes execution decisions, coordinates the team. |
| **Leon** | Secondary executor — steps in when Toni is unavailable. Handles implementation and coding tasks. |
| **Theo (OpenCode)** | Tertiary executor — activated last. Handles optimization, profiling, refactoring, and code quality validation. Has his own `theo_handoff.md` task file. |

---

## Execution Flow

```
Jovi defines task
   ↓
Toni receives & leads execution  (1st priority)
   ↓  [if Toni unavailable]
Leon steps in as secondary executor  (2nd priority)
   ↓  [after Toni/Leon complete]
Theo optimizes, validates & refactors  (3rd — last pass)
   ↓
Toni confirms completion → reports back to Jovi
```

> **If Toni is unavailable:** Leon steps up as the primary coding executor.
> **If Leon is also unavailable:** Theo handles full execution from his own `theo_handoff.md`.

---

## Task Format

```markdown
## Incoming Tasks

### Task #N
- **Priority**: [high/medium/low]
- **Objective**: [clear goal statement]
- **Assigned To**: [Toni / Theo / Leon]
- **Requirements**:
  - [specific requirement 1]
  - [specific requirement 2]
- **Constraints**: [any limitations]
- **Expected Output**: [what success looks like]
- **Notes**: [any context or references]
```

---

## Status Tracking

| Status | Meaning |
|--------|---------| 
| `pending` | Task received, not yet started |
| `in_progress` | Being executed by assigned agent |
| `optimizing` | Theo is refining or validating |
| `review` | Awaiting Toni's final check |
| `completed` | Task finished and validated |
| `blocked` | Needs clarification or resource |

---

## Execution Rules

1. **Jovi defines, Toni leads** — Toni is the main executor and owns the task pipeline
2. **No task starts until clearly defined** — if unclear, Toni or Theo asks Jovi for clarification
3. **Theo optimizes before final delivery** — review all output for quality, correctness, and performance
4. **Leon executes with autonomy** — but flags edge cases or risks to Toni/Theo immediately
5. **Leon is the secondary fallback** — if Toni is unavailable, Leon takes over
6. **Theo is the tertiary fallback** — Theo activates only when both Toni and Leon are unavailable, or for optimization passes
7. **Theo's tasks are tracked in `theo_handoff.md`** — all optimization tasks go there
8. **Blockers get escalated immediately** — don't wait to surface issues to Jovi

---

## Current Active Tasks

*None yet*

---

## Incoming Tasks

*Waiting for Jovi to add tasks*

---

## Completed Tasks

### ✅ BuildSight v1.8 Beta — Stabilized Bounding Boxes & Florence-2 VLM
- **Date**: 2026-04-23
- **Executed By**: Toni (Claude)
- **Optimized By**: Theo (OpenCode)
- **Summary**: Replaced Moondream2 with Florence-2-base, fixed VLM inference pipeline, stabilized bounding box rendering in DetectionPanel, hardened fallback logic. Pushed to GitHub as `v1.8-beta`.

### ✅ Live PPE Detection — Multi-Camera Real-Time Surveillance (LiveSurveillance)
- **Date**: 2026-05-16
- **Executed By**: Toni (Claude)
- **Optimized By**: TBD
- **Summary**: Built and wired real-time live PPE detection into the LIVE Surveillance Workspace tab. See details below.

#### New Files
| File | Description |
|------|-------------|
| `dashboard/src/components/LiveSurveillance.tsx` | Main live surveillance component (~380 lines) |
| `dashboard/src/components/LiveSurveillance.css` | CCTV-style industrial HUD styling (~330 lines) |

#### Modified Files
| File | Change |
|------|--------|
| `dashboard/src/App.tsx` | Replaced `LiveFeed` import + render with `LiveSurveillance` |

#### Features Delivered
1. **Multi-Camera Support**: Auto-enumerates all video devices (`enumerateDevices`), hot-plug detection via `devicechange` listener, dropdown selector with instant switching, refresh button
2. **Real-Time PPE Detection**: Reuses `POST /api/detect/frame` endpoint — same ensemble pipeline (YOLOv11+YOLOv26 WBF fusion, condition-aware preprocessing, class mappings, thresholds) as Video Detection Workspace
3. **Canvas Overlay Rendering**: IoU-tracker smoothed bounding boxes with compliance color coding (green=compliant, amber=partial, red=violation), confidence labels, letterbox-aware transforms
4. **Performance Monitoring**: FPS counter, latency display, detection count, live resolution info
5. **Violation Alerts**: Real-time deduplicated alerts (ALERT_DEDUPE_MS=5000) for helmet/vest violations with severity badges, timestamps, scrollable alert list (max 20), slide-in animation
6. **Audio Alert**: Web Audio API beep toggle for critical violations
7. **Heatmap Overlay**: Radial gradient heatmap from worker centroids (7s TTL)
8. **Connection Status**: IDLE → CONNECTING → CONNECTED → ERROR with colored dot indicator
9. **Error Handling**: Permission denied, no camera found, device busy, stream errors with user-friendly messages and auto-recovery
10. **CCTV-Style UI**: Dark theme, status bar, performance grid, object class breakdown, responsive layout

#### Known Limitations / Potential Issues for Jovi
- **Backend dependency**: Requires FastAPI at `localhost:8000` to be running — no offline fallback
- **No RTSP/CCTV IP camera support** — browser `getUserMedia` only (native USB + integrated). RTSP needs backend relay or Electron native addon
- **Alert dedup is spatial-bucket based** — uses `Math.floor(box[0] / 20)` as key; works but not perfect for slow-moving workers
- **`setState` on Zustand store** — uses `store.setState(...)` in inference loop which is non-ideal but matches existing pattern in `LiveMode`; could be refactored to use named store actions
- **Inference gap tuning** — `inferenceGapRef` adaptively sets min gap to `elapsed * 1.5` clamped to [50, 200]ms; may need per-device tuning
- **No Electron native integration** — only browser `getUserMedia`; no `nativeImage` / desktop capture
