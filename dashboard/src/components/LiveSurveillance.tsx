/**
 * LiveSurveillance
 * ================
 * Live PPE detection component using the same canvas-overlay + IoU tracker
 * architecture as VideoUploadMode in DetectionPanel.tsx.
 *
 * Architecture:
 *   Backend opens camera (POST /api/stream/start)
 *   MJPEG stream  → <img ref={imgRef}>       (video display)
 *   WebSocket     → store.detections          (detection boxes)
 *   RAF drawLoop  → reads store directly      (no pendingRef, no React state in hot path)
 *   lastDetsRef   → persists last non-empty detections between backend frames
 *
 * Key decisions matching VideoUploadMode exactly:
 *   • canvas.width  = canvas.clientWidth   (CSS pixels, no DPR scaling)
 *   • canvas.height = canvas.clientHeight  (CSS pixels, no DPR scaling)
 *   • No canvas.style.width override       (lets CSS 100%×100% control layout)
 *   • _letterbox uses img.naturalWidth/naturalHeight for aspect ratio
 *   • fw/fh computed via inferScale (min(1, 640 / max(natW, natH)))
 *   • mergeTracks with IoU ≥ 0.25 (identical to DetectionPanel)
 *   • Boxes drawn via EMA lerp each RAF tick for smooth motion
 */

import {
  useState, useEffect, useRef, useCallback, useMemo,
} from 'react'
// useCameraDevices intentionally NOT imported — getUserMedia locks the physical
// camera handle on Windows for 10-60s, blocking the backend DSHOW open.
import { useDetectionStats } from '../DetectionStatsContext'
import { useDetectionStore } from '../store/detectionStore'
import './LiveSurveillance.css'

// ── Constants ─────────────────────────────────────────────────────────────────

const API             = 'http://localhost:8000'
const ALERT_DEDUPE_MS = 6_000
const MAX_ALERTS      = 15
const MAX_INFER_DIM   = 640    // matches backend ensemble pipeline
const LERP            = 0.45   // EMA factor — higher = snappier, lower = smoother

// ── Colour mapping (identical to DetectionPanel + old LiveMode) ───────────────

const CLASS_COLOR: Record<string, string> = {
  worker:        '#00c864',
  person:        '#00c864',
  helmet:        '#ffd600',
  hardhat:       '#ffd600',
  safety_vest:   '#00bfff',
  'safety-vest': '#00bfff',
  vest:          '#00bfff',
  machinery:     '#ff44ff',
  vehicle:       '#ff44ff',
}

function workerColor(h?: boolean | null, v?: boolean | null): string {
  if (h == null || v == null) return '#ffaa00'
  if (h && v)   return '#00c864'
  if (!h && !v) return '#ff2a2a'
  return '#ffaa00'
}

/** Compute letterbox offsets for object-fit:contain rendering (same as DetectionPanel) */
function _letterbox(vw: number, vh: number, dw: number, dh: number) {
  if (!vw || !vh) return { rw: dw, rh: dh, ox: 0, oy: 0 }
  const va = vw / vh, da = dw / dh
  return va > da
    ? { rw: dw, rh: dw / va, ox: 0, oy: (dh - dw / va) / 2 }
    : { rw: dh * va, rh: dh, ox: (dw - dh * va) / 2, oy: 0 }
}

function hms(ts: number) {
  const d = new Date(ts)
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map(n => String(n).padStart(2, '0')).join(':')
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface LiveAlert {
  id:       string
  time:     string
  severity: 'critical' | 'warning'
  title:    string
  ts:       number
}

// ══════════════════════════════════════════════════════════════════════════════
//  Component
// ══════════════════════════════════════════════════════════════════════════════

export function LiveSurveillance() {
  const { pushDetections, setRunning, setModelName } = useDetectionStats()
  const store = useDetectionStore()

  // ── Camera devices (from backend — never getUserMedia) ────────────────────
  interface BackendCamera { index: number; label: string; resolution: string; active: boolean }
  const [backendCameras,    setBackendCameras]    = useState<BackendCamera[]>([])
  const [selectedCamIdx,    setSelectedCamIdx]    = useState<number>(0)
  const [isFetchingCameras, setIsFetchingCameras] = useState(false)

  const fetchBackendCameras = useCallback(async () => {
    setIsFetchingCameras(true)
    try {
      const res  = await fetch(`${API}/api/stream/cameras`)
      const data = await res.json()
      const cams: BackendCamera[] = (data.cameras ?? []).sort(
        (a: BackendCamera, b: BackendCamera) => a.index - b.index
      )
      setBackendCameras(cams)
      if (cams.length > 0 && !cams.find(c => c.index === selectedCamIdx)) {
        setSelectedCamIdx(cams[0].index)
      }
    } catch { /* backend not up yet — silently ignore */ }
    finally { setIsFetchingCameras(false) }
  }, [selectedCamIdx])

  useEffect(() => { fetchBackendCameras() }, [])

  // ── Stream state ──────────────────────────────────────────────────────────
  const [isStreaming, setIsStreaming] = useState(false)
  const [isStarting,  setIsStarting]  = useState(false)
  const [error,       setError]       = useState<string | null>(null)

  // ── UI metrics ────────────────────────────────────────────────────────────
  const [fps,       setFps]      = useState(0)
  const [latencyMs, setLatency]  = useState(0)
  const [detCount,  setDetCount] = useState(0)
  const [condition, setCondition]= useState('S1_NORMAL')
  const [alerts,    setAlerts]   = useState<LiveAlert[]>([])

  const lastAlertKeyRef = useRef<Record<string, number>>({})

  // ── Refs ──────────────────────────────────────────────────────────────────
  const imgRef        = useRef<HTMLImageElement>(null)
  const overlayRef    = useRef<HTMLCanvasElement>(null)
  const rafRef        = useRef<number | null>(null)
  const isActiveRef   = useRef(false)
  // lastDetsRef — persists last non-empty detections across backend frames
  // (backend ~7-15 FPS, RAF at 60 FPS — without this, boxes flicker between updates)
  const lastDetsRef = useRef<any[]>([])

  // ── Alert generator ───────────────────────────────────────────────────────
  const emitAlert = useCallback((det: any) => {
    const now = Date.now()
    const newAlerts: LiveAlert[] = []
    const bx = det.box?.[0] ?? 0, by = det.box?.[1] ?? 0

    if (det.has_helmet === false) {
      const key = `nh-${Math.floor(bx / 40)}-${Math.floor(by / 40)}`
      if ((lastAlertKeyRef.current[key] ?? 0) + ALERT_DEDUPE_MS < now) {
        lastAlertKeyRef.current[key] = now
        newAlerts.push({ id: `${now}-${key}`, time: hms(now), severity: 'critical', title: 'Helmet Missing', ts: now })
      }
    }
    if (det.has_vest === false) {
      const key = `nv-${Math.floor(bx / 40)}-${Math.floor(by / 40)}`
      if ((lastAlertKeyRef.current[key] ?? 0) + ALERT_DEDUPE_MS < now) {
        lastAlertKeyRef.current[key] = now
        newAlerts.push({ id: `${now}-${key}`, time: hms(now), severity: 'warning', title: 'Safety Vest Missing', ts: now })
      }
    }
    if (newAlerts.length > 0) {
      setAlerts(prev => [...newAlerts, ...prev].slice(0, MAX_ALERTS))
    }
  }, [])

  // ── Store → UI metrics sync ───────────────────────────────────────────────
  // Only updates React state (sidebar numbers). Hot-path drawing uses refs.
  useEffect(() => {
    if (!isActiveRef.current) return
    const dets = store.detections
    if (!dets?.length) return

    setDetCount(dets.length)
    setLatency(store.latencyMs)
    setFps(store.fps)
    setCondition((store.sceneCondition ?? 'S1_normal').replace(/_/g, ' ').toUpperCase())
    pushDetections(dets, store.latencyMs, [], [], store.fps, store.sceneCondition)
    dets.forEach((d: any) => {
      if (d.class === 'worker' || d.class === 'person') emitAlert(d)
    })
  }, [store.detections, store.latencyMs, store.fps, store.sceneCondition,
      pushDetections, emitAlert])

  // ── RAF draw loop — matches DetectionPanel.drawOverlay exactly ──────────────
  //
  // Key design decisions (all matching DetectionPanel's proven working approach):
  //   • Canvas sized from img.clientWidth (same as video.clientWidth in DetectionPanel)
  //     Fallback to canvas.clientWidth → canvas.offsetWidth → 640
  //   • fw/fh from store.frameWidth/frameHeight (backend-authoritative inference size)
  //   • Boxes drawn DIRECTLY from store detections — no EMA, no tracking state
  //   • lastDetsRef caches last non-empty dets so boxes persist between backend frames
  //   • RAF scheduled at END of function (matches DetectionPanel line 900)
  const drawLoop = useCallback(() => {
    if (!isActiveRef.current) return

    const canvas = overlayRef.current
    const img    = imgRef.current

    if (!canvas || !img) {
      rafRef.current = requestAnimationFrame(drawLoop)
      return
    }

    // ── Size canvas to match rendered img area (matches DetectionPanel exactly) ──
    // img.clientWidth gives the rendered element size (correct for object-fit:contain).
    // Fall back to canvas.clientWidth if img is hidden, then hard-coded 640×360.
    const dw = img.clientWidth  || canvas.clientWidth  || canvas.offsetWidth  || 640
    const dh = img.clientHeight || canvas.clientHeight || canvas.offsetHeight || 360

    if (canvas.width !== dw || canvas.height !== dh) {
      canvas.width  = dw
      canvas.height = dh
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      rafRef.current = requestAnimationFrame(drawLoop)
      return
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // ── Read detection data from store ────────────────────────────────────────
    const st   = useDetectionStore.getState()
    const dets = st.detections

    // Persist last non-empty detections so boxes stay visible between backend frames
    // (backend ~7-15 FPS; RAF at 60 FPS — without this, boxes flicker)
    if (dets && dets.length > 0) lastDetsRef.current = dets
    const drawDets = lastDetsRef.current

    if (!drawDets.length) {
      rafRef.current = requestAnimationFrame(drawLoop)
      return
    }

    // ── Coordinate transform ──────────────────────────────────────────────────
    // Use backend-reported inference dimensions as the authoritative coordinate space.
    // These are sent in every detection_update as frame_width / frame_height.
    const fw = (st.frameWidth  > 0 ? st.frameWidth  : 0) || 640
    const fh = (st.frameHeight > 0 ? st.frameHeight : 0) || 480
    const { rw, rh, ox, oy } = _letterbox(fw, fh, dw, dh)

    // ── Draw each detection directly (no EMA — matches DetectionPanel) ────────
    for (const det of drawDets) {
      const box = det.box
      if (!box || box.length < 4) continue

      const x1 = (box[0] / fw) * rw + ox
      const y1 = (box[1] / fh) * rh + oy
      const x2 = (box[2] / fw) * rw + ox
      const y2 = (box[3] / fh) * rh + oy
      const bw = x2 - x1
      const bh = y2 - y1
      if (bw <= 2 || bh <= 2) continue  // skip sub-pixel or degenerate boxes

      const isWorker = det.class === 'worker' || det.class === 'person'
      const color = isWorker
        ? workerColor(det.has_helmet, det.has_vest)
        : (CLASS_COLOR[det.class ?? det.cls] ?? '#aaaaaa')

      ctx.globalAlpha = 1.0

      // ── Box ─────────────────────────────────────────────────────────────
      ctx.strokeStyle = color
      ctx.lineWidth   = isWorker ? 3 : 2.5
      ctx.strokeRect(x1, y1, bw, bh)

      // Corner accents
      const cs = Math.min(12, bw * 0.18, bh * 0.18)
      ctx.lineWidth = isWorker ? 3.5 : 2.5
      ;[
        [x1,      y1,      cs,  0,   0,  cs],
        [x1 + bw, y1,     -cs,  0,   0,  cs],
        [x1,      y1 + bh, cs,  0,   0, -cs],
        [x1 + bw, y1 + bh,-cs,  0,   0, -cs],
      ].forEach(([sx, sy, dx1, dy1, dx2, dy2]) => {
        ctx.beginPath()
        ctx.moveTo(sx + dx1, sy + dy1)
        ctx.lineTo(sx, sy)
        ctx.lineTo(sx + dx2, sy + dy2)
        ctx.stroke()
      })

      // ── Label ────────────────────────────────────────────────────────────
      const conf  = Math.round((det.confidence ?? 0) * 100)
      const label = isWorker
        ? `W ${conf}%${det.has_helmet === false ? ' ⚠H' : ''}${det.has_vest === false ? ' ⚠V' : ''}`
        : `${det.class ?? det.cls ?? '?'} ${conf}%`

      ctx.font = 'bold 13px monospace'
      const tw = ctx.measureText(label).width
      ctx.fillStyle   = color
      ctx.globalAlpha = 0.88
      ctx.fillRect(x1, Math.max(0, y1 - 20), tw + 10, 20)
      ctx.globalAlpha = 1.0
      ctx.fillStyle   = isWorker ? '#000' : '#fff'
      ctx.fillText(label, x1 + 5, Math.max(14, y1 - 4))

      // ── PPE H/V badges on workers ────────────────────────────────────────
      if (isWorker) {
        const bSize = Math.max(14, Math.min(20, bw * 0.14))
        const by2   = y1 + 4
        const bFont = `bold ${Math.max(9, bSize - 5)}px monospace`

        // Helmet badge
        let bx2 = x1 + bw - bSize - 4
        ctx.fillStyle = det.has_helmet === true  ? 'rgba(0,200,100,0.92)'
                      : det.has_helmet === false ? 'rgba(255,42,42,0.92)'
                      :                            'rgba(100,100,100,0.80)'
        ctx.fillRect(bx2, by2, bSize, bSize)
        ctx.fillStyle = '#fff'; ctx.font = bFont
        ctx.fillText('H', bx2 + 4, by2 + bSize - 4)

        // Vest badge
        bx2 -= bSize + 3
        ctx.fillStyle = det.has_vest === true  ? 'rgba(0,200,100,0.92)'
                      : det.has_vest === false ? 'rgba(255,42,42,0.92)'
                      :                          'rgba(100,100,100,0.80)'
        ctx.fillRect(bx2, by2, bSize, bSize)
        ctx.fillStyle = '#fff'
        ctx.fillText('V', bx2 + 4, by2 + bSize - 4)
      }

      ctx.globalAlpha = 1.0
    }

    // Schedule next frame at END (matches DetectionPanel — ensures complete draw cycle)
    rafRef.current = requestAnimationFrame(drawLoop)
  }, [])  // stable — reads store and refs directly, no stale closures

  // ── Start camera ──────────────────────────────────────────────────────────
  const startCamera = useCallback(async (camIdx?: number) => {
    const idx = camIdx ?? selectedCamIdx

    setError(null)
    setIsStarting(true)
    setAlerts([])
    lastAlertKeyRef.current = {}
    lastDetsRef.current = []

    try {
      const res = await fetch(`${API}/api/stream/start`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ rtsp_url: String(idx) }),
      })
      if (!res.ok) throw new Error(`Stream start failed: HTTP ${res.status}`)

      setModelName('BS-ENSEMBLE-WBF')
      setRunning(true)
      setIsStreaming(true)
      isActiveRef.current = true

      if (imgRef.current) {
        imgRef.current.src = `${API}/api/stream/live?t=${Date.now()}`
      }

      store.requestSnapshot()
      // Poll for initial state a few times — camera + detection warmup takes ~1-2s
      setTimeout(() => store.requestSnapshot(), 1000)
      setTimeout(() => store.requestSnapshot(), 2500)
      rafRef.current = requestAnimationFrame(drawLoop)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Camera stream unreachable.')
    } finally {
      setIsStarting(false)
    }
  }, [selectedCamIdx, drawLoop, setModelName, setRunning, store])

  // ── Stop camera ───────────────────────────────────────────────────────────
  const stopCamera = useCallback(async () => {
    isActiveRef.current = false
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }

    try {
      await fetch(`${API}/api/stream/stop`, { method: 'POST' })
    } catch (e) {
      console.warn('[LiveSurveillance] stream stop error:', e)
    }

    if (imgRef.current) imgRef.current.src = ''
    lastDetsRef.current = []

    const ctx = overlayRef.current?.getContext('2d')
    if (ctx && overlayRef.current) {
      ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height)
    }

    setIsStreaming(false)
    setRunning(false)
    setDetCount(0); setFps(0); setLatency(0)
  }, [setRunning])

  // ── Switch camera ─────────────────────────────────────────────────────────
  const switchCamera = useCallback(async (camIdx: number) => {
    setSelectedCamIdx(camIdx)
    if (isStreaming) {
      await stopCamera()
      // 1500ms: Windows DSHOW needs time to fully release the handle
      await new Promise<void>(r => setTimeout(r, 1500))
      await startCamera(camIdx)
    }
  }, [isStreaming, stopCamera, startCamera])

  // ── Cleanup on unmount ────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      isActiveRef.current = false
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      fetch(`${API}/api/stream/stop`, { method: 'POST' }).catch(() => {})
      setRunning(false)
    }
  }, [setRunning])

  // ── Derived sidebar values ────────────────────────────────────────────────
  const classCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const d of (store.detections ?? [])) {
      const cls = d.class ?? d.cls ?? 'unknown'
      counts[cls] = (counts[cls] ?? 0) + 1
    }
    return Object.entries(counts)
  }, [store.detections])

  const displayName = (backendCameras.find(c => c.index === selectedCamIdx)?.label
    ?? backendCameras[0]?.label ?? 'Camera').replace(/\(.*?\)/g, '').trim()

  // ── JSX ───────────────────────────────────────────────────────────────────
  return (
    <div className="lsv">

      {/* ── Status bar ──────────────────────────────────────────────────── */}
      <div className="lsv__status-bar">
        <div className="lsv__status-left">
          <span className={`lsv__dot ${isStreaming ? 'lsv__dot--live' : ''}`} />
          <span className="lsv__status-text">{isStreaming ? 'LIVE' : 'OFFLINE'}</span>
          {isStreaming && (<>
            <span className="lsv__sep" />
            <span className="lsv__fps">{fps} FPS</span>
            <span className="lsv__sep" />
            <span className="lsv__latency">{latencyMs}ms</span>
            <span className="lsv__sep" />
            <span className="lsv__dets">{detCount} detections</span>
            <span className="lsv__sep" />
            <span className="lsv__condition">{condition}</span>
          </>)}
        </div>
        <div className="lsv__status-right">
          <span className="lsv__model">BS-ENSEMBLE-WBF</span>
        </div>
      </div>

      {/* ── Main layout ─────────────────────────────────────────────────── */}
      <div className="lsv__layout">

        {/* ── Video column ─────────────────────────────────────────────── */}
        <div className="lsv__video-col">
          <div className="lsv__player-wrap">

            <img
              ref={imgRef}
              className="lsv__player"
              style={{ display: isStreaming ? 'block' : 'none' }}
              alt="Live Camera Feed"
              crossOrigin="anonymous"
            />
            <canvas ref={overlayRef} className="lsv__overlay" />

            {!isStreaming && !isStarting && (
              <div className="lsv__placeholder">
                <div className="lsv__placeholder-content">
                  <svg viewBox="0 0 48 48" width="48" height="48" fill="none"
                    stroke="currentColor" strokeWidth="1.5" opacity="0.5">
                    <rect x="4" y="10" width="40" height="28" rx="3" />
                    <circle cx="24" cy="24" r="8" />
                    <path d="M32 10l4-6M16 10l-4-6M24 4v6M4 18h40" />
                  </svg>
                  <span className="lsv__placeholder-title">Camera Offline</span>
                  <span className="lsv__placeholder-sub">
                    {backendCameras.length === 0
                      ? 'No cameras detected — connect a device and click Refresh'
                      : 'Select a camera and press START'}
                  </span>
                </div>
              </div>
            )}

            {isStarting && (
              <div className="lsv__placeholder">
                <div className="lsv__placeholder-content">
                  <div className="lsv__spinner" />
                  <span className="lsv__placeholder-sub">Initialising camera…</span>
                </div>
              </div>
            )}

            {error && (
              <div className="lsv__error-overlay">
                <span className="lsv__error-icon">⚠</span>
                <span className="lsv__error-msg">{error}</span>
              </div>
            )}
          </div>

          {/* ── Controls bar ─────────────────────────────────────────── */}
          <div className="lsv__controls">
            <div className="lsv__controls-left">
              <button
                className={`lsv__btn ${isStreaming ? 'lsv__btn--stop' : 'lsv__btn--start'}`}
                onClick={isStreaming ? stopCamera : () => startCamera()}
                disabled={isStarting}
              >
                {isStarting ? 'STARTING…' : isStreaming ? 'STOP' : 'START'}
              </button>

              <div className="lsv__camera-select">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none"
                  stroke="currentColor" strokeWidth="1.5" className="lsv__camera-icon">
                  <rect x="1" y="4" width="14" height="10" rx="2" />
                  <circle cx="8" cy="9" r="3" />
                  <path d="M11 4l2-3M5 4L3 1" />
                </svg>
                <select
                  className="lsv__select"
                  value={selectedCamIdx}
                  onChange={e => switchCamera(Number(e.target.value))}
                  disabled={backendCameras.length === 0 || isStreaming}
                >
                  {backendCameras.length === 0 && <option value="">No cameras found</option>}
                  {backendCameras.map(cam => (
                    <option key={cam.index} value={cam.index}>{cam.label}</option>
                  ))}
                </select>
              </div>

              <button
                className="lsv__btn lsv__btn--refresh"
                onClick={fetchBackendCameras}
                disabled={isFetchingCameras}
                title="Refresh available cameras"
              >
                <svg viewBox="0 0 16 16" width="13" height="13" fill="none"
                  stroke="currentColor" strokeWidth="1.5">
                  <path d="M2 8a6 6 0 0111.5-3M14 8a6 6 0 01-11.5 3" />
                  <path d="M14 2v3h-3M2 14v-3h3" />
                </svg>
                {isFetchingCameras ? 'Scanning…' : 'REFRESH'}
              </button>
            </div>

            <div className="lsv__controls-right">
              <div className="lsv__connection-status">
                <span className={`lsv__conn-dot lsv__conn-dot--${
                  isStreaming ? 'connected' : isStarting ? 'connecting' : error ? 'error' : 'idle'
                }`} />
                <span className="lsv__conn-label">
                  {isStreaming ? 'STREAM OK' : isStarting ? 'CONNECTING' : error ? 'STREAM ERROR' : 'IDLE'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* ── Sidebar ──────────────────────────────────────────────────── */}
        <div className="lsv__sidebar">

          <div className="lsv__sidebar-section">
            <div className="lsv__sidebar-heading">
              <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="1" y="4" width="14" height="10" rx="2" />
                <circle cx="8" cy="9" r="3" />
                <path d="M11 4l2-3M5 4L3 1" />
              </svg>
              <span>Camera Feed</span>
            </div>
            <div className="lsv__camera-info">
              <div className="lsv__info-row">
                <span className="lsv__info-label">Source</span>
                <span className="lsv__info-value" title={displayName}>{displayName}</span>
              </div>
              <div className="lsv__info-row">
                <span className="lsv__info-label">Status</span>
                <span className={`lsv__info-value ${isStreaming ? 'lsv__info-value--ok' : 'lsv__info-value--muted'}`}>
                  {isStreaming ? 'Active' : 'Standby'}
                </span>
              </div>
              <div className="lsv__info-row">
                <span className="lsv__info-label">Resolution</span>
                <span className="lsv__info-value">
                  {isStreaming && store.frameWidth ? `${store.frameWidth}×${store.frameHeight}` : '—'}
                </span>
              </div>
              <div className="lsv__info-row">
                <span className="lsv__info-label">Devices</span>
                <span className="lsv__info-value">{backendCameras.length} found</span>
              </div>
            </div>
          </div>

          <div className="lsv__sidebar-section">
            <div className="lsv__sidebar-heading">
              <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M1 11l4-4 4 4 5-7" />
              </svg>
              <span>Performance</span>
            </div>
            <div className="lsv__perf-grid">
              <div className="lsv__perf-item">
                <span className="lsv__perf-label">FPS</span>
                <span className="lsv__perf-value">{fps}</span>
              </div>
              <div className="lsv__perf-item">
                <span className="lsv__perf-label">Latency</span>
                <span className="lsv__perf-value">{latencyMs}<span className="lsv__perf-unit">ms</span></span>
              </div>
              <div className="lsv__perf-item">
                <span className="lsv__perf-label">Detections</span>
                <span className="lsv__perf-value">{detCount}</span>
              </div>
              <div className="lsv__perf-item">
                <span className="lsv__perf-label">Model</span>
                <span className="lsv__perf-value lsv__perf-value--sm">ENSEMBLE</span>
              </div>
            </div>
          </div>

          <div className="lsv__sidebar-section">
            <div className="lsv__sidebar-heading">
              <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M8 1l2 4h4l-3 3 1 4-4-2-4 2 1-4L2 5h4z" />
              </svg>
              <span>PPE Legend</span>
            </div>
            <div className="lsv__class-list">
              {[
                { color: '#00c864', label: 'Compliant',   sub: 'Helmet + Vest' },
                { color: '#ffaa00', label: 'Partial PPE', sub: 'Missing 1 item' },
                { color: '#ff2a2a', label: 'No PPE',      sub: 'No helmet/vest' },
                { color: '#ffd600', label: 'Helmet',      sub: 'Detected alone' },
                { color: '#00bfff', label: 'Safety Vest', sub: 'Detected alone' },
              ].map(row => (
                <div key={row.label} className="lsv__class-item">
                  <span className="lsv__class-swatch" style={{ background: row.color }} />
                  <span className="lsv__class-name">{row.label}</span>
                  <span className="lsv__class-sub">{row.sub}</span>
                </div>
              ))}
            </div>
          </div>

          {isStreaming && classCounts.length > 0 && (
            <div className="lsv__sidebar-section">
              <div className="lsv__sidebar-heading">
                <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <circle cx="5" cy="5" r="3" /><circle cx="11" cy="5" r="3" />
                  <circle cx="5" cy="11" r="3" /><circle cx="11" cy="11" r="3" />
                </svg>
                <span>Objects ({detCount})</span>
              </div>
              <div className="lsv__class-list">
                {classCounts.map(([cls, count]) => (
                  <div key={cls} className="lsv__class-item">
                    <span className="lsv__class-swatch" style={{ background: CLASS_COLOR[cls] ?? '#aaa' }} />
                    <span className="lsv__class-name">{cls.replace(/_/g, ' ').toUpperCase()}</span>
                    <span className="lsv__class-count">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {alerts.length > 0 && (
            <div className="lsv__sidebar-section lsv__sidebar-section--alerts">
              <div className="lsv__sidebar-heading lsv__sidebar-heading--alert">
                <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M8 1L1 14h14L8 1zM8 5v4M8 11v1" />
                </svg>
                <span>Alerts ({alerts.length})</span>
              </div>
              <div className="lsv__alert-list">
                {alerts.slice(0, 8).map(alert => (
                  <div key={alert.id} className={`lsv__alert lsv__alert--${alert.severity}`}>
                    <div className="lsv__alert-header">
                      <span className="lsv__alert-severity">
                        {alert.severity === 'critical' ? 'CRITICAL' : 'WARNING'}
                      </span>
                      <span className="lsv__alert-time">{alert.time}</span>
                    </div>
                    <span className="lsv__alert-title">{alert.title}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
