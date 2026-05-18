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
 *   • _letterbox uses st.frameWidth/frameHeight (backend inference dims — NOT img.naturalWidth)
 *   • fw/fh from same backend dims so letterbox and coordinate scaling are always in sync
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

const API = 'http://localhost:8000'
const ALERT_DEDUPE_MS = 6_000
const MAX_ALERTS = 15
const LERP = 0.60   // EMA factor for track position smoothing (matches DetectionPanel.LiveMode)

// ── Colour mapping (identical to DetectionPanel + old LiveMode) ───────────────

const CLASS_COLOR: Record<string, string> = {
  worker: '#00c864',
  person: '#00c864',
  helmet: '#ffd600',
  hardhat: '#ffd600',
  safety_vest: '#00bfff',
  'safety-vest': '#00bfff',
  vest: '#00bfff',
  machinery: '#ff44ff',
  vehicle: '#ff44ff',
}

function workerColor(h?: boolean | null, v?: boolean | null): string {
  if (h == null || v == null) return '#ff2a2a'
  if (h && v) return '#00c864'
  return '#ff2a2a'
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

// ── IoU tracker (identical to DetectionPanel.LiveMode) ────────────────────────

let _nextTrackId = 0

interface Track {
  id: number
  cls: string
  confidence: number
  frameBox: [number, number, number, number]
  sx1: number; sy1: number; sx2: number; sy2: number
  initialized: boolean
  missed: number
  has_helmet?: boolean
  has_vest?: boolean
}

function _boxIou(a: number[], b: number[]): number {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1])
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3])
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  if (inter === 0) return 0
  const ua = (a[2] - a[0]) * (a[3] - a[1])
  const ub = (b[2] - b[0]) * (b[3] - b[1])
  return inter / Math.max(ua + ub - inter, 1e-6)
}

function mergeTracks(tracks: Track[], newDets: any[]): Track[] {
  const matched = new Set<number>()
  for (const t of tracks) {
    let bestIou = 0.25, bestIdx = -1
    for (let i = 0; i < newDets.length; i++) {
      const d = newDets[i]
      if ((d.class ?? d.cls) !== t.cls || !d.box) continue
      const iou = _boxIou(t.frameBox as number[], d.box as number[])
      if (iou > bestIou) { bestIou = iou; bestIdx = i }
    }
    if (bestIdx >= 0) {
      const nd = newDets[bestIdx]
      t.frameBox = nd.box
      t.confidence = nd.confidence
      t.has_helmet = nd.has_helmet
      t.has_vest = nd.has_vest
      t.missed = 0
      matched.add(bestIdx)
    } else {
      t.missed++
    }
  }
  const alive = tracks.filter(t => t.missed <= 2)
  newDets.forEach((d, i) => {
    if (matched.has(i) || !d.box) return
    alive.push({
      id: _nextTrackId++,
      cls: d.class ?? d.cls ?? 'unknown',
      confidence: d.confidence,
      frameBox: d.box,
      sx1: 0, sy1: 0, sx2: 0, sy2: 0,
      initialized: false, missed: 0,
      has_helmet: d.has_helmet,
      has_vest: d.has_vest,
    })
  })
  return alive
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface LiveAlert {
  id: string
  time: string
  severity: 'critical' | 'warning'
  title: string
  ts: number
}

// ══════════════════════════════════════════════════════════════════════════════
//  Component
// ══════════════════════════════════════════════════════════════════════════════

export function LiveSurveillance() {
  const { pushDetections, setRunning, setModelName } = useDetectionStats()
  const store = useDetectionStore()

  // ── Camera devices (from backend — never getUserMedia) ────────────────────
  interface BackendCamera { index: number; label: string; resolution: string; active: boolean }
  const [backendCameras, setBackendCameras] = useState<BackendCamera[]>([])
  const [selectedCamIdx, setSelectedCamIdx] = useState<number>(0)
  const [isFetchingCameras, setIsFetchingCameras] = useState(false)

  const fetchBackendCameras = useCallback(async () => {
    setIsFetchingCameras(true)
    try {
      const res = await fetch(`${API}/api/stream/cameras`)
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
  const [isStarting, setIsStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // ── UI metrics ────────────────────────────────────────────────────────────
  const [fps, setFps] = useState(0)
  const [latencyMs, setLatency] = useState(0)
  const [detCount, setDetCount] = useState(0)
  const [condition, setCondition] = useState('S1_NORMAL')
  const [alerts, setAlerts] = useState<LiveAlert[]>([])

  const lastAlertKeyRef = useRef<Record<string, number>>({})

  // ── Refs ──────────────────────────────────────────────────────────────────
  const imgRef = useRef<HTMLImageElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number | null>(null)
  const isActiveRef = useRef(false)
  // Track-based persistence (same architecture as DetectionPanel.LiveMode)
  const tracksRef = useRef<Track[]>([])
  const pendingRef = useRef<any[] | null>(null)
  const frameWRef = useRef(640)
  const frameHRef = useRef(480)

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
  // Effect 1: FPS / latency / condition — updates even with 0 detections
  useEffect(() => {
    if (!isActiveRef.current) return
    if (store.fps > 0 || store.latencyMs > 0) {
      setFps(store.fps)
      setLatency(store.latencyMs)
    }
    if (store.sceneCondition) {
      setCondition(store.sceneCondition.replace(/_/g, ' ').toUpperCase())
    }
  }, [store.fps, store.latencyMs, store.sceneCondition])

  // Effect 2: detections — feeds pendingRef (picked up by drawLoop) + sidebar state
  useEffect(() => {
    if (!isActiveRef.current) return
    const dets = store.detections
    if (!dets?.length) return

    pendingRef.current = dets
    setDetCount(dets.length)
    pushDetections(dets, store.latencyMs, [], [], store.fps, store.sceneCondition)
    dets.forEach((d: any) => {
      if (d.class === 'worker' || d.class === 'person') emitAlert(d)
    })
  }, [store.detections, store.latencyMs, store.fps, store.sceneCondition,
    pushDetections, emitAlert])

  // ── RAF draw loop — identical architecture to DetectionPanel.LiveMode ─────────
  //
  //   pendingRef  ← useEffect feeds new detections from WebSocket
  //   tracksRef   ← IoU tracker (mergeTracks) for persistent smooth boxes
  //   LERP = 0.60 ← each RAF tick lerps displayed box toward latest track position
  //
  // This is the proven working approach from DetectionPanel.LiveMode.
  const drawLoop = useCallback(() => {
    if (!isActiveRef.current) return

    const canvas = overlayRef.current
    const img = imgRef.current

    if (!canvas || !img) {
      rafRef.current = requestAnimationFrame(drawLoop)
      return
    }

    // ── Size canvas to match rendered img area ────────────────────────────────
    const dw = img.clientWidth || canvas.clientWidth || canvas.offsetWidth || 640
    const dh = img.clientHeight || canvas.clientHeight || canvas.offsetHeight || 360

    if (canvas.width !== dw || canvas.height !== dh) {
      canvas.width = dw
      canvas.height = dh
      // Reset track display positions on resize — they'll re-initialize instantly
      tracksRef.current.forEach(t => { t.initialized = false })
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      rafRef.current = requestAnimationFrame(drawLoop)
      return
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // ── Coordinate transform ──────────────────────────────────────────────────
    // Use backend-authoritative inference dimensions for BOTH letterbox and
    // coordinate scaling. img.naturalWidth is the raw MJPEG stream resolution
    // which may differ from inference resolution (e.g. 1280×720 stream but
    // 640×360 inference). Using them separately causes boxes 2× too large.
    const st = useDetectionStore.getState()
    const fw = frameWRef.current = st.frameWidth || 640
    const fh = frameHRef.current = st.frameHeight || 480
    const { rw, rh, ox, oy } = _letterbox(fw, fh, dw, dh)

    // ── Consume pending detections — run IoU tracker ──────────────────────────
    if (pendingRef.current !== null) {
      tracksRef.current = mergeTracks(tracksRef.current, pendingRef.current)
      pendingRef.current = null
    }

    // ── Draw each live track with LERP smoothing ──────────────────────────────
    for (const t of tracksRef.current) {
      const [x1, y1, x2, y2] = t.frameBox
      const tx1 = (x1 / fw) * rw + ox, ty1 = (y1 / fh) * rh + oy
      const tx2 = (x2 / fw) * rw + ox, ty2 = (y2 / fh) * rh + oy

      // Snap on first frame, lerp thereafter
      if (!t.initialized) {
        t.sx1 = tx1; t.sy1 = ty1; t.sx2 = tx2; t.sy2 = ty2
        t.initialized = true
      } else {
        t.sx1 += (tx1 - t.sx1) * LERP
        t.sy1 += (ty1 - t.sy1) * LERP
        t.sx2 += (tx2 - t.sx2) * LERP
        t.sy2 += (ty2 - t.sy2) * LERP
      }

      const bw = t.sx2 - t.sx1
      const bh = t.sy2 - t.sy1
      if (bw < 2 || bh < 2) continue

      // Fade missed tracks (up to 2 cycles before drop)
      ctx.globalAlpha = t.missed > 0 ? 0.45 : 1.0

      const isWorker = t.cls === 'worker' || t.cls === 'person'
      const color = isWorker
        ? workerColor(t.has_helmet, t.has_vest)
        : (CLASS_COLOR[t.cls] ?? '#aaaaaa')

      // ── Box ──────────────────────────────────────────────────────────────
      ctx.strokeStyle = color
      ctx.lineWidth = isWorker ? 3 : 2.5
      ctx.strokeRect(t.sx1, t.sy1, bw, bh)

      // Corner accents
      const cs = Math.min(12, bw * 0.18, bh * 0.18)
      ctx.lineWidth = isWorker ? 3.5 : 2.5
        ;[
          [t.sx1, t.sy1, cs, 0, 0, cs],
          [t.sx1 + bw, t.sy1, -cs, 0, 0, cs],
          [t.sx1, t.sy1 + bh, cs, 0, 0, -cs],
          [t.sx1 + bw, t.sy1 + bh, -cs, 0, 0, -cs],
        ].forEach(([sx, sy, dx1, dy1, dx2, dy2]) => {
          ctx.beginPath()
          ctx.moveTo(sx + dx1, sy + dy1)
          ctx.lineTo(sx, sy)
          ctx.lineTo(sx + dx2, sy + dy2)
          ctx.stroke()
        })

      // ── Label ─────────────────────────────────────────────────────────────
      const conf = Math.round((t.confidence ?? 0) * 100)
      const label = isWorker
        ? `W ${conf}%${t.has_helmet === false ? ' ⚠H' : ''}${t.has_vest === false ? ' ⚠V' : ''}`
        : `${t.cls} ${conf}%`

      ctx.font = 'bold 13px monospace'
      const tw = ctx.measureText(label).width
      ctx.fillStyle = color
      ctx.globalAlpha = t.missed > 0 ? 0.40 : 0.88
      ctx.fillRect(t.sx1, Math.max(0, t.sy1 - 20), tw + 10, 20)
      ctx.globalAlpha = t.missed > 0 ? 0.45 : 1.0
      ctx.fillStyle = isWorker ? '#000' : '#fff'
      ctx.fillText(label, t.sx1 + 5, Math.max(14, t.sy1 - 4))

      // ── PPE H/V badges on workers ─────────────────────────────────────────
      if (isWorker) {
        const bSize = Math.max(14, Math.min(20, bw * 0.14))
        const by2 = t.sy1 + 4
        const bFont = `bold ${Math.max(9, bSize - 5)}px monospace`

        let bx2 = t.sx1 + bw - bSize - 4
        ctx.globalAlpha = t.missed > 0 ? 0.45 : 0.92
        ctx.fillStyle = t.has_helmet === true ? 'rgba(0,200,100,0.92)'
          : t.has_helmet === false ? 'rgba(255,42,42,0.92)'
            : 'rgba(100,100,100,0.80)'
        ctx.fillRect(bx2, by2, bSize, bSize)
        ctx.fillStyle = '#fff'; ctx.font = bFont
        ctx.fillText('H', bx2 + 4, by2 + bSize - 4)

        bx2 -= bSize + 3
        ctx.fillStyle = t.has_vest === true ? 'rgba(0,200,100,0.92)'
          : t.has_vest === false ? 'rgba(255,42,42,0.92)'
            : 'rgba(100,100,100,0.80)'
        ctx.fillRect(bx2, by2, bSize, bSize)
        ctx.fillStyle = '#fff'
        ctx.fillText('V', bx2 + 4, by2 + bSize - 4)
      }

      ctx.globalAlpha = 1.0
    }

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
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rtsp_url: String(idx) }),
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
      fetch(`${API}/api/stream/stop`, { method: 'POST' }).catch(() => { })
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
                <span className={`lsv__conn-dot lsv__conn-dot--${isStreaming ? 'connected' : isStarting ? 'connecting' : error ? 'error' : 'idle'
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
                { color: '#00c864', label: 'Compliant', sub: 'Helmet + Vest' },
                { color: '#ff2a2a', label: 'Partial PPE', sub: 'Missing 1 item' },
                { color: '#ff2a2a', label: 'No PPE', sub: 'No helmet/vest' },
                { color: '#ffd600', label: 'Helmet', sub: 'Detected alone' },
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
