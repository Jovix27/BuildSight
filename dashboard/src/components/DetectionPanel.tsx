import {
  useRef, useState, useEffect, useCallback, useMemo,
} from 'react'
import { useDetectionPipeline, useDetectionStats } from '../DetectionStatsContext'
import { useDetectionStore } from '../store/detectionStore'
import { useSettings } from '../SettingsContext'
import PPEStatusPanel from './PPEStatusPanel'
import {
  buildPeakRiskMoment,
  detectionsToHeatmapPoints,
  mergePeakRiskMoments,
} from '../lib/detectionIntelligence'
import './DetectionPanel.css'

const API = 'http://localhost:8000/api'

const CONDITIONS = ['S1_normal', 'S2_dusty', 'S3_low_light', 'S4_crowded'] as const
type Condition = typeof CONDITIONS[number]

interface Detection {
  class: string
  confidence: number
  box: [number, number, number, number]
  /** Only present on worker/person detections — set by backend association */
  has_helmet?: boolean
  has_vest?: boolean
}

interface DetectionHeatmapPoint {
  x: number
  y: number
  value: number
  type: 'worker' | 'violation'
  risk_level?: RiskLevel
}

type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'

interface DetectionRiskZone {
  box: [number, number, number, number]
  center: [number, number]
  risk_score: number
  risk_level: RiskLevel
  label: string
  worker_count: number
  ppe_violations: number
}

interface DetectionHeatmapPayload {
  frame_width?: number
  frame_height?: number
  points: DetectionHeatmapPoint[]
  zones: DetectionRiskZone[]
  summary: {
    worker_count: number
    risk_zone_count: number
    critical_zone_count: number
    max_risk_score: number
  }
}

interface DetectResult {
  detections: Detection[]
  class_counts: Record<string, number>
  total: number
  image_b64: string
  mode: string
  condition: Condition
  elapsed_ms: number
  heatmap?: DetectionHeatmapPayload
}

const CLASS_COLORS: Record<string, string> = {
  helmet: '#ffd600',
  hardhat: '#ffd600',
  safety_vest: '#00bfff',
  'safety-vest': '#00bfff',
  vest: '#00bfff',
  worker: '#00c864',
  person: '#00c864',
}

/** 
 * Helper to resolve CSS variables like 'var(--color-accent)' to hex/rgb 
 * since Canvas fillStyle doesn't support them natively.
 */
function resolveCssVar(varName: string): string {
  if (!varName.startsWith('var(')) return varName
  const name = varName.match(/\((--[^)]+)\)/)?.[1]
  if (!name) return '#ff9900' // Fail-safe fallback
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#ff9900'
}

// ── Lightweight IoU tracker (module-level, no React state) ─────────────────────

let _nextTrackId = 0

interface Track {
  id: number
  cls: string
  confidence: number
  /** Bounding box in the sent-frame's pixel space (e.g. 640×360) */
  frameBox: [number, number, number, number]
  /** Smoothed display coords on the overlay canvas (lerped each RAF tick) */
  sx1: number; sy1: number; sx2: number; sy2: number
  initialized: boolean
  /** How many consecutive inference cycles had no match */
  missed: number
  /** PPE association flags — only set on worker/person tracks */
  has_helmet?: boolean
  has_vest?: boolean
}

interface HeatmapPoint {
  x: number
  y: number
  time: number
  type: 'worker' | 'violation'
  value: number
  riskLevel?: RiskLevel
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

function mergeTracks(tracks: Track[], newDets: Detection[]): Track[] {
  const matched = new Set<number>()

  for (const t of tracks) {
    let bestIou = 0.25, bestIdx = -1
    for (let i = 0; i < newDets.length; i++) {
      if (newDets[i].class !== t.cls || !newDets[i].box) continue
      const iou = _boxIou(t.frameBox as number[], newDets[i].box as number[])
      if (iou > bestIou) { bestIou = iou; bestIdx = i }
    }
    if (bestIdx >= 0) {
      const nd = newDets[bestIdx]
      t.frameBox = nd.box
      t.confidence = nd.confidence
      t.missed = 0
      // Carry forward PPE association flags from the latest detection
      t.has_helmet = nd.has_helmet
      t.has_vest = nd.has_vest
      matched.add(bestIdx)
    } else {
      t.missed++
    }
  }

  const alive = tracks.filter(t => t.missed <= 2)
  newDets.forEach((d, i) => {
    if (matched.has(i) || !d.box) return
    alive.push({
      id: _nextTrackId++, cls: d.class, confidence: d.confidence,
      frameBox: d.box,
      sx1: 0, sy1: 0, sx2: 0, sy2: 0,
      initialized: false, missed: 0,
      has_helmet: d.has_helmet,
      has_vest: d.has_vest,
    })
  })
  return alive
}

/** Compute letterbox offsets for object-fit:contain video rendering */
function _letterbox(vw: number, vh: number, dw: number, dh: number) {
  if (!vw || !vh) return { rw: dw, rh: dh, ox: 0, oy: 0 }
  const va = vw / vh, da = dw / dh
  return va > da
    ? { rw: dw, rh: dw / va, ox: 0, oy: (dh - dw / va) / 2 }
    : { rw: dh * va, rh: dh, ox: (dw - dh * va) / 2, oy: 0 }
}

function _fmtTime(s: number): string {
  const m = Math.floor(s / 60)
  return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`
}

function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  points: HeatmapPoint[],
  now: number,
  rw: number,
  rh: number,
  ox: number,
  oy: number,
  ttlMs = 1500,
  radius = 24,
  maxOpacity = 0.06
): HeatmapPoint[] {
  const activePoints = points.filter(point => now - point.time < ttlMs)
  if (activePoints.length === 0) return activePoints

  ctx.save()
  ctx.globalCompositeOperation = 'screen'

  activePoints.forEach(point => {
    const hx = point.x * rw + ox
    const hy = point.y * rh + oy
    const age = (now - point.time) / ttlMs
    const intensity = Math.max(0.15, Math.min(1, point.value || 0.25))
    const opacity = Math.max(0, (1 - age) * maxOpacity * (0.5 + intensity * 0.5))
    const scaledRadius = radius * (0.6 + intensity * 0.4)
    const gradient = ctx.createRadialGradient(hx, hy, 0, hx, hy, scaledRadius)
    const color = point.riskLevel === 'CRITICAL'
      ? '255, 42, 42'
      : point.riskLevel === 'HIGH'
        ? '255, 140, 0'
        : point.riskLevel === 'MODERATE'
          ? '255, 214, 0'
          : point.type === 'violation'
            ? '255, 96, 60'
            : '0, 255, 136'

    gradient.addColorStop(0, `rgba(${color}, ${opacity})`)
    gradient.addColorStop(1, `rgba(${color}, 0)`)

    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(hx, hy, scaledRadius, 0, Math.PI * 2)
    ctx.fill()
  })

  ctx.restore()
  return activePoints
}

function drawRiskZones(
  ctx: CanvasRenderingContext2D,
  zones: DetectionRiskZone[],
  fw: number,
  fh: number,
  rw: number,
  rh: number,
  ox: number,
  oy: number,
) {
  if (!zones.length || !fw || !fh) return

  ctx.save()
  zones.forEach(zone => {
    if (zone.risk_score < 0.4) return
    const [x1, y1, x2, y2] = zone.box
    const sx1 = (x1 / fw) * rw + ox
    const sy1 = (y1 / fh) * rh + oy
    const sx2 = (x2 / fw) * rw + ox
    const sy2 = (y2 / fh) * rh + oy
    const color = zone.risk_level === 'CRITICAL'
      ? '255, 42, 42'
      : zone.risk_level === 'HIGH'
        ? '255, 140, 0'
        : '255, 214, 0'

    // Stroke-only outline — no filled rectangle to avoid haze
    ctx.strokeStyle = `rgba(${color}, ${0.35 + zone.risk_score * 0.3})`
    ctx.lineWidth = 1.5
    ctx.setLineDash([5, 4])
    ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1)
    ctx.setLineDash([])

    const label = `${zone.risk_level} ${(zone.risk_score * 100).toFixed(0)}%`
    ctx.font = 'bold 10px monospace'
    const tw = ctx.measureText(label).width
    ctx.fillStyle = `rgba(${color}, 0.85)`
    ctx.fillRect(sx1, Math.max(0, sy1 - 16), tw + 8, 16)
    ctx.fillStyle = '#000'
    ctx.fillText(label, sx1 + 4, Math.max(10, sy1 - 4))
  })
  ctx.restore()
}

function pointsFromHeatmap(payload: DetectionHeatmapPayload | undefined, now: number): HeatmapPoint[] {
  if (!payload?.points?.length) return []
  return payload.points.map(point => ({
    x: point.x,
    y: point.y,
    time: now,
    type: point.type,
    value: point.value,
    riskLevel: point.risk_level,
  }))
}

// ── Shared condition picker ─────────────────────────────────────────────────────
const CONDITION_COLORS: Record<string, string> = {
  S1_normal: '#00cc66',
  S2_dusty: '#ffaa00',
  S3_low_light: '#8866ff',
  S4_crowded: '#ff4444',
  S1_NORMAL: '#00cc66',
  S2_DUSTY: '#ffaa00',
  S3_LOW_LIGHT: '#8866ff',
  S4_CROWDED: '#ff4444',
}

function SceneAutoIndicator({
  condition, autoMode, onToggleAuto, onManualChange,
}: {
  condition: string
  autoMode: boolean
  onToggleAuto: () => void
  onManualChange: (c: Condition) => void
}) {
  const color = CONDITION_COLORS[condition] ?? '#aaaaaa'
  const label = condition.replace(/_/g, ' ').toUpperCase()
  return (
    <div className="det-scene-indicator">
      <div className="det-scene-indicator__header">
        <p className="section-label" style={{ marginBottom: 0 }}>SCENE DETECTION</p>
        <button
          className={`det-scene-indicator__toggle${autoMode ? ' det-scene-indicator__toggle--on' : ''}`}
          onClick={onToggleAuto}
          title={autoMode ? 'Switch to manual condition selection' : 'Enable automatic scene classification'}
        >
          {autoMode ? 'AUTO' : 'MANUAL'}
        </button>
      </div>
      {autoMode ? (
        <div className="det-scene-indicator__badge" style={{ borderColor: color, color }}>
          <span className="det-scene-indicator__dot" style={{ background: color }} />
          {label}
        </div>
      ) : (
        <ConditionPicker value={condition as Condition} onChange={onManualChange} />
      )}
    </div>
  )
}

function ConditionPicker({
  value, onChange,
}: { value: Condition; onChange: (c: Condition) => void }) {
  return (
    <div className="det-condition">
      <p className="section-label" style={{ marginBottom: '0.75rem' }}>ENVIRONMENT CONDITIONS</p>
      <div className="det-condition__chips">
        {CONDITIONS.map(c => (
          <button
            key={c}
            className={`site-chip ${value === c ? 'site-chip--active' : ''}`}
            onClick={() => onChange(c)}
          >
            {c.replace(/_/g, ' ')}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Detection sidebar (class counts + list) ────────────────────────────────────
function DetectionSidebar({
  detections, elapsed, mode, extra, renderControls,
}: {
  detections: Detection[]
  elapsed?: number
  mode?: string
  extra?: React.ReactNode
  renderControls?: React.ReactNode
}) {
  const counts: Record<string, number> = {}
  detections.forEach(d => { counts[d.class] = (counts[d.class] ?? 0) + 1 })

  return (
    <div className="det-results__panel">
      {renderControls && (
        <div className="det-sidebar-controls">
          {renderControls}
        </div>
      )}

      {/* ── Summary Metrics Card ── */}
      <div className="det-summary-card">
        <div className="det-summary-item">
          <span className="det-summary-label">TOTAL DETECTIONS</span>
          <span className="det-summary-value">{detections.length}</span>
        </div>
        <div className="det-summary-sep" />
        <div className="det-summary-item">
          <span className="det-summary-label">LATENCY</span>
          <span className="det-summary-value">{elapsed !== undefined ? `${elapsed}ms` : '--'}</span>
        </div>
        {mode && (
          <>
            <div className="det-summary-sep" />
            <div className="det-summary-item">
              <span className="det-summary-label">MODEL</span>
              <span className="det-summary-value">{mode.toUpperCase()}</span>
            </div>
          </>
        )}
      </div>

      {/* ── Class Breakdown (Pills) ── */}
      <div className="det-class-pills">
        {Object.entries(counts).map(([cls, count]) => (
          <div key={cls} className="det-class-pill">
            <span className="det-class-swatch" style={{ background: CLASS_COLORS[cls] ?? '#aaa' }} />
            <span className="det-class-name">{cls}</span>
            <span className="det-class-count">{count}</span>
          </div>
        ))}
        {detections.length === 0 && (
          <p className="det-hint" style={{ opacity: 0.5, fontSize: 'var(--fs-xs)' }}>
            Waiting for inference...
          </p>
        )}
      </div>

      {/* ── Individual Detection List (Scrollable) ── */}
      <div className="det-det-header">
        <p className="section-label">LIVE DETECTIONS</p>
      </div>
      <div className="det-det-list">
        {detections.map((d, i) => {
          const isWorker = d.class === 'worker' || d.class === 'person';
          const hasPpeIssue = isWorker && (!d.has_helmet || !d.has_vest);

          return (
            <div key={i} className={`det-det-row ${hasPpeIssue ? 'det-det-row--alert' : ''}`}>
              <span className="det-class-swatch" style={{ background: CLASS_COLORS[d.class] ?? '#aaa' }} />
              <div className="det-det-info">
                <span className="det-det-class">{d.class}</span>
                {isWorker && (
                  <div className="det-ppe-badges">
                    <span
                      className={`det-ppe-badge ${d.has_helmet ? 'det-ppe-badge--ok' : 'det-ppe-badge--missing'}`}
                      title={d.has_helmet ? "Helmet: Pass" : "Helmet: MISSING"}
                    >
                      {d.has_helmet ? 'H' : 'H'}
                    </span>
                    <span
                      className={`det-ppe-badge ${d.has_vest ? 'det-ppe-badge--ok' : 'det-ppe-badge--missing'}`}
                      title={d.has_vest ? "Vest: Pass" : "Vest: MISSING"}
                    >
                      {d.has_vest ? 'V' : 'V'}
                    </span>
                  </div>
                )}
              </div>
              <span className="det-det-conf">{(d.confidence * 100).toFixed(0)}%</span>
            </div>
          );
        })}
        {detections.length === 0 && (
          <div className="det-empty-state">
            <span className="det-hint">No active objects detected in the site zone.</span>
          </div>
        )}
      </div>

      {extra && <div style={{ marginTop: 'auto', paddingTop: '1.25rem' }}>{extra}</div>}
    </div>
  )
}

const IMAGE_CLASS_COLORS: Record<string, string> = {
  worker: '#00c864',
  person: '#00c864',
  helmet: '#ffd600',
  hardhat: '#ffd600',
  safety_vest: '#00bfff',
  'safety-vest': '#00bfff',
  vest: '#00bfff',
}

function ImageOverlay({ imageB64, detections }: { imageB64: string, detections: Detection[] }) {
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const img = imgRef.current
    const canvas = canvasRef.current
    if (!img || !canvas || !detections.length) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const draw = () => {
      if (!img.complete) {
        img.onload = draw
        return
      }

      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      for (const det of detections) {
        if (!det.box) continue
        const [x1, y1, x2, y2] = det.box
        const isWorkerBox = det.class === 'worker' || det.class === 'person'

        let borderCol = IMAGE_CLASS_COLORS[det.class] ?? '#aaaaaa'
        if (isWorkerBox) {
          if (det.has_helmet === true && det.has_vest === true) {
            borderCol = '#00c864'
          } else if (det.has_helmet === false && det.has_vest === false) {
            borderCol = '#ff2a2a'
          } else {
            borderCol = '#ffaa00'
          }
        }

        ctx.strokeStyle = borderCol
        ctx.lineWidth = 3
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

        const helmBadge = isWorkerBox ? (det.has_helmet === false ? ' ⚠H' : '') : ''
        const vestBadge = isWorkerBox ? (det.has_vest === false ? ' ⚠V' : '') : ''
        const label = isWorkerBox
          ? `W ${(det.confidence * 100).toFixed(0)}%${helmBadge}${vestBadge}`
          : `${det.class} ${(det.confidence * 100).toFixed(0)}%`

        ctx.font = 'bold 14px monospace'
        const tw = ctx.measureText(label).width
        ctx.fillStyle = borderCol
        ctx.fillRect(x1, Math.max(0, y1 - 20), tw + 10, 20)
        ctx.fillStyle = isWorkerBox ? '#000' : '#fff'
        ctx.fillText(label, x1 + 5, Math.max(14, y1 - 5))
      }
    }

    draw()
  }, [imageB64, detections])

  return (
    <div className="det-results__image-wrap">
      <img ref={imgRef} src={`data:image/jpeg;base64,${imageB64}`}
        className="det-results__image" alt="detection result" style={{ display: 'block' }} />
      <canvas ref={canvasRef} className="det-results__overlay" />
    </div>
  )
}

function ImageUploadMode() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<DetectResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [condition, setCondition] = useState<Condition>('S1_normal')
  const { settings } = useSettings()
  const { pushDetections, setRunning } = useDetectionStats()
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    setFile(f); setResult(null); setError(null)
    setPreview(URL.createObjectURL(f))
  }

  const handleDetect = async () => {
    if (!file) return
    setLoading(true); setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('condition', condition)
      form.append('confidence', settings.confidenceThreshold.toString())
      form.append('model', settings.selectedModel)
      const res = await fetch(`${API}/detect/image`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const data = await res.json()
      setResult(data)
      // Push to shared stats context
      pushDetections(data.detections, data.elapsed_ms, [], [], 0, data.condition)
      setRunning(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Detection failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="det-upload">
      {!result && !file && (
        <div
          className="det-dropzone"
          onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleFile(f) }}
          onDragOver={e => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
        >
          <div className="det-dropzone__empty">
            <span>DROP IMAGE HERE</span>
            <span className="det-hint">or click to browse</span>
          </div>
          <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }}
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
        </div>
      )}

      {file && !result && (
        <div className="det-results">
          <div className="det-results__image-wrap">
            <img src={preview!} className="det-results__image" alt="preview" />
          </div>
          <DetectionSidebar
            detections={[]}
            renderControls={
              <>
                <ConditionPicker value={condition} onChange={setCondition} />
                <button
                  className={`det-run-btn ${loading ? 'det-run-btn--loading' : ''}`}
                  onClick={handleDetect}
                  disabled={loading}
                  style={{ width: '100%', marginTop: '1rem' }}
                >
                  {loading ? 'DETECTING...' : 'RUN DETECTION'}
                </button>
                <button className="det-reset-btn" onClick={() => { setFile(null); setPreview(null) }}>
                  CANCEL
                </button>
              </>
            }
          />
        </div>
      )}

      {error && <p className="det-error">{error}</p>}

      {result && (
        <div className="det-results">
          <ImageOverlay imageB64={result.image_b64} detections={result.detections} />
          <DetectionSidebar
            detections={result.detections}
            elapsed={result.elapsed_ms}
            mode={result.mode}
            renderControls={<ConditionPicker value={condition} onChange={setCondition} />}
            extra={
              <button className="det-reset-btn"
                onClick={() => { setResult(null); setFile(null); setPreview(null) }}>
                DETECT ANOTHER
              </button>
            }
          />
        </div>
      )}
    </div>
  )
}

// ── Video Upload Mode ──────────────────────────────────────────────────────────
//
// Architecture:
//   • No native <video controls> — custom play/pause + seek bar so the canvas
//     overlay covers exactly the video element (no controls-height offset bug).
//   • Self-scheduling async inference loop (no setInterval pile-up).
//     Fires once, awaits response, then schedules next call with
//     max(0, MIN_GAP_MS - inferenceTime) delay → at most ~2-3 inferences/sec.
//   • Frames resized to max 640 px before sending — matches YOLO's native
//     resolution, reduces JPEG size & backend decode time, does NOT change
//     ensemble or WBF logic on the server.
//   • IoU tracker (mergeTracks): matches new detections to existing tracks by
//     class + IoU ≥ 0.25; carries missed tracks for 2 cycles (faded opacity)
//     before dropping them. Eliminates box flicker and false-negative blinks.
//   • Letterbox-aware coordinate transform: accounts for object-fit:contain
//     bars so boxes land exactly on detected objects regardless of video
//     aspect ratio or display size.
//   • Position lerp (α = 0.25/frame at 60 fps): smoothly slides each track's
//     drawn box toward its latest detected position — no sudden jumps.
//
function VideoUploadMode({
  heatmapOverlayEnabled,
  onToggleHeatmap,
}: {
  heatmapOverlayEnabled: boolean
  onToggleHeatmap: () => void
}) {
  const store = useDetectionStore()
  const pipeline = useDetectionPipeline()
  const { settings } = useSettings()

  // Use Zustand store as source of truth for the session
  const file = store.videoFile
  const isPlaying = store.isRunning && !store.isPaused
  const currentTime = store.currentTime
  const duration = store.videoDuration
  
  // Local UI state (transient)
  const [downloading, setDownloading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [roiDrawMode, setRoiDrawMode] = useState(false)
  const [roiDraft, setRoiDraft] = useState<[number, number][]>([])
  // Condition is still partially local but synced to store
  const [condition, setCondition] = useState<Condition>(
    () => (sessionStorage.getItem('bs_condition') as Condition) ?? 'S1_normal'
  )
  const [autoMode, setAutoMode] = useState(
    () => sessionStorage.getItem('bs_auto_mode') !== '0'
  )

  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const rafRef = useRef<number | null>(null)
  const showHeatmapRef = useRef(false)  // updated by effect below

  // ROI persists in store
  const roiPoly = store.roiPoly

  const heatmapHistoryRef = useRef<HeatmapPoint[]>([])

  useEffect(() => { showHeatmapRef.current = heatmapOverlayEnabled }, [heatmapOverlayEnabled])

  useEffect(() => {
    // Persist user preference to sessionStorage
    sessionStorage.setItem('bs_condition', condition)
    sessionStorage.setItem('bs_auto_mode', autoMode ? '1' : '0')
  }, [condition, autoMode])

  const videoUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])
  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl) }, [videoUrl])

  useEffect(() => {
    const video = videoRef.current
    if (!video || !store.detections.length) return

    const MAX_INFER = 640
    const vw = video.videoWidth || 640
    const vh = video.videoHeight || 360
    const inferScale = Math.min(1, MAX_INFER / Math.max(vw, vh, 1))
    const frameWidth = Math.max(1, Math.round(vw * inferScale))
    const frameHeight = Math.max(1, Math.round(vh * inferScale))

    const nextPoints = detectionsToHeatmapPoints(
      store.detections,
      frameWidth,
      frameHeight,
      performance.now(),
    )

    if (nextPoints.length === 0) return
    heatmapHistoryRef.current = [...heatmapHistoryRef.current, ...nextPoints].slice(-300)
  }, [store.detections, currentTime])

  useEffect(() => {
    if (!file) {
      heatmapHistoryRef.current = []
    }
  }, [file])


  // ── Sync display video with pipeline state ──────────────────────────────────
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    if (isPlaying && video.paused) {
      video.play().catch(() => {})
    } else if (!isPlaying && !video.paused) {
      video.pause()
    }
  }, [isPlaying])

  // ── Seek display video when store currentTime jumps (e.g. seekTo) ────────────
  useEffect(() => {
    const video = videoRef.current
    if (!video || Math.abs(video.currentTime - store.currentTime) < 0.5) return
    video.currentTime = store.currentTime
  }, [store.currentTime])

  // ── Stable RAF draw loop ─────────────────────────────────────────────────────
  // Uses refs and getState() so the callback never needs to be recreated.
  // Starts on mount (if playing) and stops on unmount.
  const drawOverlay = useCallback(() => {
    rafRef.current = null
    const canvas = overlayRef.current
    const video = videoRef.current
    if (!canvas || !video) {
      rafRef.current = requestAnimationFrame(drawOverlay)
      return
    }

    // CRITICAL: Ensure canvas dimensions match video display size exactly
    // This fixes issues where bounding boxes don't appear due to dimension mismatches
    const dw = video.clientWidth || video.offsetWidth || 640
    const dh = video.clientHeight || video.offsetHeight || 360
    
    // Only resize if dimensions actually changed
    if (canvas.width !== dw || canvas.height !== dh) {
      canvas.width = dw
      canvas.height = dh
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) { rafRef.current = requestAnimationFrame(drawOverlay); return }
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Skip drawing if no video source yet
    if (!video.videoWidth || !video.videoHeight || isNaN(video.videoWidth) || isNaN(video.videoHeight)) {
      rafRef.current = requestAnimationFrame(drawOverlay)
      return
    }

    const { rw, rh, ox, oy } = _letterbox(video.videoWidth, video.videoHeight, canvas.width, canvas.height)
    
    // Compute inference frame dimensions matching DetectionStatsContext (max 640 on longest side)
    const MAX_INFER = 640
    const vw = video.videoWidth  || 640
    const vh = video.videoHeight || 360
    const inferScale = Math.min(1, MAX_INFER / Math.max(vw, vh, 1))
    const fw = Math.max(1, Math.round(vw * inferScale))
const fh = Math.max(1, Math.round(vh * inferScale))

    // Always read fresh state to avoid stale closure issues
    const st = useDetectionStore.getState()
    const currentDetections = st.detections
    const currentRoiPoly = st.roiPoly
    
    // ── Draw detections DIRECTLY without tracking transformation ─────────────────
    // This ensures boxes appear immediately without waiting for track initialization
    if (currentDetections && currentDetections.length > 0) {
      for (const det of currentDetections) {
        const box = det.box
        if (!box || box.length < 4) continue
        
        // Directly scale box coordinates to display canvas
        const x1 = (box[0] / fw) * rw + ox
        const y1 = (box[1] / fh) * rh + oy
        const x2 = (box[2] / fw) * rw + ox
        const y2 = (box[3] / fh) * rh + oy
        
        // Determine box color
        const isWorker = det.class === 'worker' || det.class === 'person'
        let borderCol = '#00c864' // green for worker by default
        if (isWorker) {
          if (det.has_helmet === true && det.has_vest === true) {
            borderCol = '#00c864' // green - compliant
          } else if (det.has_helmet === false && det.has_vest === false) {
            borderCol = '#ff2a2a' // red - full violation
          } else {
            borderCol = '#ffaa00' // orange - partial
          }
        } else if (det.class === 'helmet') {
          borderCol = '#ffd600'
        } else if (det.class === 'safety_vest' || det.class === 'vest') {
          borderCol = '#00bfff'
        }
        
        // Draw the box
        ctx.strokeStyle = borderCol
        ctx.lineWidth = 3
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
        
        // Draw label
        const conf = Math.round(det.confidence * 100)
        const label = `${det.class} ${conf}%`
        ctx.font = 'bold 14px monospace'
        ctx.fillStyle = borderCol
        ctx.fillRect(x1, Math.max(0, y1 - 20), ctx.measureText(label).width + 10, 20)
        ctx.fillStyle = '#000'
        ctx.fillText(label, x1 + 5, Math.max(14, y1 - 4))
      }
    }

    // ── Draw ROI ─────────────────────────────────────────────────────────────
    if (currentRoiPoly && currentRoiPoly.length >= 2) {
      ctx.save()
      ctx.strokeStyle = '#ffcc00'
      ctx.lineWidth = 2
      ctx.setLineDash([6, 4])
      ctx.beginPath()
      const [px0, py0] = currentRoiPoly[0]
      ctx.moveTo(px0 * rw + ox, py0 * rh + oy)
      for (let i = 1; i < currentRoiPoly.length; i++) {
        const [pxi, pyi] = currentRoiPoly[i]
        ctx.lineTo(pxi * rw + ox, pyi * rh + oy)
      }
      ctx.closePath()
      ctx.stroke()
      ctx.fillStyle = 'rgba(255, 204, 0, 0.06)'
      ctx.fill()
      ctx.restore()
    }

    // ── Draw Heatmap ─────────────────────────────────────────────────────────
    if (showHeatmapRef.current && heatmapHistoryRef.current.length > 0) {
      heatmapHistoryRef.current = drawHeatmap(
        ctx,
        heatmapHistoryRef.current,
        performance.now(),
        rw,
        rh,
        ox,
        oy,
        7000,
        28,
        0.1,
      )
    }

    // Always continue the animation loop
    rafRef.current = requestAnimationFrame(drawOverlay)
  }, []) // end drawOverlay

  // Start/stop the RAF based on play state; restarts cleanly after tab remount
  useEffect(() => {
    if (isPlaying) {
      if (!rafRef.current) {
        rafRef.current = requestAnimationFrame(drawOverlay)
      }
    } else {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }
  }, [isPlaying, drawOverlay])

  // ── Overlay Interaction ──────────────────────────────────────────────────────
  const handleOverlayClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!roiDrawMode) {
      // Toggle global play/pause if not in draw mode
      isPlaying ? pipeline.pauseDetection() : pipeline.resumeDetection()
      return
    }
    const canvas = overlayRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    const rect = canvas.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top
    const { rw, rh, ox, oy } = _letterbox(video.videoWidth, video.videoHeight, canvas.width, canvas.height)
    const nx = (cx - ox) / rw
    const ny = (cy - oy) / rh
    setRoiDraft(prev => [...prev, [nx, ny] as [number, number]])
  }, [roiDrawMode, isPlaying, pipeline])

  const handleOverlayDblClick = useCallback(() => {
    if (!roiDrawMode || roiDraft.length < 3) return
    store.setRoiPoly(roiDraft)
    setRoiDraft([])
    setRoiDrawMode(false)
  }, [roiDrawMode, roiDraft, store])

  const clearRoi = useCallback(() => {
    store.setRoiPoly(null)
    setRoiDraft([])
    setRoiDrawMode(false)
  }, [store])

  // Draw draft polygon points while user is placing them
  useEffect(() => {
    if (!roiDrawMode || roiDraft.length === 0) return
    const canvas = overlayRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext('2d')!
    const { rw, rh, ox, oy } = _letterbox(video.videoWidth, video.videoHeight, canvas.width, canvas.height)
    ctx.save()
    ctx.strokeStyle = '#ffcc00'
    ctx.fillStyle = 'rgba(255,204,0,0.8)'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    const [dx0, dy0] = roiDraft[0]
    ctx.moveTo(dx0 * rw + ox, dy0 * rh + oy)
    roiDraft.forEach(([dx, dy]) => ctx.lineTo(dx * rw + ox, dy * rh + oy))
    ctx.stroke()
    roiDraft.forEach(([dx, dy]) => {
      ctx.beginPath()
      ctx.arc(dx * rw + ox, dy * rh + oy, 4, 0, Math.PI * 2)
      ctx.fill()
    })
    ctx.restore()
  }, [roiDraft, roiDrawMode])

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value)
    pipeline.seekTo(time)
  }, [pipeline])

  const handleFile = (f: File) => {
    pipeline.startDetection(f)
    setError(null)
  }

  // ── Download annotated video (unchanged — full server-side pipeline) ──────────
  const handleDownload = async () => {
    if (!file) return
    setDownloading(true); setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('condition', condition)
      form.append('confidence', settings.confidenceThreshold.toString())
      form.append('model', settings.selectedModel)
      form.append('sample_every', '3')
      const res = await fetch(`${API}/detect/video`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = file.name.replace(/\.[^.]+$/, '') + '_buildsight_detected.mp4'
      a.click(); URL.revokeObjectURL(url)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Download failed')
    } finally { setDownloading(false) }
  }

  if (!file) {
    return (
      <div className="det-upload">
        <div
          className="det-dropzone"
          onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleFile(f) }}
          onDragOver={e => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
        >
          <div className="det-dropzone__empty">
            <span>DROP VIDEO HERE</span>
            <span className="det-hint">or click to browse · MP4, MOV, AVI, MKV</span>
          </div>
          <input ref={inputRef} type="file" accept="video/*" style={{ display: 'none' }}
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
        </div>
      </div>
    )
  }

  return (
    <div className="det-video">
      {error && <p className="det-error">{error}</p>}

      <div className="det-video__status">
        <span className={`det-video__indicator ${isPlaying ? 'det-video__indicator--live' : ''}`} />
        <span>{isPlaying ? 'LIVE DETECTION' : 'PAUSED — CLICK TO RESUME'}</span>
        
        {store.latencyMs > 0 && (
          <>
            <span className="det-video__status-sep" />
            <span>{store.latencyMs}ms</span>
          </>
        )}
        {store.fps > 0 && (
          <>
            <span className="det-video__status-sep" />
            <span>{store.fps} FPS</span>
          </>
        )}
        <span className="det-video__status-sep" />
        <span>{store.detections.length} det</span>
        
        {store.condition && (
          <>
            <span className="det-video__status-sep" />
            <span style={{ color: CONDITION_COLORS[store.condition] ?? '#aaa', fontWeight: 600 }}>
              {store.condition.replace(/_/g, ' ').toUpperCase()}
            </span>
          </>
        )}
        
        {roiPoly && (
          <>
            <span className="det-video__status-sep" />
            <span style={{ color: '#ffcc00' }}>ZONE ACTIVE</span>
          </>
        )}
      </div>

      <div className="det-video__layout">
        <div className="det-video__col">
          <div
            className="det-video__player-wrap"
            onClick={roiDrawMode ? undefined : () => isPlaying ? pipeline.pauseDetection() : pipeline.resumeDetection()}
            style={{ cursor: roiDrawMode ? 'crosshair' : 'pointer' }}
          >
            {videoUrl && (
              <video
                ref={videoRef}
                src={videoUrl}
                className="det-video__player"
                muted
                playsInline
                onLoadedMetadata={() => {
                  const video = videoRef.current
                  if (!video) return
                  // Reconnect display video to the active background session position
                  const target = useDetectionStore.getState().currentTime
                  if (target > 0.5) video.currentTime = target
                  if (useDetectionStore.getState().isRunning && !useDetectionStore.getState().isPaused) {
                    video.play().catch(() => {})
                  }
                }}
              />
            )}
            <canvas
              ref={overlayRef}
              className="det-video__overlay"
              onClick={handleOverlayClick}
              onDoubleClick={handleOverlayDblClick}
              style={{ cursor: roiDrawMode ? 'crosshair' : 'pointer' }}
            />
            {isPlaying && !roiDrawMode && (
              <div className="det-video__legend">
                <div className="det-legend-item">
                  <span className="det-legend-dot" style={{ background: '#0088ff' }} />
                  <span>COMPLIANT</span>
                </div>
                <div className="det-legend-item">
                  <span className="det-legend-dot" style={{ background: '#ffaa00' }} />
                  <span>PARTIAL PPE</span>
                </div>
                <div className="det-legend-item">
                  <span className="det-legend-dot" style={{ background: '#ff2a2a' }} />
                  <span>VIOLATION</span>
                </div>
              </div>
            )}
            {!isPlaying && !roiDrawMode && <div className="det-video__play-hint">▶</div>}
          </div>
          
          {store.peakRiskMoments.length > 0 && (
            <div className="det-video__risk-strip">
              <div className="det-risk-strip__label">PEAK RISK MOMENTS</div>
              <div className="det-risk-strip__items">
                {store.peakRiskMoments.map((m, i) => (
                  <button key={i} className="det-risk-item" onClick={() => pipeline.seekTo(m.time)}>
                    <span className="det-risk-icon">⚠️</span>
                    <span className="det-risk-time">{_fmtTime(m.time)}</span>
                    <span className="det-risk-score">{m.score} vlns</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="det-video__seekbar">
            <button
              className="det-video__play-btn"
              onClick={() => isPlaying ? pipeline.pauseDetection() : pipeline.resumeDetection()}
            >
              {isPlaying ? '⏸' : '▶'}
            </button>
            <input
              type="range" className="det-video__seek"
              min={0} max={duration || 1} step={0.05} value={currentTime}
              onChange={handleSeek}
            />
            <span className="det-video__time">
              {_fmtTime(currentTime)} / {_fmtTime(duration)}
            </span>
          </div>
        </div>

        <DetectionSidebar
          detections={store.detections}
          elapsed={store.latencyMs}
          mode="live-ensemble"
          renderControls={
            <>
              <PPEStatusPanel
                workers={store.ppeWorkers}
                sceneCondition={store.condition || condition}
              />
              <SceneAutoIndicator
                condition={store.condition || condition}
                autoMode={autoMode}
                onToggleAuto={() => setAutoMode(!autoMode)}
                onManualChange={setCondition}
              />
              <div className="det-sidebar-actions">
                <button
                  className={`det-run-btn ${roiDrawMode ? 'det-run-btn--stop' : ''}`}
                  onClick={() => setRoiDrawMode(!roiDrawMode)}
                >
                  {roiDrawMode ? 'CANCEL DRAW' : 'DRAW ZONE'}
                </button>
                {roiPoly && (
                  <button className="det-run-btn det-run-btn--stop" onClick={clearRoi}>
                    CLEAR ZONE
                  </button>
                )}
                <button className="det-run-btn" onClick={handleDownload} disabled={downloading}>
                  {downloading ? 'PROCESSING...' : 'DOWNLOAD ANNOTATED'}
                </button>
                <button
                  className={`det-run-btn ${heatmapOverlayEnabled ? 'det-run-btn--active' : ''}`}
                  onClick={onToggleHeatmap}
                >
                  {heatmapOverlayEnabled ? 'HIDE HEATMAP' : 'SHOW HEATMAP'}
                </button>
                <button className="det-reset-btn" onClick={() => pipeline.stopDetection()}>
                  CHANGE VIDEO
                </button>
              </div>
            </>
          }
        />
      </div>
    </div>
  )
}

// ── Live Camera Mode ───────────────────────────────────────────────────────────
// Refactored (Jovi handoff 2026-04-06) to use the same canvas-overlay + IoU
// tracker architecture as VideoUploadMode instead of the old base64 round-trip:
//
//   OLD path: capture → toDataURL → POST → server draws OpenCV boxes → encode
//             → send back image_b64 → decode → display  (≈3-4× extra latency)
//
//   NEW path: capture → toDataURL → POST → receive JSON dets → tracker →
//             lerp → draw on transparent canvas overlay  (no image round-trip)
//
export function LiveMode() {
  const { setRunning } = useDetectionStats()
  const pipeline = useDetectionPipeline()
  const store = useDetectionStore()
  const { settings } = useSettings()
  void pipeline; void settings

  const [active, setActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showHeatmap, setShowHeatmap] = useState(false)
  const heatmapHistoryRef = useRef<HeatmapPoint[]>([])

  const [condition, setCondition] = useState<Condition>('S1_normal')
  const [rtspUrl, setRtspUrl] = useState(() => {
    const saved = localStorage.getItem('bs_rtsp_url')
    if (saved) return saved
    return settings.cctvStreamUrl || '0'
  })

  useEffect(() => {
    localStorage.setItem('bs_rtsp_url', rtspUrl)
  }, [rtspUrl])

  // Auto-start camera on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      startCamera()
    }, 500)
    return () => {
      clearTimeout(timer)
      stopCamera()
    }
  }, [])

  // Sync with global settings if they change
  useEffect(() => {
    if (settings.cctvStreamUrl && settings.cctvStreamUrl !== rtspUrl) {
      setRtspUrl(settings.cctvStreamUrl)
    }
  }, [settings.cctvStreamUrl])

  const detections = store.detections
  const elapsed = store.latencyMs

  const imgRef = useRef<HTMLImageElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)

  const rafRef = useRef<number | null>(null)
  const isRunningRef = useRef(false)
  const tracksRef = useRef<Track[]>([])
  const pendingRef = useRef<Detection[] | null>(null)
  const riskZonesRef = useRef<DetectionRiskZone[]>([])
  const frameWRef = useRef(1)
  const frameHRef = useRef(1)

  useEffect(() => {
    if (active && detections && detections.length > 0) {
      pendingRef.current = detections
      const nextMoment = buildPeakRiskMoment(detections, 0)
      if (nextMoment) {
        useDetectionStore.setState(state => ({
          peakRiskMoments: mergePeakRiskMoments(state.peakRiskMoments, nextMoment, 6, 1)
        }))
      }
    }
  }, [detections, active])

  // ── RAF draw loop — unified with VideoUploadMode color scheme ─────────────
  const drawOverlay = useCallback(() => {
    const canvas = overlayRef.current
    const img = imgRef.current
    if (!canvas || !img) return

    const dw = img.clientWidth || img.offsetWidth || 640
    const dh = img.clientHeight || img.offsetHeight || 360
    
    if (canvas.width !== dw || canvas.height !== dh) {
      canvas.width = dw
      canvas.height = dh
      tracksRef.current.forEach(t => { t.initialized = false })
    }

    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, dw, dh)

    if (!img.naturalWidth || !img.naturalHeight || isNaN(img.naturalWidth)) {
      rafRef.current = requestAnimationFrame(drawOverlay)
      return
    }

    const { rw, rh, ox, oy } = _letterbox(img.naturalWidth, img.naturalHeight, dw, dh)

    // Use backend-reported inference dimensions for coordinate mapping
    // Falls back to img.naturalWidth if not yet received
    const st = useDetectionStore.getState()
    const fw = frameWRef.current = st.frameWidth || img.naturalWidth
    const fh = frameHRef.current = st.frameHeight || img.naturalHeight

    if (showHeatmap && heatmapHistoryRef.current.length > 0) {
      heatmapHistoryRef.current = drawHeatmap(
        ctx,
        heatmapHistoryRef.current,
        performance.now(),
        rw,
        rh,
        ox,
        oy
      )
    }

    if (showHeatmap) {
      drawRiskZones(ctx, riskZonesRef.current, fw, fh, rw, rh, ox, oy)
    }

    if (pendingRef.current !== null) {
      const dets = pendingRef.current
      tracksRef.current = mergeTracks(tracksRef.current, dets)

      const now = performance.now()
      dets.forEach(d => {
        if (!d.box) return
        const isWorker = d.class === 'worker' || d.class === 'person'
        const hasViolation = isWorker && (!d.has_helmet || !d.has_vest)
        heatmapHistoryRef.current.push({
          x: (d.box[0] + d.box[2]) / 2 / fw,
          y: (d.box[1] + d.box[3]) / 2 / fh,
          time: now,
          type: hasViolation ? 'violation' : 'worker',
          value: hasViolation ? 0.78 : 0.28,
          riskLevel: hasViolation ? 'HIGH' : 'LOW',
        })
      })

      pendingRef.current = null
    }

    const LERP = 0.60

    for (const t of tracksRef.current) {
      const [x1, y1, x2, y2] = t.frameBox
      const tx1 = (x1 / fw) * rw + ox, ty1 = (y1 / fh) * rh + oy
      const tx2 = (x2 / fw) * rw + ox, ty2 = (y2 / fh) * rh + oy

      if (!t.initialized) {
        t.sx1 = tx1; t.sy1 = ty1; t.sx2 = tx2; t.sy2 = ty2
        t.initialized = true
      } else {
        t.sx1 += (tx1 - t.sx1) * LERP
        t.sy1 += (ty1 - t.sy1) * LERP
        t.sx2 += (tx2 - t.sx2) * LERP
        t.sy2 += (ty2 - t.sy2) * LERP
      }

      ctx.globalAlpha = t.missed > 0 ? 0.45 : 1.0

      // ── Unified color scheme (matches VideoUploadMode exactly) ──────
      const isWorkerBoxLive = t.cls === 'worker' || t.cls === 'person'
      let borderColLive: string

      if (isWorkerBoxLive) {
        if (t.has_helmet === true && t.has_vest === true) {
          borderColLive = '#00c864'   // green – fully compliant
        } else if (t.has_helmet === false && t.has_vest === false) {
          borderColLive = '#ff2a2a'   // red – full violation
        } else {
          borderColLive = '#ffaa00'   // orange – partial violation
        }
      } else if (t.cls === 'helmet' || t.cls === 'hardhat') {
        borderColLive = '#ffd600'     // yellow – helmet
      } else if (t.cls === 'safety_vest' || t.cls === 'vest' || t.cls === 'safety-vest') {
        borderColLive = '#00bfff'     // cyan – vest
      } else {
        borderColLive = '#aaaaaa'
      }

      // Draw the box
      ctx.strokeStyle = borderColLive
      ctx.lineWidth = isWorkerBoxLive ? 3 : 2.5
      ctx.strokeRect(t.sx1, t.sy1, t.sx2 - t.sx1, t.sy2 - t.sy1)

      // ── Label format (matches VideoUploadMode exactly) ──────────────
      const conf = Math.round((t.confidence ?? 0) * 100)
      let label: string
      if (isWorkerBoxLive) {
        const helmBadge = t.has_helmet === false ? ' ⚠H' : ''
        const vestBadge = t.has_vest === false ? ' ⚠V' : ''
        label = `W ${conf}%${helmBadge}${vestBadge}`
      } else {
        label = `${t.cls} ${conf}%`
      }

      ctx.font = 'bold 14px monospace'
      const tw = ctx.measureText(label).width
      ctx.fillStyle = borderColLive
      ctx.fillRect(t.sx1, Math.max(0, t.sy1 - 20), tw + 10, 20)
      ctx.fillStyle = isWorkerBoxLive ? '#000' : '#fff'
      ctx.fillText(label, t.sx1 + 5, Math.max(14, t.sy1 - 4))

      ctx.globalAlpha = 1
    }

    rafRef.current = requestAnimationFrame(drawOverlay)
  }, [showHeatmap])

  // ── Camera lifecycle ─────────────────────────────────────────────────────
  const startCamera = async () => {
    setError(null)
    try {
      const res = await fetch(`${API}/stream/start`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rtsp_url: rtspUrl })
      })
      if (!res.ok) throw new Error('Failed to start camera stream')
      
      setActive(true)
      setRunning(true)
      isRunningRef.current = true
      
      if (imgRef.current) {
        imgRef.current.src = `${API}/stream/live?t=${Date.now()}`
      }
      
      store.requestSnapshot()
      rafRef.current = requestAnimationFrame(drawOverlay)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Camera stream unreachable.')
    }
  }

  const stopCamera = useCallback(async () => {
    isRunningRef.current = false
    try {
      await fetch(`${API}/stream/stop`, { method: 'POST' })
    } catch (e) {
      console.error("Failed to stop camera on backend:", e)
    }

    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
    tracksRef.current = []
    pendingRef.current = null
    riskZonesRef.current = []
    heatmapHistoryRef.current = []
    
    if (imgRef.current) {
      imgRef.current.src = ''
    }
    
    const canvas = overlayRef.current
    canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
    setActive(false)
    setRunning(false)
  }, [setRunning])

  useEffect(() => () => stopCamera(), [stopCamera])

  return (
    <div className="det-live">
      {error && <div className="det-error-banner">{error}</div>}

      <div className="live-feed__hud">
        <div className="hud-status">
          <span className={`hud-dot ${active ? 'hud-dot--active' : ''}`} />
          <span>{active ? 'LIVE STREAMING' : 'OFFLINE'}</span>
        </div>
        <div className="hud-meta">
          <span>SOURCE: {rtspUrl === '0' ? 'INTERNAL WEBCAM' : rtspUrl === '1' ? 'EXTERNAL USB CAMERA' : 'RTSP STREAM'}</span>
          <span className="hud-sep">/</span>
          <span>MODEL: BS-ENSEMBLE-WBF</span>
        </div>
        <div className="hud-settings">
          <button className="hud-btn" onClick={() => setShowHeatmap(!showHeatmap)}>
            {showHeatmap ? 'HEATMAP ON' : 'HEATMAP OFF'}
          </button>
        </div>
      </div>

      {/* Feed + sidebar */}
      <div className="det-video__layout">
        <div className="det-video__col">
          <div className="det-video__player-wrap" style={{ cursor: 'default' }}>
            <img
              ref={imgRef}
              className="det-video__player"
              style={{ display: active ? 'block' : 'none', width: '100%', height: '100%', objectFit: 'contain' }}
              alt="Live Camera Feed"
              crossOrigin="anonymous"
            />
            <canvas ref={overlayRef} className="det-video__overlay" />
            {!active && (
              <div className="det-live__placeholder" style={{ position: 'absolute', inset: 0 }}>
                Camera Off — click START CAMERA
              </div>
            )}
          </div>
          {active && (
            <div className="det-video__seekbar" style={{ border: 'none', background: 'none' }}>
              <button
                className={`det-run-btn ${showHeatmap ? 'det-run-btn--active' : ''}`}
                onClick={() => setShowHeatmap(!showHeatmap)}
                style={{ width: 'auto', padding: '0.4rem 1rem' }}
              >
                {showHeatmap ? 'HIDE HEATMAP' : 'SHOW HEATMAP'}
              </button>
              <div style={{ flex: 1 }} />
              <span className="det-video__time" style={{ color: 'var(--color-safe)' }}>LIVE STREAMING</span>
            </div>
          )}
        </div>

        <DetectionSidebar
          detections={detections}
          elapsed={active ? elapsed : undefined}
          mode={active ? 'live-ensemble' : undefined}
          renderControls={
            <>
              <div className="det-rtsp-input-group" style={{ marginBottom: '1rem' }}>
                <p className="section-label" style={{ marginBottom: '0.5rem' }}>CAMERA SOURCE</p>
                <input 
                  type="text" 
                  className="det-input"
                  style={{ width: '100%', padding: '0.5rem', background: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', color: 'var(--color-text)', fontSize: '0.85rem' }}
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  placeholder="rtsp://ip:port/stream OR local index (0, 1)"
                  disabled={active}
                />
                <p style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', marginTop: '0.4rem' }}>
                  Enter an RTSP URL or a local webcam index (e.g., 0 for laptop camera, 1 for USB webcam).
                </p>
              </div>
              <ConditionPicker value={condition} onChange={setCondition} />
              <button
                className={`det-run-btn ${active ? 'det-run-btn--stop' : ''}`}
                onClick={active ? stopCamera : startCamera}
                style={{ width: '100%', marginTop: '1.25rem' }}
              >
                {active ? 'STOP CAMERA' : 'START CAMERA'}
              </button>
            </>
          }
        />

        {/* Peak Risk Moments Sidebar (Live) */}
        <div className="det-risk-sidebar">
          <div className="det-risk-sidebar__header">
            <span>PEAK RISK MOMENTS</span>
          </div>
          <div className="det-risk-sidebar__list">
            {store.peakRiskMoments.length === 0 ? (
              <div style={{ padding: '1rem', fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>
                Waiting for high-risk detection data...
              </div>
            ) : (
              store.peakRiskMoments.map((moment, idx) => (
                <div key={idx} className="det-risk-item" style={{ cursor: 'default' }}>
                  <div className="det-risk-item__meta">
                    <span className="det-risk-item__label">{moment.type}</span>
                    <span className="det-risk-item__score">RISK: {moment.score}</span>
                  </div>
                  <div className="det-risk-item__time">LIVE CAPTURE #{idx + 1}</div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main Panel ─────────────────────────────────────────────────────────────────
type DetectionPanelMode = 'VIDEO' | 'IMAGE'

interface DetectionPanelProps {
  mode: DetectionPanelMode
}

export function DetectionPanel({ mode }: DetectionPanelProps) {
  const [apiOk, setApiOk] = useState<boolean | null>(null)
  const [modelMode, setModelMode] = useState('')
  const [heatmapOverlayEnabled, setHeatmapOverlayEnabled] = useState(false)
  const { setModelName } = useDetectionStats()
  const title = mode === 'VIDEO' ? 'PPE Detection - Video Workspace' : 'PPE Detection - Image Workspace'
  const meta = mode === 'VIDEO'
    ? 'Review uploaded construction footage with live overlays and export annotated MP4 evidence.'
    : 'Inspect still captures for PPE compliance, incident snapshots, and audit evidence.'

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => { setApiOk(true); setModelMode(d.mode); setModelName(d.mode || 'ensemble-wbf') })
      .catch(() => setApiOk(false))
  }, [setModelName])

  return (
    <div className="det-panel panel">
      <div className="panel-heading">
        <div>
          <p className="section-label">AI Detection Engine</p>
          <p className="det-panel__mode">{title}</p>
          <h3>PPE Detection — Image · Video · Live</h3>
        </div>
        <div className="det-panel__status">
          <p className="panel-meta">{meta}</p>
          <span className={`api-badge ${apiOk === true ? 'api-badge--ok' : apiOk === false ? 'api-badge--err' : 'api-badge--pending'}`}>
            {apiOk === true ? `BACKEND OK · ${modelMode.toUpperCase()}` : apiOk === false ? 'BACKEND OFFLINE' : 'CONNECTING...'}
          </span>
        </div>
      </div>

      <div className="det-body">
        {mode === 'IMAGE' ? (
          <ImageUploadMode />
        ) : (
          <VideoUploadMode
            heatmapOverlayEnabled={heatmapOverlayEnabled}
            onToggleHeatmap={() => setHeatmapOverlayEnabled((enabled) => !enabled)}
          />
        )}
      </div>
    </div>
  )
}
