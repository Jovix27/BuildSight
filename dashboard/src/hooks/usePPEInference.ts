/**
 * usePPEInference
 * ---------------
 * Self-scheduling async inference loop for live PPE detection.
 * Captures a frame from a <video> element, encodes it as JPEG, and POSTs
 * it to the backend /api/detect/frame endpoint.
 *
 * Design decisions:
 *  - Uses setTimeout (not setInterval or RAF) for self-scheduling.
 *    This means slow inference never queues up overlapping requests —
 *    the next request waits for the previous one to complete.
 *  - An AbortController cancels any in-flight fetch when the loop is
 *    stopped, preventing stale callbacks from firing after unmount.
 *  - The offscreen capture canvas is allocated once and reused across frames.
 *  - onDetections and onError are stored in refs so that inferOnce is NOT
 *    recreated when the parent re-renders — eliminating the abort-restart
 *    cycle that was preventing any detections from being returned.
 *  - classConf / nmsIou / wbfIou are forwarded to the backend so the live
 *    feed uses the same per-class confidence gates as the video workspace.
 *    Without these, the backend falls back to very lenient defaults that let
 *    hair (→helmet) and T-shirts (→vest) through as false-positives.
 *  - Inference frame dimensions (fw, fh) are exposed via infrDimsRef so
 *    the renderer can perform the correct coordinate transform.
 */

import { useRef, useCallback, useEffect } from 'react'
import type { RawDetection } from './useBoxTracker'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface InferenceMetrics {
  latencyMs:  number
  fps:        number
  condition:  string
}

/** Dimensions of the frame that was sent to the backend. */
export interface InferenceDims {
  fw: number  // inference frame width (pixels)
  fh: number  // inference frame height (pixels)
}

interface UsePPEInferenceOptions {
  /** Ref to the <video> element supplying frames */
  videoRef:    React.RefObject<HTMLVideoElement | null>
  /** Loop runs only while true */
  isActive:    boolean
  /** Minimum ms gap between consecutive inference calls (default 120) */
  minGapMs?:   number
  /** Max dimension for inference resize — larger frames are downscaled (default 640) */
  maxInferDim?: number
  /** Scene condition sent to backend; 'auto' lets server classify (default 'auto') */
  condition?:  string
  /** JPEG encode quality [0..1] (default 0.75) */
  jpegQuality?: number

  /**
   * Per-class confidence gate applied post-WBF on the backend.
   * Keys: "worker" | "helmet" | "vest"
   * Example: { worker: 0.40, helmet: 0.45, vest: 0.35 }
   *
   * IMPORTANT: Without this the backend uses lenient defaults (helmet 0.30,
   * vest 0.14) that allow hair → helmet and T-shirt → vest false positives.
   * Always pass settings.helmetConf / vestConf / workerConf here so live
   * inference uses the same calibration as the video workspace.
   */
  classConf?:  Record<string, number>

  /**
   * Per-class NMS IoU overrides forwarded to the backend.
   * Example: { worker: 0.60, helmet: 0.55, vest: 0.55 }
   */
  nmsIou?:     Record<string, number>

  /**
   * Per-class WBF fusion IoU overrides forwarded to the backend.
   * Example: { worker: 0.65, helmet: 0.45, vest: 0.50 }
   */
  wbfIou?:     Record<string, number>

  onDetections?: (dets: RawDetection[], metrics: InferenceMetrics) => void
  onError?:      (err: Error) => void
}

const API_DETECT_FRAME = 'http://localhost:8000/api/detect/frame'

// ── FPS estimator ─────────────────────────────────────────────────────────────

class FpsEstimator {
  private window: number[] = []
  private readonly maxSamples = 12

  record(now: number): number {
    this.window.push(now)
    if (this.window.length > this.maxSamples) this.window.shift()
    if (this.window.length < 2) return 0
    const span = now - this.window[0]
    return Math.round(((this.window.length - 1) / span) * 1000)
  }

  reset() { this.window = [] }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function usePPEInference({
  videoRef,
  isActive,
  minGapMs    = 120,
  maxInferDim = 640,
  condition   = 'auto',
  jpegQuality = 0.80,   // raised 0.75→0.80 to match video workspace quality
  classConf,
  nmsIou,
  wbfIou,
  onDetections,
  onError,
}: UsePPEInferenceOptions) {
  // Offscreen capture canvas — allocated once, reused every frame
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const abortRef         = useRef<AbortController | null>(null)
  const timerRef         = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isActiveRef      = useRef(isActive)
  const fpsEstimator     = useRef(new FpsEstimator())

  /** Inference frame dimensions — written each frame, read by renderer */
  const infrDimsRef = useRef<InferenceDims>({ fw: 640, fh: 480 })

  // ── Callback refs — always point to the latest callbacks without making
  //    inferOnce depend on them.  Previously onDetections and onError were
  //    listed as useCallback deps, so any re-render that produced a new
  //    function identity (e.g. an inline arrow for onError) would recreate
  //    inferOnce → trigger the useEffect cleanup → abort the in-flight
  //    request → perpetual abort-restart cycle with 0 detections / 0ms latency.
  const onDetectionsRef = useRef(onDetections)
  const onErrorRef      = useRef(onError)
  useEffect(() => { onDetectionsRef.current = onDetections }, [onDetections])
  useEffect(() => { onErrorRef.current = onError },           [onError])

  // Keep threshold refs current so inferOnce always sends the latest values
  // without being recreated when settings change.
  const classConfRef = useRef(classConf)
  const nmsIouRef    = useRef(nmsIou)
  const wbfIouRef    = useRef(wbfIou)
  useEffect(() => { classConfRef.current = classConf }, [classConf])
  useEffect(() => { nmsIouRef.current    = nmsIou    }, [nmsIou])
  useEffect(() => { wbfIouRef.current    = wbfIou    }, [wbfIou])

  // Keep isActive ref current with latest prop (no stale-closure issues)
  useEffect(() => { isActiveRef.current = isActive }, [isActive])

  const getCaptureCanvas = useCallback((): HTMLCanvasElement => {
    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement('canvas')
    }
    return captureCanvasRef.current
  }, [])

  // ── Core inference call ───────────────────────────────────────────────────
  //
  // IMPORTANT: onDetections, onError, classConf, nmsIou, wbfIou are
  // intentionally NOT listed as deps — they are read through refs so that
  // this callback stays stable across renders.  Listing them here would
  // recreate inferOnce on every parent re-render, triggering the useEffect
  // cleanup and aborting every in-flight request.
  const inferOnce = useCallback(async () => {
    if (!isActiveRef.current) return

    const video = videoRef.current
    if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      // Video not ready — back off briefly and retry
      timerRef.current = setTimeout(inferOnce, 60)
      return
    }

    const t0 = performance.now()

    // ── Capture & resize ────────────────────────────────────────────────────
    const scale = Math.min(1, maxInferDim / Math.max(video.videoWidth, video.videoHeight))
    const fw    = Math.max(1, Math.round(video.videoWidth  * scale))
    const fh    = Math.max(1, Math.round(video.videoHeight * scale))

    infrDimsRef.current = { fw, fh }

    const canvas = getCaptureCanvas()
    canvas.width  = fw
    canvas.height = fh

    const ctx2d = canvas.getContext('2d')
    if (!ctx2d) {
      timerRef.current = setTimeout(inferOnce, minGapMs)
      return
    }
    ctx2d.drawImage(video, 0, 0, fw, fh)
    const b64 = canvas.toDataURL('image/jpeg', jpegQuality)

    // ── Cancel previous in-flight request ──────────────────────────────────
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    try {
      const form = new FormData()
      form.append('image_b64',      b64)
      form.append('condition',      condition === 'auto' ? 'S1_normal' : condition)
      form.append('auto_condition', condition === 'auto' ? '1' : '0')

      // ── Per-class confidence / IoU overrides ──────────────────────────────
      // These tell the backend to apply the same per-class gates used by the
      // video workspace, preventing hair/clothing false-positives that slip
      // through the lenient global POST_WBF defaults.
      const cc = classConfRef.current
      const ni = nmsIouRef.current
      const wi = wbfIouRef.current

      if (cc && Object.keys(cc).length > 0) {
        form.append('class_conf', JSON.stringify(cc))
      }
      if (ni && Object.keys(ni).length > 0) {
        form.append('nms_iou', JSON.stringify(ni))
      }
      if (wi && Object.keys(wi).length > 0) {
        form.append('wbf_iou', JSON.stringify(wi))
      }

      const res = await fetch(API_DETECT_FRAME, {
        method: 'POST',
        body:   form,
        signal: abortRef.current.signal,
      })

      if (!res.ok) {
        throw new Error(`HTTP ${res.status} ${res.statusText}`)
      }

      const data = await res.json()

      if (!isActiveRef.current) return  // stopped while awaiting

      const latencyMs = Math.round(performance.now() - t0)
      const fps       = fpsEstimator.current.record(performance.now())

      onDetectionsRef.current?.(data.detections ?? [], {
        latencyMs,
        fps,
        condition: data.condition ?? condition,
      })
    } catch (err: any) {
      if (err?.name === 'AbortError') return  // intentional cancel — not an error
      onErrorRef.current?.(err instanceof Error ? err : new Error(String(err)))
    } finally {
      if (isActiveRef.current) {
        // Self-schedule: ensure at least minGapMs between completions
        const elapsed   = performance.now() - t0
        const nextDelay = Math.max(0, minGapMs - elapsed)
        timerRef.current = setTimeout(inferOnce, nextDelay)
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoRef, maxInferDim, condition, jpegQuality, minGapMs, getCaptureCanvas])
  //   ↑ callbacks and threshold dicts deliberately excluded — see comment above

  // ── Start / stop the loop when isActive changes ───────────────────────────
  useEffect(() => {
    if (isActive) {
      fpsEstimator.current.reset()
      // Start immediately (0 ms delay)
      timerRef.current = setTimeout(inferOnce, 0)
    } else {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
      abortRef.current?.abort()
      abortRef.current = null
    }

    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current)
      abortRef.current?.abort()
    }
  }, [isActive, inferOnce])

  return { infrDimsRef }
}
