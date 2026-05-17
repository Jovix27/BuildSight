/**
 * useBoxTracker
 * -------------
 * Lightweight IoU-based multi-object tracker with EMA (exponential moving
 * average) position smoothing. Assigns stable integer IDs across frames to
 * eliminate the visual flicker caused by per-frame ID reassignment.
 *
 * Algorithm per update() call:
 *  1. For each existing track, find the unmatched detection of the same class
 *     with the highest IoU (greedy matching, threshold IOU_MATCH_THRESH).
 *  2. Matched tracks: update box + confidence, reset missed counter.
 *  3. Unmatched tracks: increment missed counter.
 *  4. Drop tracks whose missed counter exceeds MAX_MISSED.
 *  5. Unmatched detections: create new tracks.
 *
 * EMA smoothing is applied in the *renderer* (LiveSurveillance.tsx), not here,
 * so this hook stays pure and renderer-agnostic. The track exposes both the
 * raw inference-space box (for coordinate transforms) and smoothed display-space
 * coords (sx1/sy1/sx2/sy2) that the renderer updates each RAF tick.
 *
 * No external dependencies — pure TypeScript geometry.
 */

import { useRef, useCallback } from 'react'

// ── Public types ──────────────────────────────────────────────────────────────

export interface RawDetection {
  class:       string
  confidence:  number
  /** [x1, y1, x2, y2] in inference-frame pixel space */
  box:         [number, number, number, number]
  has_helmet?: boolean
  has_vest?:   boolean
}

/**
 * A tracked object. The renderer reads `box` + infer dims for coordinate
 * transforms, and maintains sx1/sy1/sx2/sy2 for EMA-smoothed display coords.
 */
export interface TrackedBox {
  id:          number
  cls:         string
  confidence:  number
  /** Raw inference-space box — updated each inference cycle */
  box:         [number, number, number, number]
  /** EMA-smoothed display-space coords — updated each RAF tick by renderer */
  sx1: number; sy1: number; sx2: number; sy2: number
  /** False until the renderer has initialised sx/sy from the first mapping */
  initialized: boolean
  /** Frames since last matched detection (>MAX_MISSED → dropped) */
  missed:      number
  has_helmet?: boolean
  has_vest?:   boolean
}

// ── Constants ─────────────────────────────────────────────────────────────────

/** Minimum IoU to associate a detection to an existing track. */
const IOU_MATCH_THRESH = 0.18

/** Carry a track for this many missed inference cycles before dropping it. */
const MAX_MISSED = 3

// ── Monotonic track ID counter ────────────────────────────────────────────────
let _globalTrackId = 0

// ── Geometry helper ───────────────────────────────────────────────────────────

function boxIou(a: number[], b: number[]): number {
  const ix1 = Math.max(a[0], b[0])
  const iy1 = Math.max(a[1], b[1])
  const ix2 = Math.min(a[2], b[2])
  const iy2 = Math.min(a[3], b[3])
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1)
  if (inter === 0) return 0
  const aArea = (a[2] - a[0]) * (a[3] - a[1])
  const bArea = (b[2] - b[0]) * (b[3] - b[1])
  const union = aArea + bArea - inter
  return inter / Math.max(union, 1e-6)
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export interface UseBoxTrackerResult {
  /** Current live tracks — read by the RAF renderer */
  tracksRef: React.MutableRefObject<TrackedBox[]>
  /** Feed new raw detections; returns the updated track list */
  update: (detections: RawDetection[]) => TrackedBox[]
  /** Reset all tracks (call on camera stop/switch) */
  clear: () => void
}

export function useBoxTracker(): UseBoxTrackerResult {
  const tracksRef = useRef<TrackedBox[]>([])

  const update = useCallback((detections: RawDetection[]): TrackedBox[] => {
    const tracks = tracksRef.current
    const matched = new Set<number>()  // detection indices already assigned

    // ── Step 1–2: Match existing tracks ──────────────────────────────────────
    for (const track of tracks) {
      let bestIou = IOU_MATCH_THRESH
      let bestIdx = -1

      for (let i = 0; i < detections.length; i++) {
        if (matched.has(i)) continue
        if (detections[i].class !== track.cls) continue
        const iou = boxIou(track.box as number[], detections[i].box as number[])
        if (iou > bestIou) { bestIou = iou; bestIdx = i }
      }

      if (bestIdx >= 0) {
        const d = detections[bestIdx]
        track.box        = d.box
        track.confidence = d.confidence
        track.has_helmet = d.has_helmet
        track.has_vest   = d.has_vest
        track.missed     = 0
        matched.add(bestIdx)
      } else {
        track.missed++
      }
    }

    // ── Step 3–4: Drop stale tracks ───────────────────────────────────────────
    const alive = tracks.filter(t => t.missed <= MAX_MISSED)

    // ── Step 5: Spawn new tracks for unmatched detections ─────────────────────
    for (let i = 0; i < detections.length; i++) {
      if (matched.has(i)) continue
      const d = detections[i]
      alive.push({
        id:          _globalTrackId++,
        cls:         d.class,
        confidence:  d.confidence,
        box:         d.box,
        sx1: 0, sy1: 0, sx2: 0, sy2: 0,
        initialized: false,
        missed:      0,
        has_helmet:  d.has_helmet,
        has_vest:    d.has_vest,
      })
    }

    tracksRef.current = alive
    return alive
  }, [])

  const clear = useCallback(() => {
    tracksRef.current = []
  }, [])

  return { tracksRef, update, clear }
}
