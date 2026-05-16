export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'

export interface DetectionLike {
  class: string
  confidence: number
  box: [number, number, number, number]
  has_helmet?: boolean
  has_vest?: boolean
}

export interface HeatmapPoint {
  x: number
  y: number
  time: number
  type: 'worker' | 'violation'
  value: number
  riskLevel?: RiskLevel
}

export interface PeakRiskMoment {
  time: number
  score: number
  type: 'PPE_VIOLATION' | 'CROWD_DANGER' | 'RESTRICTED_ENTRY'
}

function isWorkerDetection(detection: DetectionLike): boolean {
  return detection.class === 'worker' || detection.class === 'person'
}

function isViolationDetection(detection: DetectionLike): boolean {
  return isWorkerDetection(detection) && (!detection.has_helmet || !detection.has_vest)
}

function roundRiskScore(score: number): number {
  return Math.round(score * 10) / 10
}

function averagePairDistance(points: Array<{ x: number; y: number }>): number {
  if (points.length < 2) return 0

  let total = 0
  let pairs = 0
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dx = points[i].x - points[j].x
      const dy = points[i].y - points[j].y
      total += Math.hypot(dx, dy)
      pairs++
    }
  }

  return pairs > 0 ? total / pairs : 0
}

export function detectionsToHeatmapPoints(
  detections: DetectionLike[],
  frameWidth: number,
  frameHeight: number,
  time: number,
): HeatmapPoint[] {
  if (!frameWidth || !frameHeight) return []

  return detections.flatMap((detection) => {
    if (!isWorkerDetection(detection) || !detection.box) return []

    const [x1, y1, x2, y2] = detection.box
    const point: HeatmapPoint = {
      x: ((x1 + x2) / 2) / frameWidth,
      y: ((y1 + y2) / 2) / frameHeight,
      time,
      type: isViolationDetection(detection) ? 'violation' : 'worker',
      value: isViolationDetection(detection) ? 0.9 : 0.28,
      riskLevel: isViolationDetection(detection) ? 'HIGH' : 'LOW',
    }

    return [point]
  })
}

export function buildPeakRiskMoment(
  detections: DetectionLike[],
  time: number,
): PeakRiskMoment | null {
  const violations = detections.filter(isViolationDetection)
  if (violations.length === 0) return null

  const centers = violations
    .filter(v => !!v.box)
    .map(({ box }) => ({
      x: (box[0] + box[2]) / 2,
      y: (box[1] + box[3]) / 2,
    }))
  const avgDistance = averagePairDistance(centers)
  const densityBoost = violations.length > 1 ? Math.max(0, 1.5 - avgDistance / 180) : 0

  return {
    time,
    score: roundRiskScore(violations.length + densityBoost),
    type: 'PPE_VIOLATION',
  }
}

export function mergePeakRiskMoments(
  existing: PeakRiskMoment[],
  candidate: PeakRiskMoment | null,
  maxMoments = 5,
  dedupeWindowSeconds = 2,
): PeakRiskMoment[] {
  if (!candidate) return existing

  const next = [...existing]
  const nearbyIndex = next.findIndex((moment) => (
    Math.abs(moment.time - candidate.time) < dedupeWindowSeconds
      && moment.type === candidate.type
  ))

  if (nearbyIndex >= 0) {
    if (candidate.score >= next[nearbyIndex].score) {
      next[nearbyIndex] = candidate
    }
  } else {
    next.push(candidate)
  }

  return next
    .sort((a, b) => (b.score - a.score) || (a.time - b.time))
    .slice(0, maxMoments)
}
