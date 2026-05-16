/**
 * BuildSight Global Detection State Store
 * =========================================
 * Persists background detection state across ALL tab navigation.
 * Uses Zustand for lightweight global state.
 * Updated via a single native WebSocket connection to the FastAPI
 * backend (/ws/detection) that is opened once at app startup and
 * never closed on tab change.
 */

import { create } from 'zustand'

export interface WorkerPosition {
  worker_id:      string
  confidence:     number
  lat:            number
  lng:            number
  utm_e:          number
  utm_n:          number
  has_helmet:     boolean | null
  has_vest:       boolean | null
  ppe_compliant:  boolean
  violation_type: string[]
  pixel_x:        number
  pixel_y:        number
  zone_id:        string | null
  zone_name:      string | null
  source?:        string
}

export interface ZoneViolation {
  type:       string
  severity:   'CRITICAL' | 'WARNING' | 'INFO'
  worker_id?: string
  zone_id:    string
  zone_name:  string
  details:    string
  lat:        number
  lng:        number
  timestamp:  number
}

export interface PeakRiskMoment {
  time:  number
  score: number
  type:  'PPE_VIOLATION' | 'CROWD_DANGER' | 'RESTRICTED_ENTRY'
}

interface DetectionState {
  // Connection
  isConnected:     boolean
  _socket:         WebSocket | null

  // Video / service state
  isRunning:       boolean
  isPaused:        boolean
  currentFrame:    number
  totalFrames:     number
  progressPercent: number

  // Detection metrics
  workerCount:     number
  sceneCondition:  string
  fps:             number
  processingRate:  number
  latencyMs:       number
  frameCount:      number

  // Background Video Session (Local persistence)
  videoFile:       File | null
  sessionId:       string | null
  currentTime:     number
  videoDuration:   number

  // Spatial state — drives GeoAI map live markers
  workerPositions: WorkerPosition[]
  zoneOccupancy:   Record<string, string[]>
  violations:      ZoneViolation[]
  peakRiskMoments: PeakRiskMoment[]
  roiPoly:         [number, number][] | null

  // Raw frame detections (for DetectionSidebar display)
  detections:      any[]
  ppeWorkers:      any[]
  // Inference frame dimensions (from backend broadcast)
  frameWidth:      number
  frameHeight:     number
  condition:       string

  // Actions
  connect:         () => void
  disconnect:      () => void
  requestSnapshot: () => void
  /** Directly set isRunning (called by video-upload inference to trigger GeoAI live mode) */
  setRunning:         (running: boolean) => void
  setWorkerPositions: (workers: WorkerPosition[]) => void
  setViolations:      (violations: ZoneViolation[]) => void
  setPeakRiskMoments: (moments: PeakRiskMoment[]) => void
  setWorkerCount:     (count: number) => void
  setFPS:             (fps: number) => void
  setProcessingRate:  (fps: number) => void
  setLatencyMs:       (ms: number) => void
  setSceneCondition:  (cond: string) => void
  setRoiPoly:         (poly: [number, number][] | null) => void
  setDetections:      (dets: any[]) => void
  setPpeWorkers:      (workers: any[]) => void

  // Session Actions
  setVideoSession:    (file: File | null) => void
  setPaused:          (paused: boolean) => void
  setPlayback:        (time: number, duration: number, progress: number) => void
  resetSession:       () => void
}

const WS_URL = 'ws://localhost:8000/ws/detection'

export const useDetectionStore = create<DetectionState>((set, get) => ({
  isConnected:     false,
  _socket:         null,
  isRunning:       false,
  isPaused:        false,
  currentFrame:    0,
  totalFrames:     0,
  progressPercent: 0,
  workerCount:     0,
  sceneCondition:  'S1_normal',
  fps:             0,
  processingRate:  0,
  latencyMs:       0,
  frameCount:      0,
  videoFile:       null,
  sessionId:       null,
  currentTime:     0,
  videoDuration:   0,
  workerPositions: [],
  zoneOccupancy:   {},
  violations:      [],
  peakRiskMoments: [],
  roiPoly:         null,
  detections:      [],
  ppeWorkers:      [],
  frameWidth:      640,
  frameHeight:     480,
  condition:       'S1_normal',

  connect: () => {
    const existing = get()._socket
    if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
      return
    }

    const socket = new WebSocket(WS_URL)

    socket.addEventListener('open', () => {
      set({ isConnected: true })
      socket.send('request_state_snapshot')
    })

    socket.addEventListener('close', () => {
      set({ isConnected: false, _socket: null })
      // Auto-reconnect after 2 s
      setTimeout(() => {
        if (!get()._socket) get().connect()
      }, 2000)
    })

    socket.addEventListener('error', () => {
      // close event follows automatically — reconnect handled there
    })

    socket.addEventListener('message', (event) => {
      let data: any
      try { data = JSON.parse(event.data as string) } catch { return }

      const type: string = data.type ?? ''

      if (type === 'detection_update') {
        const total = data.total_frames ?? 0
        const frame = data.frame_number ?? 0
        set({
          workerCount:     data.worker_count      ?? 0,
          sceneCondition:  data.scene_condition   ?? 'S1_normal',
          fps:             data.fps               ?? 0,
          latencyMs:       data.latency_ms        ?? 0,
          currentFrame:    frame,
          totalFrames:     total,
          progressPercent: total > 0 ? Math.round((frame / total) * 100) : 0,
          workerPositions: data.worker_positions  ?? [],
          zoneOccupancy:   data.zone_occupancy    ?? {},
          violations:      data.violations        ?? [],
          detections:      data.detections        ?? [],
          frameWidth:      data.frame_width       ?? get().frameWidth,
          frameHeight:     data.frame_height      ?? get().frameHeight,
          frameCount:      get().frameCount + 1,
        })
      } else if (type === 'detection_state_snapshot') {
        const total = data.total_frames ?? 0
        const frame = data.current_frame ?? 0
        set({
          isRunning:       data.is_running        ?? false,
          isPaused:        data.is_paused         ?? false,
          currentFrame:    frame,
          totalFrames:     total,
          progressPercent: total > 0 ? Math.round((frame / total) * 100) : 0,
          workerCount:     data.worker_count      ?? 0,
          sceneCondition:  data.scene_condition   ?? 'S1_normal',
          fps:             data.fps               ?? 0,
          latencyMs:       data.latency_ms        ?? 0,
          workerPositions: data.worker_positions  ?? [],
          zoneOccupancy:   data.zone_occupancy    ?? {},
          violations:      data.violations        ?? [],
        })
      } else if (type === 'detection_started') {
        set({ isRunning: true, isPaused: false, totalFrames: data.total_frames ?? 0 })
      } else if (type === 'detection_paused') {
        set({ isPaused: true })
      } else if (type === 'detection_resumed') {
        set({ isPaused: false })
      } else if (type === 'detection_stopped') {
        set({
          isRunning:       false,
          workerPositions: [],
          zoneOccupancy:   {},
          violations:      [],
          progressPercent: 0,
        })
      }
    })

    set({ _socket: socket })
  },

  disconnect: () => {
    get()._socket?.close()
    set({ _socket: null, isConnected: false })
  },

  requestSnapshot: () => {
    const socket = get()._socket
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send('request_state_snapshot')
    }
  },

  setRunning: (running: boolean) => {
    set({ isRunning: running })
  },

  setWorkerPositions: (workers: WorkerPosition[]) => {
    set({ workerPositions: workers })
  },

  setViolations: (violations: ZoneViolation[]) => {
    set({ violations })
  },
  
  setPeakRiskMoments: (moments: PeakRiskMoment[]) => {
    set({ peakRiskMoments: moments })
  },

  setWorkerCount: (count: number) => {
    set({ workerCount: count })
  },

  setFPS: (fps: number) => {
    set({ fps })
  },

  setProcessingRate: (fps: number) => {
    set({ processingRate: fps })
  },

  setLatencyMs: (ms: number) => {
    set({ latencyMs: ms })
  },

  setSceneCondition: (cond: string) => {
    set({ sceneCondition: cond, condition: cond })
  },

  setRoiPoly: (poly: [number, number][] | null) => {
    set({ roiPoly: poly })
  },

  setDetections: (dets: any[]) => {
    set({ detections: dets })
  },

  setPpeWorkers: (workers: any[]) => {
    set({ ppeWorkers: workers })
  },

  setVideoSession: (file: File | null) => {
    if (!file) {
      set({ videoFile: null, sessionId: null, isRunning: false, isPaused: false })
      return
    }
    set({ 
      videoFile: file, 
      sessionId: `v-${Date.now()}`, 
      isRunning: true, 
      isPaused: false,
      progressPercent: 0,
      currentTime: 0
    })
  },

  setPaused: (paused: boolean) => {
    set({ isPaused: paused })
  },

  setPlayback: (time: number, duration: number, progress: number) => {
    set({ currentTime: time, videoDuration: duration, progressPercent: progress })
  },

  resetSession: () => {
    set({
      videoFile: null,
      sessionId: null,
      isRunning: false,
      isPaused: false,
      progressPercent: 0,
      currentTime: 0,
      workerPositions: [],
      violations: [],
      peakRiskMoments: [],
      detections: [],
      ppeWorkers: [],
      workerCount: 0,
      fps: 0,
      processingRate: 0,
    })
  },
}))
