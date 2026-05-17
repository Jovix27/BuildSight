/**
 * useCameraDevices
 * ----------------
 * Enumerates available video-input devices and re-enumerates whenever
 * a device is plugged in or removed (USB hot-plug via devicechange event).
 *
 * Design notes:
 *  - Triggers a brief getUserMedia probe on first call so the browser
 *    populates device labels (labels are hidden before permission grant).
 *  - The probe stream is torn down immediately — this hook owns no stream.
 *  - Returns a stable `refresh` function so callers can force re-enumeration.
 */

import { useState, useEffect, useCallback, useRef } from 'react'

export interface CameraDevice {
  deviceId: string
  label:    string
  groupId:  string
}

interface UseCameraDevicesResult {
  devices:       CameraDevice[]
  isEnumerating: boolean
  refresh:       () => Promise<void>
}

export function useCameraDevices(): UseCameraDevicesResult {
  const [devices, setDevices]           = useState<CameraDevice[]>([])
  const [isEnumerating, setEnumerating] = useState(false)
  const permissionProbedRef             = useRef(false)

  const enumerate = useCallback(async () => {
    setEnumerating(true)
    try {
      // One-time permission probe so labels are populated.
      // Subsequent calls skip the probe because permission is already granted.
      if (!permissionProbedRef.current) {
        try {
          const probe = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
          probe.getTracks().forEach(t => t.stop())
          permissionProbedRef.current = true
        } catch {
          // Permission denied or no camera — still enumerate (deviceIds present, labels blank).
          permissionProbedRef.current = true
        }
      }

      const all = await navigator.mediaDevices.enumerateDevices()
      const videoInputs: CameraDevice[] = all
        .filter(d => d.kind === 'videoinput')
        .map((d, idx) => ({
          deviceId: d.deviceId,
          label:    d.label || `Camera ${idx + 1}`,
          groupId:  d.groupId,
        }))

      setDevices(videoInputs)
    } catch (err) {
      console.warn('[useCameraDevices] enumerate failed:', err)
    } finally {
      setEnumerating(false)
    }
  }, [])

  // Initial enumeration + hot-plug listener.
  useEffect(() => {
    enumerate()
    navigator.mediaDevices.addEventListener('devicechange', enumerate)
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', enumerate)
    }
  }, [enumerate])

  return { devices, isEnumerating, refresh: enumerate }
}
