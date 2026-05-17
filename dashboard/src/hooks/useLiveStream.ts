/**
 * useLiveStream
 * -------------
 * Opens and manages a single MediaStream for a given camera deviceId.
 * Returns a videoRef that always points at the correct <video> element.
 *
 * Design notes:
 *  - Caller attaches the ref to a <video> element; this hook wires srcObject.
 *  - Stopping / switching is handled atomically: old tracks are killed before
 *    a new getUserMedia call is made, preventing concurrent stream leaks.
 *  - Cleanup on unmount is guaranteed via the returned stop() and useEffect.
 *  - No React state updates happen inside async paths after unmount
 *    (guarded by the `alive` flag pattern).
 */

import { useState, useRef, useCallback, useEffect } from 'react'

export type StreamStatus = 'idle' | 'requesting' | 'active' | 'error'

export interface LiveStreamResult {
  status:   StreamStatus
  error:    string | null
  videoRef: React.RefObject<HTMLVideoElement | null>
  start:    (deviceId: string) => Promise<void>
  stop:     () => void
  /** true while the video element has live frames (readyState >= 2) */
  hasVideo: boolean
}

const IDEAL_WIDTH  = 1280
const IDEAL_HEIGHT = 720
const IDEAL_FPS    = 30

function buildConstraints(deviceId: string): MediaStreamConstraints {
  return {
    video: deviceId
      ? {
          deviceId:  { exact: deviceId },
          width:     { ideal: IDEAL_WIDTH,  max: 1920 },
          height:    { ideal: IDEAL_HEIGHT, max: 1080 },
          frameRate: { ideal: IDEAL_FPS,    max: 30   },
        }
      : {
          width:     { ideal: IDEAL_WIDTH  },
          height:    { ideal: IDEAL_HEIGHT },
          frameRate: { ideal: IDEAL_FPS    },
        },
    audio: false,
  }
}

function friendlyError(err: any): string {
  switch (err?.name) {
    case 'NotAllowedError':
      return 'Camera permission denied — allow access in your browser settings.'
    case 'NotFoundError':
      return 'No camera found for this device ID.'
    case 'NotReadableError':
      return 'Camera is busy or unavailable. Try closing other apps using it.'
    case 'OverconstrainedError':
      return 'Camera does not support the requested resolution or frame rate.'
    default:
      return `Camera error: ${err?.message ?? 'Unknown error'}`
  }
}

export function useLiveStream(): LiveStreamResult {
  const [status, setStatus] = useState<StreamStatus>('idle')
  const [error,  setError]  = useState<string | null>(null)
  const [hasVideo, setHasVideo] = useState(false)

  const videoRef  = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      // Tear down stream on unmount
      streamRef.current?.getTracks().forEach(t => t.stop())
    }
  }, [])

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    const v = videoRef.current
    if (v) {
      v.srcObject = null
      v.load()
    }
    if (mountedRef.current) {
      setStatus('idle')
      setError(null)
      setHasVideo(false)
    }
  }, [])

  const start = useCallback(async (deviceId: string) => {
    // Kill any existing stream before opening a new one
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null

    if (mountedRef.current) {
      setStatus('requesting')
      setError(null)
      setHasVideo(false)
    }

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia(buildConstraints(deviceId))
    } catch (err) {
      if (mountedRef.current) {
        setStatus('error')
        setError(friendlyError(err))
      }
      return
    }

    // Component may have unmounted during the async call
    if (!mountedRef.current) {
      stream.getTracks().forEach(t => t.stop())
      return
    }

    streamRef.current = stream

    const video = videoRef.current
    if (!video) {
      // videoRef not mounted yet — clean up and surface error
      stream.getTracks().forEach(t => t.stop())
      streamRef.current = null
      setStatus('error')
      setError('Video element not available. Please retry.')
      return
    }

    video.srcObject = stream

    try {
      await video.play()
    } catch (playErr) {
      // play() can fail if the component unmounted during await
      if (!mountedRef.current) return
      setStatus('error')
      setError(`Could not start playback: ${(playErr as Error).message}`)
      return
    }

    if (!mountedRef.current) return

    setStatus('active')
    setHasVideo(true)

    // Listen for track-end events (USB disconnect, permission revoked)
    stream.getVideoTracks().forEach(track => {
      track.addEventListener('ended', () => {
        if (mountedRef.current) {
          setStatus('error')
          setError('Camera disconnected. Reconnect the device and retry.')
          setHasVideo(false)
        }
        streamRef.current = null
      })
    })
  }, [])

  return { status, error, videoRef, start, stop, hasVideo }
}
