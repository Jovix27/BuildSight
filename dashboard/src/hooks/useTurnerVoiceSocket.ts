import { useCallback, useEffect, useRef, useState } from 'react'

export type TurnerVoiceEventType =
  | 'state_update'
  | 'greeting'
  | 'transcript'
  | 'response'
  | 'audio'        // separate audio-only event
  | 'error'
  | 'health'
  | 'ping'

export interface TurnerVoiceEvent {
  type:            TurnerVoiceEventType
  session_id?:     string
  timestamp?:      string
  state?:          string
  detail?:         string
  text?:           string
  audio_b64?:      string   // base64 MP3 from ElevenLabs
  stage?:          string
  message?:        string
  code?:           string
  engine_running?: boolean
}

export type VoiceState      = 'idle' | 'listening' | 'thinking' | 'responding' | 'transcribing' | 'starting' | 'stopped' | 'error'
export type ConnectionState = 'connecting' | 'open' | 'closed' | 'retrying'

const WS_URL             = 'ws://localhost:8000/ws/turner-voice'
const INITIAL_BACKOFF_MS = 1000
const MAX_BACKOFF_MS     = 16000
const ERROR_DEBOUNCE_MS  = 2500

interface SocketOptions {
  /**
   * Fired when Turner sends a text response.
   * audio_b64 is provided when the backend bundled ElevenLabs TTS audio.
   * The handler is responsible for playing audio and resuming the voice session.
   */
  onResponse?: (text: string, audio_b64?: string) => void
}

export function useTurnerVoiceSocket(
  options?: SocketOptions,
  createSocket?: () => WebSocket,
) {
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')
  const [voiceState,      setVoiceState]      = useState<VoiceState>('idle')
  const [stateDetail,     setStateDetail]     = useState('Connecting to Turner AI...')
  const [transcript,      setTranscript]      = useState('')
  const [response,        setResponse]        = useState('')
  const [greeting,        setGreeting]        = useState('')
  const [error,           setError]           = useState('')
  const [engineRunning,   setEngineRunning]   = useState(false)

  const socketRef         = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<number | null>(null)
  const errorTimerRef     = useRef<number | null>(null)
  const backoffRef        = useRef(INITIAL_BACKOFF_MS)
  const disposedRef       = useRef(false)

  // Always-current callback ref — avoids stale closure issues
  const onResponseRef = useRef(options?.onResponse)
  useEffect(() => { onResponseRef.current = options?.onResponse }, [options?.onResponse])

  useEffect(() => {
    disposedRef.current = false

    const connect = () => {
      if (disposedRef.current) return

      setConnectionState('connecting')
      if (errorTimerRef.current) window.clearTimeout(errorTimerRef.current)

      const socket = createSocket ? createSocket() : new WebSocket(WS_URL)
      socketRef.current = socket

      socket.onopen = () => {
        if (disposedRef.current) { socket.close(); return }
        setConnectionState('open')
        setError('')
        setStateDetail('Turner AI connected')
        backoffRef.current = INITIAL_BACKOFF_MS
      }

      socket.onclose = () => {
        if (disposedRef.current) return
        setConnectionState('closed')
        setEngineRunning(false)
        setVoiceState('idle')

        errorTimerRef.current = window.setTimeout(() => {
          if (!disposedRef.current && socketRef.current?.readyState !== WebSocket.OPEN) {
            setError('Connection lost. Reconnecting...')
          }
        }, ERROR_DEBOUNCE_MS)

        const nextBackoff = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS)
        backoffRef.current = nextBackoff
        setConnectionState('retrying')
        setStateDetail(`Reconnecting in ${Math.round(nextBackoff / 1000)}s...`)
        reconnectTimerRef.current = window.setTimeout(connect, nextBackoff)
      }

      socket.onerror = () => {
        if (disposedRef.current) return
        if (errorTimerRef.current) window.clearTimeout(errorTimerRef.current)
        errorTimerRef.current = window.setTimeout(() => {
          if (!disposedRef.current && socketRef.current?.readyState !== WebSocket.OPEN) {
            setError('Turner voice socket error. Retrying...')
          }
        }, ERROR_DEBOUNCE_MS)
      }

      socket.onmessage = (event) => {
        if (disposedRef.current) return
        try {
          const payload = JSON.parse(event.data as string) as TurnerVoiceEvent

          if (payload.type === 'ping') return

          setError('')
          if (errorTimerRef.current) window.clearTimeout(errorTimerRef.current)

          switch (payload.type) {
            case 'state_update':
              setVoiceState((payload.state ?? 'idle') as VoiceState)
              setStateDetail(payload.detail ?? '')
              break

            case 'greeting':
              setGreeting(payload.text ?? '')
              break

            case 'transcript':
              setTranscript(payload.text ?? '')
              break

            case 'response': {
              const text    = payload.text ?? ''
              const audio   = payload.audio_b64
              setResponse(text)
              setVoiceState('responding')
              // Hand off to the component — it owns TTS and session reset
              onResponseRef.current?.(text, audio)
              break
            }

            case 'audio':
              // Standalone audio packet (sent separately from text response)
              if (payload.audio_b64) {
                onResponseRef.current?.('', payload.audio_b64)
              }
              break

            case 'error':
              setError(payload.message ?? 'Turner voice pipeline error')
              setVoiceState('idle')
              break

            case 'health':
              setVoiceState((payload.state ?? 'idle') as VoiceState)
              setEngineRunning(Boolean(payload.engine_running))
              setStateDetail(
                payload.detail ??
                (payload.engine_running ? 'Wake word active' : 'Text mode active'),
              )
              break
          }
        } catch {
          // Ignore malformed messages
        }
      }
    }

    connect()

    return () => {
      disposedRef.current = true
      if (reconnectTimerRef.current !== null) window.clearTimeout(reconnectTimerRef.current)
      if (errorTimerRef.current !== null) window.clearTimeout(errorTimerRef.current)
      socketRef.current?.close()
    }
  }, [createSocket])

  const sendAction = useCallback((action: string, extras?: Record<string, unknown>) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ action, ...extras }))
    }
  }, [])

  /** Send a text query to Turner — AI response + TTS handled via onResponse callback */
  const sendText = useCallback((text: string, context?: Record<string, unknown>) => {
    const trimmed = text.trim()
    if (!trimmed) return
    setTranscript(trimmed)
    setVoiceState('thinking')
    setStateDetail('Turner is thinking...')
    setResponse('')
    setError('')
    sendAction('push_to_talk_end', { text: trimmed, ...(context ? { context } : {}) })
  }, [sendAction])

  return {
    connectionState,
    voiceState,
    stateDetail,
    transcript,
    response,
    greeting,
    error,
    engineRunning,
    sendAction,
    sendText,
  }
}
