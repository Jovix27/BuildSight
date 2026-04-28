import { useEffect, useRef, useState } from 'react'

export type TurnerVoiceEventType =
  | 'state_update'
  | 'greeting'
  | 'transcript'
  | 'response'
  | 'error'
  | 'health'
  | 'ping'

export interface TurnerVoiceEvent {
  type: TurnerVoiceEventType
  session_id?: string
  timestamp?: string
  state?: string
  detail?: string
  text?: string
  stage?: string
  message?: string
  code?: string
  engine_running?: boolean
}

const WS_URL = 'ws://localhost:8000/ws/turner-voice'

const INITIAL_BACKOFF_MS = 500
const MAX_BACKOFF_MS = 8000

export function useTurnerVoiceSocket(createSocket?: () => WebSocket) {
  const [connectionState, setConnectionState] = useState<'connecting' | 'open' | 'closed' | 'retrying'>('connecting')
  const [voiceState, setVoiceState] = useState('starting')
  const [stateDetail, setStateDetail] = useState('')
  const [transcript, setTranscript] = useState('')
  const [response, setResponse] = useState('')
  const [greeting, setGreeting] = useState('')
  const [error, setError] = useState('')
  const [engineRunning, setEngineRunning] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<number | null>(null)
  const backoffRef = useRef(INITIAL_BACKOFF_MS)
  const disposedRef = useRef(false)

  useEffect(() => {
    disposedRef.current = false

    const connect = () => {
      if (disposedRef.current) return
      setConnectionState('connecting')
      const socket = createSocket ? createSocket() : new WebSocket(WS_URL)
      socketRef.current = socket

      socket.onopen = () => {
        if (disposedRef.current) {
          socket.close()
          return
        }
        setConnectionState('open')
        backoffRef.current = INITIAL_BACKOFF_MS
      }

      socket.onclose = (e) => {
        if (disposedRef.current) return
        setConnectionState('closed')
        setEngineRunning(false)
        const nextBackoff = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS)
        backoffRef.current = nextBackoff
        setConnectionState('retrying')
        reconnectTimerRef.current = window.setTimeout(connect, nextBackoff)
      }

      socket.onerror = () => {
        if (disposedRef.current) return
        setError('Turner voice socket error. Retrying backend connection...')
      }

      socket.onmessage = (event) => {
        if (disposedRef.current) return
        const payload = JSON.parse(event.data) as TurnerVoiceEvent

        if (payload.type === 'ping') return

        if (payload.type === 'state_update') {
          setVoiceState(payload.state ?? 'idle')
          setStateDetail(payload.detail ?? '')
          setError('')
          return
        }

        if (payload.type === 'greeting') {
          setGreeting(payload.text ?? '')
          return
        }

        if (payload.type === 'transcript') {
          setTranscript(payload.text ?? '')
          return
        }

        if (payload.type === 'response') {
          setResponse(payload.text ?? '')
          return
        }

        if (payload.type === 'error') {
          setError(payload.message ?? 'Turner voice pipeline error')
          return
        }

        if (payload.type === 'health') {
          setVoiceState(payload.state ?? 'idle')
          setEngineRunning(Boolean(payload.engine_running))
        }
      }
    }

    connect()

    return () => {
      disposedRef.current = true
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
      }
      socketRef.current?.close()
    }
  }, [createSocket])

  return {
    connectionState,
    voiceState,
    stateDetail,
    transcript,
    response,
    greeting,
    error,
    engineRunning,
    sendAction: (action: string) => {
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({ action }))
      }
    },
  }
}