/**
 * useTurnerVoiceSession
 *
 * Single master hook for the entire Turner AI voice pipeline.
 * Owns state machine, TTS, conversation history, and session lifecycle.
 *
 * State machine:
 *   idle → standby → wake_detected → listening → processing → speaking
 *          ↑                                                        │
 *          └──── conversation (15 s follow-up, no re-wake) ────────┘
 *          └──── standby (window expired, wake word needed again) ──┘
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { useVoiceInput }        from './useVoiceInput'
import { useTurnerVoiceSocket } from './useTurnerVoiceSocket'
import type { ConnectionState } from './useTurnerVoiceSocket'

// ── Public types ──────────────────────────────────────────────────────────────

export type SessionPhase =
  | 'idle'
  | 'standby'        // continuous wake-word monitoring
  | 'wake_detected'  // "Hey Turner" heard — brief visual confirmation
  | 'listening'      // capturing user command
  | 'conversation'   // post-response window: speak without re-waking (15 s)
  | 'processing'     // query sent to AI, waiting for response
  | 'speaking'       // TTS playing

export interface ConversationTurn {
  role:      'user' | 'assistant'
  content:   string
  timestamp: number
}

export interface TurnerVoiceSessionState {
  phase:            SessionPhase
  liveText:         string     // interim transcript shown while listening
  displayText:      string     // Turner's response (or user's query while processing)
  error:            string
  isSpeaking:       boolean
  isSupported:      boolean
  connectionState:  ConnectionState
  stateDetail:      string
  engineRunning:    boolean
  conversationTurns: ConversationTurn[]
}

export interface TurnerVoiceSessionActions {
  triggerPTT:   () => void
  cancelAll:    () => void
  sendQuery:    (text: string) => void
  clearHistory: () => void
}

// ── Constants ─────────────────────────────────────────────────────────────────

const CONVERSATION_WINDOW_MS = 15_000  // stay in conversation mode after each response
const MAX_HISTORY_TURNS      = 8       // last 4 exchanges kept for context
const TTS_HTTP_TIMEOUT_MS    = 9_000
const CONVO_RESUME_DELAY_MS  = 600    // brief pause before re-listening after response

// ── Helpers ───────────────────────────────────────────────────────────────────

function pickBrowserVoice(): SpeechSynthesisVoice | undefined {
  if (!window.speechSynthesis) return undefined
  const voices = window.speechSynthesis.getVoices()
  return (
    voices.find(v => v.lang === 'en-US' && /microsoft david|microsoft guy|microsoft mark/i.test(v.name)) ??
    voices.find(v => v.lang === 'en-US' && /google us english/i.test(v.name)) ??
    voices.find(v => v.lang === 'en-US' && !v.localService) ??
    voices.find(v => v.lang.startsWith('en'))
  )
}

function decodeB64Audio(b64: string): HTMLAudioElement {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0))
  const blob  = new Blob([bytes], { type: 'audio/mpeg' })
  return new Audio(URL.createObjectURL(blob))
}

// ── Main hook ─────────────────────────────────────────────────────────────────

export function useTurnerVoiceSession(): TurnerVoiceSessionState & TurnerVoiceSessionActions {

  const [phase,       setPhase]       = useState<SessionPhase>('idle')
  const [liveText,    setLiveText_]   = useState('')
  const [displayText, setDisplayText] = useState('')
  const [error,       setError]       = useState('')
  const [isSpeaking,  setIsSpeaking]  = useState(false)
  const [turns,       setTurns]       = useState<ConversationTurn[]>([])

  // Stable refs — avoids stale closure issues across async chains
  const phaseRef         = useRef<SessionPhase>('idle')
  const historyRef       = useRef<ConversationTurn[]>([])
  const audioRef         = useRef<HTMLAudioElement | null>(null)
  const convTimerRef     = useRef<number | null>(null)
  const startPTTRef      = useRef<() => void>(() => {})
  const resumeStandbyRef = useRef<() => void>(() => {})
  const sendTextRef      = useRef<(text: string) => void>(() => {})

  // ── Phase management ────────────────────────────────────────────────────────

  const go = useCallback((next: SessionPhase) => {
    phaseRef.current = next
    setPhase(next)
  }, [])

  // ── History management ──────────────────────────────────────────────────────

  const addTurn = useCallback((role: 'user' | 'assistant', content: string) => {
    const turn: ConversationTurn = { role, content, timestamp: Date.now() }
    historyRef.current = [...historyRef.current.slice(-(MAX_HISTORY_TURNS - 1)), turn]
    setTurns([...historyRef.current])
  }, [])

  // ── Timer management ────────────────────────────────────────────────────────

  const clearConvTimer = useCallback(() => {
    if (convTimerRef.current !== null) {
      window.clearTimeout(convTimerRef.current)
      convTimerRef.current = null
    }
  }, [])

  // ── Conversation mode ───────────────────────────────────────────────────────
  // After Turner responds, stay in conversation mode for CONVERSATION_WINDOW_MS.
  // During this window the user can speak a follow-up without the wake phrase.

  const enterConversationMode = useCallback(() => {
    go('conversation')
    clearConvTimer()

    // Start listening for follow-up after a brief pause (lets audio settle)
    setTimeout(() => {
      if (phaseRef.current === 'conversation') {
        startPTTRef.current()
      }
    }, CONVO_RESUME_DELAY_MS)

    // After the window, return to wake-word standby
    convTimerRef.current = window.setTimeout(() => {
      if (phaseRef.current === 'conversation') {
        go('standby')
        resumeStandbyRef.current()
      }
    }, CONVERSATION_WINDOW_MS)
  }, [go, clearConvTimer])

  // ── TTS pipeline ────────────────────────────────────────────────────────────
  // 3 layers — ElevenLabs inline → ElevenLabs HTTP → Browser speechSynthesis.
  // Always ends by calling enterConversationMode() regardless of which layer fired.

  const playB64Audio = useCallback((b64: string): Promise<void> =>
    new Promise((resolve) => {
      try {
        if (audioRef.current) {
          audioRef.current.pause()
          URL.revokeObjectURL(audioRef.current.src)
        }
        const audio = decodeB64Audio(b64)
        audioRef.current = audio
        audio.onended = () => { URL.revokeObjectURL(audio.src); resolve() }
        audio.onerror = () => resolve()
        void audio.play()
      } catch { resolve() }
    }),
  [])

  const speakBrowser = useCallback((text: string): Promise<void> =>
    new Promise((resolve) => {
      if (!text || !window.speechSynthesis) { resolve(); return }
      window.speechSynthesis.cancel()
      const u    = new SpeechSynthesisUtterance(text)
      u.lang     = 'en-US'
      u.rate     = 0.90
      u.pitch    = 0.85
      u.volume   = 1.0
      u.onend    = () => resolve()
      u.onerror  = () => resolve()
      const pref = pickBrowserVoice()
      if (pref) u.voice = pref
      window.speechSynthesis.speak(u)
    }),
  [])

  const runTTSPipeline = useCallback(async (text: string, audio_b64?: string) => {
    setIsSpeaking(true)
    go('speaking')

    // Layer 1 — ElevenLabs audio bundled in the WebSocket response
    if (audio_b64) {
      await playB64Audio(audio_b64)
      setIsSpeaking(false)
      enterConversationMode()
      return
    }

    // Layer 2 — Fetch ElevenLabs from /api/ai/tts endpoint
    if (text) {
      try {
        const res = await fetch('http://localhost:8000/api/ai/tts', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ message: text }),
          signal:  AbortSignal.timeout(TTS_HTTP_TIMEOUT_MS),
        })
        if (res.ok) {
          const data = await res.json() as { audio_b64?: string }
          if (data.audio_b64) {
            await playB64Audio(data.audio_b64)
            setIsSpeaking(false)
            enterConversationMode()
            return
          }
        }
      } catch { /* fall through to browser TTS */ }
    }

    // Layer 3 — Browser Web Speech API (always works, zero dependencies)
    if (text) await speakBrowser(text)
    setIsSpeaking(false)
    enterConversationMode()
  }, [go, playB64Audio, speakBrowser, enterConversationMode])

  // ── Voice input callbacks ───────────────────────────────────────────────────

  const onWake = useCallback(() => {
    clearConvTimer()
    setDisplayText('')
    setError('')
    go('wake_detected')
  }, [go, clearConvTimer])

  const onCommand = useCallback((text: string) => {
    clearConvTimer()
    if (!text.trim()) return  // empty = conversation timeout expired; VoiceInputManager already resumed standby
    addTurn('user', text)
    setDisplayText(text)
    go('processing')
    sendTextRef.current(text)
  }, [go, clearConvTimer, addTurn])

  const onVoiceError = useCallback((msg: string) => {
    setError(msg)
    if (phaseRef.current !== 'speaking' && phaseRef.current !== 'processing') {
      go('standby')
    }
  }, [go])

  // ── Voice input hook ────────────────────────────────────────────────────────

  const {
    phase: voicePhase, liveText: rawLiveText, isSupported,
    startStandby, startPTT, resumeStandby, stopAll,
  } = useVoiceInput({ onWake, onCommand, onError: onVoiceError })

  // Mirror live interim text
  useEffect(() => { setLiveText_(rawLiveText) }, [rawLiveText])

  // Keep action refs current
  useEffect(() => { startPTTRef.current      = startPTT      }, [startPTT])
  useEffect(() => { resumeStandbyRef.current = resumeStandby }, [resumeStandby])

  // Bridge voicePhase → sessionPhase for listening state
  useEffect(() => {
    if (voicePhase === 'listening' && phaseRef.current !== 'conversation') {
      go('listening')
    }
  }, [voicePhase, go])

  // ── WebSocket / onResponse callback ────────────────────────────────────────

  const onResponse = useCallback(async (text: string, audio_b64?: string) => {
    if (text) {
      addTurn('assistant', text)
      setDisplayText(text)
    }
    await runTTSPipeline(text, audio_b64)
  }, [addTurn, runTTSPipeline])

  const {
    connectionState, stateDetail, engineRunning,
    error: socketError, sendAction, sendText,
  } = useTurnerVoiceSocket({ onResponse })

  // Build sendText with conversation history injected
  useEffect(() => {
    sendTextRef.current = (text: string) => {
      const history = historyRef.current
        .slice(-MAX_HISTORY_TURNS)
        .map(t => ({ role: t.role, content: t.content }))
      sendText(text, { history })
    }
  }, [sendText])

  // ── Auto-start standby ──────────────────────────────────────────────────────

  const standbyStarted = useRef(false)
  const isConnected    = connectionState === 'open'

  useEffect(() => {
    if (isConnected && !standbyStarted.current && isSupported) {
      standbyStarted.current = true
      startStandby()
      go('standby')
    }
    if (!isConnected && standbyStarted.current) {
      standbyStarted.current = false
      stopAll()
      go('idle')
    }
  }, [isConnected, isSupported, startStandby, stopAll, go])

  // ── Cleanup ─────────────────────────────────────────────────────────────────

  useEffect(() => () => {
    stopAll()
    clearConvTimer()
    if (audioRef.current) { audioRef.current.pause(); audioRef.current = null }
    window.speechSynthesis?.cancel()
  }, [stopAll, clearConvTimer])

  // ── Public actions ──────────────────────────────────────────────────────────

  const triggerPTT = useCallback(() => {
    clearConvTimer()
    setDisplayText('')
    setError('')
    go('listening')
    startPTT()
  }, [go, clearConvTimer, startPTT])

  const cancelAll = useCallback(() => {
    if (audioRef.current) { audioRef.current.pause(); audioRef.current = null }
    window.speechSynthesis?.cancel()
    clearConvTimer()
    sendAction('cancel')
    setIsSpeaking(false)
    go('standby')
    resumeStandby()
  }, [go, clearConvTimer, sendAction, resumeStandby])

  const sendQuery = useCallback((text: string) => {
    if (!text.trim()) return
    addTurn('user', text)
    setDisplayText(text)
    go('processing')
    sendTextRef.current(text)
  }, [go, addTurn])

  const clearHistory = useCallback(() => {
    historyRef.current = []
    setTurns([])
  }, [])

  // ── Return ───────────────────────────────────────────────────────────────────

  return {
    // State
    phase:             phase,
    liveText:          rawLiveText,
    displayText,
    error:             error || (connectionState === 'retrying' ? socketError : ''),
    isSpeaking,
    isSupported,
    connectionState,
    stateDetail,
    engineRunning,
    conversationTurns: turns,

    // Actions
    triggerPTT,
    cancelAll,
    sendQuery,
    clearHistory,
  }
}
