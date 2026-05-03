import React, { useMemo, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import TurnerOrb3D from './TurnerOrb3D'
import { useAudioAnalyzer }        from '../hooks/useAudioAnalyzer'
import { useTurnerVoiceSession }   from '../hooks/useTurnerVoiceSession'
import { useDetectionStore }       from '../store/detectionStore'
import type { SessionPhase }       from '../hooks/useTurnerVoiceSession'
import './TurnerVoiceMode.css'

// ── Orb state mapping ─────────────────────────────────────────────────────────

type OrbState = 'idle' | 'presenting' | 'thinking' | 'speaking'

function orbStateFor(phase: SessionPhase, isSpeaking: boolean): OrbState {
  if (isSpeaking || phase === 'speaking')   return 'speaking'
  if (phase === 'processing')               return 'thinking'
  if (phase === 'listening' || phase === 'conversation' || phase === 'wake_detected')
                                            return 'presenting'
  return 'idle'
}

// ── Status label ──────────────────────────────────────────────────────────────

const PHASE_LABELS: Record<SessionPhase, string> = {
  idle:          'Offline',
  standby:       'Say "Hey Turner" to wake me up',
  wake_detected: 'Hey — I heard you',
  listening:     "Go ahead, I'm listening...",
  conversation:  "I'm here — ask your follow-up",
  processing:    "Give me a sec...",
  speaking:      'Turner is speaking',
}

// ── Component ─────────────────────────────────────────────────────────────────

export const TurnerVoiceMode: React.FC = () => {
  const [showCC,    setShowCC]    = useState(true)
  const [showHistory, setShowHistory] = useState(false)
  const [question,  setQuestion]  = useState('')
  const [pttActive, setPttActive] = useState(false)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const metrics  = useAudioAnalyzer(audioRef.current, null)
  const { isPaused, setPaused } = useDetectionStore()

  const {
    phase,
    liveText,
    displayText,
    error,
    isSpeaking,
    isSupported,
    connectionState,
    stateDetail,
    engineRunning,
    conversationTurns,
    triggerPTT,
    cancelAll,
    sendQuery,
    clearHistory,
  } = useTurnerVoiceSession()

  const isConnected  = connectionState === 'open'
  const isListening  = phase === 'listening' || phase === 'conversation' || pttActive
  const isProcessing = phase === 'processing'
  const orbState     = useMemo(() => orbStateFor(phase, isSpeaking), [phase, isSpeaking])

  // What to show in the transcript panel
  const panelText = liveText || displayText

  const badge = {
    connecting: { label: 'CONNECTING', color: '#f59e0b' },
    open:       { label: 'CONNECTED',  color: '#10b981' },
    retrying:   { label: 'RETRYING',   color: '#ef4444' },
    closed:     { label: 'OFFLINE',    color: '#6b7280' },
  }[connectionState]

  // ── PTT click handler ─────────────────────────────────────────────────────

  const handlePttClick = () => {
    if (!isConnected || isProcessing || isSpeaking) return
    if (pttActive) {
      setPttActive(false)
      // Force-stop: session manager handles the commit
    } else {
      setPttActive(true)
      triggerPTT()
    }
  }

  // Reset pttActive when session leaves listening state
  React.useEffect(() => {
    if (phase !== 'listening') setPttActive(false)
  }, [phase])

  // ── Text input fallback ───────────────────────────────────────────────────

  const handleAsk = () => {
    const q = question.trim()
    if (!q || isProcessing) return
    sendQuery(q)
    setQuestion('')
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="turner-voice-mode">

      {/* Grid background handled by CSS ::before */}

      {/* ── Orb ── */}
      <div className="turner-voice-avatar-wrap">
        <TurnerOrb3D
          size={500}
          amplitude={metrics.amplitude}
          state={isListening ? 'listening' : orbState}
        />
        <div className={`turner-voice-status-ring turner-voice-status-ring--${orbState} ${isListening ? 'listening' : ''}`} />
        <div className="avatar-scan-line" />
      </div>

      <div className={`voice-backdrop-pulse ${isListening ? 'listening' : orbState}`} />

      {/* ── Top indicator bar ── */}
      <AnimatePresence mode="wait">
        {phase === 'standby' && (
          <motion.div
            key="standby"
            className="turner-wake-indicator"
            initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
          >
            <span className="turner-wake-dot" />
            <span className="turner-wake-label">Listening for wake word</span>
          </motion.div>
        )}

        {(phase === 'wake_detected' || phase === 'listening') && (
          <motion.div
            key="listening"
            className="turner-wake-indicator turner-wake-indicator--active"
            initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
          >
            <span className="turner-wake-dot turner-wake-dot--active" />
            <span className="turner-wake-label">
              {phase === 'wake_detected' ? 'Wake word detected...' : 'Speak now...'}
            </span>
          </motion.div>
        )}

        {phase === 'conversation' && (
          <motion.div
            key="conversation"
            className="turner-wake-indicator turner-wake-indicator--conversation"
            initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
          >
            <span className="turner-wake-dot turner-wake-dot--conversation" />
            <span className="turner-wake-label">Conversation mode — ask your follow-up</span>
          </motion.div>
        )}

        {phase === 'speaking' && (
          <motion.div
            key="speaking"
            className="turner-wake-indicator turner-wake-indicator--speaking"
            initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
          >
            <span className="turner-wake-dot turner-wake-dot--speaking" />
            <span className="turner-wake-label">Turner is speaking</span>
          </motion.div>
        )}

        {!isSupported && (
          <motion.div
            key="unsupported"
            className="turner-wake-indicator turner-wake-indicator--warn"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          >
            <span className="turner-wake-label">Chrome or Edge required for voice · text input active</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Status readout ── */}
      <div className="turner-voice-status-readout">
        <div className={`turner-voice-status turner-voice-status--${orbState} ${isListening ? 'listening' : ''}`}>
          <span className="turner-voice-status__dot" />
          {PHASE_LABELS[phase] ?? phase}
        </div>
        <div className="turner-voice-status-meta">
          <span style={{ color: badge.color }}>● {badge.label}</span>
          <span>{stateDetail || (engineRunning ? 'Wake word active' : 'Text mode active')}</span>
        </div>
      </div>

      {/* ── Transcript / response panel ── */}
      <AnimatePresence mode="wait">
        {showCC && panelText && (
          <motion.div
            key="transcript"
            className={`turner-voice-transcript ${liveText ? 'transcript--live' : ''}`}
            initial={{ opacity: 0, scale: 0.97, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 1.02, y: -6 }}
            transition={{ type: 'spring', damping: 22, stiffness: 280 }}
          >
            <div className="transcript-hud-line" />
            <p>{panelText}</p>
          </motion.div>
        )}

        {error && (
          <motion.div
            key="error"
            className="turner-voice-error-card"
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
          >
            <div className="error-card-header">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
              </svg>
              <span>System Alert</span>
            </div>
            <p>{error}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Conversation history drawer ── */}
      <AnimatePresence>
        {showHistory && conversationTurns.length > 0 && (
          <motion.div
            className="turner-history-drawer"
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 40 }}
            transition={{ type: 'spring', damping: 28 }}
          >
            <div className="turner-history-header">
              <span>Conversation</span>
              <button className="turner-history-clear" onClick={clearHistory}>Clear</button>
            </div>
            <div className="turner-history-scroll">
              {conversationTurns.map((t, i) => (
                <div key={i} className={`turner-history-turn turner-history-turn--${t.role}`}>
                  <span className="turner-history-role">{t.role === 'user' ? 'You' : 'Turner'}</span>
                  <p className="turner-history-content">{t.content}</p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Controls ── */}
      <div className="turner-voice-controls">

        {/* PTT — click to start, phase auto-resets when done */}
        <motion.button
          id="turner-ptt-btn"
          className={`turner-voice-btn turner-voice-btn--ptt ${pttActive || phase === 'listening' ? 'ptt-active' : ''} ${!isConnected ? 'disabled' : ''}`}
          onClick={handlePttClick}
          whileHover={isConnected ? { scale: 1.08 } : {}}
          whileTap={isConnected   ? { scale: 0.93 } : {}}
          title={pttActive ? 'Listening — click to stop' : 'Click to speak'}
          disabled={!isConnected || isProcessing || isSpeaking}
        >
          {pttActive || phase === 'listening'
            ? <div className="mic-listening-waves"><span /><span /><span /></div>
            : (
              <svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
              </svg>
            )
          }
          <span className="btn-label">{pttActive || phase === 'listening' ? 'LISTENING' : 'TALK'}</span>
        </motion.button>

        {/* CC toggle */}
        <motion.button
          className={`turner-voice-btn turner-voice-btn--cc ${showCC ? 'active' : ''}`}
          onClick={() => setShowCC(!showCC)}
          whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
          title={showCC ? 'Hide captions' : 'Show captions'}
        >
          <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor">
            <path d="M19 4H5c-1.11 0-2 .9-2 2v12c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm-8 7H9.5v-.5h-2v3h2V13H11v1c0 .55-.45 1-1 1H7c-.55 0-1-.45-1-1v-4c0-.55.45-1 1-1h3c.55 0 1 .45 1 1v1zm7 0h-1.5v-.5h-2v3h2V13H18v1c0 .55-.45 1-1 1h-3c-.55 0-1-.45-1-1v-4c0-.55.45-1 1-1h3c.55 0 1 .45 1 1v1z" />
          </svg>
          <span className="btn-label">CC</span>
        </motion.button>

        {/* History toggle */}
        <motion.button
          className={`turner-voice-btn ${showHistory ? 'active' : ''}`}
          onClick={() => setShowHistory(!showHistory)}
          whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
          title="Conversation history"
        >
          <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor">
            <path d="M13 3a9 9 0 0 0-9 9H1l3.89 3.89.07.14L9 12H6c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7c-1.93 0-3.68-.79-4.94-2.06l-1.42 1.42A8.954 8.954 0 0 0 13 21a9 9 0 0 0 0-18zm-1 5v5l4.28 2.54.72-1.21-3.5-2.08V8H12z" />
          </svg>
          <span className="btn-label">LOG</span>
        </motion.button>

        {/* Pause monitoring */}
        <motion.button
          className={`turner-voice-btn turner-voice-btn--pause ${isPaused ? 'active' : ''}`}
          onClick={() => setPaused(!isPaused)}
          whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
          title={isPaused ? 'Resume monitoring' : 'Pause monitoring'}
        >
          {isPaused
            ? <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
            : <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg>
          }
          <span className="btn-label">{isPaused ? 'RESUME' : 'PAUSE'}</span>
        </motion.button>

        {/* Stop / cancel */}
        <motion.button
          className="turner-voice-btn turner-voice-btn--stop"
          onClick={cancelAll}
          whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
          title="Stop Turner"
        >
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M6 6h12v12H6z" />
          </svg>
          <span className="btn-label">STOP</span>
        </motion.button>
      </div>

      {/* ── Text input fallback ── */}
      <div className="turner-voice-manual">
        <input
          id="turner-text-input"
          type="text"
          className="turner-voice-manual__input"
          placeholder={
            phase === 'listening' || phase === 'conversation'
              ? "Turner's listening..."
              : 'Type here or say "Hey Turner"'
          }
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleAsk() }}
          disabled={isProcessing || isSpeaking}
        />
      </div>
    </div>
  )
}
