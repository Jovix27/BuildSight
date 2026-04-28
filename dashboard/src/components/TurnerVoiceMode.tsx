import React, { useMemo, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import TurnerOrb3D from './TurnerOrb3D'
import { useAudioAnalyzer } from '../hooks/useAudioAnalyzer'
import { useTurnerVoiceSocket } from '../hooks/useTurnerVoiceSocket'
import { useDetectionStore } from '../store/detectionStore'
import './TurnerVoiceMode.css'

type OrbState = 'idle' | 'presenting' | 'thinking' | 'speaking'

export const TurnerVoiceMode: React.FC = () => {
  const [question, setQuestion] = useState('')
  const [panelText, setPanelText] = useState('')
  const [uiError, setUiError] = useState('')
  const [showCC, setShowCC] = useState(true)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const metrics = useAudioAnalyzer(audioRef.current, null)
  const { isPaused, setPaused } = useDetectionStore()
  const {
    connectionState,
    voiceState,
    stateDetail,
    transcript,
    response,
    greeting,
    error: socketError,
    engineRunning,
    sendAction,
  } = useTurnerVoiceSocket()

  const isListening = voiceState === 'listening'
  const orbState = useMemo<OrbState>(() => {
    if (voiceState === 'responding') return 'speaking'
    if (voiceState === 'thinking' || voiceState === 'transcribing') return 'thinking'
    return 'idle'
  }, [voiceState])

  const displayText = panelText || response || transcript || greeting
  const displayError = uiError || socketError

  const playAudioB64 = (b64: string): Promise<void> => new Promise((resolve) => {
    try {
      const binary = atob(b64)
      const bytes = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
      const blob = new Blob([bytes], { type: 'audio/mpeg' })
      const url = URL.createObjectURL(blob)
      if (audioRef.current) {
        audioRef.current.pause()
        URL.revokeObjectURL(audioRef.current.src)
      }
      const audio = new Audio(url)
      audioRef.current = audio
      audio.onended = () => {
        URL.revokeObjectURL(url)
        resolve()
      }
      audio.onerror = () => resolve()
      void audio.play()
    } catch {
      resolve()
    }
  })

  const handlePresent = async () => {
    setPanelText('')
    setUiError('')
    try {
      const resp = await fetch('http://localhost:8000/api/ai/introduce')
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`)
      const data = await resp.json() as { script?: string; audio_b64?: string }
      if (data.script) setPanelText(data.script)
      if (data.audio_b64) {
        await playAudioB64(data.audio_b64)
      } else {
        setUiError('No audio returned — check ElevenLabs API key')
      }
    } catch (e) {
      setUiError(`Introduction failed: ${String(e)}`)
    }
  }

  const handleAsk = async () => {
    const q = question.trim()
    if (!q) return

    setUiError('')
    try {
      const resp = await fetch('http://localhost:8000/turner/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: q }),
      })
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`)
      const data = await resp.json() as { response?: string }
      setPanelText(data.response ?? '')
      setQuestion('')
    } catch (e) {
      setUiError(`Voice response failed: ${String(e)}`)
    }
  }

  const statusLabel: Record<string, string> = {
    starting: 'Starting backend voice engine...',
    wake_detected: 'Wake word detected',
    greeting: 'Greeting user',
    listening: 'Listening for request',
    transcribing: 'Transcribing request',
    thinking: 'Processing request',
    responding: 'Responding',
    idle: engineRunning ? 'Wake-word monitor active' : 'Voice engine offline',
    stopped: 'Voice engine stopped',
    error: 'Voice pipeline error',
  }

  return (
    <div className="turner-voice-mode">
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

      <div className="turner-voice-status-readout">
        <div className={`turner-voice-status turner-voice-status--${orbState} ${isListening ? 'listening' : ''}`}>
          <span className="turner-voice-status__dot" />
          {statusLabel[voiceState] ?? voiceState}
        </div>
        <div className="turner-voice-status-meta">
          <span>{connectionState.toUpperCase()}</span>
          <span>{stateDetail || (engineRunning ? 'Backend supervision active' : 'Waiting for backend')}</span>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {showCC && displayText && (
          <motion.div
            key="transcript"
            className="turner-voice-transcript"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            transition={{ type: 'spring', damping: 20 }}
          >
            <div className="transcript-hud-line" />
            <p>{displayText}</p>
          </motion.div>
        )}
        {displayError && (
          <motion.div
            key="error"
            className="turner-voice-error-card"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <div className="error-card-header">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
              </svg>
              <span>System Diagnostic</span>
            </div>
            <p>{displayError}</p>
            <button className="error-reset-btn" onClick={() => sendAction('status')}>
              Refresh Status
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="turner-voice-controls">
        <motion.button
          className={`turner-voice-btn turner-voice-btn--mic ${engineRunning ? 'active' : ''}`}
          onClick={() => sendAction('status')}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          title="Refresh backend voice status"
        >
          {engineRunning ? (
            <div className="mic-listening-waves">
              <span /><span /><span />
            </div>
          ) : (
            <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
          )}
        </motion.button>

        <motion.button
          className={`turner-voice-btn turner-voice-btn--cc ${showCC ? 'active' : ''}`}
          onClick={() => setShowCC(!showCC)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          title={showCC ? 'Hide Captions' : 'Show Captions'}
        >
          <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor">
            <path d="M19 4H5c-1.11 0-2 .9-2 2v12c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm-8 7H9.5v-.5h-2v3h2V13H11v1c0 .55-.45 1-1 1H7c-.55 0-1-.45-1-1v-4c0-.55.45-1 1-1h3c.55 0 1 .45 1 1v1zm7 0h-1.5v-.5h-2v3h2V13H18v1c0 .55-.45 1-1 1h-3c-.55 0-1-.45-1-1v-4c0-.55.45-1 1-1h3c.55 0 1 .45 1 1v1z" />
          </svg>
          <span className="btn-label">CC</span>
        </motion.button>

        <motion.button
          className="turner-voice-btn turner-voice-btn--present"
          onClick={() => void handlePresent()}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Present
        </motion.button>

        <motion.button
          className={`turner-voice-btn turner-voice-btn--pause ${isPaused ? 'active' : ''}`}
          onClick={() => setPaused(!isPaused)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          title={isPaused ? 'Resume Site Monitoring' : 'Pause Site Monitoring'}
        >
          {isPaused ? (
            <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
              <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
            </svg>
          )}
          <span className="btn-label">{isPaused ? 'RESUME' : 'PAUSE'}</span>
        </motion.button>

        <motion.button
          className="turner-voice-btn turner-voice-btn--present"
          onClick={() => sendAction('cancel')}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Stop
        </motion.button>
      </div>

      <div className="turner-voice-manual">
        <input
          type="text"
          className="turner-voice-manual__input"
          placeholder={isListening ? 'Backend listening for wake + speech...' : 'Type if voice fails...'}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void handleAsk() }}
        />
      </div>
    </div>
  )
}
