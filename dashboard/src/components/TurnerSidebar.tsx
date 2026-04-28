import { useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { TurnerAssistant } from './TurnerAssistant'
import TurnerOrb3D from './TurnerOrb3D'

type SidebarTab = 'chat' | 'voice'
type OrbState = 'idle' | 'presenting' | 'thinking' | 'speaking'

interface TurnerSidebarProps {
  onExpand: () => void
  onClose: () => void
}

/* ─── Compact voice panel (sidebar-safe, column layout) ───────────────────── */
function SidebarVoice() {
  const [orbState, setOrbState] = useState<OrbState>('idle')
  const [transcript, setTranscript] = useState('')
  const [question, setQuestion] = useState('')
  const [error, setError] = useState('')
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const playAudioB64 = (b64: string): Promise<void> =>
    new Promise((resolve) => {
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
        setOrbState('speaking')
        audio.onended = () => { setOrbState('idle'); URL.revokeObjectURL(url); resolve() }
        audio.onerror = () => { setOrbState('idle'); resolve() }
        void audio.play()
      } catch { setOrbState('idle'); resolve() }
    })

  const handlePresent = async () => {
    if (orbState !== 'idle') return
    setOrbState('presenting')
    setTranscript('')
    setError('')
    try {
      const resp = await fetch('http://localhost:8000/api/ai/introduce')
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json() as { script?: string; audio_b64?: string }
      if (data.script) setTranscript(data.script)
      if (data.audio_b64) { setOrbState('idle'); await playAudioB64(data.audio_b64) }
      else { setOrbState('idle'); setError('No audio — check ElevenLabs key') }
    } catch (e) { setOrbState('idle'); setError(String(e)) }
  }

  const handleAsk = async (q = question.trim()) => {
    if (!q || orbState !== 'idle') return
    setQuestion('')
    setOrbState('thinking')
    setTranscript('')
    setError('')
    try {
      const resp = await fetch('http://localhost:8000/api/ai/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: q }),
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json() as { response?: string; audio_b64?: string }
      if (data.response) setTranscript(data.response)
      if (data.audio_b64) { setOrbState('idle'); await playAudioB64(data.audio_b64) }
      else setOrbState('idle')
    } catch (e) { setOrbState('idle'); setError(String(e)) }
  }

  const statusLabel: Record<OrbState, string> = {
    idle: 'Ready', presenting: 'Preparing...', thinking: 'Processing...', speaking: 'Speaking',
  }

  return (
    <div className="tsb-voice">
      <div className="tsb-voice__orb">
        <TurnerOrb3D
          size={200}
          amplitude={orbState === 'speaking' ? 0.45 : 0}
          state={orbState === 'idle' ? 'idle' : orbState}
        />
      </div>

      <div className={`tsb-voice__status tsb-voice__status--${orbState}`}>
        <span className="tsb-voice__dot" />
        {statusLabel[orbState]}
      </div>

      <AnimatePresence>
        {(transcript || error) && (
          <motion.div
            className={`tsb-voice__transcript ${error ? 'tsb-voice__transcript--error' : ''}`}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <p>{error || transcript}</p>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.button
        className="tsb-voice__present-btn"
        onClick={() => void handlePresent()}
        disabled={orbState !== 'idle'}
        whileHover={{ scale: orbState === 'idle' ? 1.02 : 1 }}
        whileTap={{ scale: 0.97 }}
      >
        <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor">
          <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4Zm0 14.5a6.5 6.5 0 0 0 6.5-6.5.5.5 0 0 1 1 0 7.5 7.5 0 0 1-7 7.48V19h2a.5.5 0 0 1 0 1h-5a.5.5 0 0 1 0-1h2v-2.02A7.5 7.5 0 0 1 4.5 9a.5.5 0 0 1 1 0 6.5 6.5 0 0 0 6.5 6.5Z" />
        </svg>
        {orbState === 'presenting' ? 'Preparing...' : 'Present Turner'}
      </motion.button>

      <div className="tsb-voice__qa">
        <input
          type="text"
          className="tsb-voice__input"
          placeholder="Ask Turner anything..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void handleAsk() }}
          disabled={orbState !== 'idle'}
        />
        <button
          className="tsb-voice__send"
          onClick={() => void handleAsk()}
          disabled={!question.trim() || orbState !== 'idle'}
        >
          Ask
        </button>
      </div>
    </div>
  )
}

/* ─── Turner logo mark (hex neural icon) ─────────────────────────────────── */
function TurnerLogoMark() {
  return (
    <svg className="tsb-logo-mark" viewBox="0 0 40 40" fill="none" aria-hidden="true">
      {/* Hex outline */}
      <polygon
        points="20,2 36,11 36,29 20,38 4,29 4,11"
        stroke="currentColor"
        strokeWidth="1.2"
        fill="none"
        opacity="0.6"
      />
      {/* Neural core glow */}
      <circle cx="20" cy="20" r="7" fill="currentColor" opacity="0.15" />
      <circle cx="20" cy="20" r="4" fill="currentColor" opacity="0.3" />
      <circle cx="20" cy="20" r="2" fill="currentColor" />
      {/* Neural lines */}
      <line x1="20" y1="13" x2="20" y2="2"  stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
      <line x1="20" y1="27" x2="20" y2="38" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
      <line x1="12" y1="16" x2="4"  y2="11" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
      <line x1="28" y1="16" x2="36" y2="11" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
      <line x1="12" y1="24" x2="4"  y2="29" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
      <line x1="28" y1="24" x2="36" y2="29" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
      {/* Node dots */}
      <circle cx="20" cy="2"  r="1.5" fill="currentColor" opacity="0.7" />
      <circle cx="20" cy="38" r="1.5" fill="currentColor" opacity="0.7" />
      <circle cx="4"  cy="11" r="1.5" fill="currentColor" opacity="0.7" />
      <circle cx="36" cy="11" r="1.5" fill="currentColor" opacity="0.7" />
      <circle cx="4"  cy="29" r="1.5" fill="currentColor" opacity="0.7" />
      <circle cx="36" cy="29" r="1.5" fill="currentColor" opacity="0.7" />
    </svg>
  )
}

/* ─── Diagonal expand icon ───────────────────────────────────────────────── */
function ExpandIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <path d="M10 2h4v4M14 2l-6 6M6 14H2v-4M2 14l6-6" />
    </svg>
  )
}

/* ─── Main sidebar component ─────────────────────────────────────────────── */
export function TurnerSidebar({ onExpand, onClose }: TurnerSidebarProps) {
  const [tab, setTab] = useState<SidebarTab>('chat')

  const sidebar = (
    <AnimatePresence>
      <motion.aside
        className="tsb"
        key="turner-sidebar"
        initial={{ x: '100%', opacity: 0.6 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: '100%', opacity: 0 }}
        transition={{ type: 'spring', stiffness: 320, damping: 34 }}
      >
        {/* ── Header ── */}
        <div className="tsb__header">
          <div className="tsb__brand">
            <TurnerLogoMark />
            <div className="tsb__brand-text">
              <span className="tsb__brand-name">TURNER</span>
              <span className="tsb__brand-sub">AI Supervisor</span>
            </div>
          </div>
          <div className="tsb__header-actions">
            <button
              type="button"
              className="tsb__icon-btn tsb__icon-btn--expand"
              onClick={onExpand}
              title="Open full Turner AI view"
              aria-label="Expand to full view"
            >
              <ExpandIcon />
            </button>
            <button
              type="button"
              className="tsb__icon-btn tsb__icon-btn--close"
              onClick={onClose}
              aria-label="Close sidebar"
            >
              <svg viewBox="0 0 16 16" width="12" height="12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                <path d="M2 2l12 12M14 2L2 14" />
              </svg>
            </button>
          </div>
        </div>

        {/* ── Tab bar ── */}
        <div className="tsb__tabs" role="tablist">
          <button
            role="tab"
            type="button"
            className={`tsb__tab ${tab === 'chat' ? 'tsb__tab--active' : ''}`}
            onClick={() => setTab('chat')}
          >
            <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor">
              <path d="M14 1H2a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h2v3l4-3h6a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1ZM4 6h8v1H4V6Zm0-2h8v1H4V4Z" />
            </svg>
            Chat
          </button>
          <button
            role="tab"
            type="button"
            className={`tsb__tab ${tab === 'voice' ? 'tsb__tab--active' : ''}`}
            onClick={() => setTab('voice')}
          >
            <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor">
              <path d="M8 1a3 3 0 0 1 3 3v4a3 3 0 0 1-6 0V4a3 3 0 0 1 3-3Zm5 6.5a.5.5 0 0 1 1 0A6 6 0 0 1 8.5 13.48V15h1.5a.5.5 0 0 1 0 1h-4a.5.5 0 0 1 0-1H7.5v-1.52A6 6 0 0 1 2 7.5a.5.5 0 0 1 1 0 5 5 0 0 0 10 0Z" />
            </svg>
            Voice
          </button>
        </div>

        {/* ── Body ── */}
        <div className="tsb__body">
          {tab === 'chat' && <TurnerAssistant />}
          {tab === 'voice' && <SidebarVoice />}
        </div>
      </motion.aside>
    </AnimatePresence>
  )

  return typeof document !== 'undefined' ? createPortal(sidebar, document.body) : null
}
