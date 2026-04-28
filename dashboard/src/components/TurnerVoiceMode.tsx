import React, { useRef, useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import TurnerOrb3D from './TurnerOrb3D'
import { useAudioAnalyzer } from '../hooks/useAudioAnalyzer'
import { useDetectionStore } from '../store/detectionStore'
import './TurnerVoiceMode.css'

type OrbState = 'idle' | 'presenting' | 'thinking' | 'speaking'

// STT Type Definitions
declare global {
  interface Window {
    webkitSpeechRecognition: any;
    SpeechRecognition: any;
  }
}

export const TurnerVoiceMode: React.FC = () => {
  const [orbState, setOrbState] = useState<OrbState>('idle')
  const [transcript, setTranscript] = useState('')
  const [question, setQuestion] = useState('')
  const [error, setError] = useState('')
  const [showCC, setShowCC] = useState(true)
  const [isListening, setIsListening] = useState(false)
  
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const recognitionRef = useRef<any>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const [micStream, setMicStream] = useState<MediaStream | null>(null)

  // Audio Analysis
  const metrics = useAudioAnalyzer(audioRef.current, micStream)
  
  // Global Detection State (for HUD)
  const { isPaused, setPaused } = useDetectionStore()

  // Initialize Speech Recognition
  const initRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (SpeechRecognition) {
      if (recognitionRef.current) {
        recognitionRef.current.onstart = null
        recognitionRef.current.onend = null
        recognitionRef.current.onresult = null
        recognitionRef.current.onerror = null
      }

      const recognition = new SpeechRecognition()
      recognition.continuous = false
      recognition.interimResults = false
      recognition.lang = 'en-US'

      recognition.onstart = () => {
        setIsListening(true)
        setError('')
      }
      recognition.onend = () => setIsListening(false)
      recognition.onresult = (event: any) => {
        const text = event.results[0][0].transcript
        setQuestion(text)
        void handleAsk(text)
      }
      recognition.onerror = (event: any) => {
        console.error('STT Error:', event.error)
        if (event.error === 'network') {
          setError('Network Error: Browser STT service unreachable. Switching to Robust Backend Mode...')
          setTimeout(() => {
            setError('')
            void startRobustRecording()
          }, 2000)
        } else if (event.error === 'not-allowed') {
          setError('Microphone access denied. Please enable it in browser settings.')
        } else if (event.error === 'no-speech') {
          // Auto-retry once for no-speech to keep the session alive
          console.log('No speech detected, retrying...')
          setTimeout(() => {
            if (!isListening) recognition.start()
          }, 500)
        } else {
          setError(`Speech error: ${event.error}`)
        }
        setIsListening(false)
      }
      recognitionRef.current = recognition
    } else {
      console.warn('Web Speech API not supported in this browser.')
      setError('Browser STT not supported. Use the Mic to record directly to backend.')
    }
  }

  const startRobustRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      setMicStream(stream)
      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder
      audioChunksRef.current = []

      recorder.onstart = () => {
        setIsListening(true)
        setError('')
      }

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        setIsListening(false)
        setMicStream(null)
        setOrbState('thinking')
        
        try {
          const formData = new FormData()
          formData.append('audio', audioBlob, 'voice.webm')
          
          const resp = await fetch('http://localhost:8000/api/ai/transcribe', {
            method: 'POST',
            body: formData,
          })
          
          if (!resp.ok) throw new Error('Transcription failed')
          const data = await resp.json()
          if (data.text) {
            setQuestion(data.text)
            void handleAsk(data.text)
          } else {
            setOrbState('idle')
          }
        } catch (err) {
          console.error('Robust STT Error:', err)
          setError('Backend transcription failed. Please check server status.')
          setOrbState('idle')
        }
        
        // Cleanup stream
        stream.getTracks().forEach(track => track.stop())
      }

      recorder.start()
    } catch (err) {
      console.error('Failed to start recorder:', err)
      setError('Could not access microphone for robust mode.')
    }
  }

  useEffect(() => {
    initRecognition()
  }, [])

  const toggleListening = () => {
    if (isListening) {
      if (recorderRef.current && recorderRef.current.state === 'recording') {
        recorderRef.current.stop()
      } else {
        recognitionRef.current?.stop()
      }
    } else {
      setError('')
      // If we already know there's a network issue or browser doesn't support it, use robust mode directly
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
      if (!SpeechRecognition || error.includes('Network Error')) {
        void startRobustRecording()
        return
      }

      try {
        recognitionRef.current?.start()
      } catch (e) {
        console.error('STT Start Error:', e)
        initRecognition()
        setTimeout(() => recognitionRef.current?.start(), 100)
      }
    }
  }

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
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`)
      const data = await resp.json() as { script?: string; audio_b64?: string }
      if (data.script) setTranscript(data.script)
      if (data.audio_b64) {
        setOrbState('idle')
        await playAudioB64(data.audio_b64)
      } else {
        setOrbState('idle')
        setError('No audio returned — check ElevenLabs API key')
      }
    } catch (e) {
      setOrbState('idle')
      setError(`Introduction failed: ${String(e)}`)
    }
  }

  const handleAsk = async (textOverride?: string) => {
    const q = textOverride || question.trim()
    if (!q || orbState !== 'idle') return
    if (!textOverride) setQuestion('')
    
    setOrbState('thinking')
    setTranscript('')
    setError('')
    try {
      const resp = await fetch('http://localhost:8000/api/ai/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: q }),
      })
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`)
      const data = await resp.json() as { response?: string; audio_b64?: string }
      if (data.response) setTranscript(data.response)
      if (data.audio_b64) {
        setOrbState('idle')
        await playAudioB64(data.audio_b64)
      } else {
        setOrbState('idle')
      }
    } catch (e) {
      setOrbState('idle')
      setError(`Voice response failed: ${String(e)}`)
    }
  }

  const statusLabel: Record<OrbState, string> = {
    idle: isListening ? 'Listening...' : 'Ready',
    presenting: 'Preparing introduction...',
    thinking: 'Processing...',
    speaking: 'Speaking',
  }

  return (
    <div className="turner-voice-mode">
      {/* ── Neural Orb (3D Integrated) ── */}
      <div className="turner-voice-avatar-wrap">
        <TurnerOrb3D 
          size={500}
          amplitude={metrics.amplitude}
          state={orbState === 'idle' && isListening ? 'listening' : orbState}
        />
        <div className={`turner-voice-status-ring turner-voice-status-ring--${orbState} ${isListening ? 'listening' : ''}`} />
        <div className="avatar-scan-line" />
      </div>

      {/* ── Visual Backdrop ── */}
      <div className={`voice-backdrop-pulse ${orbState}`} />

      {/* ── Visual Feedback ── */}
      <div className="turner-voice-status-readout">
        <div className={`turner-voice-status turner-voice-status--${orbState} ${isListening ? 'listening' : ''}`}>
          <span className="turner-voice-status__dot" />
          {statusLabel[orbState]}
        </div>
      </div>

      {/* ── Transcript (CC) ── */}
      <AnimatePresence mode="wait">
        {showCC && transcript && (
          <motion.div
            key="transcript"
            className="turner-voice-transcript"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            transition={{ type: 'spring', damping: 20 }}
          >
            <div className="transcript-hud-line" />
            <p>{transcript}</p>
          </motion.div>
        )}
        {error && (
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
            <p>{error}</p>
            <button className="error-reset-btn" onClick={() => error.includes('Network Error') ? startRobustRecording() : initRecognition()}>
              {error.includes('Network Error') ? 'Switch to Robust Mode' : 'Reset Connection'}
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Tactical Controls ── */}
      <div className="turner-voice-controls">
        <motion.button
          className={`turner-voice-btn turner-voice-btn--mic ${isListening ? 'active' : ''}`}
          onClick={toggleListening}
          disabled={orbState !== 'idle' && !isListening}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          title={isListening ? 'Stop Listening' : 'Start Talking'}
        >
          {isListening ? (
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
          disabled={orbState !== 'idle' || isListening}
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
      </div>

      {/* ── Manual Q&A Fallback ── */}
      <div className="turner-voice-manual">
        <input
          type="text"
          className="turner-voice-manual__input"
          placeholder={isListening ? "Listening..." : "Type if voice fails..."}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void handleAsk() }}
          disabled={orbState !== 'idle' || isListening}
        />
      </div>
    </div>
  )
}
