/**
 * useVoiceInput — Unified voice input hook for Turner AI
 *
 * Manages a dual-mode voice pipeline:
 *   standby → (wake word | PTT click) → listening → submitted → standby
 *
 * Uses a single VoiceInputManager class (stored in a ref) to avoid
 * stale-closure issues across the wake-word → command-capture state machine.
 */
import { useEffect, useRef, useState } from 'react'

export type VoicePhase =
  | 'idle'          // not running
  | 'standby'       // continuous recognition, waiting for wake phrase
  | 'wake_detected' // wake phrase heard, brief confirmation moment
  | 'listening'     // capturing the command utterance
  | 'submitted'     // text handed to consumer, waiting for resumeStandby()

const WAKE_RE = /\b(hey\s+turner|hi\s+turner|okay\s+turner|turner)\b/i
const WAKE_COOLDOWN_MS  = 3000   // prevent double-fire within 3 s
const COMMIT_SILENCE_MS = 2200   // commit after N ms of silence post-speech
const COMMIT_MAX_MS     = 9000   // hard cap on command duration

interface VoiceInputOptions {
  onWake:    () => void
  onCommand: (text: string) => void
  onError?:  (msg: string) => void
}

// ── VoiceInputManager ────────────────────────────────────────────────────────

class VoiceInputManager {
  // Injected callbacks — updated each render so they're always fresh
  onWake:    () => void = () => {}
  onCommand: (text: string) => void = () => {}
  onError:   (msg: string) => void = () => {}

  // UI state setters (injected from hook)
  setPhase:    (p: VoicePhase) => void = () => {}
  setLiveText: (t: string) => void = () => {}

  private standbyRec: SpeechRecognition | null = null
  private commandRec:  SpeechRecognition | null = null
  private active = false
  private phase: VoicePhase = 'idle'
  private commandBuf = ''
  private lastWakeMs = 0
  private commitTimer: ReturnType<typeof window.setTimeout> | null = null
  private restartPending = false

  readonly isSupported = !!(
    (window as unknown as Record<string, unknown>).SpeechRecognition ||
    (window as unknown as Record<string, unknown>).webkitSpeechRecognition
  )

  private SR(): typeof SpeechRecognition | undefined {
    return (
      (window as unknown as Record<string, unknown>).SpeechRecognition ||
      (window as unknown as Record<string, unknown>).webkitSpeechRecognition
    ) as typeof SpeechRecognition | undefined
  }

  // ── Internal helpers ────────────────────────────────────────────────────────

  private setP(p: VoicePhase) {
    this.phase = p
    this.setPhase(p)
  }

  private clearTimer() {
    if (this.commitTimer !== null) {
      window.clearTimeout(this.commitTimer)
      this.commitTimer = null
    }
  }

  private stopStandbyRec() {
    try { this.standbyRec?.stop() } catch { /* ignore */ }
    this.standbyRec = null
  }

  private stopCommandRec() {
    try { this.commandRec?.stop() } catch { /* ignore */ }
    this.commandRec = null
  }

  private commit() {
    this.clearTimer()
    this.stopCommandRec()
    const text = this.commandBuf.trim()
    this.commandBuf = ''
    this.setLiveText('')

    if (text) {
      this.setP('submitted')
      this.onCommand(text)
    } else {
      // Nothing captured — silently return to standby
      this.resumeStandby()
    }
  }

  // ── Command recognition (one-shot after wake or PTT) ───────────────────────

  private startCommandListening() {
    const SR = this.SR()
    if (!SR || !this.active) return

    this.stopStandbyRec()
    this.commandBuf = ''
    this.setLiveText('')
    this.setP('listening')

    const rec = new SR() as SpeechRecognition
    this.commandRec = rec
    rec.continuous      = false   // auto-stops after VAD silence
    rec.interimResults  = true
    rec.lang            = 'en-US'
    rec.maxAlternatives = 1

    rec.onresult = (ev: SpeechRecognitionEvent) => {
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const result = ev.results[i]
        const raw    = result[0].transcript.trim()

        // Strip wake phrase if it bled into the command
        const stripped = raw.replace(WAKE_RE, '').replace(/^\s*,?\s*/, '').trim()
        const text = stripped || raw

        if (result.isFinal) {
          if (text) {
            this.commandBuf = text
            this.setLiveText(text)
          }
          this.clearTimer()
          // onend will fire and commit
        } else {
          if (text) {
            this.commandBuf = text
            this.setLiveText(text)
          }
          // Reset silence timer
          this.clearTimer()
          this.commitTimer = window.setTimeout(() => this.commit(), COMMIT_SILENCE_MS)
        }
      }
    }

    rec.onend = () => {
      if (this.phase === 'listening') {
        this.commit()
      }
    }

    rec.onerror = (ev: SpeechRecognitionErrorEvent) => {
      if (ev.error === 'not-allowed' || ev.error === 'service-not-allowed') {
        this.onError('Microphone access denied — allow microphone in your browser settings.')
        this.active = false
        this.setP('idle')
        return
      }
      // All other errors: let onend commit whatever was captured
    }

    try {
      rec.start()
    } catch {
      this.commit()
      return
    }

    // Hard cap
    this.commitTimer = window.setTimeout(() => this.commit(), COMMIT_MAX_MS)
  }

  // ── Standby recognition (continuous, wake-word watch) ──────────────────────

  private launchStandby() {
    const SR = this.SR()
    if (!SR || !this.active || this.phase !== 'standby') return
    if (this.restartPending) return

    const rec = new SR() as SpeechRecognition
    this.standbyRec = rec
    rec.continuous      = true
    rec.interimResults  = true
    rec.lang            = 'en-US'
    rec.maxAlternatives = 1

    rec.onresult = (ev: SpeechRecognitionEvent) => {
      if (!this.active || this.phase !== 'standby') return

      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const text = ev.results[i][0].transcript
        if (WAKE_RE.test(text)) {
          const now = Date.now()
          if (now - this.lastWakeMs < WAKE_COOLDOWN_MS) return
          this.lastWakeMs = now

          this.setP('wake_detected')
          this.onWake()

          // Brief pause so the wake-detected UI can flash before listening starts
          setTimeout(() => {
            if (this.active) this.startCommandListening()
          }, 350)
          return
        }
      }
    }

    rec.onend = () => {
      if (!this.active || this.phase !== 'standby') return
      // Schedule restart — guard against rapid re-entry
      this.restartPending = true
      setTimeout(() => {
        this.restartPending = false
        if (this.active && this.phase === 'standby') {
          this.launchStandby()
        }
      }, 250)
    }

    rec.onerror = (ev: SpeechRecognitionErrorEvent) => {
      if (ev.error === 'not-allowed' || ev.error === 'service-not-allowed') {
        this.onError('Microphone access denied — allow microphone in your browser settings.')
        this.active = false
        this.setP('idle')
      }
      // Other errors handled by onend restart
    }

    try {
      rec.start()
    } catch {
      // Already started — this instance is dead, onend will restart
    }
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  startStandby() {
    if (!this.isSupported) {
      this.onError('Speech recognition requires Chrome or Edge.')
      return
    }
    this.active = true
    this.setP('standby')
    this.launchStandby()
  }

  /** PTT: skip wake word, go straight to command capture */
  startPTT() {
    if (!this.isSupported || !this.active) return
    this.stopStandbyRec()
    this.startCommandListening()
  }

  /** Called after consumer has processed the command — resumes standby */
  resumeStandby() {
    if (!this.active) return
    this.clearTimer()
    this.commandBuf = ''
    this.setLiveText('')
    this.setP('standby')
    this.stopCommandRec()
    this.launchStandby()
  }

  stopAll() {
    this.active = false
    this.clearTimer()
    this.commandBuf = ''
    this.setLiveText('')
    this.stopStandbyRec()
    this.stopCommandRec()
    this.restartPending = false
    this.setP('idle')
  }
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useVoiceInput({ onWake, onCommand, onError }: VoiceInputOptions) {
  const [phase, setPhase]       = useState<VoicePhase>('idle')
  const [liveText, setLiveText] = useState('')

  const mgr = useRef<VoiceInputManager | null>(null)
  if (!mgr.current) {
    mgr.current = new VoiceInputManager()
  }

  // Always keep callbacks fresh without recreating the manager
  const m = mgr.current
  m.setPhase    = setPhase
  m.setLiveText = setLiveText
  m.onWake      = onWake
  m.onCommand   = onCommand
  m.onError     = onError ?? (() => {})

  useEffect(() => {
    return () => { mgr.current?.stopAll() }
  }, [])

  return {
    phase,
    liveText,
    isSupported:   m.isSupported,
    startStandby:  () => m.startStandby(),
    startPTT:      () => m.startPTT(),
    resumeStandby: () => m.resumeStandby(),
    stopAll:       () => m.stopAll(),
  }
}
