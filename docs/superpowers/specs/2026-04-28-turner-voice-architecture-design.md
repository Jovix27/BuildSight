# Turner Voice Architecture Design

Date: 2026-04-28
Status: Approved for planning
Scope: Production-grade backend-driven voice interaction for Turner AI on the local Windows workstation

## Goal

Refactor Turner voice interaction into a backend-owned, always-on voice system that:

- continuously listens on the local machine microphone
- detects the wake phrase "Hi Turner"
- responds with "Hi sir, how can I assist you?"
- immediately captures the next utterance
- routes that utterance through the existing Turner chat pipeline in the same session
- streams state and transcript updates to the dashboard frontend
- handles noisy environments, interruptions, and transcription failures cleanly

This first implementation is optimized for Windows desktop operation on the current machine, but the architecture must preserve modular boundaries so future cross-platform support does not require major refactoring.

## Non-Goals

- Browser-based speech recognition
- Browser-owned microphone wake-word loop
- Full distributed or multi-machine audio capture
- Mobile-first voice support
- Replacing the existing Turner chat logic

## Current State

The current Turner voice mode mixes browser `SpeechRecognition`, browser-side recording, and backend fallback transcription inside the frontend component. This creates unstable wake-word behavior, inconsistent latency, and unclear ownership of voice state.

The backend already contains Turner-related API routes in `dashboard/backend/server.py`, including chat, speech synthesis, introduction, and transcription endpoints. The redesign should preserve the existing Turner response path while moving all real-time voice orchestration into a dedicated backend module.

## Recommended Approach

Use a backend-resident voice supervisor running inside the FastAPI backend process boundary, but isolated in its own module: `dashboard/backend/voice_engine.py`.

The voice engine owns:

- microphone capture
- wake-word detection
- speech segmentation
- speech-to-text execution
- greeting flow
- interruption handling
- voice session state
- event emission to frontend subscribers

The FastAPI app remains responsible for:

- application startup and shutdown wiring
- REST and WebSocket exposure
- composing the voice engine with the Turner chat pipeline
- surfacing engine status and manual control endpoints

The React frontend becomes a display and control surface only.

## Target Architecture

### Backend Module Layout

Add:

- `dashboard/backend/voice_engine.py`

Responsibility split:

- `voice_engine.py`
  - all microphone and audio-device interaction
  - wake-word detector integration
  - utterance capture and cancellation
  - transcription orchestration
  - event publishing
  - internal state machine
- `server.py`
  - app lifecycle hooks
  - shared Turner chat function extraction
  - REST and WebSocket routes
  - engine startup, dependency injection, and status exposure

### Internal Voice Engine Boundaries

`voice_engine.py` should define small, replaceable units so the Windows-first implementation can evolve later:

- `AudioInput` abstraction
  - reads PCM frames from the active microphone
- `WakeWordDetector` abstraction
  - Porcupine-backed implementation for "Hi Turner"
- `SpeechTranscriber` abstraction
  - Whisper-backed implementation for local STT
- `VoiceEventBus` or callback publisher
  - emits typed events to WebSocket consumers
- `TurnerVoiceEngine`
  - orchestrates the state machine and cancellable tasks

All OS-specific audio details remain behind the `AudioInput` implementation in this module.

## Runtime Flow

### Idle Listening

On backend startup, the voice engine initializes microphone access, wake-word detection, and a long-running supervisor loop.

In `idle`:

- audio frames are read continuously from the local microphone
- wake-word detection runs on short PCM frames
- frontend subscribers receive health and state updates, but not raw audio

### Wake Greeting

When "Hi Turner" is detected:

1. emit a `greeting` event
2. emit a `state_update` event indicating wake activation
3. produce the fixed greeting:
   - `Hi sir, how can I assist you?`
4. switch immediately into active capture mode for the next utterance

The wake phrase and the subsequent utterance belong to one conversational session.

### Active Utterance Capture

After the greeting:

- the engine listens for the next spoken request
- capture uses bounded silence detection and a maximum utterance duration
- if user speech begins while TTS is still active, TTS playback is interrupted
- if no meaningful speech arrives within the listening window, the engine returns to `idle`

### Transcription And Chat Routing

After utterance capture:

1. audio is normalized and preprocessed for robustness
2. Whisper transcribes the utterance
3. if transcription succeeds, the engine invokes a shared Turner chat function directly in Python
4. the Turner response is streamed back to the frontend as typed events
5. optional TTS can use the existing voice path without changing engine ownership

Important constraint:

- do not use internal HTTP calls from the voice engine to `/api/ai/chat`
- extract shared Turner chat logic into a callable backend function and reuse it from both API routes and the voice engine

### Fallback Behavior

If transcription fails or is empty:

- emit an `error` event with a user-safe message
- emit a fallback prompt such as:
  - `I didn't catch that. Please repeat your request.`
- briefly reopen active listening once
- then return to `idle` if no valid utterance is captured

## Voice State Machine

The engine must use an explicit state machine rather than loosely coordinated flags.

Primary states:

- `starting`
- `idle`
- `wake_detected`
- `greeting`
- `listening`
- `transcribing`
- `thinking`
- `responding`
- `error`
- `stopped`

Required properties:

- each state transition is explicit and logged
- long-running operations are cancellable
- only one active conversational capture cycle may run at a time
- manual cancel and speech interrupt both force deterministic transition paths

## Interrupt Handling

Interrupt support is required in the first implementation.

### Speech Interrupt

If the user begins speaking while Turner is greeting or speaking:

- stop TTS playback immediately
- cancel the current speaking task
- transition to `listening`
- prioritize fresh user input over output completion

### Manual Cancel

If the frontend issues cancel:

- cancel current capture, transcription, thinking, or speaking tasks
- flush temporary audio buffers for the active cycle
- emit a `state_update` event
- return to `idle`

## WebSocket Design

Add a Turner-specific WebSocket endpoint:

- `WS /ws/turner-voice`

The frontend subscribes to typed events rather than inferring meaning from free-form payloads.

### Event Schema

Required event types:

- `state_update`
- `greeting`
- `transcript`
- `response`
- `error`
- `health`

Suggested payload shape:

```json
{
  "type": "state_update",
  "session_id": "uuid",
  "timestamp": "2026-04-28T12:00:00Z",
  "state": "listening",
  "detail": "Awaiting user utterance"
}
```

Examples:

```json
{
  "type": "greeting",
  "session_id": "uuid",
  "text": "Hi sir, how can I assist you?"
}
```

```json
{
  "type": "transcript",
  "session_id": "uuid",
  "stage": "final",
  "text": "Give me a PPE compliance summary for zone B."
}
```

```json
{
  "type": "response",
  "session_id": "uuid",
  "text": "Zone B has the lowest compliance with two missing helmets and one missing vest."
}
```

```json
{
  "type": "error",
  "session_id": "uuid",
  "code": "transcription_failed",
  "message": "I didn't catch that. Please repeat your request."
}
```

The frontend should not depend on backend implementation details beyond this event contract.

## REST API Design

Add:

- `POST /turner/voice`

Supported modes:

- text input for manual fallback or debugging
- audio upload for manual backend transcription path

Expected behavior:

- normalize the request into a shared Turner voice/chat path
- return a structured response payload
- avoid duplicating core Turner response logic

Suggested request forms:

- JSON:
  - `{ "text": "Summarize current site alerts." }`
- multipart form:
  - audio file upload for transcription and response generation

## Frontend Refactor

Refactor `dashboard/src/components/TurnerVoiceMode.tsx` into a UI-only component.

Remove:

- browser `SpeechRecognition`
- browser `MediaRecorder` fallback logic
- direct browser-managed STT state

Keep or improve:

- waveform or activity visualization
- mic/engine state indicators
- transcript display
- response display
- manual text fallback
- pause, resume, and cancel controls
- engine connection health display

Frontend responsibilities:

- connect to `WS /ws/turner-voice`
- render typed backend events
- send manual commands to backend
- reflect backend state instead of inferring it locally

## Noise Robustness Strategy

The first implementation should optimize for practical robustness on a construction-adjacent workstation environment.

Use:

- mono 16 kHz PCM input
- frame sizing compatible with Porcupine
- gain normalization
- lightweight denoising or high-pass filtering
- silence timeout and max utterance limits
- conservative retry path on failed transcription

The design should allow later substitution of stronger VAD or denoising components without changing the frontend contract.

## Latency Targets

Primary latency target:

- wake-word detection to greeting/state response in under 1 second on the local Windows machine

Secondary latency target:

- captured utterance to Turner response should be as low as practical, with the largest remaining contributors being transcription and Turner chat generation

Implementation implications:

- keep wake-word detector hot at all times
- avoid browser STT
- avoid self-HTTP calls inside the backend
- avoid expensive per-request model reinitialization
- use persistent model instances where memory allows

## Scalability And Future Evolution

This design should support future expansion without major refactoring:

- cross-platform audio input implementations can replace the Windows-first backend audio adapter
- GeoAI-triggered voice interactions can publish into the same event bus
- a future sidecar process can host the voice engine if fault isolation becomes necessary
- streaming partial transcripts or partial Turner responses can be added without breaking event typing

## Dependencies

Initial expected dependencies:

- FastAPI
- Porcupine or equivalent lightweight wake-word engine
- Whisper implementation for local transcription
- audio I/O library suitable for Windows desktop use

Dependency rules:

- keep the wake-word dependency isolated behind an abstraction
- keep transcription isolated behind an abstraction
- avoid spreading audio-library types outside `voice_engine.py`

## Testing Strategy

Minimum implementation-time testing:

- backend unit tests for state transitions
- backend tests for typed event emission
- backend tests for cancel and interrupt behavior
- backend tests for transcription failure fallback path
- integration test for shared Turner chat function used by both REST and voice engine
- frontend test or smoke coverage for WebSocket-driven state rendering

Manual validation checklist:

- wake phrase triggers greeting reliably
- next utterance is captured after greeting
- Turner response uses existing chat pipeline context
- manual cancel returns engine to `idle`
- speaking interruption stops output and listens
- transcription failure produces fallback prompt
- frontend remains stable when WebSocket reconnects

## Risks And Mitigations

Risk: microphone/device contention on Windows
Mitigation: isolate all device access in one backend module and expose clear startup diagnostics

Risk: transcription latency exceeds user expectations
Mitigation: use a small local model, keep it warm, and avoid internal HTTP routing overhead

Risk: noisy environment causes false triggers or poor transcription
Mitigation: tune wake sensitivity separately from transcription thresholds and add bounded retry logic

Risk: frontend becomes coupled to internal backend state handling
Mitigation: keep the WebSocket schema typed, stable, and minimal

## Implementation Summary

The first production version should deliver a backend-owned always-on Turner voice experience with:

- isolated `voice_engine.py` orchestration
- direct reuse of shared Turner chat logic
- typed WebSocket events
- UI-only frontend voice mode
- built-in interrupt and fallback handling
- Windows-first device behavior with portable boundaries

This is the approved architecture baseline for implementation planning.
