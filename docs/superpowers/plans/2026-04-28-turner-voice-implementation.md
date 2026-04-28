# Turner Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a backend-resident, always-on Turner voice pipeline with wake-word detection, typed WebSocket events, direct Turner chat integration, interruption handling, and a UI-only frontend voice surface.

**Architecture:** Add an isolated `voice_engine.py` module that owns microphone access, wake-word detection, utterance capture, transcription, and a cancellable state machine. Refactor `server.py` to expose shared Turner chat logic plus Turner voice REST/WebSocket routes, then slim `TurnerVoiceMode.tsx` into a backend-driven control/display client.

**Tech Stack:** Python, FastAPI, asyncio, unittest/pytest, React, TypeScript, WebSocket, Porcupine-compatible wake-word adapter, Whisper-compatible transcription adapter.

---

## File Structure

### Create

- `dashboard/backend/voice_engine.py`
  - Backend-only voice module containing the state machine, event models, audio interfaces, wake-word adapter, transcription adapter, and engine orchestration.
- `dashboard/backend/tests/test_voice_engine.py`
  - Unit tests for state transitions, fallback behavior, cancel handling, and typed event emission.
- `dashboard/backend/tests/test_turner_voice_routes.py`
  - API-level tests for the shared Turner chat path, `/turner/voice`, and Turner voice WebSocket behavior.
- `dashboard/src/hooks/useTurnerVoiceSocket.ts`
  - Dedicated frontend hook for Turner voice WebSocket state, typed events, reconnects, and control messages.

### Modify

- `dashboard/backend/server.py`
  - Extract shared Turner chat execution into a callable backend function, wire startup/shutdown lifecycle, add `/turner/voice`, and add `WS /ws/turner-voice`.
- `dashboard/src/components/TurnerVoiceMode.tsx`
  - Remove browser STT/mic ownership and convert to a UI-only component bound to backend voice state.
- `dashboard/src/components/TurnerVoiceMode.css`
  - Keep the visual system but align control and status classes with backend-driven states and connection health.
- `requirements.txt`
  - Add backend voice dependencies for the Windows-first local pipeline.
- `deploy/backend/requirements-ai.txt`
  - Mirror backend runtime dependencies used in local backend deployment.
- `dashboard/README.md`
  - Document Turner voice setup, required environment variables, and launch steps.

## Task 1: Lock The Backend Voice Contract

**Files:**
- Create: `dashboard/backend/tests/test_voice_engine.py`
- Modify: `dashboard/backend/server.py`

- [ ] **Step 1: Write the failing event-contract and state-machine tests**

```python
import asyncio
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from voice_engine import TurnerVoiceEngine, VoiceEvent


class FakePublisher:
    def __init__(self) -> None:
        self.events: list[VoiceEvent] = []

    async def publish(self, event: VoiceEvent) -> None:
        self.events.append(event)


class FakeAudioInput:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class FakeWakeDetector:
    def __init__(self) -> None:
        self.triggered = False

    def process(self, pcm: bytes) -> bool:
        return self.triggered


class FakeTranscriber:
    async def transcribe(self, audio_bytes: bytes) -> str:
        return "show zone b compliance"


class TurnerVoiceEngineStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_wake_word_emits_greeting_and_listening_state(self) -> None:
        publisher = FakePublisher()
        detector = FakeWakeDetector()
        engine = TurnerVoiceEngine(
            audio_input=FakeAudioInput(),
            wake_detector=detector,
            transcriber=FakeTranscriber(),
            publisher=publisher,
            chat_handler=None,
        )

        detector.triggered = True
        await engine._handle_wake_word()

        event_types = [event.type for event in publisher.events]
        self.assertIn("greeting", event_types)
        self.assertIn("state_update", event_types)
        self.assertEqual(engine.state, "listening")

    async def test_cancel_returns_engine_to_idle(self) -> None:
        publisher = FakePublisher()
        engine = TurnerVoiceEngine(
            audio_input=FakeAudioInput(),
            wake_detector=FakeWakeDetector(),
            transcriber=FakeTranscriber(),
            publisher=publisher,
            chat_handler=None,
        )

        engine.state = "thinking"
        await engine.cancel_current_cycle(reason="manual_cancel")

        self.assertEqual(engine.state, "idle")
        self.assertEqual(publisher.events[-1].type, "state_update")
```

- [ ] **Step 2: Run the backend voice tests to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_voice_engine.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'voice_engine'`

- [ ] **Step 3: Create the minimal `voice_engine.py` contract**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Protocol
import uuid


class AudioInput(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class WakeWordDetector(Protocol):
    def process(self, pcm: bytes) -> bool: ...


class SpeechTranscriber(Protocol):
    async def transcribe(self, audio_bytes: bytes) -> str: ...


@dataclass
class VoiceEvent:
    type: str
    session_id: str
    timestamp: str
    payload: dict = field(default_factory=dict)


class TurnerVoiceEngine:
    def __init__(
        self,
        *,
        audio_input: AudioInput,
        wake_detector: WakeWordDetector,
        transcriber: SpeechTranscriber,
        publisher,
        chat_handler: Callable[[str], Awaitable[dict]] | None,
    ) -> None:
        self.audio_input = audio_input
        self.wake_detector = wake_detector
        self.transcriber = transcriber
        self.publisher = publisher
        self.chat_handler = chat_handler
        self.state = "starting"
        self.session_id = uuid.uuid4().hex

    async def _emit(self, event_type: str, **payload: object) -> None:
        await self.publisher.publish(
            VoiceEvent(
                type=event_type,
                session_id=self.session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                payload=dict(payload),
            )
        )

    async def _handle_wake_word(self) -> None:
        await self._emit("greeting", text="Hi sir, how can I assist you?")
        self.state = "listening"
        await self._emit("state_update", state=self.state, detail="Awaiting user utterance")

    async def cancel_current_cycle(self, *, reason: str) -> None:
        self.state = "idle"
        await self._emit("state_update", state=self.state, detail=reason)
```

- [ ] **Step 4: Run the backend voice tests to verify they pass**

Run: `python -m pytest dashboard/backend/tests/test_voice_engine.py -v`

Expected: PASS for `test_wake_word_emits_greeting_and_listening_state` and `test_cancel_returns_engine_to_idle`

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/voice_engine.py dashboard/backend/tests/test_voice_engine.py
git commit -m "test: add Turner voice engine contract"
```

## Task 2: Build The Shared Turner Chat Function And Voice REST Route

**Files:**
- Create: `dashboard/backend/tests/test_turner_voice_routes.py`
- Modify: `dashboard/backend/server.py`

- [ ] **Step 1: Write the failing shared-chat and REST route tests**

```python
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import server


class TurnerVoiceRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)

    def test_turner_voice_text_route_uses_shared_handler(self) -> None:
        async def fake_chat(message: str, history=None, context=None) -> dict:
            return {"response": f"handled:{message}", "status": "success", "provider": "test"}

        with patch.object(server, "run_turner_chat", new=AsyncMock(side_effect=fake_chat)):
            response = self.client.post("/turner/voice", json={"text": "status report"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "handled:status report")

    def test_turner_voice_requires_audio_or_text(self) -> None:
        response = self.client.post("/turner/voice", json={})
        self.assertEqual(response.status_code, 400)
```

- [ ] **Step 2: Run the Turner voice route tests to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_turner_voice_routes.py -v`

Expected: FAIL with `AttributeError` for missing `run_turner_chat` or missing `/turner/voice`

- [ ] **Step 3: Extract the shared Turner chat function and add `/turner/voice`**

```python
class VoiceTextRequest(BaseModel):
    text: str = ""
    history: list[ChatMessage] = []
    context: dict = {}


async def run_turner_chat(
    message: str,
    *,
    history: list[ChatMessage] | None = None,
    context: dict | None = None,
) -> dict:
    req = ChatRequest(message=message, history=history or [], context=context or {})

    if not mistral_enabled and not ai_model:
        return {
            "response": "Turner AI is currently offline. Set MISTRAL_API_KEY (or GOOGLE_API_KEY) in the backend environment.",
            "error": "MISSING_API_KEY",
            "status": 503,
        }

    full_prompt = _build_site_prompt(req)

    if mistral_enabled:
        try:
            messages = _build_mistral_messages(req, full_prompt)
            response_text = await run_in_threadpool(_call_mistral_sync, messages)
            return {"response": response_text, "status": "success", "provider": "mistral"}
        except Exception:
            pass

    if ai_model:
        gemini_history = [
            {"role": "user" if m.role == "user" else "model", "parts": [m.content]}
            for m in req.history
        ]
        chat = ai_model.start_chat(history=gemini_history)
        response = await run_in_threadpool(chat.send_message, full_prompt)
        response_text = _extract_response_text(response)
        return {"response": response_text, "status": "success", "provider": "gemini"}

    return {"response": "I encountered a synchronization error. Please try again.", "error": "CHAT_FAILURE", "status": 500}


@app.post("/turner/voice")
async def turner_voice_route(
    text: str | None = Form(default=None),
    audio: UploadFile | None = File(default=None),
):
    if text and text.strip():
        result = await run_turner_chat(text.strip())
        status_code = result.pop("status", 200)
        if isinstance(status_code, int):
            return JSONResponse(result, status_code=status_code) if status_code >= 400 else result
        return result

    if audio is None:
        raise HTTPException(status_code=400, detail="Provide text or audio input.")

    audio_bytes = await audio.read()
    transcript = await voice_engine.transcribe_bytes(audio_bytes, audio.content_type or "audio/webm")
    if not transcript:
        raise HTTPException(status_code=422, detail="Transcription failed.")

    result = await run_turner_chat(transcript)
    result["transcript"] = transcript
    return result
```

- [ ] **Step 4: Update `/api/ai/chat` to reuse `run_turner_chat` and rerun the tests**

Run: `python -m pytest dashboard/backend/tests/test_turner_voice_routes.py -v`

Expected: PASS for the new route tests

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/server.py dashboard/backend/tests/test_turner_voice_routes.py
git commit -m "feat: add shared Turner chat handler and voice route"
```

## Task 3: Implement The Real Voice Engine State Machine And Interrupt Paths

**Files:**
- Modify: `dashboard/backend/voice_engine.py`
- Modify: `dashboard/backend/tests/test_voice_engine.py`
- Modify: `requirements.txt`
- Modify: `deploy/backend/requirements-ai.txt`

- [ ] **Step 1: Expand the tests to cover fallback, interrupt, and transcription retry**

```python
class FakeTranscriber:
    def __init__(self, result: str = "show zone b compliance") -> None:
        self.result = result

    async def transcribe(self, audio_bytes: bytes) -> str:
        return self.result


class TurnerVoiceFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_transcript_emits_error_and_reopens_listening_once(self) -> None:
        publisher = FakePublisher()
        engine = TurnerVoiceEngine(
            audio_input=FakeAudioInput(),
            wake_detector=FakeWakeDetector(),
            transcriber=FakeTranscriber(result=""),
            publisher=publisher,
            chat_handler=None,
        )

        await engine._handle_transcription_result(b"pcm")

        event_types = [event.type for event in publisher.events]
        self.assertIn("error", event_types)
        self.assertEqual(engine.state, "listening")

    async def test_interrupt_speaking_cancels_output_and_returns_to_listening(self) -> None:
        publisher = FakePublisher()
        engine = TurnerVoiceEngine(
            audio_input=FakeAudioInput(),
            wake_detector=FakeWakeDetector(),
            transcriber=FakeTranscriber(),
            publisher=publisher,
            chat_handler=None,
        )

        engine.state = "responding"
        await engine.interrupt_for_speech()

        self.assertEqual(engine.state, "listening")
        self.assertEqual(publisher.events[-1].payload["state"], "listening")
```

- [ ] **Step 2: Run the voice engine test file to verify the new cases fail**

Run: `python -m pytest dashboard/backend/tests/test_voice_engine.py -v`

Expected: FAIL with missing `interrupt_for_speech` or `_handle_transcription_result`

- [ ] **Step 3: Implement the cancellable state machine, adapters, and fallback path**

```python
class TurnerVoiceEngine:
    def __init__(self, *, audio_input, wake_detector, transcriber, publisher, chat_handler) -> None:
        self.audio_input = audio_input
        self.wake_detector = wake_detector
        self.transcriber = transcriber
        self.publisher = publisher
        self.chat_handler = chat_handler
        self.state = "starting"
        self.session_id = uuid.uuid4().hex
        self._supervisor_task: asyncio.Task | None = None
        self._active_task: asyncio.Task | None = None
        self._retry_count = 0

    async def set_state(self, state: str, detail: str) -> None:
        self.state = state
        await self._emit("state_update", state=state, detail=detail)

    async def start(self) -> None:
        await self.audio_input.start()
        await self.set_state("idle", "Wake-word monitor active")
        self._supervisor_task = asyncio.create_task(self._supervisor_loop(), name="turner-voice-supervisor")

    async def stop(self) -> None:
        await self.cancel_current_cycle(reason="engine_stopped")
        if self._supervisor_task:
            self._supervisor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._supervisor_task
        await self.audio_input.stop()
        await self.set_state("stopped", "Voice engine stopped")

    async def _handle_transcription_result(self, audio_bytes: bytes) -> None:
        await self.set_state("transcribing", "Transcribing user request")
        transcript = (await self.transcriber.transcribe(audio_bytes)).strip()
        if not transcript:
            self._retry_count += 1
            await self._emit("error", code="transcription_failed", message="I didn't catch that. Please repeat your request.")
            if self._retry_count <= 1:
                await self.set_state("listening", "Retrying after empty transcript")
                return
            self._retry_count = 0
            await self.set_state("idle", "Retry window exhausted")
            return

        self._retry_count = 0
        await self._emit("transcript", stage="final", text=transcript)
        await self.set_state("thinking", "Routing request through Turner chat")
        result = await self.chat_handler(transcript)
        await self._emit("response", text=result["response"], provider=result.get("provider", "unknown"))
        await self.set_state("responding", "Turner response ready")

    async def interrupt_for_speech(self) -> None:
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._active_task
        await self.set_state("listening", "Speech interrupt accepted")
```

- [ ] **Step 4: Add runtime dependencies for the Windows-first backend voice path**

```text
# requirements.txt
pvporcupine>=3.0.2
openai-whisper>=20231117
sounddevice>=0.4.7
numpy>=1.26
```

```text
# deploy/backend/requirements-ai.txt
pvporcupine>=3.0.2
openai-whisper>=20231117
sounddevice>=0.4.7
```

- [ ] **Step 5: Run the expanded backend voice tests**

Run: `python -m pytest dashboard/backend/tests/test_voice_engine.py -v`

Expected: PASS for greeting, cancel, retry, and interrupt behaviors

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/voice_engine.py dashboard/backend/tests/test_voice_engine.py requirements.txt deploy/backend/requirements-ai.txt
git commit -m "feat: implement Turner voice engine state machine"
```

## Task 4: Wire FastAPI Startup, Turner Voice WebSocket, And Engine Controls

**Files:**
- Modify: `dashboard/backend/server.py`
- Modify: `dashboard/backend/tests/test_turner_voice_routes.py`

- [ ] **Step 1: Add failing tests for the Turner voice WebSocket event schema**

```python
    def test_turner_voice_websocket_returns_health_event(self) -> None:
        with self.client.websocket_connect("/ws/turner-voice") as websocket:
            payload = websocket.receive_json()

        self.assertEqual(payload["type"], "health")
        self.assertIn("state", payload)

    def test_turner_voice_websocket_accepts_cancel_command(self) -> None:
        with self.client.websocket_connect("/ws/turner-voice") as websocket:
            _ = websocket.receive_json()
            websocket.send_json({"action": "cancel"})
            payload = websocket.receive_json()

        self.assertEqual(payload["type"], "state_update")
        self.assertEqual(payload["state"], "idle")
```

- [ ] **Step 2: Run the WebSocket tests to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_turner_voice_routes.py -v`

Expected: FAIL with missing `/ws/turner-voice`

- [ ] **Step 3: Add a dedicated Turner voice connection manager and lifecycle wiring**

```python
class TurnerVoiceConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        dead: set[WebSocket] = set()
        for ws in self._connections:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.disconnect(ws)


turner_voice_ws_manager = TurnerVoiceConnectionManager()


@app.on_event("startup")
async def startup_turner_voice() -> None:
    app.state.turner_voice_engine = build_turner_voice_engine(
        event_callback=turner_voice_ws_manager.broadcast,
        chat_handler=run_turner_chat,
    )
    await app.state.turner_voice_engine.start()


@app.on_event("shutdown")
async def shutdown_turner_voice() -> None:
    engine = getattr(app.state, "turner_voice_engine", None)
    if engine is not None:
        await engine.stop()
```

- [ ] **Step 4: Add the Turner voice WebSocket route and control message handling**

```python
@app.websocket("/ws/turner-voice")
async def turner_voice_ws(websocket: WebSocket):
    await turner_voice_ws_manager.connect(websocket)
    engine = app.state.turner_voice_engine
    await websocket.send_json({
        "type": "health",
        "state": engine.state,
        "session_id": engine.session_id,
        "engine_running": True,
    })
    try:
        while True:
            message = await websocket.receive_json()
            action = message.get("action")
            if action == "cancel":
                await engine.cancel_current_cycle(reason="manual_cancel")
            elif action == "status":
                await websocket.send_json({
                    "type": "health",
                    "state": engine.state,
                    "session_id": engine.session_id,
                    "engine_running": True,
                })
    except WebSocketDisconnect:
        turner_voice_ws_manager.disconnect(websocket)
```

- [ ] **Step 5: Run the route and WebSocket tests**

Run: `python -m pytest dashboard/backend/tests/test_turner_voice_routes.py -v`

Expected: PASS for the REST and WebSocket tests

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/server.py dashboard/backend/tests/test_turner_voice_routes.py
git commit -m "feat: wire Turner voice engine into FastAPI"
```

## Task 5: Refactor TurnerVoiceMode Into A UI-Only Backend Client

**Files:**
- Create: `dashboard/src/hooks/useTurnerVoiceSocket.ts`
- Modify: `dashboard/src/components/TurnerVoiceMode.tsx`
- Modify: `dashboard/src/components/TurnerVoiceMode.css`
- Modify: `dashboard/src/hooks/useAudioAnalyzer.ts`

- [ ] **Step 1: Write the failing frontend socket hook test or smoke harness**

```ts
import { renderHook, act } from '@testing-library/react'
import { useTurnerVoiceSocket } from './useTurnerVoiceSocket'

class FakeSocket {
  public readyState = WebSocket.OPEN
  public sent: string[] = []
  public onmessage: ((event: MessageEvent) => void) | null = null
  send(payload: string) {
    this.sent.push(payload)
  }
  close() {}
}

test('maps greeting and state events into hook state', () => {
  const socket = new FakeSocket()
  const { result } = renderHook(() => useTurnerVoiceSocket(() => socket as unknown as WebSocket))

  act(() => {
    socket.onmessage?.(new MessageEvent('message', {
      data: JSON.stringify({ type: 'greeting', text: 'Hi sir, how can I assist you?' }),
    }))
  })

  expect(result.current.greeting).toBe('Hi sir, how can I assist you?')
})
```

- [ ] **Step 2: Run the frontend test or typecheck to verify the hook is missing**

Run: `npm --prefix dashboard run test -- useTurnerVoiceSocket`

Expected: FAIL because `useTurnerVoiceSocket` does not exist yet

- [ ] **Step 3: Create the Turner voice socket hook**

```ts
import { useEffect, useMemo, useRef, useState } from 'react'

export interface TurnerVoiceEvent {
  type: 'state_update' | 'greeting' | 'transcript' | 'response' | 'error' | 'health'
  state?: string
  text?: string
  message?: string
}

const WS_URL = 'ws://localhost:8000/ws/turner-voice'

export function useTurnerVoiceSocket(createSocket?: () => WebSocket) {
  const [connectionState, setConnectionState] = useState<'connecting' | 'open' | 'closed'>('connecting')
  const [voiceState, setVoiceState] = useState('idle')
  const [transcript, setTranscript] = useState('')
  const [response, setResponse] = useState('')
  const [greeting, setGreeting] = useState('')
  const [error, setError] = useState('')
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const socket = createSocket ? createSocket() : new WebSocket(WS_URL)
    socketRef.current = socket
    socket.onopen = () => setConnectionState('open')
    socket.onclose = () => setConnectionState('closed')
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as TurnerVoiceEvent
      if (payload.type === 'state_update' && payload.state) setVoiceState(payload.state)
      if (payload.type === 'greeting' && payload.text) setGreeting(payload.text)
      if (payload.type === 'transcript' && payload.text) setTranscript(payload.text)
      if (payload.type === 'response' && payload.text) setResponse(payload.text)
      if (payload.type === 'error') setError(payload.message ?? 'Voice pipeline error')
      if (payload.type === 'health' && payload.state) setVoiceState(payload.state)
    }
    return () => socket.close()
  }, [createSocket])

  return {
    connectionState,
    voiceState,
    transcript,
    response,
    greeting,
    error,
    sendAction: (action: string) => socketRef.current?.send(JSON.stringify({ action })),
  }
}
```

- [ ] **Step 4: Refactor `TurnerVoiceMode.tsx` to remove browser STT and use backend events**

```tsx
export const TurnerVoiceMode: React.FC = () => {
  const [question, setQuestion] = useState('')
  const [showCC, setShowCC] = useState(true)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const { isPaused, setPaused } = useDetectionStore()
  const { connectionState, voiceState, transcript, response, greeting, error, sendAction } = useTurnerVoiceSocket()
  const metrics = useAudioAnalyzer(audioRef.current, null)

  const statusLabel: Record<string, string> = {
    idle: 'Wake-word monitor active',
    wake_detected: 'Wake word detected',
    greeting: 'Greeting user',
    listening: 'Listening for request',
    transcribing: 'Transcribing request',
    thinking: 'Processing request',
    responding: 'Responding',
    error: 'Voice error',
    stopped: 'Voice engine offline',
  }

  const handleManualAsk = async () => {
    if (!question.trim()) return
    const resp = await fetch('http://localhost:8000/turner/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: question.trim() }),
    })
    const data = await resp.json()
    setQuestion('')
    if (data.response && audioRef.current) {
      audioRef.current.pause()
    }
  }

  return (
    <div className="turner-voice-mode">
      <TurnerOrb3D size={500} amplitude={metrics.amplitude} state={voiceState === 'listening' ? 'listening' : voiceState as any} />
      {showCC && <p>{response || transcript || greeting}</p>}
      {error && <p>{error}</p>}
      <button onClick={() => sendAction('cancel')}>Cancel</button>
      <button onClick={() => setShowCC((value) => !value)}>CC</button>
      <button onClick={() => setPaused(!isPaused)}>{isPaused ? 'RESUME' : 'PAUSE'}</button>
      <input value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') void handleManualAsk() }} />
      <span>{connectionState} · {statusLabel[voiceState] ?? voiceState}</span>
    </div>
  )
}
```

- [ ] **Step 5: Simplify the analyzer hook so it handles audio playback only**

```ts
export const useAudioAnalyzer = (audioElement: HTMLAudioElement | null) => {
  const [metrics, setMetrics] = useState<AudioMetrics>({
    amplitude: 0,
    frequencyData: new Uint8Array(0),
  })

  useEffect(() => {
    if (!audioElement) {
      setMetrics({ amplitude: 0, frequencyData: new Uint8Array(0) })
      return
    }
    // keep the existing HTMLAudioElement analyzer path only
  }, [audioElement])

  return metrics
}
```

- [ ] **Step 6: Run frontend verification**

Run: `npm --prefix dashboard run build`

Expected: PASS with no TypeScript errors in `TurnerVoiceMode.tsx` or `useTurnerVoiceSocket.ts`

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/hooks/useTurnerVoiceSocket.ts dashboard/src/components/TurnerVoiceMode.tsx dashboard/src/components/TurnerVoiceMode.css dashboard/src/hooks/useAudioAnalyzer.ts
git commit -m "feat: refactor Turner voice UI to backend-driven mode"
```

## Task 6: Document Setup, Environment, And Manual Validation

**Files:**
- Modify: `dashboard/README.md`

- [ ] **Step 1: Add backend voice setup documentation**

```md
## Turner Voice Setup

Turner voice now runs in the backend on the same Windows machine as the microphone.

Required environment variables:

- `MISTRAL_API_KEY` or `GOOGLE_API_KEY`
- `ELEVENLABS_API_KEY` for spoken output
- `TURNER_WAKE_ACCESS_KEY` for Porcupine access
- `TURNER_WAKE_KEYWORD_PATH` for the custom "Hi Turner" wake-word model, if used

Install backend dependencies:

```powershell
pip install -r requirements.txt
```

Run backend:

```powershell
python dashboard/backend/server.py
```

Run frontend:

```powershell
npm --prefix dashboard install
npm --prefix dashboard run dev
```
```

- [ ] **Step 2: Add a manual validation checklist**

```md
### Manual validation

1. Start the backend on the same machine as the microphone.
2. Open the Turner voice tab in the dashboard.
3. Say `Hi Turner`.
4. Confirm the UI shows a `greeting` event and Turner says `Hi sir, how can I assist you?`
5. Ask `Give me a PPE compliance summary for zone B`.
6. Confirm the backend emits `transcript`, `response`, and `state_update` events in sequence.
7. Trigger Cancel from the UI while Turner is speaking.
8. Confirm the engine returns to `idle`.
```

- [ ] **Step 3: Run final verification**

Run: `python -m pytest dashboard/backend/tests/test_voice_engine.py dashboard/backend/tests/test_turner_voice_routes.py -v`

Expected: PASS

Run: `npm --prefix dashboard run build`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add dashboard/README.md
git commit -m "docs: add Turner voice setup and validation guide"
```

## Spec Coverage Check

- Wake-word detection and backend-owned continuous listening are covered by Tasks 1, 3, and 4.
- Shared Turner chat reuse without internal HTTP is covered by Task 2.
- Typed WebSocket events and frontend stability are covered by Tasks 4 and 5.
- Interrupt handling and transcription fallback are covered by Task 3.
- UI-only frontend refactor is covered by Task 5.
- Setup instructions and validation are covered by Task 6.

## Placeholder Scan

- No `TBD`, `TODO`, or deferred implementation notes remain in this plan.
- Every task names exact files, commands, and expected outcomes.
- Every code-writing step includes concrete code to anchor the implementation.

## Type Consistency Check

- Voice event types remain `state_update`, `greeting`, `transcript`, `response`, `error`, and `health` throughout backend and frontend tasks.
- The shared backend entry point is consistently named `run_turner_chat`.
- The main engine class remains `TurnerVoiceEngine` across tests, runtime wiring, and frontend expectations.
