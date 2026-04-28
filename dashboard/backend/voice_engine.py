from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
import os
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Awaitable, Callable, Protocol
import wave
import uuid

logger = logging.getLogger("buildsight.turner.voice")


class AudioInput(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def read_frame(self) -> bytes: ...
    async def capture_utterance(self) -> bytes: ...


class WakeWordDetector(Protocol):
    def process(self, pcm: bytes) -> bool: ...


class SpeechTranscriber(Protocol):
    async def transcribe(self, audio_bytes: bytes) -> str: ...


class SoundDeviceAudioInput:
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_length: int = 512,
        silence_frames: int = 24,
        max_frames: int = 220,
        speech_rms_threshold: float = 350.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.silence_frames = silence_frames
        self.max_frames = max_frames
        self.speech_rms_threshold = speech_rms_threshold
        self._stream = None
        self._queue: Queue[bytes] = Queue(maxsize=512)

    async def start(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                logger.debug("sounddevice status: %s", status)
            try:
                self._queue.put_nowait(bytes(indata))
            except Exception:
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
                with contextlib.suppress(Exception):
                    self._queue.put_nowait(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_length,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def read_frame(self) -> bytes:
        return await asyncio.to_thread(self._queue.get, True, 1.0)

    async def capture_utterance(self) -> bytes:
        chunks: list[bytes] = []
        speech_started = False
        trailing_silence = 0

        for _ in range(self.max_frames):
            frame = await self.read_frame()
            rms = _pcm_rms(frame)
            if rms >= self.speech_rms_threshold:
                speech_started = True
                trailing_silence = 0
                chunks.append(frame)
                continue

            if speech_started:
                chunks.append(frame)
                trailing_silence += 1
                if trailing_silence >= self.silence_frames:
                    break

        return b"".join(chunks)


class PorcupineWakeWordDetector:
    def __init__(self, *, access_key: str, keyword_path: str | None = None, sensitivity: float = 0.6) -> None:
        import pvporcupine

        create_kwargs: dict[str, Any] = {
            "access_key": access_key,
            "sensitivities": [sensitivity],
        }
        if keyword_path:
            create_kwargs["keyword_paths"] = [keyword_path]
        else:
            create_kwargs["keywords"] = ["porcupine"]

        self._porcupine = pvporcupine.create(**create_kwargs)

    def process(self, pcm: bytes) -> bool:
        if len(pcm) < self._porcupine.frame_length * 2:
            return False
        frame = memoryview(pcm)[: self._porcupine.frame_length * 2]
        pcm_frame = [int.from_bytes(frame[i:i + 2], byteorder="little", signed=True) for i in range(0, len(frame), 2)]
        return self._porcupine.process(pcm_frame) >= 0


class WhisperSpeechTranscriber:
    def __init__(self, *, model_name: str = "tiny.en", sample_rate: int = 16000) -> None:
        import whisper

        self.sample_rate = sample_rate
        self._model = whisper.load_model(model_name)

    async def transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""

        def _run() -> str:
            with _temporary_wave_file(audio_bytes, self.sample_rate) as wav_path:
                result = self._model.transcribe(wav_path, fp16=False, language="en")
            return str(result.get("text", "")).strip()

        return await asyncio.to_thread(_run)


@dataclass
class VoiceEvent:
    type: str
    session_id: str
    timestamp: str
    payload: dict[str, Any] = field(default_factory=dict)


class TurnerVoiceEngine:
    def __init__(
        self,
        *,
        audio_input: AudioInput,
        wake_detector: WakeWordDetector,
        transcriber: SpeechTranscriber,
        publisher: Any,
        chat_handler: Callable[[str], Awaitable[dict[str, Any]]] | None,
    ) -> None:
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
        self._cancel_requested = False

    async def _emit(self, event_type: str, **payload: object) -> None:
        await self.publisher.publish(
            VoiceEvent(
                type=event_type,
                session_id=self.session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                payload=dict(payload),
            )
        )

    async def set_state(self, state: str, detail: str) -> None:
        self.state = state
        await self._emit("state_update", state=state, detail=detail)

    async def _handle_wake_word(self) -> None:
        await self.set_state("wake_detected", "Wake phrase detected")
        await self._emit("greeting", text="Hi sir, how can I assist you?")
        await self.set_state("listening", "Awaiting user utterance")

    async def start(self) -> None:
        await self.audio_input.start()
        await self.set_state("idle", "Wake-word monitor active")
        if hasattr(self.audio_input, "read_frame"):
            self._supervisor_task = asyncio.create_task(self._supervisor_loop(), name="turner-voice-supervisor")

    async def stop(self) -> None:
        await self.cancel_current_cycle(reason="engine_stopped")
        if self._supervisor_task:
            self._supervisor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._supervisor_task
        await self.audio_input.stop()
        await self.set_state("stopped", "Voice engine stopped")

    async def transcribe_bytes(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        del mime_type
        return (await self.transcriber.transcribe(audio_bytes)).strip()

    async def _supervisor_loop(self) -> None:
        while True:
            try:
                frame = await self.audio_input.read_frame()
                if self.state != "idle":
                    continue
                if self.wake_detector.process(frame):
                    self._active_task = asyncio.create_task(self._run_conversation_cycle(), name="turner-voice-cycle")
                    await self._active_task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Turner voice supervisor loop error: %s", exc)
                await self._emit("error", code="voice_supervisor_error", message=str(exc))
                await self.set_state("idle", "Recovered from supervisor error")

    async def _run_conversation_cycle(self) -> None:
        self._cancel_requested = False
        await self._handle_wake_word()
        utterance = await self.audio_input.capture_utterance()
        if self._cancel_requested:
            await self.set_state("idle", "Cancelled during capture")
            return
        if not utterance:
            await self.set_state("idle", "No speech captured after wake word")
            return
        await self._handle_transcription_result(utterance)
        if not self._cancel_requested and self.state == "responding":
            await self.set_state("idle", "Awaiting wake word")

    async def _handle_transcription_result(self, audio_bytes: bytes) -> None:
        await self.set_state("transcribing", "Transcribing user request")
        transcript = await self.transcribe_bytes(audio_bytes)
        if not transcript:
            self._retry_count += 1
            await self._emit(
                "error",
                code="transcription_failed",
                message="I didn't catch that. Please repeat your request.",
            )
            if self._retry_count <= 1:
                await self.set_state("listening", "Retrying after empty transcript")
                return

            self._retry_count = 0
            await self.set_state("idle", "Retry window exhausted")
            return

        self._retry_count = 0
        await self._emit("transcript", stage="final", text=transcript)
        await self.set_state("thinking", "Routing request through Turner chat")

        if self.chat_handler is None:
            await self._emit("response", text="", provider="none")
            await self.set_state("idle", "No chat handler configured")
            return

        result = await self.chat_handler(transcript)
        await self._emit(
            "response",
            text=result.get("response", ""),
            provider=result.get("provider", "unknown"),
        )
        await self.set_state("responding", "Turner response ready")

    async def interrupt_for_speech(self) -> None:
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._active_task
        await self.set_state("listening", "Speech interrupt accepted")

    async def cancel_current_cycle(self, *, reason: str) -> None:
        self._cancel_requested = True
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._active_task
        self._retry_count = 0
        await self.set_state("idle", reason)


def _pcm_rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    total = 0.0
    sample_count = 0
    for idx in range(0, len(pcm) - 1, 2):
        sample = int.from_bytes(pcm[idx:idx + 2], byteorder="little", signed=True)
        total += float(sample * sample)
        sample_count += 1
    if sample_count == 0:
        return 0.0
    return math.sqrt(total / sample_count)


class _WaveFileContext:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()


def _temporary_wave_file(audio_bytes: bytes, sample_rate: int) -> _WaveFileContext:
    runtime_dir = Path(os.environ.get("BUILDSIGHT_RUNTIME_DIR", Path.cwd() / "data" / "runtime"))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    temp_path = runtime_dir / f"turner_voice_{uuid.uuid4().hex}.wav"
    with wave.open(str(temp_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    return _WaveFileContext(temp_path)
