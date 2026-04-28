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
    def __init__(self, result: str = "show zone b compliance") -> None:
        self.result = result

    async def transcribe(self, audio_bytes: bytes) -> str:
        return self.result


async def fake_chat_handler(message: str) -> dict:
    return {"response": f"handled:{message}", "provider": "test"}


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

    async def test_empty_transcript_emits_error_and_reopens_listening_once(self) -> None:
        publisher = FakePublisher()
        engine = TurnerVoiceEngine(
            audio_input=FakeAudioInput(),
            wake_detector=FakeWakeDetector(),
            transcriber=FakeTranscriber(result=""),
            publisher=publisher,
            chat_handler=fake_chat_handler,
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
            chat_handler=fake_chat_handler,
        )

        engine.state = "responding"
        await engine.interrupt_for_speech()

        self.assertEqual(engine.state, "listening")
        self.assertEqual(publisher.events[-1].payload["state"], "listening")


if __name__ == "__main__":
    unittest.main()
