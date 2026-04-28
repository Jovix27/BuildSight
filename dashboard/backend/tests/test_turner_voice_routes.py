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
    def test_turner_voice_text_route_uses_shared_handler(self) -> None:
        async def fake_chat(message: str, history=None, context=None) -> dict:
            return {"response": f"handled:{message}", "status": "success", "provider": "test"}

        with patch.object(server, "run_turner_chat", new=AsyncMock(side_effect=fake_chat)):
            with TestClient(server.app) as client:
                response = client.post("/turner/voice", json={"text": "status report"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "handled:status report")

    def test_turner_voice_requires_audio_or_text(self) -> None:
        with TestClient(server.app) as client:
            response = client.post("/turner/voice", json={})
        self.assertEqual(response.status_code, 400)

    def test_turner_voice_websocket_returns_health_event(self) -> None:
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/turner-voice") as websocket:
                payload = websocket.receive_json()

        self.assertEqual(payload["type"], "health")
        self.assertIn("state", payload)

    def test_turner_voice_websocket_accepts_cancel_command(self) -> None:
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/turner-voice") as websocket:
                _ = websocket.receive_json()
                websocket.send_json({"action": "cancel"})
                payload = websocket.receive_json()

        self.assertEqual(payload["type"], "state_update")
        self.assertEqual(payload["state"], "idle")


if __name__ == "__main__":
    unittest.main()
