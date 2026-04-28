# BuildSight Dashboard

This dashboard hosts the Turner AI and GeoAI frontend for BuildSight.

## Turner Voice Setup

Turner voice now runs in the backend on the same Windows machine as the microphone. The browser no longer owns speech recognition.

Required environment variables:

- `MISTRAL_API_KEY` or `GOOGLE_API_KEY`
- `ELEVENLABS_API_KEY` for spoken output
- `TURNER_WAKE_ACCESS_KEY` for Porcupine access
- `TURNER_WAKE_KEYWORD_PATH` for the custom `Hi Turner` wake-word model if used

Install backend dependencies:

```powershell
pip install -r requirements.txt
```

Run the backend:

```powershell
python dashboard/backend/server.py
```

Install frontend dependencies:

```powershell
npm.cmd --prefix dashboard install
```

Run the frontend:

```powershell
npm.cmd --prefix dashboard run dev
```

## Manual Validation

1. Start the backend on the same machine as the microphone.
2. Open the Turner voice tab in the dashboard.
3. Say `Hi Turner`.
4. Confirm the UI shows a greeting event and Turner says `Hi sir, how can I assist you?`
5. Ask `Give me a PPE compliance summary for zone B`.
6. Confirm the backend emits `transcript`, `response`, and `state_update` events in sequence.
7. Trigger `Stop` from the UI while Turner is speaking or processing.
8. Confirm the engine returns to `idle`.

## Verification Commands

Backend tests:

```powershell
python -m pytest dashboard/backend/tests/test_voice_engine.py dashboard/backend/tests/test_turner_voice_routes.py -v
```

Frontend typecheck:

```powershell
.\dashboard\node_modules\.bin\tsc.cmd -b .\dashboard\tsconfig.json
```

Note: full `vite build` may still fail in restricted Windows environments if the host blocks child-process spawning inside Vite config resolution.
