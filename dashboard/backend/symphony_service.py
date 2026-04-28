import asyncio
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import socketio

app = FastAPI(title="Symphony Orchestrator")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO ASGI app
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# In-memory DB
TASKS = [
    {
        "id": 1,
        "issue_number": 42,
        "title": "Fix Auth Token Expiration",
        "state": "pending",
        "priority": "high",
        "branch_name": "symphony/issue-42",
        "assigned_agent": "Leon (Coder)",
        "turns_count": 0,
        "created_at": datetime.now(timezone.utc).isoformat()
    },
    {
        "id": 2,
        "issue_number": 45,
        "title": "Update GIS Layer for Zone B",
        "state": "completed",
        "priority": "medium",
        "branch_name": "symphony/issue-45",
        "assigned_agent": "Toni (Executor)",
        "turns_count": 4,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
]

EVENTS = [
    {
        "id": 1,
        "task_id": 2,
        "event_type": "log",
        "message": "Initialized workspace",
        "payload": "{}",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
]

@app.get("/api/symphony/tasks")
async def get_tasks():
    return {"status": "ok", "tasks": TASKS}

@app.get("/api/symphony/tasks/{task_id}/events")
async def get_events(task_id: int):
    task_events = [e for e in EVENTS if e["task_id"] == task_id]
    return {"status": "ok", "events": task_events}

@app.post("/api/symphony/tasks/{task_id}/approve")
async def approve_task(task_id: int):
    for t in TASKS:
        if t["id"] == task_id:
            if t["state"] == "pending":
                t["state"] = "dispatched"
                await emit_event(task_id, "dispatched", "Task approved by human.", {"approved_by": "brigh"})
            return {"status": "ok"}
    return {"status": "error", "message": "Task not found"}

async def emit_event(task_id: int, event_type: str, message: str, payload: dict):
    event = {
        "id": len(EVENTS) + 1,
        "task_id": task_id,
        "event_type": event_type, 
        "message": message,
        "payload": json.dumps(payload),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    EVENTS.append(event)
    
    await sio.emit("symphony_event", {
        "taskId": task_id,
        "type": event_type,
        "message": message,
        "payload": payload,
        "created_at": event["created_at"]
    })

@sio.event
async def connect(sid, environ, auth):
    print(f"Symphony Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Symphony Client disconnected: {sid}")

# Background simulation task
async def simulate_agent_activity():
    while True:
        await asyncio.sleep(15)
        # Randomly update a running task or start a dispatched one
        for t in TASKS:
            if t["state"] == "dispatched":
                t["state"] = "running"
                await emit_event(t["id"], "running", "Agent Toni began execution.", {"action": "checkout branch"})
            elif t["state"] == "running":
                t["turns_count"] += 1
                if t["turns_count"] > 3:
                    t["state"] = "completed"
                    await emit_event(t["id"], "completed", "Task finished successfully.", {"tests_passed": True})
                else:
                    await emit_event(t["id"], "log", f"Executing turn {t['turns_count']}", {"tool": "write_to_file", "file": "src/app.tsx"})

@app.on_event("startup")
async def startup_event():
    print("Initializing Symphony background simulation...")
    asyncio.create_task(simulate_agent_activity())

if __name__ == "__main__":
    print("Starting Symphony Orchestrator on port 3001...")
    uvicorn.run("symphony_service:socket_app", host="0.0.0.0", port=3001, reload=False)
