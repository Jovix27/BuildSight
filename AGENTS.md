# Repository Guidelines

## Operating Protocol
Address the human operator as Boss. In this workflow, Codex is Leon, the secondary executor. The four-agent execution architecture is:

- Jovi, Google's Antigravity: planner, orchestrator, reviewer, and source-of-truth owner for execution plans.
- Toni, Claude Code: primary executor for implementation, terminal work, tests, and first-pass handbacks.
- Leon, Codex: secondary executor for implementation, review, stabilization, and follow-through when Toni is unavailable or a task is assigned to Codex.
- Theo, Opencode: Codex optimizer for profiling, refactoring, validation, and tertiary execution.

Every project and module should keep this order of responsibility unless Boss explicitly reassigns ownership. Read `TASK_STATE.json`, `docs/AGENT_BRIDGE.md`, `docs/handshake.md`, and relevant files under `docs/handovers/` before acting on cross-agent work. When finishing delegated work, update the appropriate handoff or handback document with files changed, commands run, verification results, blockers, and next owner.

## Project Structure & Module Organization
BuildSight is an AI-enabled construction safety monitoring system. The root Python files `buildsight_intelligence.py` and `geoai_pipeline.py` hold core prototype logic. `dashboard/` contains the Vite React frontend plus the Python backend under `dashboard/backend/`. `dashboard/src/` owns UI state and GeoAI/PPE visualization logic. `spatial/` stores GIS schema, calibration, and GeoJSON zone assets. `research/` contains experiments and training scripts; treat it as exploratory unless a handoff says otherwise. `deploy/` contains deployment scripts. `docs/` stores workflow, reports, handoffs, and technical notes.

## Build, Test, and Development Commands
Install backend dependencies with `pip install -r requirements.txt`. Start the backend with `python dashboard/backend/server.py`. Install frontend dependencies with `npm.cmd --prefix dashboard install`, then run the dashboard with `npm.cmd --prefix dashboard run dev`.

Frontend commands from `dashboard/package.json`: `npm.cmd --prefix dashboard run build`, `npm.cmd --prefix dashboard run lint`, and `npm.cmd --prefix dashboard run preview`. Type-check with `.\dashboard\node_modules\.bin\tsc.cmd -b .\dashboard\tsconfig.json`.

Run Turner voice backend tests with `python -m pytest dashboard/backend/tests/test_voice_engine.py dashboard/backend/tests/test_turner_voice_routes.py -v`.

## Coding Style & Naming Conventions
Frontend TypeScript uses ESLint 9 flat config in `dashboard/eslint.config.js`, including `@eslint/js`, `typescript-eslint`, React Hooks, and React Refresh rules for `**/*.{ts,tsx}`. Vite/React files are ES modules. Python type checking is configured by `pyrightconfig.json` for Python 3.10, Windows, `.venv`, and dashboard backend paths; missing imports are warnings.

## Testing Guidelines
Prefer the targeted pytest command documented in `dashboard/README.md` for Turner voice changes. For dashboard changes, run TypeScript checks and `npm.cmd --prefix dashboard run lint`. Full Vite build may fail in restricted Windows environments when child-process spawning is blocked; report that limitation if it occurs.

## Commit & Pull Request Guidelines
Recent history uses concise messages such as `feat: ...`, `fix: ...`, `chore: ...`, and occasional milestone snapshots. Keep commits scoped to one module or handoff task. PRs or handbacks should name the owning agent, summarize behavior changes, list verification, and call out risks.
