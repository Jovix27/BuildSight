import { useState, useEffect, useMemo, type CSSProperties, Component, type ReactNode, type ErrorInfo } from 'react'
// Force refresh

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null }
  static getDerivedStateFromError(error: Error) { return { error } }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('[BuildSight] Render error:', error, info) }
  render() {
    if (this.state.error) {
      return (
        <div style={{ color: '#ff4444', background: '#0a0c12', minHeight: '100vh', padding: '2rem', fontFamily: 'monospace', fontSize: '14px' }}>
          <h2 style={{ color: '#ff6666' }}>BuildSight — Render Error</h2>
          <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{String(this.state.error)}</pre>
          <button onClick={() => this.setState({ error: null })} style={{ marginTop: '1rem', padding: '0.5rem 1rem', background: '#1a2030', color: '#fff', border: '1px solid #334', borderRadius: '4px', cursor: 'pointer' }}>
            Retry
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
import { AnimatePresence } from 'framer-motion'
import './App.css'
import { AlertLog, type AlertItem } from './components/AlertLog'
import { MetricCard } from './components/MetricCard'
import { DetectionPanel } from './components/DetectionPanel'
import { LiveSurveillance } from './components/LiveSurveillance'
import { SettingsPanel } from './components/SettingsPanel'
import { SettingsProvider, useSettings } from './SettingsContext'
import { DetectionStatsProvider, useDetectionStats } from './DetectionStatsContext'
import { TurnerPage } from './components/TurnerPage'
import { TurnerSidebar } from './components/TurnerSidebar'
import { AnalyticsPage } from './components/AnalyticsPage'
import { GeoAIPage } from './components/GeoAIPage'
import { BuildSightBrain } from './components/BuildSightBrain'
import { GodLock } from './components/GodLock'
import { PageTransition } from './components/AnimatedLayout'
import DetectionStatusBar from './components/DetectionStatusBar'
import { useDetectionStore } from './store/detectionStore'
/* Static fallback metrics — replaced by live data when detection is running */
const STATIC_METRICS = [
  {
    label: 'Hardhat Compliance',
    value: '98.2%',
    delta: '+1.4%',
    status: 'stable' as const,
    progress: 98.2,
    footnote: '412 workers scanned in the last 30 min',
  },
  {
    label: 'Unsafe Proximity',
    value: '03',
    delta: 'LIVE',
    status: 'alert' as const,
    progress: 34,
    footnote: '2 near crane swing radius, 1 vehicle conflict',
  },
  {
    label: 'High-Vis Vest',
    value: '85.6%',
    delta: '-2.1%',
    status: 'risk' as const,
    progress: 85.6,
    footnote: 'East scaffold zone trending below threshold',
  },
  {
    label: 'AI Confidence',
    value: '96.4%',
    delta: 'MODEL V2.5',
    status: 'stable' as const,
    progress: 96.4,
    footnote: 'Detection confidence across active cameras',
  },
] satisfies Array<{
  label: string
  value: string
  delta: string
  status: 'stable' | 'risk' | 'alert'
  progress: number
  footnote: string
}>

const alerts: AlertItem[] = [
  {
    id: 'AL-219',
    time: '14:23:19',
    camera: 'CAM-03 / Tower A',
    severity: 'critical',
    title: 'No hardhat inside hoist landing',
    detail: 'Worker remained in restricted bay for 00:18 before supervisor acknowledgment.',
  },
  {
    id: 'AL-214',
    time: '14:20:07',
    camera: 'CAM-07 / East Scaffold',
    severity: 'warning',
    title: 'High-vis vest missing',
    detail: 'Repeated vest non-compliance near edge protection zone on level 06.',
  },
  {
    id: 'AL-209',
    time: '14:16:42',
    camera: 'CAM-01 / Main Gate',
    severity: 'info',
    title: 'Unauthorized dwell near entry lane',
    detail: 'Vehicle/pedestrian overlap detected and auto-cleared within safety threshold.',
  },
]



type View = 'dashboard' | 'settings' | 'analytics' | 'geoai' | 'brain' | 'turner'
type DashboardMode = 'LIVE' | 'VIDEO' | 'IMAGE'

function formatIstSnapshot() {
  const now = new Date()
  const weekday = new Intl.DateTimeFormat('en-IN', {
    timeZone: 'Asia/Kolkata',
    weekday: 'long',
  }).format(now)
  const date = new Intl.DateTimeFormat('en-IN', {
    timeZone: 'Asia/Kolkata',
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  }).format(now).toUpperCase()
  const time = new Intl.DateTimeFormat('en-IN', {
    timeZone: 'Asia/Kolkata',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  }).format(now)

  return `${weekday} / ${date} / ${time} IST`
}

function AppInner() {
  const { settings, update } = useSettings()
  const { stats, liveAlerts } = useDetectionStats()
  const connectDetection = useDetectionStore(s => s.connect)
  const [view, setView] = useState<View>('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Connect background detection WebSocket once at app startup.
  // This single connection persists across ALL tab navigation.
  useEffect(() => { connectDetection() }, [])
  const [dashboardMode, setDashboardMode] = useState<DashboardMode>('LIVE')
  const [summaryCollapsed, setSummaryCollapsed] = useState(false)
  const [snapshotTimestamp, setSnapshotTimestamp] = useState(() => formatIstSnapshot())
  
  // ── GOD MODE Security State ───────────────────────────────────────────────
  const [isGodUnlocked, setIsGodUnlocked] = useState(false)
  const [showGodLock, setShowGodLock]   = useState(false)

  // Check persistence on load
  useEffect(() => {
    const unlockedAt = localStorage.getItem('godModeUnlockedAt')
    if (unlockedAt) {
      const timeElapsed = Date.now() - parseInt(unlockedAt, 10)
      const TWENTY_FOUR_HOURS = 24 * 60 * 60 * 1000
      if (timeElapsed < TWENTY_FOUR_HOURS) {
        setIsGodUnlocked(true)
      } else {
        localStorage.removeItem('godModeUnlockedAt')
        localStorage.removeItem('godModeUnlocked')
      }
    }
  }, [])

  const handleGodModeAccess = () => {
    if (isGodUnlocked) {
      setView('brain')
    } else {
      setShowGodLock(true)
    }
  }

  const onGodUnlockSuccessful = () => {
    setIsGodUnlocked(true)
    setShowGodLock(false)
    localStorage.setItem('godModeUnlocked', 'true')
    localStorage.setItem('godModeUnlockedAt', Date.now().toString())
    setView('brain')
  }

  const handleLockSystem = () => {
    setIsGodUnlocked(false)
    localStorage.removeItem('godModeUnlocked')
    localStorage.removeItem('godModeUnlockedAt')
    setView('dashboard')
  }

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setSnapshotTimestamp(formatIstSnapshot())
    }, 1000)

    return () => window.clearInterval(intervalId)
  }, [])

  // ── Live compliance metrics computed from detection stats ──────────────────
  type MetricData = { label: string; value: string; delta: string; status: 'stable' | 'risk' | 'alert'; progress: number; footnote: string }
  const liveMetrics: MetricData[] = useMemo(() => {
    if (!stats.isRunning || stats.framesScanned === 0) return STATIC_METRICS

    const workers = stats.totalWorkers || 1 // avoid division by zero
    const helmetPct = Math.min(100, (stats.helmetsDetected / workers) * 100)
    const vestPct   = Math.min(100, (stats.vestsDetected / workers) * 100)
    const confPct   = stats.avgConfidence * 100

    const helmetStatus: 'stable' | 'risk' | 'alert' =
      helmetPct >= 90 ? 'stable' : helmetPct >= 70 ? 'risk' : 'alert'
    const vestStatus: 'stable' | 'risk' | 'alert' =
      vestPct >= 90 ? 'stable' : vestPct >= 70 ? 'risk' : 'alert'
    const confStatus: 'stable' | 'risk' | 'alert' =
      confPct >= 80 ? 'stable' : confPct >= 50 ? 'risk' : 'alert'

    return [
      {
        label: 'Hardhat Compliance',
        value: `${helmetPct.toFixed(1)}%`,
        delta: stats.isRunning ? 'LIVE' : '+0.0%',
        status: helmetStatus,
        progress: helmetPct,
        footnote: `${stats.helmetsDetected} helmets / ${workers} workers · ${stats.framesScanned} frames`,
      },
      {
        label: 'Unsafe Proximity',
        value: String(stats.proximityViolations).padStart(2, '0'),
        delta: stats.isRunning ? 'TRACKING' : 'IDLE',
        status: stats.proximityViolations > 0 ? 'alert' : 'stable',
        progress: Math.min(100, stats.proximityViolations * 20),
        footnote: `${stats.proximityViolations} zone violation${stats.proximityViolations !== 1 ? 's' : ''} detected`,
      },
      {
        label: 'High-Vis Vest',
        value: `${vestPct.toFixed(1)}%`,
        delta: stats.isRunning ? 'LIVE' : '+0.0%',
        status: vestStatus,
        progress: vestPct,
        footnote: `${stats.vestsDetected} vests / ${workers} workers · ${stats.framesScanned} frames`,
      },
      {
        label: 'AI Confidence',
        value: `${confPct.toFixed(1)}%`,
        delta: stats.modelName ? stats.modelName.toUpperCase() : 'MODEL V2.5',
        status: confStatus,
        progress: confPct,
        footnote: `Mean confidence · ${stats.elapsedMs}ms latency · ${stats.framesScanned} frames`,
      },
    ]
  }, [stats])

  // Apply theme to document root
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', settings.theme)
  }, [settings.theme])

  // Apply accent color live to CSS custom property
  useEffect(() => {
    document.documentElement.style.setProperty('--color-accent', settings.accentColor)
  }, [settings.accentColor])

  // Mode switches do NOT reset detection — background session persists across tabs

  const workspaceCopy: Record<DashboardMode, { label: string; title: string; meta: string }> = {
    LIVE: {
      label: 'Unified Monitoring Hub',
      title: 'Live Surveillance Workspace',
      meta: 'Interactive crane-approach watch map with compliance and restricted-zone overlays',
    },
    VIDEO: {
      label: 'Unified Monitoring Hub',
      title: 'Video Detection Workspace',
      meta: 'Server-assisted PPE analysis for uploaded site footage and annotated export review',
    },
    IMAGE: {
      label: 'Unified Monitoring Hub',
      title: 'Image Detection Workspace',
      meta: 'Single-frame PPE review for audits, incident capture, and evidence validation',
    },
  }

  const workspace = workspaceCopy[dashboardMode]
  const mergedAlerts = useMemo(() => [...liveAlerts, ...alerts], [liveAlerts])





  // ── Helper: Render Escalation Queue ───────────────────────────────────────
  const renderEscalationQueue = (showToggle = false) => (
    <div className="panel-content-wrapper">
      <div className="panel-heading">
        <div>
          <p className="section-label">Escalation Queue</p>
          <h3>AI-Flagged Alert Log</h3>
        </div>
        {showToggle && (
          <button type="button" className="panel-toggle" onClick={() => setSummaryCollapsed(!summaryCollapsed)}>
            {summaryCollapsed ? 'View Alert Grid' : 'Compact View'}
          </button>
        )}
        {!showToggle && (
           <p className="panel-meta">Most recent events sorted by operational severity</p>
        )}
      </div>
      
      {!summaryCollapsed || !showToggle ? (
        <AlertLog alerts={mergedAlerts} />
      ) : (
        <div className="hero-metrics-grid hero-metrics-grid--compact">
          {liveMetrics.map((metric) => (
            <MetricCard key={`hero-compact-${metric.label}`} {...metric} />
          ))}
        </div>
      )}
    </div>
  )

  return (
    <div className={`app-shell ${settings.compactView ? 'app-shell--compact' : ''} ${view === 'brain' ? 'app-shell--god-mode' : ''}`}>
      {view !== 'brain' && (
      <aside className="sidebar panel">
        <div className="sidebar__brand" onClick={() => { setView('dashboard'); setDashboardMode('LIVE'); }} role="button" tabIndex={0} aria-label="Go to dashboard home" style={{ cursor: 'pointer' }}>
          <img src="/logo.png" alt="BuildSight Logo" className="brand-logo" />
          <div className="brand-text">
            <h1>BuildSight</h1>
          </div>
        </div>

        <div className="sidebar__content">
          <div className="sidebar__section">
            <p className="section-label">Site Matrix</p>
            <button className="site-chip site-chip--active">CHN-042 / Metro Core</button>
            <button className="site-chip">BLR-019 / South Yard</button>
            <button className="site-chip">HYD-011 / Precast Plant</button>
          </div>

          <nav className="sidebar__section" aria-label="Primary">
            <p className="section-label">Control Stack</p>
            <button
              className={`nav-link ${view === 'dashboard' ? 'nav-link--active' : ''}`}
              onClick={() => {
                setView('dashboard')
                setDashboardMode('LIVE')
              }}
            >
              Live Surveillance
            </button>
            <a className={`nav-link ${view === 'dashboard' ? '' : 'nav-link--dim'}`} href="#metrics" onClick={() => setView('dashboard')}>
              Compliance Metrics
            </a>
            <button
              className={`nav-link ${view === 'turner' ? 'nav-link--active' : ''}`}
              onClick={() => setView('turner')}
            >
              AI Supervisor Terminal
            </button>
            <a className={`nav-link ${view === 'dashboard' ? '' : 'nav-link--dim'}`} href="#alerts" onClick={() => setView('dashboard')}>
              Alert Escalation
            </a>
          </nav>

          {/* ── Detection Threshold (moved higher for accessibility) ──────── */}
          {view === 'dashboard' && (
            <div className="sidebar__section">
              <p className="section-label">Detection Threshold</p>
              <div className="conf-slider">
                <div className="conf-slider__readout">
                  <span>Min Confidence</span>
                  <strong>{(settings.confidenceThreshold * 100).toFixed(0)}%</strong>
                </div>
                <input
                  type="range"
                  className="conf-slider__input"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={settings.confidenceThreshold}
                  style={{ '--val': `${settings.confidenceThreshold * 100}%` } as CSSProperties}
                  onChange={(event) => update('confidenceThreshold', Number(event.target.value))}
                  aria-label="Detection confidence threshold"
                />
                <div className="conf-slider__labels">
                  <span>5%</span>
                  <span>Detect All</span>
                  <span>95%</span>
                </div>
              </div>
            </div>
          )}

          <nav className="sidebar__section" aria-label="Intelligence">
            <p className="section-label">Intelligence</p>
            <button
              className={`nav-link ${view === 'analytics' ? 'nav-link--active' : ''}`}
              onClick={() => setView('analytics')}
            >
              📊 Site Intelligence
            </button>
            <button
              className={`nav-link ${view === 'geoai' ? 'nav-link--active' : ''}`}
              onClick={() => setView('geoai')}
            >
              🗺️ GeoAI
            </button>
          </nav>

          <div className="sidebar__section">
            <p className="section-label">System</p>
            <button
              className="nav-link"
              onClick={handleGodModeAccess}
            >
              GOD MODE
            </button>
          </div>
        </div>

          <div className="sidebar__footer">
            <p className="section-label">System Access</p>
            <button
              className={`nav-link nav-link--settings ${view === 'settings' ? 'nav-link--active' : ''}`}
              onClick={() => setView('settings')}
            >
              ⚙ Settings
            </button>
          </div>
        </aside>
      )}

      <PageTransition viewKey={view}>
      {view === 'dashboard' && (
        <main className="dashboard">
          <header className="topbar">
            <div className="topbar__identity">
              <p className="eyebrow">Construction Safety Monitoring System</p>
              <h2>Command Dashboard</h2>
            </div>

            <div className="topbar__actions">
              <div className="mode-switch">
                {(['LIVE', 'VIDEO', 'IMAGE'] as DashboardMode[]).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    className={`mode-switch__button ${dashboardMode === mode ? 'mode-switch__button--active' : ''}`}
                    onClick={() => setDashboardMode(mode)}
                  >
                    {mode}
                  </button>
                ))}
              </div>

              <div className="topbar__status">
                <div className="status-pill status-pill--live">
                  <span className="status-pill__dot status-pill__dot--live" />
                  <span>SYSTEM LIVE</span>
                </div>
                <div className="telemetry-block">
                  <span>Feed Sync</span>
                  <strong>18ms</strong>
                </div>
                <div className="telemetry-block">
                  <span>Snapshot</span>
                  <strong>{snapshotTimestamp}</strong>
                </div>
              </div>
            </div>
          </header>

          <section className="dashboard__content">
            <section className="hero-grid">
              <div className="hero-grid__main panel" id="live">
                <div className="panel-heading">
                  <div>
                    <p className="section-label">{workspace.label}</p>
                    <h3>{workspace.title}</h3>
                  </div>
                  {!summaryCollapsed && <p className="panel-meta">{workspace.meta}</p>}
                </div>
                <div className="video-viewport">
                  {dashboardMode === 'LIVE' ? (
                  <LiveSurveillance />
                ) : (
                  <DetectionPanel mode={dashboardMode} />
                )}
                </div>
              </div>

              <aside className="hero-grid__side hero-grid__side--metrics panel" id="metrics">
                <div className="panel-heading">
                  <div>
                    <p className="section-label">Real-Time Indicators</p>
                    <h3>Compliance Metrics</h3>
                  </div>
                </div>
                <section className="metrics-vertical-stack">
                  {liveMetrics.map((metric, idx) => (
                    <MetricCard key={metric.label} {...metric} index={idx} />
                  ))}
                </section>
              </aside>
            </section>

            <section className="lower-grid">
              <div className="panel" id="alerts" style={{ gridColumn: '1 / -1' }}>
                {renderEscalationQueue(false)}
              </div>
            </section>
          </section>
        </main>
      )}

      {view === 'analytics' && (
        <main className="dashboard dashboard--analytics">
          <header className="topbar">
            <div className="topbar__brand">
              <p className="eyebrow">Enterprise Intelligence Engine</p>
              <h2>Site Intelligence</h2>
            </div>
            <div className="topbar__status">
              <div className="telemetry-block">
                <span>Data Sync</span>
                <strong>LIVE</strong>
              </div>
              <button className="stg-back-btn" onClick={() => setView('dashboard')}>
                <span className="stg-back-btn__icon">←</span>
                <span>Back to Dashboard</span>
              </button>
            </div>
          </header>
          <section className="analytics-section">
            <AnalyticsPage />
          </section>
        </main>
      )}

      {view === 'geoai' && (
        <main className="dashboard dashboard--geoai">
          <header className="topbar">
            <div className="topbar__brand">
              <p className="eyebrow">Spatial Intelligence</p>
              <h2>GeoAI Engine</h2>
            </div>
            <div className="topbar__status">
              <div className="telemetry-block">
                <span>Spatial Sync</span>
                <strong>CONNECTED</strong>
              </div>
              <button className="stg-back-btn" onClick={() => setView('dashboard')}>
                <span className="stg-back-btn__icon">←</span>
                <span>Back to Dashboard</span>
              </button>
            </div>
          </header>
          <section className="geoai-section" style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, padding: 0, overflow: 'hidden' }}>
            <GeoAIPage />
          </section>
        </main>
      )}

      {view === 'turner' && (
        <TurnerPage onBack={() => setView('dashboard')} onOpenSettings={() => setView('settings')} />
      )}

      {view === 'settings' && (
        <main className="dashboard dashboard--settings">
          <header className="topbar">
            <div className="topbar__brand">
              <p className="eyebrow">System Configuration</p>
              <h2>Terminal Settings</h2>
            </div>
            <div className="topbar__status">
              <button className="stg-back-btn" onClick={() => setView('dashboard')}>
                <span className="stg-back-btn__icon">←</span>
                <span>Back to Dashboard</span>
              </button>






            </div>
          </header>
          <section className="settings-section">
            <SettingsPanel />
          </section>
        </main>
      )}
        {view === 'brain' && (
          <BuildSightBrain 
            onBack={() => setView('dashboard')} 
            onLock={handleLockSystem}
          />
        )}
      </PageTransition>

      {/* ── SECURITY OVERLAYS ─────────────────────────────────────────────────── */}
      <AnimatePresence>
        {showGodLock && (
          <GodLock
            onUnlock={onGodUnlockSuccessful}
            onClose={() => setShowGodLock(false)}
          />
        )}
      </AnimatePresence>

      {/* ── PERSISTENT DETECTION STATUS BAR ──────────────────────────────────── */}
      <DetectionStatusBar />

      {/* ── FLOATING TURNER AI BUTTON ─────────────────────────────────────────── */}
      {view !== 'brain' && view !== 'turner' && (
        <button
          type="button"
          className={`turner-fab ${sidebarOpen ? 'turner-fab--open' : ''}`}
          onClick={() => setSidebarOpen((o) => !o)}
          aria-label={sidebarOpen ? 'Close Turner AI' : 'Open Turner AI'}
          title={sidebarOpen ? 'Close Turner AI' : 'Open Turner AI'}
        >
          <span className="turner-fab__ring" aria-hidden="true" />
          {sidebarOpen ? (
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          ) : (
            <svg viewBox="0 0 40 40" fill="none" width="22" height="22" aria-hidden="true">
              <polygon points="20,2 36,11 36,29 20,38 4,29 4,11" stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.7" />
              <circle cx="20" cy="20" r="5" fill="currentColor" opacity="0.25" />
              <circle cx="20" cy="20" r="2.5" fill="currentColor" />
              <line x1="20" y1="15" x2="20" y2="2"  stroke="currentColor" strokeWidth="1" opacity="0.6" />
              <line x1="20" y1="25" x2="20" y2="38" stroke="currentColor" strokeWidth="1" opacity="0.6" />
              <line x1="13" y1="17" x2="4"  y2="11" stroke="currentColor" strokeWidth="1" opacity="0.6" />
              <line x1="27" y1="17" x2="36" y2="11" stroke="currentColor" strokeWidth="1" opacity="0.6" />
              <line x1="13" y1="23" x2="4"  y2="29" stroke="currentColor" strokeWidth="1" opacity="0.6" />
              <line x1="27" y1="23" x2="36" y2="29" stroke="currentColor" strokeWidth="1" opacity="0.6" />
              <circle cx="20" cy="2"  r="1.8" fill="currentColor" opacity="0.8" />
              <circle cx="20" cy="38" r="1.8" fill="currentColor" opacity="0.8" />
              <circle cx="4"  cy="11" r="1.8" fill="currentColor" opacity="0.8" />
              <circle cx="36" cy="11" r="1.8" fill="currentColor" opacity="0.8" />
              <circle cx="4"  cy="29" r="1.8" fill="currentColor" opacity="0.8" />
              <circle cx="36" cy="29" r="1.8" fill="currentColor" opacity="0.8" />
            </svg>
          )}
          <span className="turner-fab__label">{sidebarOpen ? 'Close' : 'Turner AI'}</span>
        </button>
      )}

      {/* ── TURNER SIDEBAR ────────────────────────────────────────────────────── */}
      {sidebarOpen && view !== 'brain' && view !== 'turner' && (
        <TurnerSidebar
          onExpand={() => { setSidebarOpen(false); setView('turner') }}
          onClose={() => setSidebarOpen(false)}
        />
      )}
    </div>
)
}

function App() {
return (
  <ErrorBoundary>
    <SettingsProvider>
      <ErrorBoundary>
        <DetectionStatsProvider>
          <ErrorBoundary>
            <AppInner />
          </ErrorBoundary>
        </DetectionStatsProvider>
      </ErrorBoundary>
    </SettingsProvider>
  </ErrorBoundary>
)
}

export default App
