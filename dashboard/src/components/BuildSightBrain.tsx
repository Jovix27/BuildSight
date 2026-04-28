import { useState, useMemo, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useSettings } from '../SettingsContext'
import { useDetectionStats } from '../DetectionStatsContext'
import { useGeoAIWebSocket } from '../hooks/useGeoAIWebSocket'
import { GeoAIMap } from './GeoAIMap'
import { VLMActivityFeed } from './VLMActivityFeed'
import { DynamicZoneEditor } from './DynamicZoneEditor'
import type { DynamicZone } from '../types/geoai'
import TurnerOrb3D from './TurnerOrb3D'
import './BuildSightBrain.css'

interface BuildSightBrainProps {
  onBack: () => void
  onLock: () => void
}

const RECOMMENDATIONS = [
  { id: 'rec-1', text: 'Deploy supervisor to East Scaffold Zone', tag: 'HIGH',    priority: 'high' },
  { id: 'rec-2', text: 'Increase PPE enforcement on Level 06',    tag: 'MED',     priority: 'medium' },
  { id: 'rec-3', text: 'Pause excavation activity in North Yard', tag: 'CRIT',    priority: 'critical' },
  { id: 'rec-4', text: 'Inspect scaffolding integrity post wind', tag: 'MED',     priority: 'medium' },
  { id: 'rec-5', text: 'Re-route compute to Sector 4 high density', tag: 'HIGH', priority: 'high' },
]

const ORBITING_NODES = [
  { id: 'n1',  label: 'Detection Engine',    status: 'stable', controlGroup: 'thresholds' },
  { id: 'n2',  label: 'GeoAI',              status: 'stable', controlGroup: 'geoai' },
  { id: 'n3',  label: 'Turner AI',          status: 'stable', controlGroup: 'routing' },
  { id: 'n4',  label: 'Risk Engine',        status: 'warn',   controlGroup: 'sensitivity' },
  { id: 'n5',  label: 'Worker Tracking',    status: 'stable', controlGroup: 'tracking' },
  { id: 'n6',  label: 'PPE Compliance',     status: 'alert',  controlGroup: 'thresholds' },
  { id: 'n7',  label: 'Escalation',         status: 'stable', controlGroup: 'sensitivity' },
  { id: 'n8',  label: 'Site Intelligence',  status: 'stable', controlGroup: 'geoai' },
  { id: 'n9',  label: 'CCTV Streams',       status: 'stable', controlGroup: 'tracking' },
  { id: 'n10', label: 'Heatmaps',           status: 'warn',   controlGroup: 'geoai' },
  { id: 'n11', label: 'Zone Monitor',       status: 'stable', controlGroup: 'geoai' },
  { id: 'n12', label: 'Alert System',       status: 'stable', controlGroup: 'system' },
]

const SPATIAL_ZONES = [
  { id: 'sz-1', label: 'EXCAVATION_A', coords: { x: -300, y: -200 }, color: '#3b82f6' },
  { id: 'sz-2', label: 'CRANE_RADIUS_01', coords: { x: 320,  y: -150 }, color: '#ff3b30' },
  { id: 'sz-3', label: 'SCAFFOLD_Z4',    coords: { x: 280,  y: 220 },  color: '#cbd5e1' },
  { id: 'sz-4', label: 'RESTRICTED_B',   coords: { x: -250, y: 180 },  color: '#ff3b30' },
]

// Stable camera statuses — computed once, not on every render
const CAM_STATUSES = [1, 2, 3, 4, 5, 6].map(id => ({
  id,
  label: `CAM-0${id}`,
  status: id === 3 ? 'Lagging' : 'Optimal',
}))

export function BuildSightBrain({ onBack, onLock }: BuildSightBrainProps) {
  const { settings, update } = useSettings()
  const { stats } = useDetectionStats()
  const [controlsOpen, setControlsOpen]     = useState(false)
  const [activeControlGroup, setActiveControlGroup] = useState<string | null>(null)
  const [activeTab, setActiveTab]           = useState<'geoai' | 'turner' | 'cameras'>('geoai')
  const [timeStr, setTimeStr]               = useState('')
  const [turnerInput, setTurnerInput]       = useState('')
  const [turnerChat, setTurnerChat]         = useState([
    { id: 1, text: 'Analyzing site stream. Ensemble model confirms 98% tracking confidence.', type: 'normal' },
    { id: 2, text: 'Anomalous movement in Sector B detected. Re-routing compute for enhanced fidelity.', type: 'warn' },
    { id: 3, text: 'Daily Briefing: Elevated risk near Alpha Zone due to heavy machinery.', type: 'normal' },
  ])

  // GeoAI live data
  const { data: geoData, isDemoMode: geoDemo } = useGeoAIWebSocket()

  // GEO AI sub-tab: map | vlm | zones
  const [geoSubTab, setGeoSubTab] = useState<'map' | 'vlm' | 'zones'>('map')

  // Dynamic zones (owned here so GeoAIMap and DynamicZoneEditor share the same list)
  const [dynamicZones, setDynamicZones] = useState<DynamicZone[]>([])

  // Map layer toggles
  const [mapLayers, setMapLayers] = useState({
    zones: true,
    labels: true,
    cameraFOV: true,
    heatmap: true,
    workers: true,
  })
  const [mapViewMode, setMapViewMode] = useState<'tactical' | 'satellite'>('tactical')

  const toggleLayer = (key: keyof typeof mapLayers) =>
    setMapLayers(prev => ({ ...prev, [key]: !prev[key] }))

  const [godSettings, setGodSettings] = useState({
    vestConf: 0.5, sceneConf: 0.6, s1s4Tuning: 0.8, wbfThreshold: 0.45,
    crowdSens: 0.7, heatmapSens: 0.8, zoneSens: 0.9, ppeSens: 0.85, riskSens: 0.7,
    workerPersist: 30, ghostPersist: 15, scenePersist: 60, polyOpacity: 0.4,
    heatmapOpacity: 0.6, fpsTarget: 60, tileInference: true,
    yolo11Weight: 'best_v11.pt', yolo26Weight: 'experimental_v26.pt',
    geminiVerify: true, alertRouting: 'all', theme: 'cyber-dark',
    camPriority: 'smart', optimizeMode: 'live',
  })

  // ── Geo-Neural Integration State ──
  const geoRiskIntensity = useMemo(() => {
    if (!geoData?.kpi?.avg_site_risk) return 0;
    return Math.min(geoData.kpi.avg_site_risk / 10, 1);
  }, [geoData]);

  // Smooth the risk signal only until it converges; do not keep a perpetual RAF loop alive.
  const [smoothedRisk, setSmoothedRisk] = useState(0)
  useEffect(() => {
    let animationFrame: number | null = null

    const step = () => {
      let shouldContinue = false

      setSmoothedRisk((prev) => {
        const diff = geoRiskIntensity - prev
        if (Math.abs(diff) < 0.001) {
          return geoRiskIntensity
        }

        shouldContinue = true
        return prev + diff * 0.08
      })

      if (shouldContinue) {
        animationFrame = requestAnimationFrame(step)
      }
    }

    animationFrame = requestAnimationFrame(step)

    return () => {
      if (animationFrame !== null) {
        cancelAnimationFrame(animationFrame)
      }
    }
  }, [geoRiskIntensity])

  const [lastGeoAlert, setLastGeoAlert] = useState<string | null>(null);

  // Turner reactions to GeoAI events
  useEffect(() => {
    if (geoRiskIntensity > 0.7 && activeTab === 'geoai') {
      const msg = `Elevated spatial risk detected (${(geoRiskIntensity * 10).toFixed(1)}). Turner AI prioritizing GeoAI analysis.`;
      if (lastGeoAlert !== msg) {
        setTurnerChat(prev => [{ id: Date.now(), text: msg, type: 'warn' }, ...prev.slice(0, 4)]);
        setLastGeoAlert(msg);
      }
    }
  }, [geoRiskIntensity, activeTab, lastGeoAlert]);

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeStr(new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' IST')
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  const escalationState = useMemo(() => {
    if (!stats) return 'stable'
    const workers = Math.max(1, stats.totalWorkers || 1)
    if (stats.proximityViolations > 5 || (stats.totalWorkers > 20 && (stats.helmetsDetected / workers) < 0.5)) return 'critical'
    if (stats.proximityViolations > 0 || (stats.helmetsDetected / workers) < 0.8) return 'elevated'
    return 'stable'
  }, [stats])

  const riskIntensity = useMemo(() => {
    if (!stats) return 0
    const workers = Math.max(1, stats.totalWorkers || 1)
    const compliance = (stats.helmetsDetected + stats.vestsDetected) / (workers * 2)
    return Math.max(0, 1 - compliance)
  }, [stats])

  // Derived metrics — null-safe, no misleading 100% when no data
  const helmetRate = stats?.totalWorkers
    ? ((stats.helmetsDetected / stats.totalWorkers) * 100).toFixed(1) + '%'
    : '—'
  const vestRate = stats?.totalWorkers
    ? ((stats.vestsDetected / stats.totalWorkers) * 100).toFixed(1) + '%'
    : '—'
  const proximityViol = stats ? String(stats.proximityViolations || 0).padStart(2, '0') : '—'

  const updateGodSetting = (key: keyof typeof godSettings, value: any) =>
    setGodSettings(prev => ({ ...prev, [key]: value }))

  const handleTurnerSubmit = () => {
    if (!turnerInput.trim()) return
    setTurnerChat(prev => [...prev, { id: Date.now(), text: turnerInput, type: 'user' }])
    setTurnerInput('')
    setTimeout(() => {
      setTurnerChat(prev => [...prev, { id: Date.now(), text: 'Command acknowledged. Adjusting neural parameters...', type: 'normal' }])
    }, 1000)
  }

  const openControlsFor = (group: string) => {
    setControlsOpen(true)
    setActiveControlGroup(group)
    setTimeout(() => {
      const el = document.getElementById(`god-group-${group}`)
      if (el && group !== 'all') el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 100)
  }

  const showGroup = (group: string) =>
    !activeControlGroup || activeControlGroup === 'all' || activeControlGroup === group

  return (
    <div className="god-container">
      <div className="god-atmosphere" />
      <div className="god-particles" />

      {/* ── TOP COMMAND STRIP ─────────────────────────────────────────────── */}
      <header className="god-topbar">
        <div className="god-topbar-left">
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="god-brand">
            <h1>GOD MODE</h1>
            <span className="god-version">MASTER INTELLIGENCE HUB · v5.0</span>
          </motion.div>
          <div className="god-status-block">
            <span className={`god-pulse-dot god-pulse-dot--${escalationState}`} />
            <span className="god-status-text">SYSTEM {escalationState.toUpperCase()}</span>
          </div>
        </div>

        <div className="god-kpi-strip">
          <div className="god-kpi-item" onClick={() => openControlsFor('sensitivity')} style={{ cursor: 'pointer' }}>
            <span className="god-kpi-label">SITE RISK SCORE</span>
            <span className="god-kpi-value" style={{ color: riskIntensity > 0.5 ? '#ff2a2a' : '#f8fafc' }}>
              {(riskIntensity * 100).toFixed(1)}%
            </span>
          </div>
          <div className="god-kpi-sep" />
          <div className="god-kpi-item" onClick={() => openControlsFor('thresholds')} style={{ cursor: 'pointer' }}>
            <span className="god-kpi-label">ACTIVE WORKERS</span>
            <span className="god-kpi-value">{stats?.totalWorkers ?? '—'}</span>
          </div>
          <div className="god-kpi-sep" />
          <div className="god-kpi-item" onClick={() => openControlsFor('thresholds')} style={{ cursor: 'pointer' }}>
            <span className="god-kpi-label">HELMET COMPLIANCE</span>
            <span className="god-kpi-value">{helmetRate}</span>
          </div>
          <div className="god-kpi-sep" />
          <div className="god-kpi-item">
            <span className="god-kpi-label">SYS CONFIDENCE</span>
            <span className="god-kpi-value" style={{ color: '#00ff80' }}>
              {stats?.avgConfidence ? (stats.avgConfidence * 100).toFixed(1) + '%' : '—'}
            </span>
          </div>
        </div>

        <div className="god-topbar-right">
          <div className="god-time">{timeStr}</div>
          <button className="god-btn" onClick={onLock}>LOCK SYSTEM</button>
          <button className="god-btn" onClick={onBack}>EXIT GOD MODE</button>
        </div>
      </header>

      {/* ── MAIN VIEWPORT ─────────────────────────────────────────────────── */}
      <main className="god-layout">

        {/* LEFT HUD */}
        <aside className="god-column god-column-left">

          {/* Safety & Compliance */}
          <motion.div className="god-panel" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
            <div className="god-panel-header">
              <h3>SAFETY &amp; COMPLIANCE</h3>
            </div>
            <div className="god-safety-metrics">
              <div className="god-metric-row" onClick={() => openControlsFor('sensitivity')}>
                <span>Proximity Violations</span>
                <strong style={{ color: stats?.proximityViolations > 0 ? '#ff2a2a' : '#00ff80' }}>
                  {proximityViol}
                </strong>
              </div>
              <div className="god-metric-row" onClick={() => openControlsFor('thresholds')}>
                <span>Vest Coverage Rate</span>
                <strong style={{ color: vestRate === '—' ? 'rgba(255,255,255,0.35)' : undefined }}>{vestRate}</strong>
              </div>
              <div className="god-metric-row" onClick={() => openControlsFor('thresholds')}>
                <span>Helmet Coverage Rate</span>
                <strong style={{ color: helmetRate === '—' ? 'rgba(255,255,255,0.35)' : undefined }}>{helmetRate}</strong>
              </div>
              <div className="god-metric-row" onClick={() => openControlsFor('routing')}>
                <span>Model Drift (7d avg)</span>
                <strong style={{ color: '#00ff80' }}>Stable</strong>
              </div>
              <div className="god-metric-row" onClick={() => openControlsFor('tracking')}>
                <span>Global FPS Target</span>
                <strong>{godSettings.fpsTarget} FPS</strong>
              </div>
            </div>
          </motion.div>

          {/* Turner AI Recommendations — separate panel */}
          <motion.div className="god-panel god-flex-grow" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
            <div className="god-panel-header">
              <h3>TURNER AI RECOMMENDATIONS</h3>
              <span className="god-badge god-badge--amber">{RECOMMENDATIONS.length} ACTIVE</span>
            </div>
            <div className="god-action-list">
              {RECOMMENDATIONS.map((rec, i) => (
                <motion.div
                  key={rec.id}
                  className={`god-action-card god-action-card--${rec.priority}`}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.25 + i * 0.07 }}
                  onClick={() => openControlsFor('routing')}
                >
                  <span className={`god-action-tag god-action-tag--${rec.priority}`}>{rec.tag}</span>
                  <span className="god-action-text">{rec.text}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>

        </aside>

        {/* CENTER STAGE — Neural Brain */}
        <section className="god-core-zone">
          <div className="god-brain-container">
            {/* ── FLAGSHIP 3D BRAIN CORE ── */}
            <div className="god-brain-orb-wrap">
              <TurnerOrb3D 
                amplitude={0} 
                state="idle"
                size={500}
                variant="god"
                riskLevel={smoothedRisk}
              />
            </div>

            <svg className="god-brain-connections">
              <circle cx="50%" cy="50%" r="220" stroke="rgba(226,232,240,0.1)" strokeWidth="1" fill="none" />
              
              {/* Dynamic Geo-Neural Link */}
              <motion.line 
                x1="50%" y1="50%" x2="50%" y2="100%" 
                stroke={smoothedRisk > 0.5 ? "rgba(226, 232, 240, 0.4)" : "rgba(226, 232, 240, 0.1)"}
                strokeWidth={2 + smoothedRisk * 3}
                animate={{
                  opacity: [0.1, 0.4, 0.1],
                  strokeDashoffset: [0, -20]
                }}
                transition={{
                  repeat: Infinity,
                  duration: 2 - (smoothedRisk * 0.5), // Subtle change, not jerky
                  ease: "linear"
                }}
                strokeDasharray="10 5"
              />

              <line x1="50%" y1="50%" x2="20%" y2="20%" />
              <line x1="50%" y1="50%" x2="80%" y2="80%" />
              <line x1="50%" y1="50%" x2="80%" y2="20%" />
              <line x1="50%" y1="50%" x2="20%" y2="80%" />
            </svg>

            {ORBITING_NODES.map((node, i) => {
              const angle  = (i / ORBITING_NODES.length) * Math.PI * 2
              const radius = i % 2 === 0 ? 180 : 250
              const x = Math.cos(angle) * radius
              const y = Math.sin(angle) * radius
              return (
                <motion.div
                  key={node.id}
                  className="god-brain-node"
                  style={{ left: `calc(50% + ${x}px)`, top: `calc(50% + ${y}px)` }}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.5 + i * 0.05 }}
                  onClick={() => openControlsFor(node.controlGroup)}
                >
                  <span className={`god-brain-node-dot god-brain-node-dot--${node.status}`} />
                  {node.label}
                </motion.div>
              )
            })}

            {/* ── SPATIAL AWARENESS LAYERS (SAM Integration) ────────────────── */}
            {SPATIAL_ZONES.map((zone, i) => (
              <motion.div
                key={zone.id}
                className="god-spatial-zone"
                style={{ 
                  left: `calc(50% + ${zone.coords.x}px)`, 
                  top: `calc(50% + ${zone.coords.y}px)`,
                  '--zone-color': zone.color 
                } as any}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ 
                  opacity: [0.3, 0.6, 0.3],
                  scale: [1, 1.05, 1],
                  transition: { repeat: Infinity, duration: 3 + i, ease: "easeInOut" }
                }}
              >
                <div className="god-spatial-tag">
                  <span className="god-spatial-ptr" />
                  <span className="god-spatial-label">{zone.label}</span>
                  <span className="god-spatial-status">SAM_L2_ACTIVE</span>
                </div>
                <svg className="god-spatial-shape" width="120" height="80">
                  <path 
                    d="M10,10 L110,10 L110,70 L10,70 Z" 
                    fill="none" 
                    stroke={zone.color} 
                    strokeWidth="1" 
                    strokeDasharray="4 2"
                  />
                </svg>
              </motion.div>
            ))}

            <motion.div
              className="god-brain-center"
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 1, ease: 'easeOut' }}
              onClick={() => openControlsFor('all')}
              style={{ zIndex: 10 }}
            >
              <div className="god-brain-core-text">
                BUILDSIGHT
                <span>MASTER OVERRIDE</span>
              </div>
            </motion.div>
          </div>
          <div className="god-core-overlay-text">NEURAL NETWORK SYNCHRONIZED</div>
        </section>

        {/* RIGHT HUD */}
        <aside className="god-column god-column-right">

          <div className="god-tabs">
            <button className={`god-btn ${activeTab === 'geoai'   ? 'god-btn-primary' : ''}`} onClick={() => setActiveTab('geoai')}>GEO AI</button>
            <button className={`god-btn ${activeTab === 'turner'  ? 'god-btn-primary' : ''}`} onClick={() => setActiveTab('turner')}>TURNER AI</button>
            <button className={`god-btn ${activeTab === 'cameras' ? 'god-btn-primary' : ''}`} onClick={() => setActiveTab('cameras')}>CAMERAS</button>
          </div>

          <motion.div className="god-panel god-flex-grow" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>

            {activeTab === 'geoai' && (
              <div className="god-tab-content god-tab-content--geoai">

                {/* ── GEO AI sub-tab bar ──────────────────────────────── */}
                <div className="god-subtab-bar">
                  <button className={`god-btn god-btn-sm ${geoSubTab === 'map'   ? 'god-btn-primary' : ''}`} onClick={() => setGeoSubTab('map')}>MAP</button>
                  <button className={`god-btn god-btn-sm ${geoSubTab === 'vlm'   ? 'god-btn-primary' : ''}`} onClick={() => setGeoSubTab('vlm')}>VLM</button>
                  <button className={`god-btn god-btn-sm ${geoSubTab === 'zones' ? 'god-btn-primary' : ''}`} onClick={() => setGeoSubTab('zones')}>ZONES</button>
                  <span className={`god-badge god-subtab-badge ${geoDemo ? 'god-badge--amber' : 'god-badge--ok'}`}>
                    {geoDemo ? 'DEMO' : 'LIVE'}
                  </span>
                </div>

                {/* ── MAP sub-tab ─────────────────────────────────────── */}
                {geoSubTab === 'map' && (
                  <>
                    {/* View mode + layer toggles */}
                    <div className="god-geoai-toolbar">
                      <button className={`god-btn god-btn-xs ${mapViewMode === 'tactical'  ? 'god-btn-primary' : ''}`} onClick={() => setMapViewMode('tactical')}>TACT</button>
                      <button className={`god-btn god-btn-xs ${mapViewMode === 'satellite' ? 'god-btn-primary' : ''}`} onClick={() => setMapViewMode('satellite')}>SAT</button>
                      {(Object.keys(mapLayers) as Array<keyof typeof mapLayers>).map(key => (
                        <button
                          key={key}
                          className={`god-btn god-btn-xs ${mapLayers[key] ? 'god-btn-primary' : ''}`}
                          onClick={() => toggleLayer(key)}
                        >
                          {key === 'cameraFOV' ? 'FOV' : key === 'heatmap' ? 'HEAT' : key === 'workers' ? 'W' : key === 'zones' ? 'Z' : 'L'}
                        </button>
                      ))}
                    </div>

                    {/* Live map */}
                    <div className="god-geoai-map">
                      <GeoAIMap
                        data={geoData}
                        showZones={mapLayers.zones}
                        showLabels={mapLayers.labels}
                        showCameraFOV={mapLayers.cameraFOV}
                        showHeatmap={mapLayers.heatmap}
                        showWorkers={mapLayers.workers}
                        heatmapOpacity={godSettings.heatmapOpacity}
                        viewMode={mapViewMode}
                        dynamicZones={dynamicZones}
                      />
                    </div>

                    {/* KPI strip */}
                    <div className="god-safety-metrics" style={{ flexShrink: 0 }}>
                      <div className="god-metric-row">
                        <span>Active Workers</span>
                        <strong>{geoData?.kpi?.active_workers ?? '—'}</strong>
                      </div>
                      <div className="god-metric-row">
                        <span>PPE Compliance</span>
                        <strong style={{ color: (geoData?.kpi?.ppe_compliance ?? 100) < 60 ? '#ff2a2a' : '#00ff80' }}>
                          {geoData?.kpi ? geoData.kpi.ppe_compliance.toFixed(0) + '%' : '—'}
                        </strong>
                      </div>
                      <div className="god-metric-row">
                        <span>Critical Alerts</span>
                        <strong style={{ color: (geoData?.kpi?.critical_alerts ?? 0) > 0 ? '#ff2a2a' : '#00ff80' }}>
                          {geoData?.kpi?.critical_alerts ?? '—'}
                        </strong>
                      </div>
                      <div className="god-metric-row">
                        <span>Avg Site Risk</span>
                        <strong style={{ color: (geoData?.kpi?.avg_site_risk ?? 0) > 0.5 ? '#f8fafc' : '#00ff80' }}>
                          {geoData?.kpi ? (geoData.kpi.avg_site_risk * 100).toFixed(0) + '%' : '—'}
                        </strong>
                      </div>
                    </div>

                    {/* Active spatial events */}
                    {geoData?.events && geoData.events.filter(e => e.state === 'ACTIVE').length > 0 && (
                      <div style={{ flexShrink: 0, marginTop: 6 }}>
                        <p className="god-section-label">ACTIVE EVENTS</p>
                        {geoData.events.filter(e => e.state === 'ACTIVE').slice(0, 3).map(ev => (
                          <div
                            key={ev.event_id}
                            className={`god-action-card god-action-card--${ev.priority === 'CRITICAL' ? 'critical' : ev.priority === 'WARNING' ? 'high' : 'medium'}`}
                            style={{ marginBottom: 4 }}
                          >
                            <span className={`god-action-tag god-action-tag--${ev.priority === 'CRITICAL' ? 'critical' : ev.priority === 'WARNING' ? 'high' : 'medium'}`}>
                              {ev.priority === 'CRITICAL' ? 'CRIT' : ev.priority === 'WARNING' ? 'WARN' : 'INFO'}
                            </span>
                            <span className="god-action-text">{ev.message}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}

                {/* ── VLM sub-tab ─────────────────────────────────────── */}
                {geoSubTab === 'vlm' && <VLMActivityFeed />}

                {/* ── ZONES sub-tab ───────────────────────────────────── */}
                {geoSubTab === 'zones' && (
                  <DynamicZoneEditor onZonesChange={setDynamicZones} />
                )}
              </div>
            )}

            {activeTab === 'turner' && (
              <div className="god-tab-content">
                <div className="god-panel-header">
                  <h3>TURNER AI COMMAND</h3>
                </div>
                <div className="god-turner-chat">
                  {turnerChat.map(chat => (
                    <div
                      key={chat.id}
                      className={`god-chat-bubble ${chat.type === 'warn' ? 'god-chat-bubble--warn' : ''} ${chat.type === 'user' ? 'god-chat-bubble--user' : ''}`}
                    >
                      {chat.text}
                    </div>
                  ))}
                </div>
                <div className="god-turner-input">
                  <input
                    type="text"
                    placeholder="Command Turner AI..."
                    value={turnerInput}
                    onChange={e => setTurnerInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleTurnerSubmit()}
                  />
                  <button className="god-btn-primary" onClick={handleTurnerSubmit}>EXECUTE</button>
                </div>
              </div>
            )}

            {activeTab === 'cameras' && (
              <div className="god-tab-content">
                <div className="god-panel-header">
                  <h3>SITE HEALTH MONITORING</h3>
                </div>
                <div className="god-camera-list">
                  {CAM_STATUSES.map(cam => (
                    <div key={cam.id} className="god-camera-item" onClick={() => openControlsFor('tracking')}>
                      <div className="god-camera-stream" />
                      <div className="god-camera-info">
                        <strong>{cam.label}</strong>
                        <span className={cam.status === 'Lagging' ? 'god-cam-lag' : 'god-cam-ok'}>
                          {cam.status} · 60 FPS
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </motion.div>
        </aside>

      </main>

      {/* ── FOOTER CONTROLS DRAWER ────────────────────────────────────────── */}
      <AnimatePresence>
        {controlsOpen && (
          <motion.div
            className="god-drawer god-drawer--open"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="god-drawer-overlay" onClick={() => { setControlsOpen(false); setActiveControlGroup(null) }} />
            <motion.div
              className="god-drawer-content"
              initial={{ y: '100%' }}
              animate={{ y: 0 }}
              exit={{ y: '100%' }}
              transition={{ type: 'spring', stiffness: 260, damping: 28 }}
            >
              <div className="god-drawer-header">
                <h3>{activeControlGroup === 'all' ? 'MASTER SYSTEM OVERRIDES' : 'SPECIFIC SYSTEM OVERRIDE'}</h3>
                <button className="god-btn god-btn-primary" onClick={() => { setControlsOpen(false); setActiveControlGroup(null) }}>
                  CLOSE OVERRIDES
                </button>
              </div>

              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <div className={`god-param-grid ${activeControlGroup && activeControlGroup !== 'all' ? 'god-param-grid--single' : ''}`}>

                  {showGroup('thresholds') && (
                    <div id="god-group-thresholds" className="god-param-group">
                      <p className="god-section-label">DETECTION THRESHOLDS</p>
                      <label className="god-slider">
                        <div><span>Worker</span><strong>{((settings?.workerConf ?? 0.5) * 100).toFixed(0)}%</strong></div>
                        <input type="range" min="0" max="1" step="0.01" value={settings?.workerConf ?? 0.5} onChange={e => update('workerConf', Number(e.target.value))} />
                      </label>
                      <label className="god-slider">
                        <div><span>Helmet</span><strong>{((settings?.helmetConf ?? 0.5) * 100).toFixed(0)}%</strong></div>
                        <input type="range" min="0" max="1" step="0.01" value={settings?.helmetConf ?? 0.5} onChange={e => update('helmetConf', Number(e.target.value))} />
                      </label>
                      <label className="god-slider">
                        <div><span>Safety Vest</span><strong>{(godSettings.vestConf * 100).toFixed(0)}%</strong></div>
                        <input type="range" min="0" max="1" step="0.01" value={godSettings.vestConf} onChange={e => updateGodSetting('vestConf', Number(e.target.value))} />
                      </label>
                      <label className="god-slider">
                        <div><span>Scene Classification</span><strong>{(godSettings.sceneConf * 100).toFixed(0)}%</strong></div>
                        <input type="range" min="0" max="1" step="0.01" value={godSettings.sceneConf} onChange={e => updateGodSetting('sceneConf', Number(e.target.value))} />
                      </label>
                    </div>
                  )}

                  {showGroup('sensitivity') && (
                    <div id="god-group-sensitivity" className="god-param-group">
                      <p className="god-section-label">ENGINE SENSITIVITY</p>
                      <label className="god-slider">
                        <div><span>S1–S4 Tuning</span><strong>{(godSettings.s1s4Tuning * 100).toFixed(0)}%</strong></div>
                        <input type="range" min="0" max="1" step="0.01" value={godSettings.s1s4Tuning} onChange={e => updateGodSetting('s1s4Tuning', Number(e.target.value))} />
                      </label>
                      <label className="god-slider">
                        <div><span>WBF Threshold</span><strong>{(godSettings.wbfThreshold * 100).toFixed(0)}%</strong></div>
                        <input type="range" min="0" max="1" step="0.01" value={godSettings.wbfThreshold} onChange={e => updateGodSetting('wbfThreshold', Number(e.target.value))} />
                      </label>
                      <label className="god-slider">
                        <div><span>Crowd Density</span><strong>{(godSettings.crowdSens * 100).toFixed(0)}%</strong></div>
                        <input type="range" min="0" max="1" step="0.01" value={godSettings.crowdSens} onChange={e => updateGodSetting('crowdSens', Number(e.target.value))} />
                      </label>
                      <label className="god-slider">
                        <div><span>Risk Escalation</span><strong>{(godSettings.riskSens * 100).toFixed(0)}%</strong></div>
                        <input type="range" min="0" max="1" step="0.01" value={godSettings.riskSens} onChange={e => updateGodSetting('riskSens', Number(e.target.value))} />
                      </label>
                    </div>
                  )}

                  {showGroup('routing') && (
                    <div id="god-group-routing" className="god-param-group">
                      <p className="god-section-label">AI ROUTING &amp; MODELS</p>
                      <label className="god-select-wrap">
                        <span>Model Selection</span>
                        <select className="god-select" value={settings?.selectedModel || 'ensemble'} onChange={e => update('selectedModel', e.target.value)}>
                          <option value="ensemble">Balanced Ensemble v4.2</option>
                          <option value="yolov11">YOLOv11 Precision</option>
                          <option value="yolov26">YOLOv26 Speed</option>
                        </select>
                      </label>
                      <label className="god-select-wrap">
                        <span>YOLOv11 Weights</span>
                        <select className="god-select" value={godSettings.yolo11Weight} onChange={e => updateGodSetting('yolo11Weight', e.target.value)}>
                          <option value="best_v11.pt">best_v11.pt (Production)</option>
                          <option value="experimental_v11.pt">experimental_v11.pt</option>
                        </select>
                      </label>
                      <label className="god-toggle-wrap">
                        <span>Gemini Verification</span>
                        <div className={`god-toggle ${godSettings.geminiVerify ? 'god-toggle--active' : ''}`} onClick={() => updateGodSetting('geminiVerify', !godSettings.geminiVerify)}>
                          <div className="god-toggle-knob" />
                        </div>
                      </label>
                      <label className="god-toggle-wrap">
                        <span>Tile Inference</span>
                        <div className={`god-toggle ${godSettings.tileInference ? 'god-toggle--active' : ''}`} onClick={() => updateGodSetting('tileInference', !godSettings.tileInference)}>
                          <div className="god-toggle-knob" />
                        </div>
                      </label>
                    </div>
                  )}

                  {showGroup('geoai') && (
                    <div id="god-group-geoai" className="god-param-group">
                      <p className="god-section-label">GEO-AI CONTROLS</p>
                      <label className="god-slider">
                        <div><span>Hazard Polygon Opacity</span><strong>{(godSettings.polyOpacity * 100).toFixed(0)}%</strong></div>
                        <input type="range" min="0" max="1" step="0.01" value={godSettings.polyOpacity} onChange={e => updateGodSetting('polyOpacity', Number(e.target.value))} />
                      </label>
                      <label className="god-slider">
                        <div><span>Heatmap Opacity</span><strong>{(godSettings.heatmapOpacity * 100).toFixed(0)}%</strong></div>
                        <input type="range" min="0" max="1" step="0.01" value={godSettings.heatmapOpacity} onChange={e => updateGodSetting('heatmapOpacity', Number(e.target.value))} />
                      </label>
                      <label className="god-slider">
                        <div><span>Zone Sensitivity</span><strong>{(godSettings.zoneSens * 100).toFixed(0)}%</strong></div>
                        <input type="range" min="0" max="1" step="0.01" value={godSettings.zoneSens} onChange={e => updateGodSetting('zoneSens', Number(e.target.value))} />
                      </label>
                    </div>
                  )}

                  {showGroup('tracking') && (
                    <div id="god-group-tracking" className="god-param-group">
                      <p className="god-section-label">TRACKING PERSISTENCE</p>
                      <label className="god-slider">
                        <div><span>Worker Persistence (frames)</span><strong>{godSettings.workerPersist}</strong></div>
                        <input type="range" min="1" max="120" step="1" value={godSettings.workerPersist} onChange={e => updateGodSetting('workerPersist', Number(e.target.value))} />
                      </label>
                      <label className="god-slider">
                        <div><span>Ghost Persistence (frames)</span><strong>{godSettings.ghostPersist}</strong></div>
                        <input type="range" min="1" max="60" step="1" value={godSettings.ghostPersist} onChange={e => updateGodSetting('ghostPersist', Number(e.target.value))} />
                      </label>
                      <label className="god-slider">
                        <div><span>Target FPS</span><strong>{godSettings.fpsTarget}</strong></div>
                        <input type="range" min="15" max="120" step="5" value={godSettings.fpsTarget} onChange={e => updateGodSetting('fpsTarget', Number(e.target.value))} />
                      </label>
                    </div>
                  )}

                  {showGroup('system') && (
                    <div id="god-group-system" className="god-param-group">
                      <p className="god-section-label">SYSTEM PREFERENCES</p>
                      <label className="god-select-wrap">
                        <span>Alert Routing</span>
                        <select className="god-select" value={godSettings.alertRouting} onChange={e => updateGodSetting('alertRouting', e.target.value)}>
                          <option value="all">All Channels</option>
                          <option value="dashboard">Dashboard Only</option>
                          <option value="sms">SMS + Dashboard</option>
                        </select>
                      </label>
                      <label className="god-select-wrap">
                        <span>Optimization Mode</span>
                        <select className="god-select" value={godSettings.optimizeMode} onChange={e => updateGodSetting('optimizeMode', e.target.value)}>
                          <option value="live">Live CCTV (Low Latency)</option>
                          <option value="upload">Video Upload (Max Accuracy)</option>
                        </select>
                      </label>
                      <label className="god-select-wrap">
                        <span>Theme</span>
                        <select className="god-select" value={godSettings.theme} onChange={e => updateGodSetting('theme', e.target.value)}>
                          <option value="cyber-dark">Cyber Dark</option>
                          <option value="contrast">High Contrast</option>
                        </select>
                      </label>
                    </div>
                  )}

                </div>
              </motion.div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
