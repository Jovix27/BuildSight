import React, { useEffect, useRef, useMemo } from 'react'
import { useSettings } from '../SettingsContext'

interface TurnerOrb3DProps {
  amplitude: number
  state: 'idle' | 'presenting' | 'thinking' | 'speaking' | 'listening'
  size?: number
  variant?: 'standard' | 'god'
  riskLevel?: number // 0 to 1 integration with GeoAI
}

// ── Types for Neural Geometry ──
interface NeuralPoint {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  size: number;
  brightness: number;
  connections: number[]; // Indices of connected points
}

interface SynapticPath {
  points: { x: number; y: number; z: number }[];
  speed: number;
  offset: number;
  width: number;
  opacity: number;
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '')
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

const TurnerOrb3D: React.FC<TurnerOrb3DProps> = ({
  amplitude,
  state,
  size = 400,
  variant = 'standard',
  riskLevel = 0
}) => {
  const { settings } = useSettings()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const requestRef = useRef<number>(0)
  const rotationRef = useRef({ x: 0, y: 0, z: 0 })

  // ── Theme Mapping ──
  const theme = useMemo(() => {
    // God Mode keeps its own fixed palette — never follows accent color
    if (variant === 'god') {
      return {
        primary: '#f8fafc',
        secondary: '#94a3b8',
        core: '#ffffff',
        glow: 'rgba(226, 232, 240, 0.2)'
      }
    }

    const accent = settings.accentColor || '#00ffff'
    // Thinking state is semantic (cognitive state) — always white/grey
    if (state === 'thinking') {
      return { primary: '#f8fafc', secondary: '#94a3b8', core: '#ffffff', glow: 'rgba(248, 250, 252, 0.15)' }
    }

    const glowAlpha = state === 'speaking' ? 0.25 : state === 'presenting' ? 0.15 : 0.08
    return {
      primary:   accent,
      secondary: accent,
      core:      '#ffffff',
      glow:      hexToRgba(accent, glowAlpha),
    }
  }, [state, variant, settings.accentColor])

  // ── Generate Neural Geometry once ──
  const neuralData = useMemo(() => {
    const points: NeuralPoint[] = []
    const paths: SynapticPath[] = []
    const sphereRadius = size * (variant === 'god' ? 0.45 : 0.38)
    const nodeCount = variant === 'god' ? 240 : 120 // Highly dense for God Mode

    // 1. Generate nodes on sphere surface
    for (let i = 0; i < nodeCount; i++) {
      const phi = Math.acos(-1 + (2 * i) / nodeCount)
      const theta = Math.sqrt(nodeCount * Math.PI) * phi
      
      const p = {
        x: sphereRadius * Math.cos(theta) * Math.sin(phi),
        y: sphereRadius * Math.sin(theta) * Math.sin(phi),
        z: sphereRadius * Math.cos(phi),
        vx: (Math.random() - 0.5) * 0.01,
        vy: (Math.random() - 0.5) * 0.01,
        vz: (Math.random() - 0.5) * 0.01,
        size: Math.random() * 1.5 + 0.5,
        brightness: Math.random(),
        connections: [] as number[]
      }
      points.push(p)
    }

    // Assign connections to nearby points
    points.forEach((p, i) => {
      for (let j = i + 1; j < points.length; j++) {
        const other = points[j]
        const dist = Math.sqrt((p.x-other.x)**2 + (p.y-other.y)**2 + (p.z-other.z)**2)
        const connectLimit = variant === 'god' ? 5 : 3
        if (dist < sphereRadius * 0.4 && p.connections.length < connectLimit) {
          p.connections.push(j)
        }
      }
    })

    // 2. Generate synaptic paths (arcs)
    const arcCount = variant === 'god' ? 32 : 18
    for (let i = 0; i < arcCount; i++) {
      const pathPoints = []
      const segments = 24
      const startPhi = Math.random() * Math.PI * 2
      const startTheta = Math.random() * Math.PI
      const length = Math.random() * Math.PI * 1.2 + 0.4
      
      for (let j = 0; j <= segments; j++) {
        const t = j / segments
        const phi = startPhi + t * length
        const theta = startTheta + Math.sin(t * Math.PI) * 0.3
        pathPoints.push({
          x: sphereRadius * Math.cos(phi) * Math.sin(theta),
          y: sphereRadius * Math.sin(phi) * Math.sin(theta),
          z: sphereRadius * Math.cos(theta)
        })
      }
      paths.push({
        points: pathPoints,
        speed: Math.random() * 0.03 + 0.01,
        offset: Math.random() * Math.PI * 2,
        width: Math.random() * 1.2 + 0.3,
        opacity: Math.random() * 0.4 + 0.15
      })
    }

    // 3. Clockwork Rings (exclusive to God Mode for that "Brain" feel)
    const rings: { radiusX: number; radiusY: number; speed: number; tilt: number; opacity: number }[] = []
    if (variant === 'god') {
      for (let i = 0; i < 8; i++) {
        rings.push({
          radiusX: sphereRadius * (1.2 + i * 0.15),
          radiusY: sphereRadius * (0.6 + i * 0.1),
          speed: (Math.random() * 0.005 + 0.002) * (i % 2 === 0 ? 1 : -1),
          tilt: (i / 8) * Math.PI,
          opacity: 0.1 + (i * 0.05)
        })
      }
    }

    return { points, paths, rings, sphereRadius }
  }, [size, variant])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = size * dpr
    canvas.height = size * dpr
    ctx.scale(dpr, dpr)

    const draw = (time: number) => {
      ctx.clearRect(0, 0, size, size)
      const centerX = size / 2
      const centerY = size / 2
      
      // Rotation logic - integrated with GeoAI riskLevel
      const baseRotation = state === 'idle' ? 0.0015 : (state === 'thinking' ? 0.02 : 0.006)
      const riskBoost = riskLevel * 0.03
      rotationRef.current.y += baseRotation + (amplitude * 0.04) + riskBoost
      rotationRef.current.x = Math.sin(time * 0.0004) * (0.15 + riskLevel * 0.1)
      const { x: rotX, y: rotY } = rotationRef.current

      // Projection Helper
      const project = (p: { x: number; y: number; z: number }) => {
        // Rotate Y
        let x1 = p.x * Math.cos(rotY) - p.z * Math.sin(rotY)
        let z1 = p.x * Math.sin(rotY) + p.z * Math.cos(rotY)
        // Rotate X
        let y2 = p.y * Math.cos(rotX) - z1 * Math.sin(rotX)
        let z2 = p.y * Math.sin(rotX) + z1 * Math.cos(rotX)
        
        const fov = size * 1.5
        const scale = fov / (fov + z2)
        return {
          sx: centerX + x1 * scale,
          sy: centerY + y2 * scale,
          sz: z2,
          alpha: scale * 0.8 // simple alpha based on depth
        }
      }

      // ── Render Clockwork Rings (God Mode) ──
      if (variant === 'god' && neuralData.rings) {
        neuralData.rings.forEach((ring, idx) => {
          ctx.beginPath()
          ctx.strokeStyle = theme.primary
          ctx.globalAlpha = ring.opacity * (0.5 + Math.sin(time * 0.001 + idx) * 0.5)
          ctx.lineWidth = 0.5
          
          const ringRot = time * ring.speed
          for (let i = 0; i <= 60; i++) {
            const angle = (i / 60) * Math.PI * 2 + ringRot
            const rx = Math.cos(angle) * ring.radiusX
            const ry = Math.sin(angle) * ring.radiusY
            
            // Apply tilt and rotation
            const tx = rx * Math.cos(ring.tilt) - ry * Math.sin(ring.tilt)
            const ty = rx * Math.sin(ring.tilt) + ry * Math.cos(ring.tilt)
            
            // Rotate the entire ring system with the global Y rotation
            const p = project({ x: tx, y: ty, z: 0 })
            if (i === 0) ctx.moveTo(p.sx, p.sy)
            else ctx.lineTo(p.sx, p.sy)
          }
          ctx.stroke()
        })
      }
      ctx.globalAlpha = 1

      // 1. Global Glow - intensifies with risk
      const bgGlow = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, size * (0.5 + riskLevel * 0.1))
      bgGlow.addColorStop(0, theme.glow)
      bgGlow.addColorStop(1, 'rgba(0,0,0,0)')
      ctx.fillStyle = bgGlow
      ctx.fillRect(0, 0, size, size)

      // 2. Synaptic Pathways (Background)
      neuralData.paths.forEach(path => {
        ctx.beginPath()
        ctx.lineWidth = path.width + (amplitude * 4)
        path.points.forEach((p, idx) => {
          const { sx, sy, alpha } = project(p)
          ctx.globalAlpha = path.opacity * alpha * (0.4 + amplitude * 0.6)
          ctx.strokeStyle = theme.primary
          if (idx === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()

        // Data pulses
        const pulseT = (time * path.speed + path.offset) % 1
        const pulseIdx = Math.floor(pulseT * (path.points.length - 1))
        const { sx: psx, sy: psy, alpha: palpha } = project(path.points[pulseIdx])
        if (palpha > 0.3) {
          ctx.beginPath()
          ctx.arc(psx, psy, 1.5 + (amplitude * 5), 0, Math.PI * 2)
          ctx.fillStyle = '#fff'
          ctx.globalAlpha = 0.8 * palpha
          ctx.fill()
        }
      })

      // 3. Neural Mesh (Connections between nodes)
      ctx.lineWidth = 0.5
      neuralData.points.forEach(p => {
        const proj1 = project(p)
        if (proj1.sz < -20) return // Culling back-facing connections
        
        p.connections.forEach(targetIdx => {
          const target = neuralData.points[targetIdx]
          const proj2 = project(target)
          
          const opacity = (Math.sin(time * 0.002 + p.brightness * 10) + 1) * 0.5
          ctx.globalAlpha = proj1.alpha * opacity * 0.2 * (0.5 + amplitude * 1.5)
          ctx.strokeStyle = theme.primary
          ctx.beginPath()
          ctx.moveTo(proj1.sx, proj1.sy)
          ctx.lineTo(proj2.sx, proj2.sy)
          ctx.stroke()
        })
      })

      // 4. Neural Nodes
      neuralData.points.forEach(p => {
        const { sx, sy, alpha } = project(p)
        if (alpha < 0.2) return

        const flicker = Math.sin(time * 0.01 + p.brightness * 100) > 0.8 ? 1 : 0.4
        const finalSize = p.size * (1 + amplitude * 3) * flicker
        
        ctx.globalAlpha = alpha * (0.5 + amplitude * 0.5)
        ctx.fillStyle = theme.primary
        ctx.beginPath()
        ctx.arc(sx, sy, finalSize, 0, Math.PI * 2)
        ctx.fill()

        if (alpha > 0.8 && amplitude > 0.3 && Math.random() > 0.98) {
          ctx.beginPath()
          ctx.arc(sx, sy, finalSize * 3, 0, Math.PI * 2)
          ctx.strokeStyle = '#fff'
          ctx.globalAlpha = 0.3
          ctx.stroke()
        }
      })

      // 5. Central Brain Core
      const corePulse = 1 + Math.sin(time * 0.01) * 0.05 + (amplitude * 1.2)
      const coreRadius = size * 0.07 * corePulse
      
      const coreGrad = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, coreRadius)
      coreGrad.addColorStop(0, '#fff')
      coreGrad.addColorStop(0.3, theme.primary)
      coreGrad.addColorStop(0.7, theme.secondary)
      coreGrad.addColorStop(1, 'rgba(0,0,0,0)')
      
      ctx.globalAlpha = 1
      ctx.fillStyle = coreGrad
      ctx.beginPath()
      ctx.arc(centerX, centerY, coreRadius, 0, Math.PI * 2)
      ctx.fill()

      // Core Structural Rings
      ctx.lineWidth = 1
      for (let i = 0; i < 4; i++) {
        const r = coreRadius * (0.2 + i * 0.25)
        const rot = time * 0.002 * (i % 2 === 0 ? 1 : -1)
        ctx.setLineDash([2, 4])
        ctx.strokeStyle = theme.primary
        ctx.globalAlpha = 0.3
        ctx.beginPath()
        ctx.ellipse(centerX, centerY, r, r * 0.6, rot, 0, Math.PI * 2)
        ctx.stroke()
      }
      ctx.setLineDash([])

      requestRef.current = requestAnimationFrame(draw)
    }

    requestRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(requestRef.current)
  }, [amplitude, state, size, neuralData, theme])

  const dropShadow = variant === 'god'
    ? 'drop-shadow(0 0 50px rgba(226, 232, 240, 0.12))'
    : `drop-shadow(0 0 50px ${hexToRgba(settings.accentColor || '#00ffff', 0.15)})`

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size, filter: dropShadow }}
    />
  )
}

export default TurnerOrb3D


