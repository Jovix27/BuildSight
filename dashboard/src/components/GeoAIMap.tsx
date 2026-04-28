import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import * as turf from '@turf/turf'
import type { HeatmapUpdatePayload, ZoneCollection, ZoneFeature, DynamicZone, SpatialNarrationPayload } from '../types/geoai'
import type { WorkerPosition } from '../store/detectionStore'

const SITE_CENTER: [number, number] = [10.81662, 78.66891]
const INITIAL_ZOOM = 19
const TARGET_OVERLAY_ROTATION_DEG = 85
const OVERLAY_OFFSET_METERS: [number, number] = [7.5, -4.5]
const OVERLAY_SCALE = 0.72

const RISK_COLORS: Record<string, string> = {
  CRITICAL: '#ff5252',
  HIGH: '#ff7043',
  MODERATE: '#ffca28',
  LOW: '#5bb8ff',
  none: '#5bb8ff',
}

const RISK_OPACITY_RANGE: Record<string, { min: number, max: number }> = {
  CRITICAL: { min: 0.18, max: 0.25 },
  HIGH: { min: 0.15, max: 0.22 },
  MODERATE: { min: 0.12, max: 0.18 },
  LOW: { min: 0.10, max: 0.15 },
  none: { min: 0.08, max: 0.12 },
}

interface GeoAIMapProps {
  data: HeatmapUpdatePayload | null
  showZones: boolean
  showLabels: boolean
  showCameraFOV: boolean
  showHeatmap: boolean
  showWorkers: boolean
  heatmapOpacity?: number
  viewMode: string
  dynamicZones?: DynamicZone[]
  narration?: SpatialNarrationPayload | null
  theme?: 'light' | 'dark'
  /** Live worker positions projected from background detection service */
  detectionWorkers?: WorkerPosition[]
  /** Active zone violations from background detection service */
  detectionViolations?: { zone_id: string; severity: string }[]
}

interface OverlayMetadata {
  rotation_deg?: number
}

type LatLngTuple = [number, number]
type LngLatTuple = [number, number]

function getMetersPerDegreeLon(lat: number) {
  return 111320 * Math.cos((lat * Math.PI) / 180)
}

function rotateLatLng(point: LatLngTuple, center: LatLngTuple, angleDeg: number): LatLngTuple {
  if (Math.abs(angleDeg) < 0.0001) return point

  const theta = (angleDeg * Math.PI) / 180
  const cosTheta = Math.cos(theta)
  const sinTheta = Math.sin(theta)
  const metersPerDegLat = 110574
  const metersPerDegLon = getMetersPerDegreeLon(center[0])

  const dx = (point[1] - center[1]) * metersPerDegLon
  const dy = (point[0] - center[0]) * metersPerDegLat

  const rx = dx * cosTheta - dy * sinTheta
  const ry = dx * sinTheta + dy * cosTheta

  return [
    center[0] + ry / metersPerDegLat,
    center[1] + rx / metersPerDegLon,
  ]
}

function scaleLatLng(point: LatLngTuple, center: LatLngTuple, scale: number): LatLngTuple {
  if (Math.abs(scale - 1) < 0.0001) return point

  const metersPerDegLat = 110574
  const metersPerDegLon = getMetersPerDegreeLon(center[0])
  const dx = (point[1] - center[1]) * metersPerDegLon
  const dy = (point[0] - center[0]) * metersPerDegLat

  return [
    center[0] + (dy * scale) / metersPerDegLat,
    center[1] + (dx * scale) / metersPerDegLon,
  ]
}

function offsetLatLng(point: LatLngTuple, offsetMeters: [number, number]): LatLngTuple {
  const [dxMeters, dyMeters] = offsetMeters
  const metersPerDegLat = 110574
  const metersPerDegLon = getMetersPerDegreeLon(point[0])
  return [
    point[0] + dyMeters / metersPerDegLat,
    point[1] + dxMeters / metersPerDegLon,
  ]
}

function denormalizeLatLng(point: LatLngTuple, center: LatLngTuple, rotationDelta: number, scale: number, offset: [number, number]): LatLngTuple {
  // 1. Reverse offset
  const pt1 = offsetLatLng(point, [-offset[0], -offset[1]])
  // 2. Reverse scale
  const pt2 = scaleLatLng(pt1, center, 1 / scale)
  // 3. Reverse rotation
  return rotateLatLng(pt2, center, -rotationDelta)
}

function computeOverlayCenter(features: ZoneFeature[] | undefined): LatLngTuple {
  if (!features || !Array.isArray(features) || features.length === 0) return SITE_CENTER;

  const cornerPoints = features
    .filter((feature) => feature.geometry?.type === 'Point' && feature.properties?.type === 'building_corner')
    .map((f) => (f.geometry as any).coordinates as [number, number]);

  if (cornerPoints.length === 0) {
    const boundary = features.find((feature) => feature.properties?.zone === 'site_boundary' && feature.geometry?.type === 'Polygon')
    if (boundary) {
      const ring = (boundary.geometry.coordinates as number[][][])[0]
      const lats = ring.map((coord) => coord[1])
      const lngs = ring.map((coord) => coord[0])
      return [
        (Math.min(...lats) + Math.max(...lats)) / 2,
        (Math.min(...lngs) + Math.max(...lngs)) / 2,
      ]
    }
    return SITE_CENTER
  }

  const lat = cornerPoints.reduce((sum, point) => sum + point[1], 0) / cornerPoints.length
  const lng = cornerPoints.reduce((sum, point) => sum + point[0], 0) / cornerPoints.length
  return [lat, lng]
}

export function GeoAIMap({
  data,
  showZones,
  showLabels,
  showCameraFOV,
  showHeatmap,
  showWorkers,
  heatmapOpacity = 0.65,
  viewMode,
  dynamicZones = [],
  narration = null,
  theme = 'dark',
  detectionWorkers = [],
  detectionViolations = [],
}: GeoAIMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<any>(null)
  const tacticalTileRef = useRef<any>(null)
  const satelliteTileRef = useRef<any>(null)
  const heatLayerRef = useRef<any>(null)
  const zonesLayerRef = useRef<L.LayerGroup | null>(null)
  const labelsLayerRef = useRef<L.LayerGroup | null>(null)
  const fovLayerRef = useRef<L.LayerGroup | null>(null)
  const workersLayerRef = useRef<L.LayerGroup | null>(null)
  const detectionWorkersLayerRef = useRef<L.LayerGroup | null>(null)
  const dynamicZonesLayerRef = useRef<L.LayerGroup | null>(null)
  const samLayerRef = useRef<L.LayerGroup | null>(null)
  const vlmLayerRef = useRef<L.LayerGroup | null>(null)

  const [analysisHistory, setAnalysisHistory] = useState<any[]>([])
  const [hoverCoords, setHoverCoords] = useState<[number, number] | null>(null)
  const [showHistory, setShowHistory] = useState(false)
  const [isQuerying, setIsQuerying] = useState(false)
  const [leafletReady, setLeafletReady] = useState(false)
  const [mapReady, setMapReady] = useState(false)
  const [geoData, setGeoData] = useState<ZoneCollection | null>(null)
  const lastValidZonesRef = useRef<ZoneCollection | null>(null)
  const [mapError, setMapError] = useState<string | null>(null)
  const [samSegments, setSamSegments] = useState<any[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [mapMode, setMapMode] = useState<'segment' | 'expert'>('segment')

  const TACTICAL_URL = theme === 'light' 
    ? 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
    : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
  const SATELLITE_URL = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'

  const overlayCenter = useMemo(
    () => (geoData ? computeOverlayCenter(geoData.features) : SITE_CENTER),
    [geoData]
  )

  const overlayRotationDelta = useMemo(() => {
    const metadata = (geoData as ZoneCollection & { metadata?: OverlayMetadata } | null)?.metadata
    const sourceRotation = metadata?.rotation_deg ?? 0
    return TARGET_OVERLAY_ROTATION_DEG - sourceRotation
  }, [geoData])

  const normalizeLatLng = useCallback(
    (point: LatLngTuple): LatLngTuple =>
      offsetLatLng(
        scaleLatLng(
          rotateLatLng(point, overlayCenter, overlayRotationDelta),
          overlayCenter,
          OVERLAY_SCALE
        ),
        OVERLAY_OFFSET_METERS
      ),
    [overlayCenter, overlayRotationDelta]
  )

  const normalizeLngLat = useCallback(
    (point: LngLatTuple): LatLngTuple => normalizeLatLng([point[1], point[0]]),
    [normalizeLatLng]
  )

  useEffect(() => {
    if (typeof window !== 'undefined' && window.L) {
      setLeafletReady(true)
      return
    }
    const check = setInterval(() => {
      if (window.L) {
        setLeafletReady(true)
        clearInterval(check)
      }
    }, 200)
    return () => clearInterval(check)
  }, [])

  useEffect(() => {
    const loadGeoZones = async () => {
      try {
        // Step 1: Try static public GeoJSON
        const response = await fetch('/buildsight_zones.geojson')
        if (!response.ok) throw new Error('Static GeoJSON fetch failed')
        const data = await response.json()

        console.log('GeoAI: Successfully loaded zones from static GeoJSON')
        setGeoData(data)
        lastValidZonesRef.current = data
        setMapError(null)
      } catch (err) {
        console.warn('GeoAI: Static GeoJSON failed, falling back to API', err)

        try {
          // Step 2: Fallback to Backend API
          const apiResponse = await fetch('http://127.0.0.1:8000/api/geoai/zones')
          if (!apiResponse.ok) throw new Error('Backend API fetch failed')
          const apiData = await apiResponse.json()

          console.log('GeoAI: Successfully loaded zones from Backend API')
          setGeoData(apiData)
          lastValidZonesRef.current = apiData
          setMapError(null)
        } catch (apiErr) {
          console.error('GeoAI: Failed to load zones from all sources', apiErr)

          // Final fallback: Use last valid or Site Center default
          if (lastValidZonesRef.current) {
            console.log('GeoAI: Reusing last valid zone data from memory')
            setGeoData(lastValidZonesRef.current)
          } else {
            setMapError('Critical: Zoning data unavailable')
          }
        }
      }
    }

    loadGeoZones()
  }, [])

  useEffect(() => {
    if (!leafletReady || !containerRef.current || mapRef.current) return

    const map = L.map(containerRef.current, {
      center: SITE_CENTER,
      zoom: INITIAL_ZOOM,
      zoomControl: false,
      attributionControl: false,
    })

    const isSat = viewMode === 'satellite'

    satelliteTileRef.current = L.tileLayer(SATELLITE_URL, {
      maxZoom: 22,
      subdomains: 'abcd',
      className: 'geoai-satellite-tile',
      opacity: isSat ? 1 : 0
    }).addTo(map)

    tacticalTileRef.current = L.tileLayer(TACTICAL_URL, {
      maxZoom: 22,
      subdomains: 'abcd',
      className: 'geoai-tactical-tile',
      opacity: isSat ? 0 : 1
    }).addTo(map)

    L.control.zoom({ position: 'bottomright' }).addTo(map)

    if (window.HeatmapOverlay) {
      const heatLayer = new window.HeatmapOverlay({
        radius: 40,
        maxOpacity: heatmapOpacity,
        minOpacity: 0.15,
        blur: 0.75,
        scaleRadius: false,
        useLocalExtrema: false,
        latField: 'lat',
        lngField: 'lng',
        valueField: 'value',
        gradient: {
          '0.0': 'rgba(0,200,100,0)',
          '0.3': 'rgba(0,200,100,0.8)',
          '0.55': 'rgba(255,214,0,0.9)',
          '0.75': 'rgba(255,140,0,0.95)',
          '1.0': 'rgba(255,30,30,1)',
        },
      })
      map.addLayer(heatLayer as any)
      heatLayerRef.current = heatLayer as unknown as typeof heatLayerRef.current
    }

    // Initialize layer groups once
    zonesLayerRef.current = L.layerGroup().addTo(map)
    labelsLayerRef.current = L.layerGroup().addTo(map)
    fovLayerRef.current = L.layerGroup().addTo(map)
    workersLayerRef.current = L.layerGroup().addTo(map)
    detectionWorkersLayerRef.current = L.layerGroup().addTo(map)
    dynamicZonesLayerRef.current = L.layerGroup().addTo(map)
    samLayerRef.current = L.layerGroup().addTo(map)
    vlmLayerRef.current = L.layerGroup().addTo(map)

    map.on('mousemove', (e: any) => {
      setHoverCoords([e.latlng.lat, e.latlng.lng])
    })

    map.on('mouseout', () => {
      setHoverCoords(null)
    })

    mapRef.current = map
    setMapReady(true)

    const resizeObserver = new ResizeObserver(() => {
      if (map) {
        map.invalidateSize()
      }
    })
    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
      map.remove()
      mapRef.current = null
      setMapReady(false)
    }
  }, [leafletReady]) // Singleton map init

  // Dynamic Event Handling for Map Clicks
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    const handleMapClick = async (e: any) => {
      const realWorldGps = denormalizeLatLng(
        [e.latlng.lat, e.latlng.lng],
        overlayCenter,
        overlayRotationDelta,
        OVERLAY_SCALE,
        OVERLAY_OFFSET_METERS
      )

      if (mapMode === 'segment') {
        if (isAnalyzing) return
        const swLat = 10.81658333, swLon = 78.66883333
        const mLat = 111132.92
        const mLon = 111412.84 * Math.cos((swLat * Math.PI) / 180)
        const ry = (realWorldGps[0] - swLat) * mLat
        const rx = (realWorldGps[1] - swLon) * mLon
        const angleRad = (-85.0 * Math.PI) / 180
        const wx = rx * Math.cos(angleRad) - ry * Math.sin(angleRad)
        const wy = rx * Math.sin(angleRad) + ry * Math.cos(angleRad)
        const px = (wx / 18.90) * 1920
        const py = (1.0 - wy / 9.75) * 1080

        setIsAnalyzing(true)
        try {
          const res = await fetch('/api/geoai/sam/prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompts: [[px, py]], min_score: 0.65 }),
          })
          if (res.ok) {
            const segments = await res.json()
            setSamSegments(prev => [...prev, ...segments])
          }
        } catch (err) {
          console.error('SAM Prompt failed:', err)
        } finally {
          setIsAnalyzing(false)
        }
      } else {
        setIsQuerying(true)
        try {
          const res = await fetch('/api/geoai/vlm/spatial-query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              lat: realWorldGps[0],
              lon: realWorldGps[1],
              question: "Assess site activity at this exact coordinate. Identify any missing PPE or safety risks."
            }),
          })
          if (res.ok) {
            const analysis = await res.json()
            setAnalysisHistory(prev => [{
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              coords: realWorldGps,
              result: analysis.description
            }, ...prev])
            setShowHistory(true)
          }
        } catch (err) {
          console.error('Spatial VLM Query failed:', err)
        } finally {
          setIsQuerying(false)
        }
      }
    }

    map.on('click', handleMapClick)
    return () => {
      map.off('click', handleMapClick)
    }
  }, [mapMode, isAnalyzing, overlayCenter, overlayRotationDelta])

  // Automated Narration Listener
  useEffect(() => {
    if (!narration) return

    setAnalysisHistory(prev => {
      // Check if this narration is already in history (by timestamp)
      const exists = prev.some(item => Math.abs(item.raw_timestamp - narration.timestamp) < 0.1)
      if (exists) return prev

      return [{
        timestamp: new Date(narration.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        raw_timestamp: narration.timestamp,
        result: narration.text,
        type: 'automated',
        vlm_active: narration.vlm_active,
        coords: [0, 0] // Default coords for automated narration to prevent UI crash
      }, ...prev].slice(0, 50) // Keep last 50
    })

    // Optionally auto-show history for important background updates
    // setShowHistory(true)
  }, [narration])

  useEffect(() => {
    if (!mapRef.current || !tacticalTileRef.current || !satelliteTileRef.current) return

    const isSat = viewMode === 'satellite'
    const targetSatOpacity = isSat ? 1 : 0
    const targetTacOpacity = isSat ? 0 : 1

    let progress = 0
    const startSatOpacity = satelliteTileRef.current.options.opacity
    const startTacOpacity = tacticalTileRef.current.options.opacity

    const fadeInterval = setInterval(() => {
      progress += 0.08
      if (progress >= 1) progress = 1

      satelliteTileRef.current.setOpacity(startSatOpacity + (targetSatOpacity - startSatOpacity) * progress)
      tacticalTileRef.current.setOpacity(startTacOpacity + (targetTacOpacity - startTacOpacity) * progress)

      if (progress === 1) {
        clearInterval(fadeInterval)
      }
    }, 20)
  }, [viewMode])

  // Unified Rendering Effect for Zones and Labels
  useEffect(() => {
    if (!mapRef.current || !geoData || !zonesLayerRef.current || !labelsLayerRef.current || !fovLayerRef.current) return

    // Always clear existing layers first to avoid stale artifacts
    zonesLayerRef.current.clearLayers()
    labelsLayerRef.current.clearLayers()
    fovLayerRef.current.clearLayers()

    if (!showZones && !showLabels && !showCameraFOV) return

    const labelOpacity = 1.0

    const features: ZoneFeature[] = geoData.features || (geoData as any).zones?.map((z: any) => ({
      type: 'Feature',
      properties: z.properties || {},
      geometry: { type: 'Polygon', coordinates: z.coordinates }
    })) || []

    features.forEach((feature: ZoneFeature) => {
      const props = feature.properties
      const geom = feature.geometry
      if (!geom) return

      const risk = (props.risk || 'none').toUpperCase()
      const isBoundary = props.zone === 'site_boundary' || props.zone_id === 'site_boundary'
      const isSegmentMode = mapMode === 'segment'

      if (isBoundary && showZones) {
        const coords = (geom.coordinates as number[][][])[0].map(
          (c: number[]) => normalizeLngLat(c as LngLatTuple)
        )
        const boundaryPoly = L.polygon(coords, {
          color: '#5bb8ff',
          weight: 2,
          dashArray: '8 6',
          fillColor: '#5bb8ff',
          fillOpacity: isSegmentMode ? 0.03 : 0.05,
          opacity: 0.8,
          className: 'geoai-boundary-layer'
        })
        zonesLayerRef.current?.addLayer(boundaryPoly)
        return
      }

      if (showZones) {
        const zoneColor = RISK_COLORS[risk] || RISK_COLORS.none
        const opacityConfig = RISK_OPACITY_RANGE[risk] || RISK_OPACITY_RANGE.none
        const dynamicFillOpacity = isSegmentMode ? opacityConfig.min : opacityConfig.max

        if (geom.type === 'Polygon') {
          const rings = (geom.coordinates as number[][][]).map(ring =>
            ring.map((c: number[]) => normalizeLngLat(c as LngLatTuple))
          )
          
          const zonePoly = L.polygon(rings, {
            color: zoneColor,
            weight: 2,
            opacity: 0.9,
            fillColor: zoneColor,
            fillOpacity: dynamicFillOpacity,
            className: 'geoai-zone-path transition-all duration-300'
          })
          zonesLayerRef.current?.addLayer(zonePoly)

          if (showLabels) {
            const bounds = zonePoly.getBounds()
            const center = bounds.getCenter()
            const rawLabel = (props.display_name || props.zone || props.zone_id || '') as string
            const labelText = rawLabel.replace(/_/g, ' ').toUpperCase()
            
            const labelMarker = L.marker([center.lat, center.lng], {
              icon: L.divIcon({
                className: 'geoai-zone-label',
                html: `
                  <div class="geoai-label-wrapper transition-opacity duration-300" style="opacity: ${labelOpacity}">
                    <span style="
                      color: ${zoneColor}; 
                      text-shadow: 0 0 8px rgba(0,0,0,0.7); 
                      font-weight: 700; 
                      font-size: 11px;
                      letter-spacing: 1px;
                      text-transform: uppercase;
                    ">${labelText}</span>
                  </div>
                `,
                iconSize: [120, 16],
                iconAnchor: [60, 8],
              }),
              interactive: false,
              zIndexOffset: 1000
            })
            labelsLayerRef.current?.addLayer(labelMarker)
          }
        }
      }
    })
  }, [geoData, normalizeLngLat, mapMode, showZones, showLabels, showCameraFOV, mapReady])

  // Remove old granular visibility effects as they are now handled by the unified rendering effect

  useEffect(() => {
    if (!heatLayerRef.current) return
    if (!showHeatmap) {
      heatLayerRef.current.setData({ min: 0, max: 1, data: [] })
      return
    }

    // Priority 1: WebSocket spatial engine grid data
    if (data?.heatmap) {
      const { cols, rows, resolution_m, data: gridData } = data.heatmap
      const hmData = []
      const angleRad = (85.0 * Math.PI) / 180
      const cosA = Math.cos(angleRad)
      const sinA = Math.sin(angleRad)
      const siteSwLat = 10.81658333
      const siteSwLon = 78.66883333
      const mLat = 110574.0
      const mLon = 111319.0 * Math.cos((siteSwLat * Math.PI) / 180)
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const value = gridData[r * cols + c]
          if (value > 0.05) {
            const wx = c * resolution_m + resolution_m / 2
            const wy = r * resolution_m + resolution_m / 2
            const rx = wx * cosA - wy * sinA
            const ry = wx * sinA + wy * cosA
            const lat = siteSwLat + ry / mLat
            const lon = siteSwLon + rx / mLon
            const [nLat, nLng] = normalizeLatLng([lat, lon])
            hmData.push({ lat: nLat, lng: nLng, value })
          }
        }
      }
      heatLayerRef.current.setData({ min: 0, max: 1, data: hmData })
      return
    }

    // Priority 2: WebSocket cell data
    if (data?.cells) {
      const hmData = {
        min: 0,
        max: 1,
        data: data.cells
          .filter(c => c.risk_score > 0.05)
          .map(c => {
            const [lat, lng] = normalizeLatLng([c.lat, c.lon])
            return { lat, lng, value: c.risk_score }
          }),
      }
      heatLayerRef.current.setData(hmData)
      return
    }

    // Priority 3: Live detection workers — generate risk-weighted heat points
    if (detectionWorkers.length > 0) {
      const hmData = detectionWorkers
        .filter(w => w.lat && w.lng && !isNaN(w.lat) && !isNaN(w.lng))
        .map(w => {
          // Risk score: non-compliant=1.0, partial=0.65, compliant=0.35
          const score = !w.ppe_compliant
            ? (w.has_helmet === false && w.has_vest === false ? 1.0 : 0.75)
            : 0.35
          const [lat, lng] = normalizeLatLng([w.lat, w.lng])
          return { lat, lng, value: score }
        })
      heatLayerRef.current.setData({ min: 0, max: 1, data: hmData })
      return
    }

    // No data available — clear
    heatLayerRef.current.setData({ min: 0, max: 1, data: [] })
  }, [data, showHeatmap, normalizeLatLng, detectionWorkers])

  useEffect(() => {
    if (!mapRef.current || !workersLayerRef.current) return
    workersLayerRef.current.clearLayers()

    // When live detection workers are present, the detectionWorkersLayer already
    // shows them as color-coded W markers. Hide the WebSocket demo workers to avoid overlap.
    if (!showWorkers || !data?.workers || !geoData || detectionWorkers.length > 0) return

    data.workers.forEach(w => {
      // 1. Spatial Guard: Only show workers within defined zones or site perimeter
      if (!w.lat || !w.lon || isNaN(w.lat) || isNaN(w.lon)) return

      // Explicit Check: Use Turf.js for robust containment verification
      const isInsideAnyZone = geoData.features.some((feature) => {
        try {
          return turf.booleanPointInPolygon(
            turf.point([w.lon, w.lat]),
            feature as any
          )
        } catch (e) {
          return false
        }
      })

      // Also filter by fallback 'outside' classification if provided by pipeline
      if (!isInsideAnyZone || w.zone === 'outside') return

      const riskColor = RISK_COLORS[w.risk] || '#00e676'
      const ppeClass = w.ppe_ok ? 'ppe-ok' : 'ppe-violation'
      const helmetIcon = w.has_helmet ? '⛑️' : '⚠'
      const vestIcon = w.has_vest ? '🦺' : '⚠'

      const isAtRisk = w.status === 'AT_RISK' || w.status === 'CRITICAL'
      const statusPulse = isAtRisk ? '<div class="geoai-worker-pulse"></div>' : ''
      const dwellBadge = w.dwell_time_s && w.dwell_time_s > 30
        ? `<span class="geoai-worker-dwell">${Math.floor(w.dwell_time_s)}s</span>`
        : ''

      const icon = L.divIcon({
        className: `geoai-worker-marker ${ppeClass}`,
        html: `
          ${statusPulse}
          <div class="geoai-worker-dot" style="background:${riskColor};box-shadow:0 0 12px ${riskColor}">
            <span class="geoai-worker-ppe">${helmetIcon}${vestIcon}</span>
          </div>
          ${dwellBadge}
        `,
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      })

      const [lat, lon] = normalizeLatLng([w.lat, w.lon])
      const marker = L.marker([lat, lon], { icon })
      marker.bindPopup(`
        <div class="geoai-popup">
          <strong>Worker:</strong> W${w.worker_id ?? '?'} &nbsp; <strong>Camera:</strong> ${w.camera_id ?? 'CAM-01'}<br/>
          <strong>Zone:</strong> ${w.zone.replace(/_/g, ' ')}<br/>
          <strong>Risk:</strong> <span style="color:${riskColor}">${w.risk}</span>
          &nbsp; <strong>Status:</strong> <span style="color:${riskColor}">${w.status ?? '—'}</span><br/>
          <strong>Height:</strong> ${w.height_m}m<br/>
          <strong>Helmet:</strong> ${w.has_helmet ? '✅' : '❌'} &nbsp;
          <strong>Vest:</strong> ${w.has_vest ? '✅' : '❌'}<br/>
          ${w.dwell_time_s && w.dwell_time_s > 0 ? `<strong>Dwell:</strong> ${w.dwell_time_s}s (${w.dwell_severity ?? ''})` : ''}
        </div>
      `, { className: 'geoai-popup-container' })

      workersLayerRef.current!.addLayer(marker)
    })
  }, [data, showWorkers, normalizeLatLng, detectionWorkers])

  // ── Live detection worker markers (from background detection service) ──
  useEffect(() => {
    if (!mapRef.current || !detectionWorkersLayerRef.current) return
    detectionWorkersLayerRef.current.clearLayers()
    if (!detectionWorkers.length) return

    // Compute set of zone_ids that have CRITICAL violations for styling
    const criticalZones = new Set(
      detectionViolations.filter(v => v.severity === 'CRITICAL').map(v => v.zone_id)
    )

    detectionWorkers.forEach(worker => {
      if (!worker.lat || !worker.lng || isNaN(worker.lat) || isNaN(worker.lng)) return

      const compliant = worker.ppe_compliant
      const partial   = !compliant && (worker.has_helmet || worker.has_vest)
      const color     = compliant ? '#00c864' : partial ? '#ffa500' : '#ff3030'
      const inCritical = worker.zone_id ? criticalZones.has(worker.zone_id) : false

      const pulseStyle = inCritical
        ? `animation: detection-worker-pulse 0.9s ease-in-out infinite;`
        : ''

      const icon = L.divIcon({
        className: '',
        html: `<div style="
          width:18px; height:18px; border-radius:50%;
          background:${color}; border:2px solid rgba(255,255,255,0.85);
          box-shadow:0 0 8px ${color}99;
          display:flex; align-items:center; justify-content:center;
          font-size:9px; font-weight:700; color:#fff;
          ${pulseStyle}
        ">W</div>`,
        iconSize:   [18, 18],
        iconAnchor: [9, 9],
      })

      const helmetTxt = worker.has_helmet === null ? 'Unknown' : worker.has_helmet ? '✓ Present' : '✗ Missing'
      const vestTxt   = worker.has_vest   === null ? 'Unknown' : worker.has_vest   ? '✓ Present' : '✗ Missing'
      const statusTxt = compliant ? '<span style="color:#00c864">✓ COMPLIANT</span>'
                                  : '<span style="color:#ff4040">⚠ VIOLATION</span>'

      const pos = normalizeLatLng([worker.lat, worker.lng])
      const marker = L.marker(pos, { icon, zIndexOffset: 2000 })
      marker.bindPopup(`
        <div class="geoai-popup" style="min-width:160px">
          <strong style="font-size:13px">${worker.worker_id}</strong>
          <div style="margin:4px 0">${statusTxt}</div>
          <div>Conf: ${Math.round(worker.confidence * 100)}%</div>
          <div>Zone: ${worker.zone_name ?? '<em>Outside zones</em>'}</div>
          <div>Helmet: ${helmetTxt}</div>
          <div>Vest: ${vestTxt}</div>
          <div style="font-size:10px;color:#666;margin-top:4px">
            UTM ${worker.utm_e.toFixed(1)}E · ${worker.utm_n.toFixed(1)}N
          </div>
        </div>
      `, { className: 'geoai-popup-container' })

      detectionWorkersLayerRef.current!.addLayer(marker)
    })
  }, [detectionWorkers, detectionViolations])

  // ── Dynamic zones layer ────────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current || !dynamicZonesLayerRef.current) return
    dynamicZonesLayerRef.current.clearLayers()

    dynamicZones.filter(z => z.is_active).forEach(zone => {
      // Apply the same normalization pipeline as static zones so dynamic zones
      // align with the rotated/scaled building overlay on the map.
      const ring = zone.coordinates.map(([lng, lat]) => normalizeLngLat([lng, lat]))
      if (ring.length < 3) return

      const poly = L.polygon(ring, {
        color: zone.color,
        weight: 2,
        dashArray: '6 4',
        fillColor: zone.color,
        fillOpacity: 0.15,
      })

      poly.bindPopup(`
        <div class="geoai-popup">
          <strong>${zone.name}</strong><br/>
          Type: ${zone.zone_type}<br/>
          Risk: <span style="color:${zone.color}">${zone.risk_level.toUpperCase()}</span><br/>
          ${zone.description ? `<em>${zone.description}</em>` : ''}
        </div>
      `, { className: 'geoai-popup-container' })

      dynamicZonesLayerRef.current!.addLayer(poly)

      // Label
      const bounds = poly.getBounds()
      const centre = bounds.getCenter()
      const label = L.marker([centre.lat, centre.lng], {
        icon: L.divIcon({
          className: 'geoai-zone-label',
          html: `<span style="color:${zone.color}; text-shadow:0 0 10px rgba(0,0,0,0.8)">${zone.name.toUpperCase()}</span>`,
          iconSize: [120, 16],
          iconAnchor: [60, 8],
        }),
      })
      dynamicZonesLayerRef.current!.addLayer(label)
    })
  }, [dynamicZones, normalizeLngLat])

  // ── SAM Segments layer ──────────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current || !samLayerRef.current) return
    samLayerRef.current.clearLayers()

    samSegments.forEach(seg => {
      const feature = seg.geojson
      const coords = feature.geometry.coordinates[0].map((c: any) => normalizeLngLat(c as LngLatTuple))

      const poly = L.polygon(coords, {
        color: '#00e5ff',
        weight: 2,
        fillColor: '#00e5ff',
        fillOpacity: 0.3,
        className: 'geoai-sam-segment'
      })

      poly.bindPopup(`
        <div class="geoai-popup">
          <strong style="color:#00e5ff">GeoAI Segment</strong><br/>
          Area: ${seg.area_m2.toFixed(1)} m²<br/>
          Confidence: ${(seg.confidence * 100).toFixed(1)}%<br/>
          Device: ${seg.device}
        </div>
      `, { className: 'geoai-popup-container' })

      samLayerRef.current?.addLayer(poly)
    })
  }, [samSegments, normalizeLngLat])

  return (
    <div className="geoai-shell" style={{ height: '100%', position: 'relative' }}>

      {/* ── BASE LAYER: Map fills entire shell ─────────────────────── */}
      <div ref={containerRef} className="geoai-map-container" />

      {mapError && (
        <div style={{
          position: 'absolute',
          top: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 2000,
          background: 'rgba(255, 0, 0, 0.2)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 0, 0, 0.5)',
          padding: '8px 16px',
          borderRadius: '4px',
          color: '#ff4444',
          fontSize: '12px',
          fontFamily: 'monospace',
          textTransform: 'uppercase',
          letterSpacing: '1px'
        }}>
          [ TACTICAL ERROR ]: {mapError}
        </div>
      )}

      {/* ── OVERLAY LAYER: All HUD elements on top of the map ──────── */}
      <div className="geoai-map-overlays">

        {/* Compact Tactical Compass HUD */}
        <div className="geoai-compass-hud geoai-glass geoai-hud-layer" style={{
          position: 'absolute',
          top: '18px',
          right: '18px',
          width: '72px',
          height: '72px',
          zIndex: 1200,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '12px',
          pointerEvents: 'auto'
        }}>
          {/* Central Crosshair */}
          <div style={{ position: 'absolute', width: '24px', height: '1px', background: 'var(--color-accent)', opacity: 0.6 }} />
          <div style={{ position: 'absolute', width: '1px', height: '24px', background: 'var(--color-accent)', opacity: 0.6 }} />
          
          {/* Direction Labels */}
          <span className="geoai-hud-label" style={{ position: 'absolute', top: '4px', fontSize: '10px', fontWeight: 900, color: '#fff' }}>N</span>
          <span className="geoai-hud-label" style={{ position: 'absolute', bottom: '4px', fontSize: '10px', fontWeight: 900, color: 'rgba(255,255,255,0.4)' }}>S</span>
          <span className="geoai-hud-label" style={{ position: 'absolute', right: '4px', fontSize: '10px', fontWeight: 900, color: 'rgba(255,255,255,0.4)' }}>E</span>
          <span className="geoai-hud-label" style={{ position: 'absolute', left: '4px', fontSize: '10px', fontWeight: 900, color: 'rgba(255,255,255,0.4)' }}>W</span>
          
          {/* Inner Glow Circle */}
          <div style={{
            width: '40px',
            height: '40px',
            border: '1px solid rgba(0, 229, 255, 0.1)',
            borderRadius: '50%',
            boxShadow: 'inset 0 0 10px rgba(0, 229, 255, 0.05)'
          }} />
        </div>

        {/* Precision Coordinate HUD */}
        <div className="geoai-hud-card geoai-glass geoai-hud-layer">
          <div className="geoai-hud-header">
            <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" style={{ boxShadow: 'var(--geoai-glow)' }} />
            <span className="geoai-hud-label geoai-glow-text">Spatial Logic Active</span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '24px' }}>
            <div className="geoai-hud-metric">
              <span className="geoai-hud-label">Projection</span>
              <span className="geoai-hud-value">UTM ZONE 44N</span>
            </div>
            <div className="geoai-hud-metric" style={{ textAlign: 'right' }}>
              <span className="geoai-hud-label">Precision</span>
              <span className="geoai-hud-value" style={{ color: 'var(--color-accent)' }}>±0.008m</span>
            </div>
          </div>

          <div style={{ height: '1px', width: '100%', background: 'rgba(255,255,255,0.05)', margin: '4px 0' }} />

          <div className="geoai-hud-metric">
            <span className="geoai-hud-label">GPS Cursor</span>
            <span className="geoai-hud-value">
              {hoverCoords ? (
                <>
                  {hoverCoords[0].toFixed(6)}°N / {hoverCoords[1].toFixed(6)}°E
                </>
              ) : '--.------°N / --.------°E'}
            </span>
          </div>
        </div>

        {/* Mode Controls - Premium Toolbar */}
        <div className="geoai-toolbar geoai-glass geoai-hud-layer">
          <button
            onClick={() => setMapMode('segment')}
            className={`geoai-action-button ${mapMode === 'segment' ? 'geoai-action-button--active' : ''}`}
          >
            <svg className="geoai-tool-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
            </svg>
            <span>Segmentation</span>
          </button>

          <button
            onClick={() => setMapMode('expert')}
            className={`geoai-action-button geoai-action-button--purple ${mapMode === 'expert' ? 'geoai-action-button--active' : ''}`}
          >
            <svg className="geoai-tool-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <span>Spatial Expert</span>
          </button>
        </div>

        {/* Intelligence Insights Sidebar */}
        <div className={`geoai-sidebar geoai-glass geoai-sidebar-layer ${showHistory ? '' : 'geoai-sidebar--hidden'}`}>
          <div className="geoai-sidebar-header">
            <div>
              <h3 style={{ fontSize: 'var(--fs-md)', fontWeight: 900, textTransform: 'uppercase', color: '#fff', letterSpacing: '-0.02em' }}>Insights</h3>
              <span className="geoai-hud-label" style={{ letterSpacing: '0.3em' }}>Spatial VLM Stream</span>
            </div>
            <button onClick={() => setShowHistory(false)} style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'rgba(255,255,255,0.05)', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'rgba(255,255,255,0.4)' }}>
              <svg className="geoai-tool-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" /></svg>
            </button>
          </div>

          <div className="geoai-sidebar-content custom-scrollbar">
            {analysisHistory.length === 0 ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: 0.2, textAlign: 'center', padding: '40px' }}>
                <div style={{ width: '64px', height: '64px', border: '2px dashed rgba(255,255,255,0.2)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '16px' }}>
                  <svg className="geoai-tool-icon" style={{ width: '32px', height: '32px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" strokeWidth={1.5} /></svg>
                </div>
                <p className="geoai-hud-label">Awaiting spatial probe data...</p>
              </div>
            ) : (
              analysisHistory.map((entry, idx) => (
                <div key={idx} className="geoai-insight-card">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div className="w-1.5 h-1.5 rounded-full bg-purple-500" style={{ boxShadow: '0 0 10px rgba(168,85,247,0.8)' }} />
                      <span style={{ fontSize: '11px', fontWeight: 900, color: '#a855f7', fontFamily: 'var(--font-mono)' }}>{entry.timestamp}</span>
                    </div>
                  </div>
                  <p style={{ fontSize: '14px', color: 'rgba(255,255,255,0.8)', lineHeight: '1.6', marginBottom: '16px' }}>{entry.result}</p>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ padding: '4px 8px', background: 'rgba(168,85,247,0.2)', color: '#a855f7', fontSize: '9px', fontWeight: 900, borderRadius: '6px', textTransform: 'uppercase', letterSpacing: '0.1em' }}>VLM Core</span>
                    <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.2)', fontFamily: 'var(--font-mono)' }}>
                      {entry.coords && entry.coords.length >= 2 
                        ? `${entry.coords[0].toFixed(5)}, ${entry.coords[1].toFixed(5)}`
                        : 'SYSTEM MONITOR'}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>

          <div style={{ padding: '24px', borderTop: '1px solid var(--geoai-border)', background: 'rgba(255,255,255,0.02)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', fontWeight: 700, color: 'rgba(255,255,255,0.2)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              <span>Total Scans: {analysisHistory.length}</span>
              <span style={{ color: 'var(--color-accent)', opacity: 0.5 }}>V1.2.4-PRO</span>
            </div>
          </div>
        </div>

        {/* History Slide Toggle */}
        {!showHistory && analysisHistory.length > 0 && (
          <button
            onClick={() => setShowHistory(true)}
            className="geoai-glass geoai-sidebar-layer"
            style={{
              position: 'absolute',
              right: 0,
              top: '50%',
              transform: 'translateY(-50%)',
              width: '48px',
              height: '180px',
              borderRadius: '24px 0 0 24px',
              borderRight: 'none',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '16px',
              color: 'rgba(255,255,255,0.4)',
              cursor: 'pointer'
            }}
          >
            <span className="vertical-text" style={{ fontSize: '10px', fontWeight: 900, textTransform: 'uppercase', letterSpacing: '0.4em' }}>Audit Log</span>
            <svg className="geoai-tool-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M15 19l-7-7 7-7" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" /></svg>
          </button>
        )}

        {/* Analysis Pulse Matrix Overlay */}
        {(isAnalyzing || isQuerying) && (
          <div className="geoai-matrix-overlay">
            <div className="geoai-matrix-spinner">
              <div style={{ position: 'absolute', inset: 0, border: '4px solid rgba(0, 229, 255, 0.1)', borderRadius: '50%', animation: 'ping 2s infinite' }} />
              <div style={{ position: 'absolute', inset: 0, border: '4px solid rgba(168, 85, 247, 0.1)', borderRadius: '50%', animation: 'ping 3s infinite' }} />
              <div style={{ position: 'absolute', inset: 0, border: '6px solid var(--geoai-border)', borderTopColor: 'var(--color-accent)', borderBottomColor: '#a855f7', borderRadius: '50%', animation: 'spin 4s linear infinite', boxShadow: '0 0 50px rgba(0, 229, 255, 0.2)' }} />
            </div>

            <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <h2 className="geoai-glow-text" style={{ fontSize: '24px', fontWeight: 900, textTransform: 'uppercase', color: '#fff' }}>
                {isQuerying ? 'Consulting Expert VLM' : 'Generating Mesh'}
              </h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', justifyContent: 'center' }}>
                <div style={{ height: '2px', width: '40px', background: 'linear-gradient(to right, transparent, var(--color-accent))' }} />
                <span className="geoai-hud-label" style={{ opacity: 0.6 }}>Spatial Intelligence Active</span>
                <div style={{ height: '2px', width: '40px', background: 'linear-gradient(to left, transparent, #a855f7)' }} />
              </div>
            </div>
          </div>
        )}

      </div> {/* .geoai-map-overlays */}

      <style>{`
        .geoai-map-container {
          position: absolute;
          inset: 0;
          z-index: 1;
        }
        .geoai-map-overlays {
          position: absolute;
          inset: 0;
          z-index: 10;
          pointer-events: none;
        }
        .geoai-map-overlays button,
        .geoai-map-overlays input,
        .geoai-map-overlays .interactive {
          pointer-events: auto;
        }
        .vertical-text {
          writing-mode: vertical-rl;
          text-orientation: mixed;
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 20px;
          border: 2px solid transparent;
          background-clip: content-box;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.1);
        }
        .geoai-tactical-tile {
          /* Clean light aesthetic: Remove dark grayscale/contrast/brightness filters */
          opacity: 0.9;
          transition: opacity 300ms ease;
        }
        .geoai-map-container.leaflet-container {
          background: #fcfcfc !important;
        }
        .geoai-zone-path {
          transition: fill-opacity 300ms ease-in-out, stroke-opacity 300ms ease-in-out;
        }
        .geoai-label-wrapper {
          transition: opacity 300ms ease-in-out, transform 300ms ease-in-out;
        }
        .geoai-fov-layer {
          transition: opacity 300ms ease-in-out;
        }
        @keyframes scan {
          from { transform: translateY(-100%); }
          to { transform: translateY(100%); }
        }
      `}</style>

    </div> /* .geoai-shell */
  )
}

export default GeoAIMap

