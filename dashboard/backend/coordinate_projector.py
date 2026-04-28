"""
BuildSight Pixel-to-World Coordinate Projector
===============================================
Unified implementation using SpatialMapper — the same pipeline as
IntelligenceEngine.  Replaces the previous GCP-based homography that
was calibrated against wrong site dimensions (30m × 20m rectangle).

Public API is unchanged: PixelToWorldProjector.pixel_to_world(x, y, frame_shape)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

# Allow both dashboard/backend/ and project-root execution contexts
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent

try:
    from geoai.utils.spatial_mapper import SpatialMapper, SITE_CONFIG
except ImportError:
    sys.path.insert(0, str(_THIS_DIR))
    from geoai.utils.spatial_mapper import SpatialMapper, SITE_CONFIG

# H matrix calibrated at 848×478 (camera_cam01_H.npy)
_H_PATH = _PROJECT_ROOT / "spatial" / "calibration" / "camera_cam01_H.npy"
_CALIB_W = SITE_CONFIG.get("calib_frame_w", 848)
_CALIB_H = SITE_CONFIG.get("calib_frame_h", 478)

_ZONE_NAMES = {
    "high_risk_staircase":    "Staircase Zone (High Risk)",
    "high_risk_scaffolding":  "Scaffolding Zone (High Risk)",
    "moderate_risk_interior": "Interior Zone (Moderate Risk)",
    "low_risk_parking":       "Parking / Storage (Low Risk)",
    "low_risk_common":        "Common Area (Low Risk)",
}

# Approximate UTM Zone 44N base for the site (metres)
_UTM_E_BASE = 430_000.0
_UTM_N_BASE = 1_196_500.0


def _resolve_zone(wx: float, wy: float) -> Tuple[str, str]:
    """Map site-local metres (18.9 × 9.75 m) to zone ID + display name."""
    if 0.0 <= wx <= 3.5 and wy >= 5.5:
        zid = "high_risk_staircase"
    elif wx >= 14.5:
        zid = "low_risk_parking"
    elif 0.0 <= wx < 14.5 and 0.0 <= wy <= 7.5:
        zid = "moderate_risk_interior"
    elif (0.5 <= wx <= SITE_CONFIG["width_m"] - 0.5
          and 0.5 <= wy <= SITE_CONFIG["depth_m"] - 0.5):
        zid = "high_risk_scaffolding"
    else:
        zid = "low_risk_common"
    return zid, _ZONE_NAMES[zid]


class PixelToWorldProjector:
    """
    Projects CCTV pixel coordinates to real-world geographic coordinates.

    Uses SpatialMapper (H matrix when available, perspective model otherwise)
    so worker positions are consistent with the GeoAI WebSocket pipeline.
    """

    def __init__(self, _config: Optional[dict] = None):
        # `_config` kept for backwards-compat but SpatialMapper is the source of truth
        h_path = str(_H_PATH) if _H_PATH.exists() else None
        self._mapper = SpatialMapper(
            frame_width=_CALIB_W,
            frame_height=_CALIB_H,
            homography_path=h_path,
        )

    def pixel_to_world(
        self,
        pixel_x: float,
        pixel_y: float,
        frame_shape: tuple,
    ) -> dict:
        """
        Convert a pixel foot-point to geographic coordinates.

        Args:
            pixel_x: x coordinate in the CCTV frame (any resolution)
            pixel_y: y coordinate in the CCTV frame (any resolution)
            frame_shape: (height, width[, channels]) of the actual frame

        Returns:
            dict with lat, lng, utm_e, utm_n, zone_id, zone_name
        """
        fh, fw = frame_shape[0], frame_shape[1]
        # Scale to calibration frame so SpatialMapper pixel_to_world is accurate
        # (pixel_to_world internally also scales by calib/frame, but mapper was
        #  constructed at calib size, so the net scale is 1.0 — explicit here for clarity)
        sx = pixel_x * (_CALIB_W / fw)
        sy = pixel_y * (_CALIB_H / fh)

        wx, wy = self._mapper.pixel_to_world(sx, sy)
        lat, lon = self._mapper.world_to_gps(wx, wy)
        zone_id, zone_name = _resolve_zone(wx, wy)

        return {
            "lat":          round(lat, 7),
            "lng":          round(lon, 7),
            "utm_e":        round(_UTM_E_BASE + wx, 2),
            "utm_n":        round(_UTM_N_BASE + wy, 2),
            "utm_e_offset": round(wx, 2),
            "utm_n_offset": round(wy, 2),
            "zone_id":      zone_id,
            "zone_name":    zone_name,
        }
