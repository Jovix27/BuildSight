from shapely.geometry import Polygon as ShapePolygon, Point as ShapePoint
import logging
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
try:
    from pyproj import Transformer
except ImportError:
    Transformer = None

log = logging.getLogger("BuildSight.SpatialData")

# ── Site configuration (Canonical source) ──────────────────────────────────
SITE_CONFIG = {
    "site_id": "CH-SITE-01-TIRUCHIRAPPALLI",
    "anchors": {
        "SW_STAIRCASE": [10.816539, 78.668835],
        "SE_KITCHEN":   [10.816714, 78.66842],
        "NE_HALL":      [10.816715, 78.668946],
        "NW_TOILET":    [10.816527, 78.668945],
    },
    # SW corner from buildsight_zones_complete.geojson metadata (authoritative)
    "sw_lat": 10.81658333,
    "sw_lon": 78.66883333,
    # Site extent from H-matrix calibration (camera_cam01_H.npy world points):
    #   SW(0,0) → NE(18.9, 9.75) in site-local metres
    "width_m": 18.9,    # X-axis = site local East (rotated 85° from true East)
    "depth_m": 9.75,    # Y-axis = site local North
    # rotation_deg=85: all GeoJSON zone polygons are pre-rotated 85° CCW from
    # axis-aligned. world_to_gps must rotate by +85° before adding to SW corner
    # so that worker GPS coords fall inside the zone polygons.
    "rotation_deg": 85,
    "centre": [10.81662742, 78.66891976],
    # H-matrix calibration frame (camera_cam01_H.npy was calibrated at 848×478)
    "calib_frame_w": 848,
    "calib_frame_h": 478,
    "utm_zone": 44,
    "utm_band": "N",
    # ── CAM-01 perspective model (fallback when H matrix is unavailable) ──
    # Camera mounted in neighbour building south of site, looking North at 7.62m
    "cam_wx":      9.45,    # m east of SW corner (≈ site width midpoint 18.9/2)
    "cam_wy":     -5.00,    # m south of SW corner (camera is outside site)
    "cam_h":       7.62,    # camera height above ground (m)
    "cam_fov_h":  60.0,     # horizontal FoV in degrees
    # Vertical coverage: bottom of frame ≈ 2m into site, top ≈ 9m (within 9.75m)
    "cam_wy_near": 2.0,     # wy at bottom frame edge (m)
    "cam_wy_far":  9.0,     # wy at top frame edge (m)
}

class SpatialMapper:
    """
    Handles coordinate transformations between camera pixels, local site meters, 
    and global GPS coordinates with high-precision local projection using pyproj/UTM.
    """
    def __init__(
        self, 
        frame_width: int = 1920, 
        frame_height: int = 1080,
        homography_path: Optional[str] = None,
        site_config: Optional[dict] = None,
    ):
        self.site = site_config or SITE_CONFIG
        self.H: Optional[np.ndarray] = None
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.calibrated = False

        # 1. Initialize UTM Transformers for High Precision
        if Transformer:
            try:
                # WGS84 (Lat/Lon) to UTM Zone 44N (suitable for Trichy, India)
                self.transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
                self.transformer_from_utm = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
                
                # Pre-calculate UTM base for relative offsets
                self.base_utm_x, self.base_utm_y = self.transformer_to_utm.transform(
                    self.site["sw_lon"], self.site["sw_lat"]
                )
                self.has_crs = True
            except Exception as e:
                log.error(f"SpatialMapper: CRS initialization failed: {e}")
                self.has_crs = False
        else:
            self.has_crs = False

        # 2. Local Linear Fallback Constants (if pyproj fails/missing)
        self._M_LAT = 111132.92
        self._M_LON = 111412.84 * math.cos(math.radians(self.site["centre"][0]))

        # 3. Load Homography if provided
        if homography_path:
            try:
                self.H = np.load(homography_path)
                self.calibrated = True
                log.info(f"SpatialMapper: Loaded calibration from {homography_path}")
            except Exception as e:
                log.warning(f"SpatialMapper: Could not load {homography_path}: {e}")

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pixel (x, y) to site-local meters (world_x, world_y)."""
        if self.calibrated and self.H is not None:
            # H matrix was calibrated at (calib_frame_w × calib_frame_h).
            # Scale incoming pixels from the live frame size to calibration size
            # so the homography is applied in the correct pixel space.
            calib_w = self.site.get("calib_frame_w", 848)
            calib_h = self.site.get("calib_frame_h", 478)
            sx = px * (calib_w / self.frame_w)
            sy = py * (calib_h / self.frame_h)
            pt = np.array([[[sx, sy]]], dtype=np.float32)
            world = cv2.perspectiveTransform(pt, self.H)
            wx, wy = float(world[0][0][0]), float(world[0][0][1])
        else:
            wx, wy = self._perspective_pixel_to_world(px, py)

        # Clamp strictly to site boundary — no margin outside
        wx = max(0.0, min(self.site["width_m"], wx))
        wy = max(0.0, min(self.site["depth_m"], wy))
        return wx, wy

    def _perspective_pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """
        Perspective-accurate pixel→world mapping for CAM-01.

        Model: pinhole camera pitched downward, looking North.
        Parameters derived from camera_CAM01_position (zones.geojson):
          • Camera at (cam_wx, cam_wy, cam_h) in site-local metres
          • Horizontal FoV = cam_fov_h degrees
          • Vertical coverage: bottom frame = cam_wy_near, top frame = cam_wy_far

        Derivation: ground-plane ray-cast from camera centre through each pixel.
          d_world = R_pitch(θ) · [u, 1, -v]  (camera looking +Y/North)
          t = cam_h / (sin θ + v·cos θ)
          wx = cam_wx + t·u
          wy = cam_wy + t·(cos θ − v·sin θ)
        """
        cam_wx   = self.site.get("cam_wx",      12.46)
        cam_wy   = self.site.get("cam_wy",       -5.00)
        cam_h    = self.site.get("cam_h",          7.62)
        fov_h    = self.site.get("cam_fov_h",     60.0)
        wy_near  = self.site.get("cam_wy_near",    3.0)
        wy_far   = self.site.get("cam_wy_far",    14.75)

        dy_near = wy_near - cam_wy          # ground depth to bottom-frame edge
        dy_far  = wy_far  - cam_wy          # ground depth to top-frame edge

        angle_near = math.atan(cam_h / dy_near)    # steep downward angle (near)
        angle_far  = math.atan(cam_h / dy_far)     # shallow downward angle (far)

        pitch_rad    = (angle_near + angle_far) / 2.0
        fov_v_half   = (angle_near - angle_far) / 2.0

        f_x = (self.frame_w / 2.0) / math.tan(math.radians(fov_h / 2.0))
        f_y = (self.frame_h / 2.0) / math.tan(fov_v_half)

        u = (px - self.frame_w / 2.0) / f_x
        v = (py - self.frame_h / 2.0) / f_y

        sin_p = math.sin(pitch_rad)
        cos_p = math.cos(pitch_rad)

        denom = sin_p + v * cos_p
        if denom <= 1e-6:
            # Ray is at or above horizon — place at far boundary
            return cam_wx, self.site["depth_m"]

        t  = cam_h / denom
        wx = cam_wx + t * u
        wy = cam_wy + t * (cos_p - v * sin_p)
        return wx, wy

    def world_to_gps(self, wx: float, wy: float) -> Tuple[float, float]:
        """Convert site meters to GPS (lat, lon) using UTM projection or high-precision linear fallback."""
        angle_rad = math.radians(self.site.get("rotation_deg", 0))
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Site rotates relative to north, so we rotate local meters to align with UTM/WGS grid
        rx = wx * cos_a - wy * sin_a
        ry = wx * sin_a + wy * cos_a
        
        if self.has_crs:
            lon, lat = self.transformer_from_utm.transform(self.base_utm_x + rx, self.base_utm_y + ry)
            return lat, lon
        else:
            lat = self.site["sw_lat"] + ry / self._M_LAT
            lon = self.site["sw_lon"] + rx / self._M_LON
            return lat, lon

    def pixel_to_gps(self, px: float, py: float) -> Tuple[float, float]:
        """Direct pixel → GPS conversion."""
        wx, wy = self.pixel_to_world(px, py)
        return self.world_to_gps(wx, wy)

    def gps_to_world(self, lat: float, lon: float) -> Tuple[float, float]:
        """Inverse conversion: GPS → Local Site Meters (rotated back)."""
        if self.has_crs:
            utm_x, utm_y = self.transformer_to_utm.transform(lon, lat)
            rx = utm_x - self.base_utm_x
            ry = utm_y - self.base_utm_y
        else:
            ry = (lat - self.site["sw_lat"]) * self._M_LAT
            rx = (lon - self.site["sw_lon"]) * self._M_LON
        
        angle_rad = math.radians(-self.site.get("rotation_deg", 0))
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        wx = rx * cos_a - ry * sin_a
        wy = rx * sin_a + ry * cos_a
        return wx, wy

    def calculate_polygon_metrics(self, gps_coords: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Calculate area, centroid, and bbox in a planar metric space."""
        # Convert all to local world meters for accurate geometry
        world_pts = [self.gps_to_world(lat, lon) for lat, lon in gps_coords]
        
        poly = ShapePolygon(world_pts)
        area = poly.area
        centroid_world = (poly.centroid.x, poly.centroid.y)
        centroid_gps = self.world_to_gps(*centroid_world)
        
        bounds_world = poly.bounds # (minx, miny, maxx, maxy)
        bbox_gps = [
            self.world_to_gps(bounds_world[0], bounds_world[1]),
            self.world_to_gps(bounds_world[2], bounds_world[3])
        ]
        
        return {
            "area_m2": round(area, 2),
            "centroid_gps": list(centroid_gps),
            "bbox_gps": [list(b) for b in bbox_gps],
            "is_valid": poly.is_valid
        }

    @property
    def status(self) -> str:
        return "CALIBRATED_PRECISION" if self.calibrated else "UTM_PRECISION"
