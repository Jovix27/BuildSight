import cv2
import time
import threading
import queue
import logging
import traceback
import os
import uuid

logger = logging.getLogger("buildsight.camera")

class CameraStreamManager:
    """
    Manages the RTSP camera stream.
    Features:
    - FFmpeg backend with TCP mode to reduce packet loss.
    - Maxsize=1 Queue to drop old frames and maintain zero buildup.
    - Reconnection loop for network drops.
    - Stream health monitoring (FPS, latency, dropped frames).
    """
    def __init__(self, rtsp_url: str):
        self.is_local_camera = False
        rtsp_url = rtsp_url.strip()
        try:
            # Check if it's a numeric ID for a local webcam (0, 1, etc)
            self.cam_source = int(rtsp_url)
            self.is_local_camera = True
            self.rtsp_url = f"Camera {self.cam_source}"
        except ValueError:
            self.is_local_camera = False
            # Ensure we append ?tcp if not already present
            if "?tcp" not in rtsp_url and "rtsp://" in rtsp_url:
                self.cam_source = f"{rtsp_url}?tcp"
            else:
                self.cam_source = rtsp_url
            self.rtsp_url = self.cam_source
            
        self.latest_frame = (None, None, None)
        self.new_frame_condition = threading.Condition()
        self.is_running = False
        self.thread = None
        
        # Metrics
        self.fps = 0.0
        self.last_frame_time = 0.0
        self.dropped_frames = 0
        self.reconnect_count = 0
        self.frame_id = 0
        
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Camera stream started for {self.rtsp_url}")
        
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Camera stream stopped.")
            
    def _capture_loop(self):
        while self.is_running:
            try:
                if self.is_local_camera:
                    logger.info(f"Connecting to local camera: {self.cam_source}")
                    if os.name == 'nt':
                        # Try DSHOW first (typical for USB webcams), then default
                        logger.info(f"Trying DSHOW backend for camera {self.cam_source}")
                        cap = cv2.VideoCapture(self.cam_source, cv2.CAP_DSHOW)
                        if not cap.isOpened():
                            logger.warning("DSHOW failed, falling back to default backend")
                            cap = cv2.VideoCapture(self.cam_source)
                    else:
                        logger.info(f"Using default backend for camera {self.cam_source}")
                        cap = cv2.VideoCapture(self.cam_source)
                    
                    # Set resolution for stability
                    logger.info(f"Setting resolution to 640x480 for {self.rtsp_url}")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                else:
                    # Use FFmpeg explicitly for RTSP
                    logger.info(f"Connecting to RTSP stream: {self.cam_source}")
                    cap = cv2.VideoCapture(self.cam_source, cv2.CAP_FFMPEG)
                    
                    # Configure buffer sizes for low latency
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not cap.isOpened():
                    logger.warning(f"Failed to open {self.rtsp_url}. Retrying in 5s...")
                    self.reconnect_count += 1
                    time.sleep(5)
                    continue
                    
                logger.info(f"{self.rtsp_url} connected successfully.")
                
                # Warm-up: discard first 5 frames to allow exposure adjustment
                if self.is_local_camera:
                    for _ in range(5):
                        cap.read()
                
                frames_read = 0
                start_time = time.time()
                
                while self.is_running:
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning(f"Failed to read frame from {self.rtsp_url}. Reconnecting...")
                        self.reconnect_count += 1
                        break
                        
                    self.frame_id += 1
                    now = time.time()
                    self.last_frame_time = now
                    
                    frames_read += 1
                    elapsed = now - start_time
                    if elapsed > 2.0:
                        self.fps = frames_read / elapsed
                        frames_read = 0
                        start_time = now
                        
                    # Update latest frame and notify all waiting clients
                    with self.new_frame_condition:
                        self.latest_frame = (self.frame_id, now, frame)
                        self.new_frame_condition.notify_all()
                    
            except Exception as e:
                logger.error(f"Error in capture loop: {e}\n{traceback.format_exc()}")
                self.reconnect_count += 1
                time.sleep(5)
            finally:
                if 'cap' in locals() and cap is not None:
                    cap.release()

    def get_latest_frame(self, block=False, timeout=None):
        """Returns (frame_id, timestamp, frame) or (None, None, None) if empty/timeout"""
        if not block:
            with self.new_frame_condition:
                return self.latest_frame
        
        with self.new_frame_condition:
            notified = self.new_frame_condition.wait(timeout=timeout)
            if notified:
                return self.latest_frame
            return None, None, None

    def get_health_metrics(self):
        return {
            "is_running": self.is_running,
            "fps": round(self.fps, 1),
            "latency_ms": round((time.time() - self.last_frame_time) * 1000, 1) if self.last_frame_time else 0,
            "dropped_frames": self.dropped_frames,
            "reconnect_count": self.reconnect_count,
            "last_frame_id": self.frame_id
        }

camera_manager = None

def init_camera(rtsp_url: str):
    global camera_manager
    if camera_manager is not None:
        camera_manager.stop()
    camera_manager = CameraStreamManager(rtsp_url)
    camera_manager.start()
    return camera_manager

def get_camera():
    return camera_manager
