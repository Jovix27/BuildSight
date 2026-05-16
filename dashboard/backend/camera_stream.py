import cv2
import time
import threading
import logging
import traceback
import os

logger = logging.getLogger("buildsight.camera")

class CameraStreamManager:
    """
    Manages camera stream (local webcam or RTSP IP stream).
    - DSHOW backend on Windows for USB webcams, FFmpeg for RTSP.
    - Reconnection loop with backoff.
    - Resets frame#+timestamp on each reconnection so consumers can detect breaks.
    """
    def __init__(self, rtsp_url: str):
        self.is_local_camera = False
        rtsp_url = rtsp_url.strip()
        try:
            self.cam_source = int(rtsp_url)
            self.is_local_camera = True
            self.rtsp_url = f"Camera {self.cam_source}"
        except ValueError:
            self.is_local_camera = False
            if "?tcp" not in rtsp_url and "rtsp://" in rtsp_url:
                self.cam_source = f"{rtsp_url}?tcp"
            else:
                self.cam_source = rtsp_url
            self.rtsp_url = self.cam_source

        self.latest_frame = (None, None, None)
        self.new_frame_condition = threading.Condition()
        self.is_running = False
        self.thread = None
        self._cap = None

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
        self._release_capture()
        if self.thread:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning(f"Capture thread for {self.rtsp_url} did not exit — detaching")
                self.thread = None
        logger.info(f"Camera stream stopped for {self.rtsp_url}")

    def _release_capture(self):
        cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                logger.warning(f"Error releasing capture: {exc}")
            self._cap = None

    def _open_capture(self):
        if self.is_local_camera:
            if os.name == 'nt':
                cap = cv2.VideoCapture(self.cam_source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    logger.warning("DSHOW failed, falling back to default backend")
                    cap = cv2.VideoCapture(self.cam_source)
            else:
                cap = cv2.VideoCapture(self.cam_source)

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap = cv2.VideoCapture(self.cam_source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    def _capture_loop(self):
        while self.is_running:
            try:
                cap = self._open_capture()
                self._cap = cap

                if not cap.isOpened():
                    logger.warning(f"Failed to open {self.rtsp_url}. Retrying in 3s...")
                    self.reconnect_count += 1
                    self._release_capture()
                    time.sleep(3)
                    continue

                logger.info(f"{self.rtsp_url} connected successfully.")

                # Reset frame tracking so consumers see a clean break
                self.frame_id = 0
                self.latest_frame = (None, None, None)
                with self.new_frame_condition:
                    self.new_frame_condition.notify_all()

                # Warm-up: discard early frames to let exposure/white balance settle
                if self.is_local_camera:
                    for _ in range(20):
                        cap.read()

                frames_read = 0
                start_time = time.time()
                last_frame_time = time.time()

                while self.is_running:
                    ret, frame = cap.read()
                    now = time.time()

                    if not self.is_running:
                        break

                    if now - last_frame_time > 5.0 and frames_read > 0:
                        logger.warning(f"No frame from {self.rtsp_url} for 5s. Reconnecting...")
                        self.reconnect_count += 1
                        break

                    if not ret:
                        logger.warning(f"Failed to read frame from {self.rtsp_url}. Reconnecting...")
                        self.reconnect_count += 1
                        break

                    last_frame_time = now
                    self.frame_id += 1
                    self.last_frame_time = now

                    frames_read += 1
                    elapsed = now - start_time
                    if elapsed > 2.0:
                        self.fps = frames_read / elapsed
                        frames_read = 0
                        start_time = now

                    with self.new_frame_condition:
                        self.latest_frame = (self.frame_id, now, frame)
                        self.new_frame_condition.notify_all()

            except Exception as e:
                logger.error(f"Error in capture loop: {e}\n{traceback.format_exc()}")
                self.reconnect_count += 1
                time.sleep(5)
            finally:
                self._release_capture()

    def wait_for_frame(self, timeout=5.0):
        """Block until a frame arrives or timeout. Returns True if frame ready."""
        with self.new_frame_condition:
            return self.new_frame_condition.wait(timeout=timeout)

    def get_latest_frame(self, block=False, timeout=None):
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
