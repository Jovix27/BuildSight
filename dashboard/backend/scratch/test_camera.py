import cv2
import time

def test_camera(index=0):
    print(f"Testing camera at index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Failed to open camera with CAP_DSHOW. Trying default...")
        cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Failed to open camera at index {index}")
        return False
    
    print(f"Camera {index} opened successfully.")
    ret, frame = cap.read()
    if ret:
        print(f"Successfully captured a frame of size {frame.shape}")
        cv2.imwrite(f"camera_test_{index}.jpg", frame)
        print(f"Saved test frame to camera_test_{index}.jpg")
    else:
        print(f"Failed to capture a frame from camera {index}")
    
    cap.release()
    return ret

if __name__ == "__main__":
    test_camera(0)
    test_camera(1)
