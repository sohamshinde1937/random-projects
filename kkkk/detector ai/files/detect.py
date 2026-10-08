import cv2
import sys
import time
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"])
    from ultralytics import YOLO

# ── Colour palette for bounding boxes ──────────────────────────────────────
COLORS = [
    (255,  56,  56), (255, 157,  51), ( 76, 213, 112), ( 64, 159, 255),
    (215,  64, 255), (255,  64, 153), ( 64, 255, 220), (255, 222,  64),
    (120, 120, 255), (255, 120, 120),
]

def color_for(class_id):
    return COLORS[class_id % len(COLORS)]

def draw_box(frame, x1, y1, x2, y2, label, conf, color):
    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    text = f"{label}  {conf:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 5
    by1 = max(y1 - th - 2*pad, 0)
    by2 = by1 + th + 2*pad
    cv2.rectangle(frame, (x1, by1), (x1 + tw + 2*pad, by2), color, -1)
    cv2.putText(frame, text, (x1 + pad, by2 - pad),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_hud(frame, fps, num_objects):
    h, w = frame.shape[:2]
    # Top-left HUD panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 70), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "AI Object Detector",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {num_objects}",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1, cv2.LINE_AA)
    # Bottom hint
    cv2.putText(frame, "Press Q to quit",
                (w - 140, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

def main():
    print("=" * 50)
    print("  AI Object Detector")
    print("  Powered by YOLOv8 + OpenCV")
    print("=" * 50)
    print("\nLoading YOLOv8 model (first run downloads ~6 MB)...")

    model = YOLO("yolov8n.pt")   # nano – fast on CPU
    print("Model loaded!\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Make sure it is connected.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print("Camera opened. Press Q in the window to quit.\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # ── Run YOLO inference ──────────────────────────────────────────────
        results = model(frame, verbose=False, conf=0.4)[0]

        num_objects = 0
        for box in results.boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            label   = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color   = color_for(cls_id)
            draw_box(frame, x1, y1, x2, y2, label, conf, color)
            num_objects += 1

        # ── FPS ────────────────────────────────────────────────────────────
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        draw_hud(frame, fps, num_objects)

        cv2.imshow("AI Object Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
