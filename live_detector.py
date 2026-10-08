import cv2
from ultralytics import YOLO
import argparse 

parser = argparse.ArgumentParser(description="YOLOv8 Live Detector for CSA2001 - Soham Shinde")
parser.add_argument("--model", default="yolov8n.pt", help="Path to model file")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0.0 to 1.0)")
args = parser.parse_args()

model = YOLO(args.model)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

print(f"Starting webcam with confidence {args.conf}... Press 'q' to quit.")

while True:
    success, frame = cap.read()

    if success:
        
        results = model(frame, stream=True, conf=args.conf)
    
        annotated_frame = frame 

        for r in results:
            annotated_frame = r.plot()
        
        cv2.imshow("CSA2001 Project: Real-time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Error: Failed to capture image.")
        break

cap.release()
cv2.destroyAllWindows()