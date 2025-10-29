import cv2
import time
import torch
from ultralytics import YOLO
from datetime import datetime
import winsound  # For alert sound (works on Windows)

# -----------------------------
# 1️⃣ Load your trained YOLO model
# -----------------------------
model_path = "models/visionguard_exp2/weights/best.pt"
model = YOLO(model_path)

# -----------------------------
# 2️⃣ Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()

# -----------------------------
# 3️⃣ Variables for FPS calculation
# -----------------------------
prev_time = 0
fps = 0

# -----------------------------
# 4️⃣ Loop over webcam frames
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Run YOLO inference
    results = model(frame, stream=True)

    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            color = (0, 255, 0) if label == "With Helmet" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # --------------- ALERT SYSTEM ---------------
            if label == "Without Helmet" and conf > 0.4:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"violations/without_helmet_{timestamp}.jpg"

                # Save snapshot of violation
                cv2.imwrite(filename, frame)
                print(f"⚠️ Snapshot saved: {filename}")

                # Play beep alert (frequency=1000Hz, duration=400ms)
                winsound.Beep(1000, 400)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam feed
    cv2.imshow("VisionGuard", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

# -----------------------------
# 5️⃣ Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()

