
import cv2
from ultralytics import YOLO
import numpy as np
from activity_utils import check_activity
import csv
from datetime import datetime


# --- Step 1: Frame Capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Load YOLOv8 Pose model
model = YOLO("/home/advik/Desktop/worker-monitor/models/yolov8n-pose.engine")


# Open CSV log file
log_file = open("/home/advik/Desktop/worker-monitor/logs/activity_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Worker_ID", "Activity"])
display = False
previous_activity = {}
prev_kpts = {}
prev_gray = None


# cv2.namedWindow("Worker Activity Monitor", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Worker Activity Monitor", 800, 600)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Step 2: Preprocessing ---
    frame_resized = cv2.resize(frame, (640, 480))  # Resize for faster inference
    curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    frame_normalized = frame_resized / 255.0  

    # --- Step 3: ROI ---
    #roi = frame_resized[50:330, 100:540]  
    roi = frame_resized[70:400, 50:600]
    cv2.rectangle(frame_resized, (50, 70), (600, 400), (255, 255, 0), 2)  # Light blue box
    cv2.putText(frame_resized, "Monitoring ROI", (100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    # --- Step 4: Detection & Classification ---
    results = model(roi,conf=0.7)

    for result in results:
        keypoints = result.keypoints
        boxes = result.boxes

        if keypoints is not None:
            for i, kpt in enumerate(keypoints.data):
                kpt = kpt.cpu().numpy()
                #activity = check_activity(kpt)

                 # Worker ID
                worker_id = f"Worker_{i+1}"

                if len(boxes) > i:
                    box = boxes[i].xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                else:
                    continue  # Skip this keypoint if no box is associated

                
                activity = check_activity(
                    kpts=kpt,
                    prev_kpts=prev_kpts.get(worker_id, None),
                    prev_frame=prev_gray,
                    curr_frame=curr_gray,
                    roi=(x1, y1, x2 - x1, y2 - y1),
                    worker_id=worker_id
                )
                color = (0, 255, 0) if activity == "Working" else (0, 0, 255)

                if previous_activity.get(worker_id) != activity:
                    previous_activity[worker_id] = activity

                    # Log activity with timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_writer.writerow([timestamp, f"Worker_{i+1}", activity])

                # Draw box and label
                cv2.rectangle(roi, (x1, y1), (x2, y2), color, 2)
                cv2.putText(roi, activity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw keypoints
                for j in range(17):
                    x, y, conf = kpt[j]
                    if conf > 0.5:
                        cv2.circle(roi, (int(x), int(y)), 3, color, -1)

                prev_kpts[worker_id] = kpt


    prev_gray = curr_gray.copy()


    #frame_resized[50:330, 100:540] = roi
    frame_resized[70:400, 50:600] = roi

    if display:
        cv2.imshow("Worker Activity Monitor", frame_resized)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()