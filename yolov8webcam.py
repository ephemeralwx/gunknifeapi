from ultralytics import YOLO
import cv2
import numpy as np
import json
import torch
import platform

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print(f"Using CPU: {platform.processor()}")

# Load the YOLOv8 model on the chosen device
model = YOLO('USEyolov8weights.pt').to(device)

# Initialize the second webcam
cap = cv2.VideoCapture(1)  # Use index 1 for the second camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

knife_detections = []
frame_count = 0
process_every_n_frames = 2

cv2.namedWindow("Webcam Capture", cv2.WINDOW_NORMAL)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    frame_count += 1
    current_time = frame_count / 30  # Assuming 30 fps, adjust if needed

    if frame_count % process_every_n_frames == 0:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        boxes = results[0].boxes.cpu().numpy()
        
        # Process detections
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Check for knife detections
            if model.names[cls].lower() == 'knife':
                knife_detections.append({
                    "time": round(current_time, 1),
                    "text": "knife"
                })

    # Display the frame
    cv2.imshow("Webcam Capture", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Write knife detections to a JSON file
with open('knife_detections.json', 'w') as f:
    json.dump(knife_detections, f, indent=2)

print(f"Processing complete.")
print(f"Knife detections have been saved to 'knife_detections.json'")