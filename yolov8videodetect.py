#BEST WORKING ONE AS OF 2:59 PM 7/22/2024

from ultralytics import YOLO
import cv2
import numpy as np
import json
import torch
import platform
# Check if GPU is available
if torch.cuda.is_available():
  # Set the desired GPU device (0 for first GPU)
  device = torch.device('cuda:0')
  print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
  # Fallback to CPU if no GPU available
  device = torch.device('cpu')
  print(f"Using CPU: {platform.processor()}")

# Load the YOLOv8 model on the chosen device
model = YOLO('USEyolov8weights.pt').to(device)





# Open the video file
video = cv2.VideoCapture('test1.mp4')

# Get video properties
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

frame_count = 0
process_every_n_frames = 2  # Process every 5th frame
last_boxes = None
knife_detections = []

while True:
    # Read a frame from the video
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    current_time = frame_count / fps  # Calculate current time in seconds

    if frame_count % process_every_n_frames == 0:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        last_boxes = results[0].boxes.cpu().numpy()
        
        # Check for knife detections
        for box in last_boxes:
            cls = int(box.cls[0])
            if model.names[cls].lower() == 'knife':
                knife_detections.append({
                    "time": round(current_time, 1),
                    "text": "knife"
                })
    
    # If we have detected boxes (either from this frame or a previous one), draw them
    if last_boxes is not None:
        for box in last_boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release everything
video.release()
out.release()

# Write knife detections to a JSON file
with open('knife_detections.json', 'w') as f:
    json.dump(knife_detections, f, indent=2)

print(f"Processing complete. Output video saved as 'output_video.mp4'")
print(f"Knife detections have been saved to 'knife_detections.json'")