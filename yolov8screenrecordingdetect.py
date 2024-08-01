from ultralytics import YOLO
import cv2
import numpy as np
import json
import torch
import platform
import mss
import numpy as np
import tkinter as tk
from tkinter import simpledialog

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print(f"Using CPU: {platform.processor()}")

# Load the YOLOv8 model on the chosen device
model = YOLO('USEyolov8weights.pt').to(device)

# Initialize the screen capture
sct = mss.mss()

# Get information about all monitors
monitors = sct.monitors

# Create a simple dialog to choose the monitor
root = tk.Tk()
root.withdraw()  # Hide the main window

print("Available monitors:")
for i, m in enumerate(monitors[1:], 1):  # Skip the first monitor (usually represents "all monitors")
    print(f"{i}: {m['left']}x{m['top']} {m['width']}x{m['height']}")

monitor_index = simpledialog.askinteger("Monitor Selection", "Enter the number of the monitor to capture:", 
                                        minvalue=1, maxvalue=len(monitors)-1)

# Get the selected monitor
monitor = monitors[monitor_index]
#whats up guys, kevin here, today im gonna show you what we're currently working on, a live detection inference system. right now our system only works for videos, but we're working to make it work for live feeds.
#In this video, i'm broadcasting a live feed from my phone as I simulate what I think a threatening person would do, and as you can see, the model works pretty well. 
#no minions were hurt in the making of this video

knife_detections = []
frame_count = 0
process_every_n_frames = 2

cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Screen Capture", monitor['width']//2, monitor['height']//2)  # Adjust as needed

while True:
    # Capture screen
    screen = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)

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
    cv2.imshow("Screen Capture", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Write knife detections to a JSON file
with open('knife_detections.json', 'w') as f:
    json.dump(knife_detections, f, indent=2)

print(f"Processing complete.")
print(f"Knife detections have been saved to 'knife_detections.json'")