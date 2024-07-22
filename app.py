from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
import tempfile

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('USEyolov8weights.pt')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if video_file:
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as input_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.json') as json_temp:

            input_filename = input_temp.name
            output_filename = output_temp.name
            json_filename = json_temp.name

        # Save the uploaded file
        video_file.save(input_filename)

        # Process the video
        video = cv2.VideoCapture(input_filename)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        frame_count = 0
        process_every_n_frames = 5
        last_boxes = None
        knife_detections = []

        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps

            if frame_count % process_every_n_frames == 0:
                results = model(frame)
                last_boxes = results[0].boxes.cpu().numpy()

                for box in last_boxes:
                    cls = int(box.cls[0])
                    if model.names[cls].lower() == 'knife':
                        knife_detections.append({
                            "time": round(current_time, 1),
                            "text": "knife"
                        })

            if last_boxes is not None:
                for box in last_boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = box.conf[0]
                    cls = int(box.cls[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = f'{model.names[cls]} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out.write(frame)

        video.release()
        out.release()

        # Write knife detections to JSON file
        with open(json_filename, 'w') as f:
            json.dump(knife_detections, f, indent=2)

        # Clean up the input temp file
        os.unlink(input_filename)

        # Send both files as attachments
        return send_file(output_filename, as_attachment=True, download_name='processed_video.mp4'), \
               send_file(json_filename, as_attachment=True, download_name='knife_detections.json')

