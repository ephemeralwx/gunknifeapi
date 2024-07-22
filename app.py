from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
import tempfile
import io
import zipfile

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
        # Create temporary files for input, output, and JSON
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as input_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.json') as json_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as zip_temp:

            input_filename = input_temp.name
            output_filename = output_temp.name
            json_filename = json_temp.name
            zip_filename = zip_temp.name

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
        chunk_size = 100  # Process 100 frames at a time
        knife_detections = []

        while True:
            frames = []
            for _ in range(chunk_size):
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1

            if not frames:
                break

            # Process chunk of frames
            results = model(frames)

            for i, result in enumerate(results):
                current_time = (frame_count - len(frames) + i + 1) / fps
                frame = frames[i]

                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{model.names[cls]} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if model.names[cls].lower() == 'knife':
                        knife_detections.append({
                            "time": round(current_time, 1),
                            "text": "knife"
                        })

                out.write(frame)

            # Clear memory
            del frames
            del results

        video.release()
        out.release()

        # Write knife detections to JSON file
        with open(json_filename, 'w') as f:
            json.dump(knife_detections, f, indent=2)

        # Create a zip file containing both the video and JSON
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            zipf.write(output_filename, 'processed_video.mp4')
            zipf.write(json_filename, 'knife_detections.json')

        # Clean up temporary files
        os.unlink(input_filename)
        os.unlink(output_filename)
        os.unlink(json_filename)

        # Send the zip file
        return send_file(
            zip_filename,
            mimetype='application/zip',
            as_attachment=True,
            download_name='processed_results.zip'
        )

if __name__ == '__main__':
    app.run(debug=True)