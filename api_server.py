import cv2
import os
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import threading

# --- Configuration ---
MODEL_PATH = 'C:/Users/arano/Desktop/ai parking/Backend_flask_AI/best.pt'  # Assumes 'best.pt' is in the same folder
VIDEO_SOURCES = {
    # !!! IMPORTANT: Update these paths to your actual video files !!!
    # These keys MUST match the 'value' from your selection.html
    # I recommend creating a 'videos' folder next to this script
    # and placing your videos inside it.
    "North_Lot": "C:/Users/arano/Desktop/ai parking/Backend_flask_AI/carPark.mp4",
    "South_Garage": "C:/Users/arano/Desktop/ai parking/Backend_flask_AI/carPark1.mp4",
    "East_Field": "videos/east_field.mp4",
}

# --- Global Variables ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Try to load the YOLO model
try:
    model = YOLO(MODEL_PATH)
    print(f"Successfully loaded YOLO model from {MODEL_PATH}")
except Exception as e:
    print(f"--- FATAL ERROR: YOLO model failed to load from {MODEL_PATH} ---")
    print(f"Make sure '{MODEL_PATH}' is in the same directory as this script.")
    print(f"Error details: {e}")
    # If the model fails, we'll use a placeholder
    model = None

# Global dictionary to store parking status
# We use a lock to prevent race conditions when updating/reading
parking_status = {"free": 0, "occupied": 0}
status_lock = threading.Lock()

# --- Video Processing Logic ---

def process_video_frames(video_path):
    """
    Generator function to process a video, yield frames for streaming,
    and update the global parking_status.
    """
    global parking_status
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please check the VIDEO_SOURCES dictionary in api_server.py")
        # Yield a placeholder image if video is missing
        # Create a simple black image as a fallback
        img = cv2.UMat(480, 640, cv2.CV_8UC3)
        img.setTo((0, 0, 0))
        cv2.putText(img, f"Video not found: {video_path}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', img.get())
        img_bytes = buffer.tobytes()
        
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video stream at {video_path}")
        return

    print(f"Starting video processing for: {video_path}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # Loop the video
            print(f"End of video '{video_path}'. Looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if model:
            # Run YOLOv8 inference on the frame
            results = model(frame, verbose=False) # verbose=False to reduce console spam

            # --- Counts ---
            # Your README says the model identifies 'cars' and 'free' spaces
            # We assume 'cars' = occupied and 'free' = free
            # Note: This logic might need tuning based on your model's exact output
            free_count = 0
            occupied_count = 0

            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Iterate over detections
            for r in results:
                for c in r.boxes.cls:
                    class_name = model.names[int(c)]
                    if class_name == 'free':
                        free_count += 1
                    elif class_name == 'car' or class_name == 'occupied': # Adjust as needed
                        occupied_count += 1

            # --- Update Global Status (Thread-Safe) ---
            with status_lock:
                parking_status["free"] = free_count
                parking_status["occupied"] = occupied_count

            # Encode the *annotated* frame as JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)
            if not flag:
                continue
                
            # Yield the output frame in the byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        
        else:
            # Fallback if model failed to load
            cv2.putText(frame, "MODEL FAILED TO LOAD", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    cap.release()
    print(f"Released video capture for: {video_path}")

# --- Flask API Endpoints ---

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    # Get the location from the URL query (e.g., ?location=North_Lot)
    location_key = request.args.get('location', 'North_Lot') # Default to North_Lot
    video_path = VIDEO_SOURCES.get(location_key)

    if not video_path:
        print(f"Error: Invalid location key received: {location_key}")
        return "Error: Invalid location specified.", 404
        
    return Response(process_video_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """API route to get the current parking status."""
    # Read the global status in a thread-safe way
    with status_lock:
        current_status = parking_status.copy()
        
    # Return JSON in the format your README specifies: { "free": X, "occupied": Y }
    return jsonify(free=current_status["free"], occupied=current_status["occupied"])

if __name__ == '__main__':
    print("--- Starting SWIFT SLOT AI Server ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Available Video Sources: {list(VIDEO_SOURCES.keys())}")
    # Run the app
    app.run(debug=True, threaded=True, use_reloader=False)

