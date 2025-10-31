import cv2
import os
import math  
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import threading


MODEL_PATH = 'best.pt'
VIDEO_SOURCES = {
    
    "North_Lot": "carPark1.mp4",
    "South_Garage": "carPark2.mp4",
    "East_Field": "carPark.mp4",   
}


ENTRY_POINTS = {
    "North_Lot": (320, 480),  
    "South_Garage": (100, 450),
    "East_Field": (600, 450),  
}


app = Flask(__name__)
CORS(app)
try:
    model = YOLO(MODEL_PATH)
    print(f"Successfully loaded YOLO model from {MODEL_PATH}")
except Exception as e:
    print(f"--- FATAL ERROR: YOLO model failed to load from {MODEL_PATH} ---")
    print(f"Make sure '{MODEL_PATH}' is in the same directory as this script.")
    print(f"Error details: {e}")
    model = None


parking_status = {"free": 0, "occupied": 0, "direction": "N/A"}
status_lock = threading.Lock()


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)



def process_video_frames(video_path, location_key):
    """
    Generator function to process a video, yield frames for streaming,
    update the global parking_status, and draw navigation lines.
    """
    global parking_status

    entry_point = ENTRY_POINTS.get(location_key)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
       
        img = cv2.UMat(480, 640, cv2.CV_8UC3)
        img.setTo((0, 0, 0))
        cv2.putText(img, f"Video not found: {video_path}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', img.get())
        img_bytes = buffer.tobytes()

        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
      
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video stream at {video_path}")
        return

    print(f"Starting video processing for: {video_path}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(f"End of video '{video_path}'. Looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if model:
            results = model(frame, verbose=False)

            free_slots_centers = []
            occupied_count = 0
            
         
            direction_message = "N/A"
            nearest_center = None

            annotated_frame = results[0].plot()

       
            for r in results:
                for box in r.boxes:
                    class_name = model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    if class_name == 'free':
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        free_slots_centers.append(center)
                    elif class_name == 'car' or class_name == 'occupied':
                        occupied_count += 1

            free_count = len(free_slots_centers)

            
            if entry_point and free_slots_centers:
               
                distances = [
                    (calculate_distance(entry_point, center), center)
                    for center in free_slots_centers
                ]
                nearest_distance, nearest_center = min(distances, key=lambda x: x[0])

                
                X_DIFF_THRESHOLD = 50 
                Y_DIFF_THRESHOLD = 50 

                x_diff = nearest_center[0] - entry_point[0]
                y_diff = nearest_center[1] - entry_point[1]
                
               

                
                if abs(y_diff) < Y_DIFF_THRESHOLD:
                    if abs(x_diff) < X_DIFF_THRESHOLD:
                        direction_message = "Forward"
                    elif x_diff < 0:
                        direction_message = "Slight Left"
                    else:
                        direction_message = "Slight Right"
                
               
                elif y_diff < 0: 
                    if abs(x_diff) < X_DIFF_THRESHOLD:
                        direction_message = "Forward Deep"
                    elif x_diff < 0:
                        direction_message = "Proceed to Front-Left"
                    else:
                        direction_message = "Proceed to Front-Right"
                
                elif y_diff > 0:
                    if abs(x_diff) < X_DIFF_THRESHOLD:
                        direction_message = "Reverse Slot"
                    elif x_diff < 0:
                        direction_message = "Turn Hard Left"
                    else:
                        direction_message = "Turn Hard Right"
                
                
                cv2.line(annotated_frame, entry_point, nearest_center, (255, 255, 0), 3) 
                
              
                cv2.circle(annotated_frame, entry_point, 10, (0, 255, 255), -1) 
                cv2.putText(annotated_frame, "ENTRY", (entry_point[0] + 15, entry_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            elif entry_point:
               
                direction_message = "Lot Full"
                
               
                cv2.circle(annotated_frame, entry_point, 10, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "LOT FULL", (entry_point[0] + 15, entry_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            

            with status_lock:
                parking_status["free"] = free_count
                parking_status["occupied"] = occupied_count
               
                parking_status["direction"] = direction_message

            (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)
            if not flag:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

        else:
            
            cv2.putText(frame, "MODEL FAILED TO LOAD", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    cap.release()
    print(f"Released video capture for: {video_path}")



@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    location_key = request.args.get('location', 'North_Lot')
    video_path = VIDEO_SOURCES.get(location_key)

    if not video_path:
        print(f"Error: Invalid location key received: {location_key}")
        return "Error: Invalid location specified.", 404

    
    return Response(process_video_frames(video_path, location_key),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """API route to get the current parking status."""
    with status_lock:
        current_status = parking_status.copy()

   
    
    return jsonify(
        free=current_status["free"], 
        occupied=current_status["occupied"],
        direction=current_status.get("direction", "N/A") # Added direction
    )

if __name__ == '__main__':
    print("--- Starting SWIFT SLOT AI Server ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Available Video Sources: {list(VIDEO_SOURCES.keys())}")
    print(f"Configured Entry Points: {ENTRY_POINTS}")
   
    app.run(debug=True, threaded=True, use_reloader=False)


