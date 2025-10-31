import cv2
from ultralytics import YOLO

def run_inference():
    """
    Loads a custom-trained YOLOv8 model and runs inference on a video.
    """
    # --- Step 1: Load the trained model ---
    # The 'best.pt' file contains the weights of the best-performing model from your training run.
    # Replace the path with the actual path to your saved 'best.pt' file.
    model_path = "best.pt"
    model = YOLO(model_path)
    
    # --- Step 2: Define video source ---
    # You can use a pre-recorded video file.
    video_path = "carPark.mp4"
    
    # You can also use a webcam for live detection by changing the path to an integer (e.g., 0).
    # video_path = 0 
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source at '{video_path}'.")
        return

    # --- Step 3: Run inference and display results ---
    print("Inference started. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference on the frame.
        # The 'conf' parameter is the confidence threshold. You can adjust this.
        # A higher value means the model will only show detections it's more sure about.
        results = model(frame, conf=0.5)
        
        # Get the annotated frame with bounding boxes and labels drawn by YOLO.
        # The 'plot()' method returns a numpy array of the annotated image.
        annotated_frame = results[0].plot()
        
        # Display the result.
        cv2.imshow("Inference", annotated_frame) 
        
        # Press 'q' to quit the display window.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()