import cv2
from ultralytics import YOLO

def run_inference():
    """
    Loads a custom-trained YOLOv8 model and runs inference on a video.
    """
   
    model_path = "best.pt"
    model = YOLO(model_path)
    
    
    video_path = "carPark.mp4"
    
   
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source at '{video_path}'.")
        return

    
    print("Inference started. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        
        results = model(frame, conf=0.5)
        
        
        annotated_frame = results[0].plot()
        
    
        cv2.imshow("Inference", annotated_frame) 
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    run_inference()
