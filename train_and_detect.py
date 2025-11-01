from ultralytics import YOLO
import cv2



data_yaml_path = 'dataset.yaml'


model = YOLO('yolov8n.pt')

def train_model():
    """
    Trains a new YOLOv8 model on your custom dataset.
    """
    print("--- Starting model training ---")
    results = model.train(
        data=data_yaml_path,
        epochs=10,  
        imgsz=640,
        name='my_custom_model'
    )
    print("--- Training complete. Model saved to 'runs/detect/my_custom_model/weights/best.pt' ---")
    return results

def detect_on_video(video_path, trained_model_path):
    """
    Uses the trained model to detect objects in a video.
    """
    print("--- Starting video detection ---")
   
    trained_model = YOLO(trained_model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        
        results = trained_model.predict(frame, save=False, classes=0) # Assuming 'car' is class 0

        
        for r in results:
            boxes = r.boxes
            if boxes:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label = trained_model.names[int(box.cls[0].item())]
                    confidence = float(box.conf[0].item())

                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    
                    label_text = f"{label} {confidence:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    
    pass

