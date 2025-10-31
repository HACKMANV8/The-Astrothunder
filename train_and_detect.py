from ultralytics import YOLO
import cv2

# --- Step 1: Define the Dataset and Model ---
# This YAML file tells YOLO where to find your data and what your classes are.
# You need to create this file in your project folder.
# Example content for 'dataset.yaml':
# train: my_dataset/images/train/
# val: my_dataset/images/val/
# nc: 1  # number of classes, e.g., 1 for 'car'
# names: ['car']
data_yaml_path = 'dataset.yaml'

# Load a pre-trained YOLOv8n model. This is the smallest and fastest version.
model = YOLO('yolov8n.pt')

def train_model():
    """
    Trains a new YOLOv8 model on your custom dataset.
    """
    print("--- Starting model training ---")
    results = model.train(
        data=data_yaml_path,
        epochs=10,  # You can increase this for better accuracy
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
    # Load the trained model
    trained_model = YOLO(trained_model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Pass the frame to the trained model for detection
        results = trained_model.predict(frame, save=False, classes=0) # Assuming 'car' is class 0

        # Draw the bounding boxes and labels
        for r in results:
            boxes = r.boxes
            if boxes:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label = trained_model.names[int(box.cls[0].item())]
                    confidence = float(box.conf[0].item())

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label_text = f"{label} {confidence:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Main function to run the script ---
if __name__ == '__main__':
    # Uncomment the line below to train the model.
    # train_model()

    # Uncomment the line below to run detection after training.
    # Replace 'path/to/your/video.mp4' with your video file.
    # The 'best.pt' file is created after successful training.
    # detect_on_video('path/to/your/video.mp4', 'runs/detect/my_custom_model/weights/best.pt')
    pass
