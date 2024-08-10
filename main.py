import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train2/weights/best.pt')

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    results = model(frame)
    
    # Annotate the frame with bounding boxes
    for result in results:
        frame = result.plot()  # Use plot() method to draw the results on the frame

    # Display the frame
    cv2.imshow('YOLOv8 Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
