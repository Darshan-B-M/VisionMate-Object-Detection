import cv2
from ultralytics import YOLO


KNOWN_WIDTH = 0.36  


OBJECT_TO_FIND = 'laptop'



model = YOLO('yolov8n.pt')


cap = cv2.VideoCapture(0)
print("Point the camera at your object. Press 'c' to capture.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame, verbose=False)
    
   
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 0, 255), 2)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if class_name == OBJECT_TO_FIND:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                
                pixel_width = x2 - x1
                
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} | Pixel Width: {pixel_width}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    
                    KNOWN_DISTANCE = 1.0 # meters
                    
                    
                    FOCAL_LENGTH = (pixel_width * KNOWN_DISTANCE) / KNOWN_WIDTH
                    
                    print("\n--- CALIBRATION SUCCESSFUL ---")
                    print(f"Object: {OBJECT_TO_FIND}")
                    print(f"Known Real Width: {KNOWN_WIDTH} meters")
                    print(f"Known Distance: {KNOWN_DISTANCE} meters")
                    print(f"Measured Pixel Width: {pixel_width} pixels")
                    print(f"\nYour magic FOCAL_LENGTH is: {FOCAL_LENGTH}")
                    print("--- COPY THIS NUMBER ---")
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()