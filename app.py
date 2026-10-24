import cv2
import time
import subprocess
from ultralytics import YOLO
import speech_recognition as sr 


SPEECH_COOLDOWN = 3.0 
MIN_CONFIDENCE = 0.50


FOCAL_LENGTH = 714.28 

KNOWN_WIDTHS = {
    
    'person': 0.45, 'cat': 0.15, 'dog': 0.30, 'horse': 0.5, 'sheep': 0.4, 'cow': 0.7,
    
    'bicycle': 0.6, 'car': 1.8, 'motorcycle': 0.8, 'bus': 2.5, 'truck': 2.6,
    'traffic light': 0.3, 'stop sign': 0.75, 'parking meter': 0.2, 'bench': 1.5,
    'fire hydrant': 0.3,
    
    'chair': 0.5, 'couch': 2.0, 'bed': 1.5, 'dining table': 0.9, 'toilet': 0.4,
    'potted plant': 0.3, 'refrigerator': 0.75, 'oven': 0.7, 'sink': 0.5,
    'microwave': 0.5, 'toaster': 0.15, 'clock': 0.3, 'vase': 0.15,
    
    'laptop': 0.36, 'book': 0.15, 'cell phone': 0.07, 'tv': 0.8, 'mouse': 0.06,
    'remote': 0.05, 'keyboard': 0.4,
    
    'bottle': 0.07, 'cup': 0.08, 'wine glass': 0.08, 'fork': 0.025, 'knife': 0.02,
    'spoon': 0.035, 'bowl': 0.15, 'banana': 0.04, 'apple': 0.08, 'sandwich': 0.12,
    'orange': 0.08, 'pizza': 0.35, 'donut': 0.1,
    
    'backpack': 0.3, 'handbag': 0.3, 'suitcase': 0.5, 'umbrella': 1.0,
    'sports ball': 0.22, 'scissors': 0.08, 'teddy bear': 0.3,
}


WAKE_COMMAND = "start command"


def listen_for_wake_word():
    """
    Listens for the wake command and returns True if it's heard.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"Listening for '{WAKE_COMMAND}'...")
        
        try:
            
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            
            
            text = recognizer.recognize_google(audio)
            text = text.lower()
            
            print(f"Heard: '{text}'")
            
            
            if WAKE_COMMAND in text:
                return True
                
        except sr.UnknownValueError:
            
            print("Could not understand audio.")
        except sr.RequestError as e:
            
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except sr.WaitTimeoutError:
            
            print("No speech detected.")
        
    return False

def start_detection_loop():
    """
    This contains all of our previous object detection code.
    """
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera. Exiting.")
        return 
        
    last_speech_time = 0
    print("Starting continuous detection... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        frame_width = frame.shape[1]
        results = model(frame, stream=True, verbose=False) 
        
        best_object_name = None
        max_confidence = 0
        best_object_distance = 0
        best_object_direction = ""

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                confidence = float(box.conf[0])
                
                if class_name in KNOWN_WIDTHS and confidence > MIN_CONFIDENCE and confidence > max_confidence:
                    max_confidence = confidence
                    best_object_name = class_name
                    x1, y1, x2, y2 = box.xyxy[0]
                    pixel_width = x2 - x1
                    
                    if pixel_width > 0:
                        distance = (KNOWN_WIDTHS[class_name] * FOCAL_LENGTH) / pixel_width
                        best_object_distance = distance
                    
                    object_center = (x1 + x2) / 2
                    if object_center < frame_width * 0.33:
                        best_object_direction = "on your left"
                    elif object_center > frame_width * 0.66:
                        best_object_direction = "on your right"
                    else:
                        best_object_direction = "ahead"

                
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
        current_time = time.time()
        if best_object_name and best_object_distance > 0 and (current_time - last_speech_time > SPEECH_COOLDOWN):
            distance_str = f"{best_object_distance:.1f}"
            speech_text = f"{best_object_name}, {distance_str} meters {best_object_direction}"
            print(f"Announcing: {speech_text}")
            last_speech_time = current_time
            
            try:
                command = f"powershell -Command \"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{speech_text}')\""
                subprocess.Popen(command, shell=True)
            except Exception as e:
                print(f"Error trying to speak: {e}")

        
        cv2.imshow("Real-Time Object Announcer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    print("Stopping detection.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        if listen_for_wake_word():
            print("Wake command detected!")
            
            
            try:
                command = f"powershell -Command \"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('Starting detection')\""
                subprocess.Popen(command, shell=True)
            except Exception as e:
                print(f"Error trying to speak: {e}")
            
            
            start_detection_loop()
            
            
            break
        else:
            
            print("Wake word not detected, listening again...")
            