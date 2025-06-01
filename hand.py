import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

import mediapipe as mp
import cv2
import numpy as np


# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    # Initialize webcam (0 for default camera, 1 for external camera)
    cap = cv2.VideoCapture(0)
    
    # Configure MediaPipe Hands
    with mp_hands.Hands(
        min_detection_confidence=0.7,    # Higher value = more confident detection
        min_tracking_confidence=0.5,     # Higher value = more stable tracking
        max_num_hands=2                  # Maximum number of hands to detect
    ) as hands:
        
        while cap.isOpened():
            # Read frame from webcam
            success, image = cap.read()
            if not success:
                print("Failed to read from webcam")
                continue

            # Flip image horizontally for natural viewing
            image = cv2.flip(image, 1)
            
            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = hands.process(image_rgb)
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks and connections
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # Landmarks
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)   # Connections
                    )
            
            # Display the frame
            cv2.imshow('Hand Detection', image)
            
            # Break loop on 'ESC' key
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    




# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ...existing code...

def get_face_measurements(image, face_detection):
    height, width = image.shape[:2]
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Estimate height (rough estimation)
            estimated_height = (h * 7.5) / height
            
            return h, w, estimated_height
    return None, None, None

def main():
    cap = cv2.VideoCapture(0)
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to read from webcam")
                continue
            
            image = cv2.flip(image, 1)
            face_height, face_width, estimated_height = get_face_measurements(image, face_detection)
            
            if face_height and face_width:
                cv2.putText(image, f'Face Height: {face_height}px', (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, f'Face Width: {face_width}px', (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, f'Est. Height Ratio: {estimated_height:.2f}', 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Face Detection & Measurements', image)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()