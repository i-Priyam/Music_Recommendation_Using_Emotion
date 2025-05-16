import mediapipe as mp
import numpy as np
import cv2
import os

# Initialize MediaPipe modules
holistic = mp.solutions.holistic
hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
holis = holistic.Holistic()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open video stream.")
    exit()

# Ask for gesture label
name = input("Enter the name of the data: ").strip()
if not name:
    print("❌ Error: Name cannot be empty!")
    cap.release()
    exit()

X = []
data_size = 0

# Ensure 'data' directory exists
os.makedirs('data', exist_ok=True)

while True:
    lst = []

    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    result = holis.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.face_landmarks:
        # Normalize face landmarks
        for lm in result.face_landmarks.landmark:
            lst.append(lm.x - result.face_landmarks.landmark[1].x)
            lst.append(lm.y - result.face_landmarks.landmark[1].y)

        # Normalize left hand
        if result.left_hand_landmarks:
            for lm in result.left_hand_landmarks.landmark:
                lst.append(lm.x - result.left_hand_landmarks.landmark[8].x)
                lst.append(lm.y - result.left_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

        # Normalize right hand
        if result.right_hand_landmarks:
            for lm in result.right_hand_landmarks.landmark:
                lst.append(lm.x - result.right_hand_landmarks.landmark[8].x)
                lst.append(lm.y - result.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

        X.append(lst)
        data_size += 1

    # Draw landmarks
    drawing.draw_landmarks(frame, result.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frame, result.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frame, result.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Show data count
    cv2.putText(frame, f"Samples: {data_size}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display window
    cv2.imshow("Collecting Data", frame)

    # Exit condition: Esc key or 100 samples
    if cv2.waitKey(1) & 0xFF == 27 or data_size >= 500:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save the collected data
np.save(f"data/{name}.npy", np.array(X))
print(f"✅ Data saved to data/{name}.npy with shape: {np.array(X).shape}")
