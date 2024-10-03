import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
# Load the pre-trained model
model = load_model('rock_paper_scissors_model1.h5')

# Initialize player scores
player1_score = 0
player2_score = 0

# Initialize MediaPipe Hands and Selfie Segmentation
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Reverse mapping for gesture labels
reverse_mapping = {0: 'rock', 1: 'paper', 2: 'scissors', 3: 'nothing'}

# Function to get the gesture label from the model's prediction
def get_gesture_label(prediction):
    return reverse_mapping[prediction]

# Function to determine the winner of the game
def determine_winner(player1_gesture, player2_gesture):
    if player1_gesture == "nothing" or player2_gesture == "nothing":
        return "Invalid Move!"
    elif player1_gesture == player2_gesture:
        return "It's a tie!"
    elif (player1_gesture == "rock" and player2_gesture == "scissors") or \
         (player1_gesture == "paper" and player2_gesture == "rock") or \
         (player1_gesture == "scissors" and player2_gesture == "paper"):
        return "Player 1 wins!"
    else:
        return "Player 2 wins!"

# Variables for controlling the game rounds
start_round = False
round_start_time = 0
round_duration = 3  # 3 seconds for each round

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Resize the frame to zoom in (e.g., 130% of the original size)
    frame = cv2.resize(frame, None, fx=1.3, fy=1.3)

    # Split the frame into two halves
    h, w, _ = frame.shape
    left_half = frame[:, :w//2]
    right_half = frame[:, w//2:]

    # Convert the image to RGB and pass it to MediaPipe Hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    player1_gesture = "nothing"
    player2_gesture = "nothing"

    # If hands are detected
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extract bounding box with more padding to make ROI larger
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w) - 50
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h) - 50
            x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w) + 50
            y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h) + 50
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            # Extract ROI and preprocess
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, (224, 224))
            roi = roi / 255.0

            # Predict gesture
            prediction = model.predict(np.expand_dims(roi, axis=0))
            gesture_label = get_gesture_label(np.argmax(prediction))

            if i == 0:
                player2_gesture = gesture_label
            elif i == 1:
                player1_gesture = gesture_label

            # Draw bounding box and label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, gesture_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    if start_round:
        elapsed_time = time.time() - round_start_time
        cv2.putText(frame, f"Time: {int(elapsed_time)}s", (w // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if elapsed_time >= round_duration:
            start_round = False
            # Determine the winner and update scores
            winner = determine_winner(player1_gesture, player2_gesture)
            if winner == "Player 1 wins!":
                player1_score += 1
            elif winner == "Player 2 wins!":
                 player2_score += 1
            cv2.putText(frame, winner, (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow('Rock Paper Scissors', frame)
            cv2.waitKey(1)  # Refresh the display
            time.sleep(2)  # Delay for 2 seconds after round ends
    else:
        cv2.putText(frame, "Press 'P' to Start", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display scores
    cv2.putText(frame, f"Player 1: {player1_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Player 2: {player2_score}", (w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Rock Paper Scissors', frame)

    # Check for key press to start a new round or exit the game
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p') and not start_round:
        start_round = True
        round_start_time = time.time()
    elif key == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()