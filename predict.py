import time
import cv2
import mediapipe as mp
import pickle
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

voices = engine.getProperty('voices')

# Female voice (change index if needed)
engine.setProperty('voice', voices[1].id)

# Speaking speed
engine.setProperty('rate', 120)


# Load trained model
with open("mudra_model.pkl", "rb") as f:
    model = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Store previous spoken prediction
last_prediction = ""

# Stable prediction variables
stable_prediction = ""
prediction_count = 0

# Speech cooldown settings
last_spoken_time = 0
cooldown = 2

# Better pronunciation
speak_name = {
    "Pathaka": "Pa-tha-ka",
    "Mushti": "Moosh-ti",
    "Thripathaaka": "Thri-pa-tha-ka",
    "Ardhapathaaka": "Ar-dha-pa-thaa-ka",
    "KartariMukham": "Kar-ta-ri-Mukh-am",
    "Mayuram": "Ma-yu-ra,",
    "Aralam": "Aa-ra-lam",    
}

while True:

    success, img = cap.read()

    if not success:
        break

    # Flip image for mirror effect
    img = cv2.flip(img, 1)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            landmark_list = []

            # Extract landmarks
            for lm in hand_landmarks.landmark:

                landmark_list.extend([lm.x, lm.y, lm.z])

            # Predict mudra
            prediction = model.predict([landmark_list])

            mudra_name = prediction[0]

            current_time = time.time()

            # Check stable prediction
            if mudra_name == stable_prediction:

                prediction_count += 1

            else:

                stable_prediction = mudra_name

                prediction_count = 0

            # Speak only if:
            # 1. Prediction stable
            # 2. Different from last spoken
            # 3. Cooldown completed
            if (
                prediction_count > 15
                and mudra_name != last_prediction
                and current_time - last_spoken_time > cooldown
            ):

                engine.say(speak_name.get(mudra_name,mudra_name))

                engine.runAndWait()

                last_prediction = mudra_name

                last_spoken_time = current_time

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Display mudra name
            cv2.putText(
                img,
                mudra_name,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Show webcam
    cv2.imshow("Mudra Prediction", img)

    # Quit on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()

cv2.destroyAllWindows()