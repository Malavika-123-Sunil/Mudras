import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            landmark_list = []

            for lm in hand_landmarks.landmark:

                landmark_list.extend([lm.x, lm.y, lm.z])

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            key = cv2.waitKey(1)

            if key == ord('p'):

                with open('dataset.csv', 'a', newline='') as f:

                    writer = csv.writer(f)

                    landmark_list.append("Pathaka")

                    writer.writerow(landmark_list)

                print("Pathaka saved")

            if key == ord('m'):

                with open('dataset.csv', 'a', newline='') as f:

                    writer = csv.writer(f)

                    landmark_list.append("Mushti")

                    writer.writerow(landmark_list)

                print("Mushti saved")
                
                
            if key == ord('t'):

                with open('dataset.csv', 'a', newline='') as f:

                    writer = csv.writer(f)

                    landmark_list.append("Thripathaaka")

                    writer.writerow(landmark_list)

                print("Thripathaaka saved")
                
            
            if key == ord('a'):

                with open('dataset.csv', 'a', newline='') as f:

                    writer = csv.writer(f)

                    landmark_list.append("Ardhapathaaka")

                    writer.writerow(landmark_list)

                print("Ardhapathaaka saved")
                
            if key == ord('k'):

                with open('dataset.csv', 'a', newline='') as f:

                    writer = csv.writer(f)

                    landmark_list.append("KartariMukham")

                    writer.writerow(landmark_list)

                print("KartariMukham saved")
            
            if key == ord('y'):

                with open('dataset2.csv', 'a', newline='') as f:

                    writer = csv.writer(f)

                    landmark_list.append("Mayuram")

                    writer.writerow(landmark_list)

                print("Mayuram saved")
                
            if key == ord('r'):

                with open('dataset2.csv', 'a', newline='') as f:

                    writer = csv.writer(f)

                    landmark_list.append("Aralam")

                    writer.writerow(landmark_list)

                print("Aralam saved")
                
            
            
            
    cv2.imshow("Collect Data", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()