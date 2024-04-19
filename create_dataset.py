import os
import pickle

import mediapipe as mp

import cv2

data_dir = "./data"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []
for class_type in os.listdir(data_dir):
    class_dir_path = os.path.join(data_dir, class_type)
    for img_path in os.listdir(class_dir_path):
        data_aux = []
        img = cv2.imread(os.path.join(class_dir_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_ = []
        y_ = []
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            if len(data_aux) != 84:
                data_aux.extend(0 for _ in range(42))
            data.append(data_aux)
            labels.append(class_type)

f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()
