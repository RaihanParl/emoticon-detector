import pickle

import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: "👍", 1: "👎🏻️", 2: "❤"}
font = "C:\Windows\seguiemj.ttf"
fnt = ImageFont.truetype(font, size=109)


# Function to draw text on an image
def draw_text_with_pil(img, text, position, font, fill):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, fill="#faa8", embedded_color=True, font=font)
    return np.array(pil_img)


while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    predicted_character = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

    frame_with_text = draw_text_with_pil(
        frame,
        predicted_character,
        (0, 32),
        fnt,
        (250, 170, 170, 255),  # RGBA color
    )

    # Convert RGB back to BGR
    frame_with_text_bgr = cv2.cvtColor(frame_with_text, cv2.COLOR_BGR2RGB)

    cv2.imshow("frame", frame_with_text)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
