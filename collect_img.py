import os

import cv2
import time


DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print("Collecting data for class {}".format(j))

    done = False
    clicked = False
    while True:
        ret, frame = cap.read()
        cv2.putText(
            frame,
            'Press "Q" to capture',
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) == ord("q"):
            break

    TIMER = int(3)
    prev = time.time()

    while TIMER >= 0:
        ret, img = cap.read()

        # Display countdown on each frame
        # specify the font and draw the
        # countdown using puttext
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img,
            f"take image in {TIMER}",
            (100, 50),
            font,
            1.3,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.imshow("frame", img)
        cv2.waitKey(25)

        # current time
        cur = time.time()

        # Update and keep track of Countdown
        # if time elapsed is one second
        # then decrease the counter
        if cur - prev >= 1:
            prev = cur
            TIMER = TIMER - 1

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), "{}.jpg".format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
