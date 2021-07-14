import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

# default parameters:
#   static_image_mode=False,
#   max_min_hands=2,
#   max_detection_confidence=0.5,
#   min_tracking_confidence=0.5
hands = mpHands.Hands()


while True:
    success, img =cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    cv2.imshow("img",img)
    cv2.waitKey(1)