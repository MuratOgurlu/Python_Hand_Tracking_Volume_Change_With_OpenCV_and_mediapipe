import cv2
import mediapipe as mp
from math import hypot
import pyautogui as pyg
width, height = 1280, 720
x1, y1, x2, y2 = 0, 0, 0, 0
# burada elimizi çizeceğimiz temel işlemleri başlatıyoruz
my_hand = mp.solutions.hands
hand = my_hand.Hands()
my_drawing_utils = mp.solutions.drawing_utils

# kamera ayarları
webcam = cv2.VideoCapture(0)
webcam.set(3, width)
webcam.set(4, height)

while webcam.isOpened():

    success, img = webcam.read()
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hand.process(imgRgb)
    if result.multi_hand_landmarks:
        for id, lm in enumerate(result.multi_hand_landmarks):
            thumb_number = lm.landmark[my_hand.HandLandmark.THUMB_TIP]
            finger_number = lm.landmark[my_hand.HandLandmark.INDEX_FINGER_TIP]
            x1, y1 = int(thumb_number.x * img.shape[1]), int(thumb_number.y * img.shape[0])
            x2, y2 = int(finger_number.x * img.shape[1]), int(finger_number.y * img.shape[0])
        my_drawing_utils.draw_landmarks(img, lm, my_hand.HAND_CONNECTIONS)
        c1 = cv2.circle(img, (x1, y1), 15, (0, 255, 0), 3)
        c2 = cv2.circle(img, (x2, y2), 15, (0, 255, 0), 3)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        length = hypot(x2 - x1, y2 - y1)#math kütüphanesindeki hipotenüs fonksiyonuyla uzunluk aldık
        if length > 50:
            pyg.press("volumeup")
        elif length < 50:
            pyg.press("volumedown")

    cv2.imshow("hand track volume change", img)
    key = cv2.waitKey(10)
    if key == 27 or key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
