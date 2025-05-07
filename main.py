import cv2 as cv

from utils import hand_detection

count = 0
frame = cv.VideoCapture(0)

detector = hand_detection.Hand_detector(number_of_hands=2)
while True:
    ret, img = frame.read()
    print(img.shape)

    k = cv.waitKey(1)
    if k == ord("q"):
        break

    img, _ = detector.detect_hand_landmarks_in_image(
        img, True, only_landmarks=False
    )
    cv.imshow(
        "video",
        img,
    )
frame.release()
cv.destroyAllWindows()
