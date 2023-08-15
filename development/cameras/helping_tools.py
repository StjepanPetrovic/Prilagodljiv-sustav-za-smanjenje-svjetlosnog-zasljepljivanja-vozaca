import cv2


def drawText(frame, txt, location, color=(50, 50, 170)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
