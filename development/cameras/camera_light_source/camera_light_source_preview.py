import cv2
import numpy as np

from development.cameras.helping_tools import drawText


def open_camera_light_source_preview():
    camera_indexes = [0, 2]

    source = cv2.VideoCapture(camera_indexes[0])

    win_name = 'Light Source Tracking Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    lower_range = np.array([0, 0, 255])
    upper_range = np.array([240, 11, 255])

    while cv2.waitKey(1) != 27:  # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        color_mask = cv2.inRange(hsv_frame, lower_range, upper_range)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 4000
        big_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

        for contour in big_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        drawText(frame, 'Press ESC to exit', (20, 20))

        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)
