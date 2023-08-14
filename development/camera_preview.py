import cv2


def drawText(frame, txt, location, color=(50, 50, 170)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def open_camera_preview():
    camera_index = 0

    source = cv2.VideoCapture(camera_index)

    win_name = 'Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cv2.waitKey(1) != 27:  # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break

        drawText(frame, 'Press ESC to exit', (20, 20))

        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)
