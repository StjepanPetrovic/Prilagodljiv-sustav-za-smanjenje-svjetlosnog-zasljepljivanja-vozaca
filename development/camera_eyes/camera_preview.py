import cv2


def drawText(frame, txt, location, color=(50, 50, 170)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def open_camera_preview():
    # Available camera indexes
    # The index for the integrated webcam is typically 0
    # The index for my external webcam is 2, but maybe for you is 1 or something else
    camera_indexes = [0, 2]

    source = cv2.VideoCapture(camera_indexes[0])

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    win_name = 'Eyes Tracking Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cv2.waitKey(1) != 27:  # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        drawText(frame, 'Press ESC to exit', (20, 20))

        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)
