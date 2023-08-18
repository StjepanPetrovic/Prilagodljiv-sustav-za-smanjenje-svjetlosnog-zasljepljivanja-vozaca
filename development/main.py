import cv2 as cv
import threading
from queue import Queue
import numpy as np

eyes_frames_queue = Queue()
light_frames_queue = Queue()

eyes_position_queue = Queue()
light_position_queue = Queue()


def read_camera_frames_and_save_in_queues(stop_event):
    camera_indexes = [0, 2]

    eyes_source = cv.VideoCapture(camera_indexes[0])
    light_source = cv.VideoCapture(camera_indexes[1])

    while not stop_event.is_set():
        has_eye_frame, eye_frame = eyes_source.read()
        has_light_frame, light_frame = light_source.read()

        if not has_eye_frame or not has_light_frame:
            print("Frame not found. Check cameras.\n")
            break

        eye_frame, light_frame = analyze_frames(eye_frame, light_frame)

        eyes_frames_queue.put(eye_frame)
        light_frames_queue.put(light_frame)

    eyes_source.release()
    light_source.release()


def analyze_frames(eyes_frame, light_frame):
    eyes_frame = detect_eyes(eyes_frame)
    light_frame = detect_light(light_frame)

    return eyes_frame, light_frame


def detect_eyes(eyes_frame):
    eyes_gray_frame = cv.cvtColor(eyes_frame, cv.COLOR_BGR2GRAY)

    eye_cascade_model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

    eyes = eye_cascade_model.detectMultiScale(eyes_gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    eyes_position_queue.put(eyes)

    for (x, y, w, h) in eyes:
        cv.rectangle(eyes_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    drawText(eyes_frame, 'Press ESC to exit', (20, 20))

    return eyes_frame


def detect_light(light_frame):
    lower_range = np.array([0, 0, 255])
    upper_range = np.array([240, 11, 255])

    light_hsv_frame = cv.cvtColor(light_frame, cv.COLOR_BGR2HSV)

    color_mask = cv.inRange(light_hsv_frame, lower_range, upper_range)

    contours, _ = cv.findContours(color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    min_contour_area = 4000
    big_contours = [contour for contour in contours if cv.contourArea(contour) > min_contour_area]

    light_positions = []

    for contour in big_contours:
        light_positions.append(cv.boundingRect(contour))
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(light_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    light_position_queue.put(light_positions)

    drawText(light_frame, 'Press ESC to exit', (20, 20))

    return light_frame


def drawText(frame, txt, location, color=(50, 50, 170)):
    cv.putText(frame, txt, location, cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def read_queues_and_show_it(eyes_window, light_window, protection_window):
    while cv.waitKey(1) != 27:
        eye_frame = eyes_frames_queue.get()
        light_frame = light_frames_queue.get()

        if eye_frame is None or light_frame is None:
            break

        protection_frame = createProtectionFrame()

        light_frames_queue.task_done()
        eyes_frames_queue.task_done()

        cv.imshow(eyes_window, eye_frame)
        cv.imshow(light_window, light_frame)
        cv.imshow(protection_window, protection_frame)

    stop_read_event.set()

    cv.destroyAllWindows()


def createProtectionFrame():
    frame_width = 640
    frame_height = 480

    protection_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

    eye_positions = eyes_position_queue.get()
    light_positions = light_position_queue.get()

    for light_position in light_positions:
        for eye_position in eye_positions:
            eye_x, eye_y, eye_w, eye_h = eye_position
            light_x, light_y, light_w, light_h = light_position

            protection_x = (eye_x + light_x) // 2

            protection_y = (eye_y + light_y) // 2

            cv.rectangle(
                protection_frame,
                (protection_x, protection_y),
                (protection_x + light_w, protection_y + light_h),
                (0, 0, 0),
                -1
            )

    eyes_position_queue.task_done()
    light_position_queue.task_done()

    return protection_frame


def open_window(name):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(name, cv.WND_PROP_AUTOSIZE, cv.WINDOW_NORMAL)


if __name__ == '__main__':
    win_name_eyes = 'Eyes Camera Preview'
    open_window(win_name_eyes)

    win_name_light = 'Light Camera Preview'
    open_window(win_name_light)

    win_name_protection = 'Protection Preview'
    open_window(win_name_protection)

    stop_read_event = threading.Event()

    threading.Thread(
        target=read_camera_frames_and_save_in_queues,
        args=(stop_read_event,),
        daemon=True
    ).start()

    read_queues_and_show_it(win_name_eyes, win_name_light, win_name_protection)
