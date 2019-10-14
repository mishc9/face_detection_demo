from pathlib import Path
from typing import Union

import cv2
import numpy as np

from face_detection_demo.model import FaceDetector


def draw_rectangle(frame: np.array, face_box: tuple):
    ymin, xmin, ymax, xmax = face_box
    top_left = (xmin, ymin)
    bottom_right = (xmax, ymax)
    cv2.rectangle(frame,
                  top_left,
                  bottom_right,
                  (0, 155, 255),
                  2
                  )
    return frame


def classify_eyes_state(model, frame):
    return model.predict(np.expand_dims(frame, axis=0) / 255.)[0]


def crop_image(frame, box):
    ymin, xmin, ymax, xmax = box
    crop = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
    return crop


def run_video_capture(cap, model_path: Union[Path, str]):
    face_detector = FaceDetector(str(model_path),
                                 gpu_memory_fraction=0.5,
                                 visible_device_list='0'
                                 )
    while True:
        try:
            ret, frame = cap.read()
            boxes, scores = face_detector(frame, score_threshold=0.3)

            for i, (box, score) in enumerate(zip(boxes, scores)):
                frame = draw_rectangle(frame, box)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            print("Releasing")
            break

    cap.release()
    cv2.destroyAllWindows()
