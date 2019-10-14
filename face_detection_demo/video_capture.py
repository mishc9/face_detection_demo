from pathlib import Path
from typing import Union

import cv2
import numpy as np
from tensorflow.python.keras.models import load_model

from face_detection_demo.apply_model import FaceDetector
from face_detection_demo.build import build_eye_state_and_angle_model
from face_detection_demo.face_box import FaceBox


def draw_rectangle(frame: np.array, face_box: FaceBox):
    cv2.rectangle(frame,
                  face_box.top_left,
                  face_box.bottom_right,
                  (0, 155, 255),
                  2
                  )
    return frame


def put_text(img, fstring):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (10, 400)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, fstring,
                bottom_left_corner_of_text,
                font,
                fontScale,
                fontColor,
                lineType)
    return img


def classify_eyes_state(model, frame):
    return model.predict(np.expand_dims(frame, axis=0) / 255.)[0]


def crop_image(frame, box):
    ymin, xmin, ymax, xmax = box
    crop = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
    return crop


def run_video_capture(cap, models_path: Union[Path, str], fd_path, eye_path):
    face_detector = FaceDetector(str(models_path / fd_path),
                                 gpu_memory_fraction=0.5,
                                 visible_device_list='0'
                                 )
    eyes_model = build_eye_state_and_angle_model()
    eyes_model.load_weights(str(models_path / eye_path))
    while True:
        try:
            ret, frame = cap.read()
            boxes, scores = face_detector(frame, score_threshold=0.3)

            predictions = list()
            for i, (box, score) in enumerate(zip(boxes, scores)):
                face_box = FaceBox.from_box(box)
                crop = crop_image(frame, box)
                resized = cv2.resize(crop, (224, 224))
                cv2.imshow(f'crop_{i}', resized)
                prediction_i = classify_eyes_state(eyes_model, resized)
                predictions.append(prediction_i)
                # im_net_box = face_box.to_quadratic_box(frame.shape)
                frame = draw_rectangle(frame, face_box)

            # font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 500)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2
            print(f"Eyes states: {predictions}")
            # cv2.addText(frame, str(predictions),
            #             bottomLeftCornerOfText,
            #             font,
            #             fontScale,
            #             fontColor,
            #             lineType
            #             )
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            print("Releasing")
            break

    cap.release()
    cv2.destroyAllWindows()
