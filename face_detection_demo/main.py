import cv2

from face_detection_demo import root_folder
from face_detection_demo.video_capture import run_video_capture


def main(model_path=None):
    face_detector_model_path = model_path or root_folder / 'model' / 'model.pb'
    cap = cv2.VideoCapture(0)
    run_video_capture(cap, face_detector_model_path)
