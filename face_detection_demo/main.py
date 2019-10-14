import cv2

from face_detection_demo import root_folder
from face_detection_demo.video_capture import run_video_capture

face_detector_model_path = root_folder / 'model'
cap = cv2.VideoCapture(0)
run_video_capture(cap, face_detector_model_path, 'model.pb', 'new_eyes_only_weights.hdf5')
