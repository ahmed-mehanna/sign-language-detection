# lib
import os
import mediapipe as mp
import cv2
import numpy as np

# utils
from utils import actions,arg_max
from utils import dispay_probability,display_counters,display_sentence,draw_styled_landmarks

# models & realTime
from pytorch_model import PytorchPredictor
from keras_model import KerasPredictor
from MultiSignDetector import MultiSignPredictor


holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

n_classes = len(actions)

WEIGHTS_PATH=os.path.join("..","..","sign_language_detection","ensemble","V1")
KERAS_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"keras_weights","V1.h5")
TORCH_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"pytorch_weights.tar")




pytorch_predictor = PytorchPredictor(path=TORCH_WEIGHTS_PATH)
keras_predictor = KerasPredictor(path=KERAS_WEIGHTS_PATH)


multi_sign_detector = MultiSignPredictor(holistic, (512,512))


class SignPredictor:
    
    def process(cap):
        sentence = []
        predictions = []

        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            if(not ret):
                break
            
            frame,body_keypoints = multi_sign_detector.detect_frame(frame)
            

            if multi_sign_detector.have_sign(frame):
                frame_list = multi_sign_detector.get_frames_indices(frames_no=16)

                for frame_idx in frame_list:
                    pytorch_predictor.add_frame(multi_sign_detector[frame_idx])
                    keras_predictor.add_frame(multi_sign_detector[frame_idx])

                res1 = pytorch_predictor.predict()
                res2 = keras_predictor.predict()
                res = res1 + res2
                arg_max = np.argmax(res)
                predictions.append(arg_max)
                predictions = predictions[-16:]
                print(predictions)
                multi_sign_detector.truncate_listed_frames()
                sentence.append(actions[arg_max])
                sentence = sentence[-4:]


            image = frame.copy()
            draw_styled_landmarks(image, body_keypoints)
            display_sentence(image,sentence)
            dispay_probability(image,multi_sign_detector.get_data())
            display_counters(image,multi_sign_detector.counter,multi_sign_detector.discarded_frames)


            cv2.imshow("Real-Time", image)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return sentence
        


