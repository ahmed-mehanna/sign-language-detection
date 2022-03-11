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

n_classes = len(actions)
WEIGHTS_PATH=os.path.join("..","..","sign_language_detection","ensemble","V1")
KERAS_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"keras_weights","V1.h5")
TORCH_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"pytorch_weights.tar")






class SignPredictor:

    def __init__(self):
        holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.pytorch_predictor = PytorchPredictor(path=TORCH_WEIGHTS_PATH)
        self.keras_predictor = KerasPredictor(path=KERAS_WEIGHTS_PATH)
        self.multi_sign_detector = MultiSignPredictor(holistic, (512,512))


    def predict(self,frame_list):

        for frame in frame_list:
            self.pytorch_predictor.add_frame(frame)
            self.keras_predictor.add_frame(frame)

        res1 = self.pytorch_predictor.predict()
        res2 = self.keras_predictor.predict()
        res =   res1 + res2
        arg_max = np.argmax(res)
        action = actions[arg_max]

        return arg_max,action

    
    def process_sign(self,cap):
        lis = []
        while cap.isOpened():
            ret, frame = cap.read()
            if(not ret):
                break

            lis.append(frame)
            # cv2.imshow("Real-Time", frame)
            
            # if cv2.waitKey(50) & 0xFF == ord('q'):
            #     break

        cap.release()
        cv2.destroyAllWindows()

    
        final_idx =  np.linspace(0, len(lis)-1, 16, dtype=np.int16)
        final_lis =  [lis[i] for i in final_idx]

        sign_id,sign_name = self.predict(final_lis)
        
        return sign_name






    
    def process_multisign(self,cap):
        sentence = []
        predictions = []

        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            if(not ret):
                break
            
            frame,body_keypoints = self.multi_sign_detector.detect_frame(frame)
            

            if self.multi_sign_detector.have_sign(frame):
                frames = self.multi_sign_detector.get_frames(frames_no=16)
                sign_num,action = self.predict(frames)
                
                predictions.append(sign_num)
                predictions = predictions[-16:]
                print(predictions)
                self.multi_sign_detector.truncate_listed_frames()
                sentence.append(action)
                


            # image = frame.copy()
            # draw_styled_landmarks(image, body_keypoints)
            # display_sentence(image,sentence[-4:])
            # dispay_probability(image,self.multi_sign_detector.get_data())
            # display_counters(image,self.multi_sign_detector.counter,self.multi_sign_detector.discarded_frames)


            # cv2.imshow("Real-Time", image)

            # if cv2.waitKey(50) & 0xFF == ord('q'):
            #     break

        cap.release()
        cv2.destroyAllWindows()

        return sentence
        


