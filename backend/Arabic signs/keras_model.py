import os
import numpy as np
import cv2 
import mediapipe as mp
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Input,Dropout
from tensorflow.keras.models import Model
import time
import pandas as pd
from utils import actions

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

num_hand_marks = 21
num_pose_marks = 33

pose_selected_landmarks = [
    [0,2,5,11,13,15,12,14,16],
    [0,2,4,5,8,9,12,13,16,17,20],
    [0,2,4,5,8,9,12,13,16,17,20],
]



def use_tensorflow_cpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')
    tf.test.gpu_device_name()

    tf.config.set_visible_devices([], 'GPU')
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def draw_updated_styled(image,results):
    image_rows, image_cols, _ = image.shape
    
    original_landmarks = [
        results.pose_landmarks,
        results.left_hand_landmarks,
        results.right_hand_landmarks
    ]

    for shape in range(3):
        if(original_landmarks[shape]):
            lis = original_landmarks[shape].landmark
            for idx in pose_selected_landmarks[shape]:
                point = lis[idx]
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(point.x, point.y,
                                                           image_cols, image_rows)

                cv2.circle(image, landmark_px, 2, (0,0,255),
                         4)
                
def extract_keypoints(results):
    
    original_landmarks = [
        results.pose_landmarks,
        results.left_hand_landmarks,
        results.right_hand_landmarks
    ]
    
    outputs = []
    for shape in range(3):
        if(original_landmarks[shape]):
            lis = original_landmarks[shape].landmark
            pose = np.array([ [lis[res].x,lis[res].y] for res in pose_selected_landmarks[shape] ]).flatten()
        else:
            pose = np.zeros(len(pose_selected_landmarks[shape])*2)
        outputs.append(pose)
    return np.concatenate([outputs[0],outputs[1],outputs[2]])



def mediapipe_detection(image,model):
    image  = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image  = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results
    
    


def draw_landmark_from_array(image, keyPoints):
    image_rows, image_cols, _ = image.shape
    
    
    for i in range(len(keyPoints)//2):
        x = keyPoints[i*2]
        y = keyPoints[i*2+1]
        if(x!=0 and y!=0): 
            landmark_px = mp_drawing._normalized_to_pixel_coordinates(x,y,
                                                       image_cols, image_rows)
            cv2.circle(image, landmark_px, 2, (0,0,255),
                     4)

                
class KerasPredictor:
    def __init__(self,path):
        use_tensorflow_cpu()
        self.create_model(path)
        self.sequence = []
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        
    def create_model(self,path):
        input_layer = Input(shape=(16,62))
        layer = LSTM(64,return_sequences=True,activation="relu")(input_layer)
        layer = LSTM(128,return_sequences=True,activation="relu")(layer)
        layer = LSTM(96,return_sequences=False,activation="relu")(layer)
        layer = Dense(64,activation="relu")(layer)
        layer = Dense(len(actions),activation="softmax")(layer)


        model = Model(inputs=input_layer,outputs=layer)
        model.compile(optimizer="Adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(os.path.join(path))
        self.model = model
        

    def can_predict(self):
        return len(self.sequence) == 16
    
    def add_frame(self,frame):
        
        f2 = cv2.resize(frame,(512,512))
        image, results = mediapipe_detection(f2, self.holistic)
        keypoints = extract_keypoints(results)
        
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-16:]
        
        
        
    def predict(self):
        return self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
        
        