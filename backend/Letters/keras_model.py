import os
import numpy as np
import cv2 
import mediapipe as mp
import tensorflow as tf 
from tensorflow.keras.layers import  Dense,Input
from tensorflow.keras.models import Model

actions = [
    "ا",
    "ب",
    "ت",
    "ث",
    "ج",
    "ح",
    "خ",
    "د",
    "ذ",
    "ر",
    "ز",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ع",
    "غ",
    "ف",
    "ق",
    "ك",
    "ل",
    "م",
    "ن",
    "ه",
    "و",
    "ي",
    "ة",
    "ال",
    "لا",
    "ى"

]






WEIGHTS_PATH=os.path.join("..","..","weights")
KERAS_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"letters.h5")


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils



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



def extract_keypoints(results):
    if results.right_hand_landmarks:
        return np.array([ [res.x,res.y] for res in results.right_hand_landmarks.landmark ]).flatten()
    else :
        return np.zeros(42,dtype=np.float32)



def mediapipe_detection(image,model):
    image  = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image  = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results
    

class KerasPredictor:
    def __init__(self,path):
        use_tensorflow_cpu()
        self.create_model(path)
        self.lis = []
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        
    def create_model(self,path):
        input_layer = Input(shape=(42,))
        layer = Dense(128,activation="relu")(input_layer)
        layer = Dense(256,activation="relu")(layer)
        layer = Dense(128,activation="relu")(layer)
        layer = Dense(len(actions),activation="softmax")(layer)

        model = Model(inputs=input_layer,outputs=layer)
        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.load_weights(os.path.join(path))
        self.model = model
        
        


    
    def add_frame(self,frame):
        
        # f2 = cv2.resize(frame,(640,480))
        image, results = mediapipe_detection(frame, self.holistic)
        keypoints = extract_keypoints(results)
        self.lis.append(keypoints)
        
        
        
    def predict(self):
        res =  self.model.predict(np.array(self.lis))
        self.lis = []
        return res
        
        
        


class SignPredictor:

    def __init__(self):
        self.keras_predictor = KerasPredictor(path=KERAS_WEIGHTS_PATH)


    def predict(self,lis):
        
        for frame in lis:
            self.keras_predictor.add_frame(frame)
        
        res = self.keras_predictor.predict().sum(axis=0)
            
        arg_max = np.argmax(res)
        action = actions[arg_max]

        return arg_max,action

