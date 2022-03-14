import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

num_hand_marks = 21
num_pose_marks = 33

pose_selected_landmarks = [
    [0,2,5,11,13,15,12,14,16],
    [0,2,4,5,8,9,12,13,16,17,20],
    [0,2,4,5,8,9,12,13,16,17,20],
]



def mediapipe_detection(image,model):
    image  = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image  = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results
    


def extract_keypoints(results,left,right):

    # extract left hand
    if results.left_hand_landmarks:
        left_hand = np.array([ [res.x,res.y] for res in results.left_hand_landmarks.landmark ]).flatten()
    else:
        if type(left) == np.ndarray:
            left_hand = left
        else:
            left_hand = np.zeros(num_hand_marks*2)


    # extract right hand
    if results.right_hand_landmarks:
        right_hand = np.array([ [res.x,res.y] for res in results.right_hand_landmarks.landmark ]).flatten()
    else:
        if type(right) == np.ndarray:
            right_hand = right
        else:
            right_hand = np.zeros(num_hand_marks*2)


    return left_hand, right_hand


class MultiSignPredictor:
    def __init__(self, holistic, fsize=(512, 512)):
        self.fsize = fsize
        self.listed_frames = []
        self.holistic = holistic
        
        # current values
        self.frame_left_hand = None
        self.frame_right_hand = None
        
        # previous values
        self.last_frame_left_hand = None
        self.last_frame_right_hand = None
        
        self.counter = 0
        self.discarded_frames = 0
        
    def detect_frame(self,frame):
        
        frame = cv2.resize(frame, self.fsize)
        image, results = mediapipe_detection(frame, self.holistic)

        frame_left_hand, frame_right_hand = extract_keypoints(results,left=self.last_frame_left_hand,right=self.last_frame_right_hand)
        
        
        self.frame_left_hand = frame_left_hand.sum().round(2)
        self.frame_right_hand = frame_right_hand.sum().round(2)
        return frame,results
        
    
    def update_last_frame(self):
        self.last_frame_left_hand = self.frame_left_hand
        self.last_frame_right_hand = self.frame_right_hand
    
    def save_frame(self, frame):
        self.listed_frames.append(frame)
    

    def have_sign(self,frame):
        valid = self.valid_frame()
        if valid:
            self.counter += 1
            self.update_last_frame()
            self.discarded_frames = 0
            self.save_frame(frame)
        else:
            self.discarded_frames += 1
            if self.discarded_frames == 10:
                self.counter = 0
                self.discarded_frames = 0
                if len(self.listed_frames) >= 16:
                    return True
                else:
                    self.truncate_listed_frames()
        return False
            
    
    def valid_frame(self,right_hand_diff_threshold=0.5, left_hand_diff_threshold=0.5):
        try:
            right_hand_diff = np.abs(self.last_frame_right_hand - self.frame_right_hand).round(2)
            left_hand_diff = np.abs(self.last_frame_left_hand - self.frame_left_hand).round(2)

            if right_hand_diff >= right_hand_diff_threshold or left_hand_diff >= left_hand_diff_threshold:
                return True
            return False
        except:
            return True




    
    def get_frames_indices(self, frames_no):
        self.listed_frames = self.listed_frames[1:]

        return np.linspace(0, len(self.listed_frames)-1, frames_no, dtype=np.int16)
        
    
    def get_frames(self,frames_no):
        self.listed_frames = self.listed_frames[1:]

        final_idx =  np.linspace(0, len(self.listed_frames)-1, frames_no, dtype=np.int16)
        return [self.listed_frames[i] for i in final_idx]
    
    def truncate_listed_frames(self):
        self.listed_frames = []

    def __getitem__(self, idx):
        return self.listed_frames[idx]

    def __len__(self):
        return len(self.listed_frames)
    
    def get_data(self):
        return dict({
            "L": self.frame_left_hand,
            "R": self.frame_right_hand,
            "L2": self.last_frame_left_hand,
            "R2": self.last_frame_right_hand,
            "L-D": np.abs(self.last_frame_left_hand - self.frame_left_hand).round(2),
            "R-D": np.abs(self.last_frame_right_hand - self.frame_right_hand).round(2)
        })
