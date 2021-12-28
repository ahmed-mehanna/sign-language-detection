# holistic model process image and return the results as keypoints
try:
    import cv2
    import mediapipe as mp
    import numpy as np
except :
    print("error importing libaray")



def mediapipe_detection(image,model):
    image  = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image  = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results
    
def draw_styled_landmarks(image,results):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    

# read the keypoints and extract them and process them
def extract_keypoints(key_points,num_hand_marks=21,num_pose_marks=33):
    # extract pose marks
    if key_points.pose_landmarks:
        pose = np.array([ [res.x,res.y,res.z,res.visibility] for res in key_points.pose_landmarks.landmark ]).flatten()
    else:
        pose = np.zeros(num_pose_marks*4)
    
    # extract left hand
    if key_points.left_hand_landmarks:
        left_hand = np.array([ [res.x,res.y,res.z] for res in key_points.left_hand_landmarks.landmark ]).flatten()
    else:
        left_hand = np.zeros(num_hand_marks*3)
        
        
    # extract right hand
    if key_points.right_hand_landmarks:
        right_hand = np.array([ [res.x,res.y,res.z] for res in key_points.right_hand_landmarks.landmark ]).flatten()
    else:
        right_hand = np.zeros(num_hand_marks*3)
    
    return np.concatenate([pose,left_hand,right_hand])
    