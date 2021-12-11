import os
import numpy as np
import cv2
import mediapipe as mp


# both these directories are created from the data collection files
data_path = os.path.join("../../data/one_video")
letters_path = os.path.join("../../data/letters")

def get_list_of_actions(data_path):
    lis = os.listdir(data_path)
    output = {}
    for i in lis:
        key = i.split('.')[0].split("_")[-1]
        output[key] = os.path.join(data_path,i)
    return output

def get_list_of_letters(data_path):
    lis = os.listdir(data_path)
    output = {}
    for i in lis:
        key = i.split('.')[0]
        output[key] = os.path.join(data_path,i)
    return output


    
actions = get_list_of_actions(data_path)
letters = get_list_of_letters(letters_path)


def write_text(image,text,position,size='l'):
    if size=='l':
        cv2.putText(image, text, position, 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
    elif size=='s':
        cv2.putText(image, text, position, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)




mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils



# model here is holistic surrounding the code
def mediapipe_detection(image,model):
    image  = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image  = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results
    

def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
def draw_styled_landmarks(image,results):
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
    
    
def draw_styled_landmarks(image,results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2)
                             ) 
    
    
    
    
    
def extract_keypoints(results):
    # extract pose marks
    if results.pose_landmarks:
        pose = np.array([ [res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark ]).flatten()
    else:
        pose = np.zeros(num_pose_marks*4)
    
    # extract left hand
    if results.left_hand_landmarks:
        left_hand = np.array([ [res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark ]).flatten()
    else:
        left_hand = np.zeros(num_hand_marks*3)
        
        
    # extract right hand
    if results.right_hand_landmarks:
        right_hand = np.array([ [res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark ]).flatten()
    else:
        right_hand = np.zeros(num_hand_marks*3)
    
    return np.concatenate([pose,left_hand,right_hand])
    


def get_letters(word):
    output = []
    for letter in word:
        video = cv2.VideoCapture(letters[letter.upper()])
        video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        count=0
        while video.isOpened():
            ret,frame = video.read()
            if not ret:
                continue
            output.append(frame)
            count+=1
            if(count>=video_length):
                video.release()
        video.release()
    return output


# text = "orange tea"
def create_video(words):
    list_of_frames = []
    for word in words:
        if word in actions:
            vid = []
            video = cv2.VideoCapture(actions[word])
            video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

            count=0
            while video.isOpened():
                ret,frame = video.read()
                if not ret:
                    continue
                vid.append(frame)
                count+=1
                if(count>=video_length):
                    video.release()
            video.release()
            list_of_frames.append(vid)
        else:
            list_of_frames.append(get_letters(word))
    
    return list_of_frames
    



def display_keypoints_video(words,video):
    list_of_viewed_words = []
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
    for i,word in enumerate(words):
        if(len(video[i])>0):
            list_of_viewed_words.append(word)
        for frame in video[i]:
            frame = cv2.resize(frame,(1280,960))
            frame, results = mediapipe_detection(frame, holistic)
            # new_frame = np.zeros((960,1280,3)) + 255
            draw_styled_landmarks(frame, results)
            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            write_text(frame," ".join(list_of_viewed_words[-5:]),(10,30))
            cv2.imshow("current frame",frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    

words = input("Enter text : ").split(" ")
video = create_video(words)
display_keypoints_video(words,video)


cv2.destroyAllWindows()