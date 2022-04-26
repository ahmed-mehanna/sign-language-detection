import numpy as np
import cv2
import mediapipe as mp



actions = ['one','you','teacher','girl','tomorrow','mom','look','crazy','walk','agree']

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def softmax(x):    
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def arg_max(array):
    arg_max = np.argmax(array)
    return arg_max,array[arg_max]





def draw_styled_landmarks(image, results):
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

def display_sentence(frame,sentence):
    cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(frame, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


def dispay_probability(frame,data):
    TEXT_COLOR = (0,0,255)
    
    cv2.putText(frame, "L:"+str(data["L"]), (0, 85+0*40), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_8)
    cv2.putText(frame, "R:"+str(data["R"]), (0, 85+1*40), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_8)

    cv2.putText(frame, "L2:"+str(data["L2"]), (0, 85+4*40), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_8)
    cv2.putText(frame, "R2:"+str(data["R2"]), (0, 85+5*40), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_8)


    cv2.putText(frame, "L-D:"+str(data["L-D"]), (0, 85+8*40), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_8)
    cv2.putText(frame, "R-D:"+str(data["R-D"]), (0, 85+9*40), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_8)


def display_counters(frame,counter,discarded_frames):
    cv2.putText(frame, str(counter), (250, 85+5*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,250,150), 2, cv2.LINE_8)
    cv2.putText(frame, str(discarded_frames), (250, 85+6*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,250,150), 2, cv2.LINE_8)

    
