import os
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

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
    

    
def display_video(words,video):    
    list_of_viewed_words = []
    
    for i,word in enumerate(words):
        if(len(video[i])>0):
            list_of_viewed_words.append(word)
        for frame in video[i]:
            
            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            write_text(frame," ".join(list_of_viewed_words[-5:]),(10,30))
            cv2.imshow("current frame",frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    


words = input("Enter word : ").split(" ")
video = create_video(words)
display_video(words,video)


cv2.destroyAllWindows()