import numpy as np
import os
import mediapipe as mp
import cv2 
import time



# Path for exported data, numpy arrays
DATA_PATH = os.path.join('data_letters') 

# Actions that we try to detect
actions = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# fifty videos worth of data
n_videos = 1

# Videos are going to be 30 frames in length
video_length = 2

frames_per_seconds=25

video_frames = video_length*frames_per_seconds

frame_time = 1000//frames_per_seconds



try: 
    os.mkdir(DATA_PATH)
except:
    pass


def write_text(image,text,position,size='s'):
    if size=='l':
        cv2.putText(image, text, position, 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
    elif size=='s':
        cv2.putText(image, text, position, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        
        
        

cap = cv2.VideoCapture(0)
continue_training=True
for action in actions:
    if not continue_training:
        break
    
    ret, frame = cap.read()
    write_text(frame,"STARTING in 2 seconds",(120,200),'l')
    write_text(frame,'Collecting frames for {}'.format(action),(15,12))
    cv2.imshow('OpenCV Feed', frame)
    cv2.waitKey(2000)
        


    start_time=time.time()
    #fourcc = VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(f'{DATA_PATH}/{action}.mp4',0x7634706d, frames_per_seconds, (640,480))

    for frame_num in range(-1,video_frames):

        # Read feed
        ret, frame = cap.read()

        # NEW Apply wait logic
        if frame_num == -1: 
            write_text(frame,"STARTING COLLECTING",(120,200),'l')
            write_text(frame,'Collecting frames for {}'.format(action),(15,12))
            write_text(frame,'time {}'.format(round(time.time()-start_time),2),(15,40))
            cv2.imshow('OpenCV Feed', frame)
            cv2.waitKey(500)

        else: 
            out.write(frame)
            write_text(frame,'Collecting frames for {}'.format(action),(15,12))
            write_text(frame,'time {}'.format(round(time.time()-start_time,2)),(15,40))
            write_text(frame,'frame {}'.format(frame_num),(15,64))
            cv2.imshow('OpenCV Feed', frame)



        # Break gracefully
        if cv2.waitKey(frame_time) & 0xFF == ord('q'):
            continue_training=False
            break

    out.release()
        
        
        
        
        
                

cap.release()
cv2.destroyAllWindows()



