# the purpose of this file is to collect one video for each sign and rename them and save them so we can use them for mapping in sign to text

import shutil
import numpy as np
import os
import mediapipe as mp
import cv2 
import time
import pandas as pd


data_path = "../data"
video_path = os.path.join(data_path,"train")

one_video_path = os.path.join(data_path,"one_video")

train_labels = pd.read_csv("../data/train_labels.csv",names=['sample','id'])
class_id = pd.read_csv("../data/class_id.csv")

n_signs = 226
actions = list(class_id['EN'].iloc[:n_signs])




def check_file(file_path):
    try:
        f = open(file_path)
        f.close()
        return True
    except IOError:
        return False
    
    

def construct_path(file):
    return os.path.join(video_path,file+"_color.mp4")
    
    
def get_data(id):
    data =  train_labels[train_labels['id']==id]
    lis =  [construct_path(i) for i in  (data['sample'])]
    data =  [i for i in lis if check_file(i)]
    return data,[id for i in data]
    
    
    
pathes = []
for id in range(n_signs):
    path = get_data(id)[0][0]
    pathes.append(path)

try :
    os.mkdir(one_video_path)
    print("created successfully")
except :
    print("directory exists")

for i,path in enumerate(pathes):
    shutil.copy(path,os.path.join(one_video_path,f"sign_{i}_{actions[i]}.mp4"))
print("files copied successfully ")

