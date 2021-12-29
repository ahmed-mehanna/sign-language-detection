import os
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

def get_frames(video_path, n_frames=1):
    frames = []
    v_cap = cv2.VideoCapture(video_path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
    frame_list = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)

    for fn in range(v_len):
        success, frame = v_cap.read()
        if success == False:
            continue
        if fn in frame_list:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
    v_cap.release()
    while len(frames) < n_frames:
        frames.insert(0, frames[0])
    return frames

def store_frames(frames, path_to_store):
    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path = os.path.join(path_to_store, "frame"+str(i)+".jpg")
        cv2.imwrite(path, frame)

def denormalize(x_, mean, std):
    x = x_.clone()
    for i in range(3):
        x[i] = x[i]*std[i]+mean[i]
    x = to_pil_image(x)        
    return x