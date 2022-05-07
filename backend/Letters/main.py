import shutil
import base64
import string
import cv2
import numpy as np
from fastapi import FastAPI, File,UploadFile
from rsa import sign_hash
from keras_model import  SignPredictor
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List



def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sign_predictor = SignPredictor()


class Data(BaseModel):
    data: List
    hand_label: str


@app.post("/test")
async def test(data:Data):
    hand_label = data.hand_label
    frame_list_data = (data.data)
    lis = []
    for frame in frame_list_data:
        frame = readb64(frame)
        if hand_label == 'Left':
            frame = cv2.flip(frame, 1)
        lis.append(frame)
    
    
    arg,letter = sign_predictor.predict(lis)
    
    return {"text":letter}





# uvicorn main:app --port 8002 --reload

