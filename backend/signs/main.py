import shutil
import cv2
import numpy as np
from fastapi import FastAPI, File,UploadFile
from SignPredictor import  SignPredictor


sign_predictor = SignPredictor()
app = FastAPI()



@app.post('/multi')
async def root(file:UploadFile = File(...)):
    with open(f'temp_video.mp4','wb') as buffer:
        shutil.copyfileobj(file.file,buffer)
    
    cap = cv2.VideoCapture("temp_video.mp4")
    output = sign_predictor.process_multisign(cap)

    
    return {"text":output}




@app.post('/single')
async def root(file:UploadFile = File(...)):
    with open(f'temp_video.mp4','wb') as buffer:
        shutil.copyfileobj(file.file,buffer)
    
    cap = cv2.VideoCapture("temp_video.mp4")
    output = sign_predictor.process_sign(cap)

    
    return {"text":output}











# from SignPredictor import  SignPredictor
# import cv2
# from fastapi import FastAPI

# app = FastAPI()

# sign_predictor = SignPredictor()

# print("every thing initalized")


# @app.get('/')
# async def root():
#     cap = cv2.VideoCapture(0)
#     output = sign_predictor.process_multisign(cap)
#     print(output)
#     return {"message":output}



