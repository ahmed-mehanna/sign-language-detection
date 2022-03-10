from SignPredictor import  SignPredictor
import cv2


cap = cv2.VideoCapture(0)

output = SignPredictor.process(cap)

print(output)

