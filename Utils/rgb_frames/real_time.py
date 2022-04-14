import pandas as pd
import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch


class RealTime():
    def __init__ (self, model, signs, device, transform):
        self.model = model
        self.actions = signs
        self.device = device
        self.transform = transform
        self.cap = None

        self.model = self.model.to(self.device)
    
    def __prob_viz(self, res, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            prob = max(0, prob)
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num%3], -1)
            cv2.putText(output_frame, str(self.actions[num]), (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        return output_frame
    
    def start(self):
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5

        self.cap = cv2.VideoCapture(0)
        self.model.eval()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            final_frame = Image.fromarray(final_frame)
            final_frame = self.transform(final_frame).to(self.device)

            sequence.append(final_frame)
            sequence[-16:]
            arg_max = -1
            if len(sequence) == 16:
                first_seq = torch.stack(sequence).to(self.device)
                output_seq = torch.unsqueeze(first_seq, dim=0).permute(0, 2, 1, 3, 4)
                with torch.no_grad():
                    res = self.model(output_seq)
                    arg_max = int(torch.argmax(res))
                predictions.append(arg_max)
            

                # Viz logic
                if np.unique(predictions[-2:])[0] == arg_max:
                    if res[0][arg_max] > threshold:
                        if len(sequence) > 0:
                            if self.actions[arg_max] != sentence[-1]:
                                sentence.append(self.actions[arg_max])
                        else:
                            sentence.append(self.actions[arg_max])
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                # Viz probabilities
                frame = prob_viz(res.cpu().detach().numpy()[0], frame, colors)
            
            cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("OpenCV Feed", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

        

