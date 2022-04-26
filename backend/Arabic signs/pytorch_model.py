import pandas as pd
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from utils import actions
from utils import softmax


n_classes = len(actions)



def convert_relu_to_swish(model: nn.Module):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.SiLU(True))
        else:
            convert_relu_to_swish(child)

class Swish(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return x.mult_(torch.sigmoid(x))
    
    
    
class r2plus1d_18(nn.Module):
    def __init__(self, pretrained=True, n_classes=3, dropout_p=0.5):
        super(r2plus1d_18, self).__init__()
        self.pretrained = pretrained
        self.n_classes = n_classes

        model = torchvision.models.video.r2plus1d_18(pretrained=self.pretrained)
        modules = list(model.children())[:-1]
        self.r2plus1d_18 = nn.Sequential(*modules)
        convert_relu_to_swish(self.r2plus1d_18)
        self.fc1 = nn.Linear(model.fc.in_features, self.n_classes)
        self.dropout = nn.Dropout(dropout_p, inplace=True)

    def forward(self, x):
        # (b, f, c, h, w) = x.size()
        # x = x.view(b, c, f, h, w)

        out = self.r2plus1d_18(x)
        out = out.flatten(1)
        out = self.dropout(out)
        out = self.fc1(out)

        return out
    

class PytorchPredictor:
    def __init__(self,path):
        self.create_torch_model(path)
        self.sequence = []
        h, w = 128, 128
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
        
        # transformeers
        self.resize_transform   = transforms.Resize((h, w))
        self.totensor_transform  = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean, std)
        
        
    def create_torch_model(self,path):
        

        pytorch_model = r2plus1d_18(pretrained=False, n_classes=n_classes)
        best_checkpoint = torch.load(path)
        pytorch_model.load_state_dict(best_checkpoint["model_state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pytorch_model = pytorch_model.to(device)
        
        self.model = pytorch_model
        self.device = device
        
    
    def can_predict(self):
        return len(self.sequence) == 16
    
    def add_frame(self,frame):
        
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_frame = Image.fromarray(new_frame)
        new_frame = self.resize_transform(new_frame)
        new_frame = self.totensor_transform(new_frame)
        new_frame = self.normalize_transform(new_frame).to(self.device)
        
        self.sequence.append(new_frame)
        self.sequence = self.sequence[-16:]
        
        
    def predict(self):
        seq = torch.stack(self.sequence).to(self.device)
        seq = torch.unsqueeze(seq, dim=0).permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
            self.model.eval()
            res = self.model(seq)
            res = res.cpu().detach().numpy()[0]
            return softmax(res)
        