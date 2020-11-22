import torch
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import requests
from io import BytesIO
from net import Net

class predict_guitar():

    def __init__(self):
        """Model is loaded on init of the class"""
       
        self.model = Net()

        # load parameters
        self.model.load_state_dict(torch.load('model.pt'))
        
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()
    
        self.model.eval()

    def softmax(self, vector):
        """Softmax function for calculating probs"""
        e = np.exp(vector)
        return e / e.sum()

    def predict(self,url):
        """Generating prediction of image url"""

        # get image
        response = requests.get(url)
        
        img = Image.open(BytesIO(response.content))

        transform = transforms.Compose([transforms.Grayscale(),
                                         transforms.Resize((128,128)),
                                         transforms.ToTensor()])

        img = transform(img).unsqueeze(0)

        if torch.cuda.is_available(): 
            img = img.cuda() 

        out = self.model(img)

        classes = ['Jazzmaster','Les Paul', 'Mustang', 'PRS SE', 'SG',
                            'Stratocaster','Telecaster']

        if torch.cuda.is_available():

            logs = out.cpu().data.numpy()
        
        else:

            logs = out.data.numpy()
        
        return [classes[logs.argmax()]]
