import torch
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import requests
from io import BytesIO

class predict_guitar():

    def __init__(self):
        """Model is loaded on init of the class"""
        # get model
        self.model = models.vgg16_bn(pretrained=True)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-2])
        self.model.classifier[0] = nn.Dropout(0.7, inplace=False)
        self.model.classifier[1] = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.model.classifier[2] = nn.ReLU(inplace=True)
        self.model.classifier[3] = nn.Dropout(0.7, inplace=False)
        #Change the final parameter to 7 categories
        self.model.classifier[4] = nn.Linear(in_features=4096, out_features=7, bias=True)

        # load parameters
        self.model.load_state_dict(torch.load('model_vgg16.pt'))
        
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

        transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

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
