import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch.nn.functional as F

import cv2
from PIL import Image

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Number of channels in the training images. For color images this is 3
nc = 3

b_size = 1


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class GAN_CELEBA():
    def __init__(self):
        self.model = Generator()
        self.model.load_state_dict(torch.load('./models/CELEBRA_1.pt'))    
        self.model.eval()
    
        if torch.cuda.is_available():
            self.Tensor = torch.cuda.FloatTensor
            self.model.cuda()
        else:
            self.Tensor = torch.FloatTensor
            
            
    def predictIMG(self, array):
        # Now, we will use some autogenerated latent spaces to produce some new image
        #img2 = self.model(self.Tensor([[0.3, 1, 1, 1, -1, 1, 5, 1, 1, -1]])) #[[1,2,-0.2,1,-3,0.3,0.2,0.1,0.2,0.5]])) 
        img2 = self.model(self.Tensor(array))
        save_image(img2.data[0], './output/generated_CELEBRA.png' , nrow= 1, normalize= True)
        self.resizeIMG()
    
    def resizeIMG(self, file= './output/generated_CELEBRA.png'):
        """
        Resize the given image
        """
        im = Image.open(file)
        im = im.resize((int(im.size[0]*5), int(im.size[1]*5)), Image.NEAREST)
        im.save('./output/CELEBRA_new.png')



import PySimpleGUI as sg
import cv2

# GUI
sg.theme('DarkAmber')   
layout = [  [sg.Text('Si, eso es GUI de Python, no me pegueis!')],
            [sg.Image(filename= './output/new.png', key="-IMAGE-", visible=True)],
            [sg.Button('Ok'), sg.Button('Cancel')],
            #[sg.Slider(range=(-2.0,2.0), resolution=.1, default_value=0.0, size=(20,15), enable_events=True, orientation='h',key="-S1-", font=('Helvetica', 12))]]
            [sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S1-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S2-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S3-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S4-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S5-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S6-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S7-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S8-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S9-", font=('Helvetica', 12))]
        
        ]

# Create the Window
window = sg.Window('Python GUI = ?', layout)

# Model init
model = GAN_CELEBA()
#array = [1 for i in range(100)]#[1,1,1,1,1,1,1,1,1]
array = torch.randn(b_size, nz, 1, 1).cuda()

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    
    # Scrollers listeners
    if event == "-S1-":
        v = values["-S1-"]
        array[0][0][0][0] = v
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/CELEBRA_new.png')
        
    elif values['-S2-']:
        v = values["-S2-"]
        array[0][1][0][0] = v
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/CELEBRA_new.png')
    
    elif values['-S3-']:
        v = values["-S3-"]
        array[0][2][0][0] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/CELEBRA_new.png')
    
    elif values['-S4-']:
        v = values["-S4-"]
        array[0][3][0][0] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/CELEBRA_new.png')
    
    elif values['-S5-']:
        v = values["-S5-"]
        array[0][4][0][0] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/CELEBRA_new.png')
    
    elif values['-S6-']:
        v = values["-S6-"]
        array[0][5][0][0] = v
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/CELEBRA_new.png')
    
    elif values['-S7-']:
        v = values["-S7-"]
        array[0][6][0][0] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/CELEBRA_new.png')

    elif values['-S8-']:
        v = values["-S8-"]
        array[0][7][0][0] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/CELEBRA_new.png')
    
    elif values['-S9-']:
        v = values["-S9-"]
        array[0][8][0][0] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/CELEBRA_new.png')
    
window.close()