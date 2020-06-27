import argparse
import math
import os
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader 
from torchvision import datasets
from torch.autograd import Variable

import torchvision.transforms as transforms 
from torchvision.utils import save_image 

os.makedirs('output', exist_ok= True)

# Black and White image size
img_shape = (1, 28, 28) 

# The number of Epochs for training
EPOCHS = 30


class Generator(nn.Module):
    """Generate fakes iages by using random noise of input to 100 nodes layer,
    in ascending layers order, generates an image

    Args:
        nn (nn.Module): Pytorch nn Module
    """
    def __init__(self, inputDim = 10):
        super(Generator, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(inputDim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 28*28)

        # Image noralization layers (before ReLU)
        self.in1 = nn.BatchNorm1d(128)
        self.in2 = nn.BatchNorm1d(512)
        self.in3 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        x = F.leaky_relu(self.in1(self.fc1(x)), 0.2)
        x = F.leaky_relu(self.in2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.in3(self.fc3(x)), 0.2)
        x = F.leaky_relu(self.in3(self.fc4(x)), 0.2)
        x = F.tanh(self.fc5(x))
        
        return x.view(x.shape[0], *img_shape)
    

class Discriminator(nn.Module):
    """Discriminates the images, if they are fake or real, descending order of layers

    Args:
        nn (nn.Module): Pytorch nn Module
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Fully Connected layers
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        
        x = F.sigmoid(self.fc4(x))
        return x
    
    
class GAN():
    """Creates and train a GAN Model wit MNIST dataset
    """
    def __init__(self, inputDim= 9, epochs= EPOCHS):
        self.epochs = epochs
        # Creating our Loss Function
        self.loss_funct = torch.nn.BCELoss()
        
        # Initializing The two models
        self.generator = Generator(inputDim = inputDim)
        self.discriminator = Discriminator()
        
        # Dataet based on MNIST Black and White colors, so, 1 channel only
        self.dataset = torch.utils.data.DataLoader(
                    datasets.MNIST('data/', train= True, download= True, transform= transforms.Compose([
                        
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, ), (0.5, )) # For RGB: (0.5, 0.5, 0.5)
                        
                    ])), batch_size= 64, shuffle= True
                )
        # Defining our Adam optimazers
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr= 0.0002, betas= (0.4, 0.999))
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr= 0.0002, betas= (0.4, 0.999))

    def trainModel(self, saveModel= False):
        """Trains the model on MNIST dataset
        """
        # Of Course wi will be using CUDA because we should love our CPU
        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.loss_funct.cuda()
            
            Tensor = torch.cuda.FloatTensor
        else:
            Tensor = torch.FloatTensor
                
        # Starting with our training
        for epoch in range(self.epochs):
            # Take a look that we don't use labels because of this is not a classifier
            for i, (imgs, _) in enumerate(self.dataset):
                # Ground Truth
                val = Tensor(imgs.size(0), 1).fill_(1.0)
                fake = Tensor(imgs.size(0), 1).fill_(0.0)

                # We pass the images to cuda
                real_imgs = imgs.cuda()
                
                # GENERATOR
                # =================================================================================================================
                # Resetting the derivatives to good updates
                self.generator_optimizer.zero_grad()
                
                # Creating the noise for our Generator with normal distribution
                generator_noice = Tensor(np.random.normal(0, 1, (imgs.shape[0], 9)))
                
                # Applying generator
                generated_imgs = self.generator(generator_noice)
                
                # Looking for the loss, or its the same, how well we predict, using for this purpose our dsicriminator 
                # Ability of the generator to fool discriminator
                generator_loss = self.loss_funct(self.discriminator(generated_imgs), val)
                
                # Backpropagation
                generator_loss.backward()
                self.generator_optimizer.step()
                
                
                # DISCRIMINATOR
                # =================================================================================================================
                # Resetting the derivatives to good updates
                self.discriminator_optimizer.zero_grad()
                
                # Loss funtions of the images
                real_loss = self.loss_funct(self.discriminator(real_imgs), val)
                fake_loss = self.loss_funct(self.discriminator(generated_imgs.detach()), fake)
                discriminator_loss = (real_loss + fake_loss) / 2
                
                # Backpropagation
                discriminator_loss.backward()
                self.discriminator_optimizer.step()
                
                print("[Epoch: %d/%d | Batch: %d/%d | Disc. Loss: %f | Gen. Loss: %f]" % (epoch, self.epochs, i, len(self.dataset), 
                                                                                        discriminator_loss.item(), 
                                                                                        generator_loss.item()))

                total_batch = epoch * len(self.dataset) + i
                if total_batch % 400 == 0:
                    save_image(generated_imgs.data[:25], 'output/generated_%d.png' % total_batch, nrow= 5, normalize= True)
        
        if saveModel:
            torch.save(self.generator.state_dict(), './model.pt')

def loadGenerator(path= './model.pt'):
    model = Generator()
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    print("Model Ready...")

a = GAN()
a.trainModel(saveModel= True)