import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from PIL import Image

inputSize = 784
outputSize = 10
numEpochs = 5
batchSize = 100
learningRate = 0.001
dataPath = './Documents/MNISTDATA'

#Define transformation and load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainSet = MNIST(root=dataPath, train=True, download=False, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True)

testSet = MNIST(root=dataPath, train=False, download=False, transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=False)