import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from PIL import Image

# Define the neural network architecture
class ANN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 512)
        self.l2 = nn.Linear(512, 64)
        self.l3 = nn.Linear(64, 16)
        self.l4 = nn.Linear(16, output_size)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(16)

    def forward(self, x):
        out = x.reshape(-1, 784)
        out = F.relu(self.bn1(self.l1(out)))
        out = F.relu(self.bn2(self.l2(out)))
        out = F.relu(self.bn3(self.l3(out)))
        out = self.l4(out)
        return out


#train model
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


#test model 
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

#classify image from path 
def whatsThatNumber(model, imagePath):
    image = Image.open(imagePath).convert('L')  # Convert to grayscale
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()


#define constant parameters 

inputSize = 784
outputSize = 10
numEpochs = 5
batchSize = 100
learningRate = 0.001
dataPath = './MNISTDATA'

#Define transformation and load the MNIST dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = MNIST(root=dataPath, train=True, download=False, transform=transform)
trainSet, valSet = random_split(dataset, [50000, 10000])
trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=4)
valLoader = DataLoader(valSet, batch_size=batchSize, shuffle=False, num_workers=4)
    
#Define loss function and optimizer
model = ANN(inputSize, outputSize)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Training Loop
for epoch in range(numEpochs):
    train(model, trainLoader, optimizer, criterion, device)
    valLoss, valAccuracy = test(model, valLoader, criterion, device)
    print(f"Epoch {epoch+1}/{numEpochs}, Val Loss: {valLoss:.4f}, Val Acc: {valAccuracy:.2f}%")
    scheduler.step()

# Test
testSet = MNIST(root=dataPath, train=False, download=False, transform=transform)
testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False, num_workers=4)
testLoss, test_accuracy = test(model, testLoader, criterion, device)
print(f"Test Loss: {testLoss:.4f}, Test Acc: {test_accuracy:.2f}%")

imagePath = input("Enter an image path")
print(whatsThatNumber(model, imagePath))