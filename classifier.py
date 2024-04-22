import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
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
train_set, val_set = random_split(dataset, [50000, 10000])
train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=batchSize, shuffle=False, num_workers=4)
    
#Define loss function and optimizer
model = ANN(inputSize, outputSize)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)