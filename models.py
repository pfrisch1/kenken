import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, gsz):
        super(SimpleCNN, self).__init__()
        self.embed = nn.Embedding(gsz, 3)
        self.conv1 = nn.Conv2d(3, 5, gsz, 1, gsz/2)
        self.conv2 = nn.Conv2d(5, 5, 5, 1, 2)
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
       x = self.embed(x).permute(0,3,1,2)
       x = self.conv1(x)
       x = F.relu(x)
       x = self.conv2(x)
       x = x.max(-1)[0].max(-1)[0]
       return self.fc(x)