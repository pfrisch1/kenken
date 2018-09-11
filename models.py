import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, gsz):
        super(SimpleCNN, self).__init__()
        self.embedones = nn.Embedding(10, 3)
        self.embedtens = nn.Embedding(10, 3)
        self.embedhunds = nn.Embedding(10, 3)
        self.embedop = nn.Embedding(5, 3)

        self.conv1 = nn.Conv2d(12, 5, 5, 1, 2)
        self.conv2 = nn.Conv2d(5, 5, 5, 1, 2)
        self.conv3 = nn.Conv2d(5, gsz, 1, 1, 0)

    def forward(self, x):
      x = torch.cat([
        self.embedones(x[:,:,:,0]),
        self.embedtens(x[:,:,:,1]),
        self.embedhunds(x[:,:,:,2]),
        self.embedop(x[:,:,:,3])
        ], -1).permute(0,3,1,2)

      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = self.conv3(x)
      return x
       # x = x.max(-1)[0].max(-1)[0]
       # return self.fc(x)