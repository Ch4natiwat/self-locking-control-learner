import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNN(nn.Module):
    
    def __init__(self, input_channels=30, num_outputs=17):
        
        super(DQNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_outputs)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x