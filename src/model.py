import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


MAX_STATES = 10
GRID_SIZE = 128


class SelfLockingControlBinder(nn.Module):
    
    def __init__(self, model: "DQNN"):
        
        super(SelfLockingControlBinder, self).__init__()
        
        self.max_states = MAX_STATES
        self.model = model
        self.number_of_states = 0
        self.states = torch.zeros((1, MAX_STATES * 3, GRID_SIZE, GRID_SIZE))
        self.is_locked = False
        self.locked_rebind = None


    def rebind(self, state: np.ndarray, action: np.ndarray):

        if self.is_locked:
            rebind = self.locked_rebind
            
        else:
            state = torch.from_numpy(np.transpose(state, (2, 0, 1)))
            
            if self.number_of_states < self.max_states:
                start_idx = self.number_of_states * 3
                end_idx = start_idx + 3
                self.states[0, start_idx:end_idx, :, :] = state
                self.number_of_states += 1
            else:
                self.states[0, :-3, :, :] = self.states[0, 3:, :, :]
                self.states[0, -3:, :, :] = state
            
            rebind = self.model(self.states)
            
            if rebind[0, -1] > 0:
                self.is_locked = True
                self.locked_rebind = rebind
                
            # TODO
        


class DQNN(nn.Module):
    
    def __init__(self, input_channels=30, num_outputs=21):
        
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