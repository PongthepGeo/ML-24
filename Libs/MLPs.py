#-----------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
#-----------------------------------------------------------------------------------------#

class MLPs(nn.Module):
    def __init__(self, n_classes):
        super(MLPs, self).__init__()
        self.hidden1 = nn.Linear(3, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, n_classes)

    def forward(self, x, apply_softmax=False):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        if apply_softmax:
            x = nn.functional.softmax(x, dim=1)
        return x

#-----------------------------------------------------------------------------------------#