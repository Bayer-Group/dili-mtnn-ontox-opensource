import torch
import torch.nn.functional as F
from torch import nn

class MultiTaskNN(nn.Module):
    def __init__(self, input_size, params, n_tasks):
        super().__init__()
        self.n_tasks = n_tasks

        # define layers
        self.fc = nn.ModuleList([nn.Linear(input_size, params['n_units'][0])])
        input_units = params['n_units'][0]

        for n_units in params['n_units'][1:]:
            self.fc.append(nn.Linear(input_units, n_units))
            input_units = n_units
        # output layer for each task
        for task in range(n_tasks):
            self.fc.append(nn.Linear(input_units, 1))
        # self.fc.append([nn.Linear(input_units, 1) for task in range(n_tasks)])

        # define dropouts
        self.dropouts = nn.ModuleList([nn.Dropout(p=drop) for drop in params['dropout']])

        # weight initialization
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(mod.bias, 0)


    def forward(self, x):
        for i in range(len(self.fc) - self.n_tasks):
            x = F.relu(self.fc[i](x))
            x = self.dropouts[i](x)

        outputs = []
        for t in range(self.n_tasks):
            outputs.append(torch.sigmoid(self.fc[-self.n_tasks + t](x))) # the last n_task layers correspond to the output layers of the different tasks
        return outputs