import torch as th
import torch.nn as nn
from network import Network
import torch.nn.functional as F
from pruningmethod import PruningMethod

class LeNet(nn.Module, Network,PruningMethod):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 10)
        self.a_type='relu'
        for m in self.modules():
            self.weight_init(m)
        self.softmax = nn.Softmax(dim=1)

        

    def forward(self, x):
        layer1 = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        layer2 = F.max_pool2d(F.relu(self.conv2(layer1)), 2)
        layer2_p = layer2.view(-1, int(layer2.nelement() / layer2.shape[0]))
        layer3= F.relu(self.fc1(layer2_p))
        layer4 = F.relu(self.fc2(layer3))
        layer5 = self.fc3(layer4)
        return layer5



