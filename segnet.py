import torch
import torch.nn as nn
import torch.nn.functional as F


class Segnet(nn.Module):
    def __init__(self):
        super(Segnet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,6,(4,4)),
            nn.ReLU(),
            nn.Conv2d(6, 12, (4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 6, (3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, (3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, (3, 3)),
            nn.Sigmoid()
        )
        self.mask1 = nn.Parameter()
        self.mask2 = nn.Parameter()

    def forward(self, x):
        x = torch.cat((x,self.mask1, self.mask2), dim=1)
        return self.net(x)
