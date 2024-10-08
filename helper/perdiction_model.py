import torch.nn as nn

class UT_HAR_LeNet(nn.Module):
    def __init__(self):
        super(UT_HAR_LeNet, self).__init__()
        self.encoder = nn.Sequential(
            # input size: (1,250,90)
            nn.Conv2d(1, 32, 7, stride=(3, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        # to store activations
        self.activations = None

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 96 * 4 * 4)
        out = self.fc(x)
        return out