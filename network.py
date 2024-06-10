import torch.nn as nn;

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.flatten = nn.Flatten()
        self.layers2 = nn.Sequential(
            nn.Linear(9 * 50 * 9, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, a):
        out = self.layers1(a)
        out = self.flatten(out)
        out = self.layers2(out)
        return out