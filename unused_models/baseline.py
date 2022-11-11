from torch import nn

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(1, 2, 5), # 224 -> 220
            nn.MaxPool2d(2), # 220 -> 110
            nn.ReLU(),
            nn.Conv2d(2, 4, 3), # 110 -> 108
            nn.MaxPool2d(2), # 108 -> 54
            nn.ReLU(),
            nn.Conv2d(4, 8, 5), # 54 -> 50
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(50 * 50 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
        )
    
    def forward(self, x):
        return self.classifier(self.sequential(x))