import torch


class MyResNet(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.base_model = base_model
        self.base_model.classifier = torch.nn.Linear(1024, 5)

    def forward(self, x):
        out = self.base_model(x)
        return out