import torch
from torch import nn

class TinyCNN(nn.Module):
    """The 24,346-parameter CNN described in the project report."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

def parameter_summary(model):
    total = sum(p.numel() for p in model.parameters())
    bn = sum(p.numel() for m in model.modules()
             if isinstance(m, nn.BatchNorm2d) for p in m.parameters(recurse=False))
    return {"total": total, "adaptable_batchnorm": bn,
            "adaptable_percent": round(100 * bn / total, 2)}
