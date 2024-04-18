from torch import nn
import torch

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

    def forward(self, x):
        return self.classifier(x), {}

class MLP(nn.Module):
    def __init__(self, num_input, num_classes, num_clips) -> None:
        super().__init__()
        self.num_clips = num_clips
        self.num_input = num_input
        self.classifier = nn.Sequential(
            nn.Linear(self.num_input, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Dropout(p=0.6)
        )

    def forward(self, x):
        logits = []
        for clip in range(self.num_clips):
            logits.append(self.classifier(x[clip,:]))
        return torch.stack(logits, dim=0).mean(dim=0), {}
