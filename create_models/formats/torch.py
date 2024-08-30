from pathlib import Path
from torch import nn
import torch

from ..model_weights.common import WBTuples


class Linear(nn.Module):
    def __init__(self, weights: WBTuples):
        super().__init__()
        assert len(weights) == 3

        self.seq = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.Sigmoid(),
            nn.Linear(8, 4),
            nn.Tanh(),
        )
        seq_inds = [0, 2, 4]

        for i, (weight, bias) in zip(seq_inds, weights):
            self.seq[i].weight.data = torch.as_tensor(weight)
            self.seq[i].bias.data = torch.as_tensor(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Conv2d(nn.Module):
    def __init__(self, weights: WBTuples):
        super().__init__()
        assert len(weights) == 3
        self.seq = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5),
            nn.Conv2d(16, 6, 3),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
        )
        seq_inds = [0, 2, 3]
        for i, (weight, bias) in zip(seq_inds, weights):
            self.seq[i].weight.data = torch.as_tensor(weight)
            self.seq[i].bias.data = torch.as_tensor(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Conv2dLinear(nn.Module):
    def __init__(self, weights: WBTuples):
        super().__init__()
        assert len(weights) == 6
        self.seq = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5),
            nn.Conv2d(16, 6, 3),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.Sigmoid(),
            nn.Linear(8, 4),
            nn.Tanh(),
        )
        seq_inds = [0, 2, 3, 6, 8, 10]
        for i, (weight, bias) in zip(seq_inds, weights):
            self.seq[i].weight.data = torch.as_tensor(weight)
            self.seq[i].bias.data = torch.as_tensor(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


MODELS = {"linear": Linear, "conv2d": Conv2d, "conv2d_linear": Conv2dLinear}


def create(model_name: str, weights: WBTuples, save_dir: Path):
    model = MODELS[model_name](weights)
    torch.save(model, save_dir / f"torch_{model_name}.pt")
