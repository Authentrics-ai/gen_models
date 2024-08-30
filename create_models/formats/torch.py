from pathlib import Path
from torch import nn
import torch

from create_models.architecture import Architecture

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
            nn.Conv2d(1, 2, 3),
            nn.ReLU(),
            nn.Conv2d(2, 6, 5),
            nn.Conv2d(6, 6, 3),
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
            nn.Conv2d(1, 2, 3),
            nn.ReLU(),
            nn.Conv2d(2, 6, 5),
            nn.Conv2d(6, 6, 3),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=0),
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.Sigmoid(),
            nn.Linear(8, 4),
            nn.Tanh(),
        )
        seq_inds = [0, 2, 3, 7, 9, 11]
        for i, (weight, bias) in zip(seq_inds, weights):
            self.seq[i].weight.data = torch.as_tensor(weight)
            self.seq[i].bias.data = torch.as_tensor(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


MODELS = {Architecture.linear: Linear, Architecture.conv2d: Conv2d, Architecture.conv2d_linear: Conv2dLinear}


def create(architecture: Architecture, weights: WBTuples, save_dir: Path) -> Path:
    model = MODELS[architecture](weights)
    save_file = save_dir / f"torch_{architecture.name}.pt"
    print(model(torch.zeros((1, 1, 10, 10), dtype=torch.double)))
    torch.save(model, save_file)
    return save_file