import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.bce_loss = nn.BCELoss()

    def forward(self, x, result):

        return self.bce_loss(torch.squeeze(x), result)
