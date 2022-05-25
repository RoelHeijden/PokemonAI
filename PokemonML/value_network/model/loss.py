import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, loss='BCE'):
        super().__init__()

        if loss == 'L1':
            self.loss_function = nn.L1Loss()
        else:
            self.loss_function = nn.BCELoss()

    def forward(self, x, result):
        return self.loss_function(torch.squeeze(x), result)
