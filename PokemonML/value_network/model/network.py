import torch
from torch import nn

from model.encoder import Encoder


class ValueNet(nn.Module):
    def __init__(self, hidden: torch.nn.Module, output: torch.nn.Module):
        super().__init__()

        self.encoder = Encoder()
        self.hidden = hidden
        self.output = output

    def forward(self, fields, sides, pokemon):
        x = self.encoder(fields, sides, pokemon)
        x = self.hidden(x)
        x = self.output(x)
        return x


class Hidden(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.bn1(self.relu(self.fc1(x)))
        x = self.bn2(self.relu(self.fc2(x)))
        x = self.bn3(self.relu(self.fc3(x)))
        return x


class Output(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.out_layer = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor):
        p1_win_chance = self.sigmoid(self.out_layer(x))
        return p1_win_chance

