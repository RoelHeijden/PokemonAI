import torch
from torch import nn
import numpy as np

from encoder import Encoder
from encoder import create_embeddings


class ValueNet(nn.Module):
    def __init__(self, hidden: torch.nn.Module, hidden_out_size):
        super().__init__()

        self.embeddings = create_embeddings(['species', 'move'], 64)
        self.embeddings.update(create_embeddings(['item', 'ability'], 16))

        self.encoder = Encoder(self.embeddings)

        self.hidden = hidden
        # self.output = output_layer(hidden_out_size)

    def forward(self, field, side1, side2):
        x = self.encoder(field, side1, side2)
        x = self.hidden(x)
        x = self.output(x)
        return x


class Hidden(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x

