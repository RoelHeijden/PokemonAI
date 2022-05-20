import torch
from torch import nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, embeddings: nn.ModuleDict):
        super().__init__()
        # Initialize a flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Get embeddings
        self.species_embedding = embeddings["species"]
        self.move_embedding = embeddings["move"]
        self.item_embedding = embeddings["item"]
        self.ability_embedding = embeddings["ability"]


def create_embeddings(categories: list, embedding_dim):
    embeddings = nn.ModuleDict()

    for category in categories:
        num_embeddings = len() + 1  # or + 2???
        embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        embeddings[category] = embedding

    return embeddings
