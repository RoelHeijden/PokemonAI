import torch
from torch import nn
from typing import Dict

from data.categories import (
    SPECIES,
    MOVES,
    ITEMS,
    ABILITIES
)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)

        self.species_embedding = nn.Embedding(len(SPECIES) + 1, 64, padding_idx=0)
        self.move_embedding = nn.Embedding(len(MOVES) + 1, 128, padding_idx=0)
        self.item_embedding = nn.Embedding(len(ITEMS) + 1, 32, padding_idx=0)
        self.ability_embedding = nn.Embedding(len(ABILITIES) + 1, 64, padding_idx=0)

    def _concat_pokemon(self, pokemon: Dict[str, torch.tensor]):
        """
        returns:
            for each player:
                for each pokemon:
                    'embeddings': concatenated pokemon embeddings
                    'attributes': concatenated pokemon attributes
        """
        species = self.species_embedding(pokemon["species"])
        moves = self.move_embedding(pokemon["moves"])
        items = self.item_embedding(pokemon["items"])
        abilities = self.ability_embedding(pokemon["abilities"])

        return [
            [
                torch.cat(
                    (
                        species[:, j][:, i],
                        items[:, j][:, i],
                        abilities[:, j][:, i],
                        self.flatten(moves[:, j][:, i]),
                        # torch.mean(moves[:, j][:, i], dim=1),
                        self.flatten(pokemon['move_attributes'][:, j][:, i]),
                        pokemon['pokemon_attributes'][:, j][:, i],
                    ),
                    dim=1
                )
                for i in range(6)
            ]
            for j in range(2)
        ]

    def forward(self, fields, sides, pokemon):
        return fields, sides, self._concat_pokemon(pokemon)


