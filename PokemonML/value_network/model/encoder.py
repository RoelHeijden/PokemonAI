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
    def __init__(self, species_dim, move_dim, item_dim, ability_dim):
        super().__init__()

        self.species_embedding = nn.Embedding(len(SPECIES) + 1, species_dim, padding_idx=0)
        self.move_embedding = nn.Embedding(len(MOVES) + 1, move_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(len(ITEMS) + 1, item_dim, padding_idx=0)
        self.ability_embedding = nn.Embedding(len(ABILITIES) + 1, ability_dim, padding_idx=0)

    def _concat_pokemon(self, pokemon: Dict[str, torch.tensor]) -> torch.tensor:
        """
        returns: tensor of size [batch, 2, 6, pokemon]
        """
        species = self.species_embedding(pokemon["species"])
        moves = self.move_embedding(pokemon["moves"])
        items = self.item_embedding(pokemon["items"])
        abilities = self.ability_embedding(pokemon["abilities"])

        return torch.cat(
            (
                species,
                items,
                abilities,
                torch.flatten(moves, start_dim=3),
                torch.flatten(pokemon['move_attributes'], start_dim=3),
                pokemon['pokemon_attributes'],
            ),
            dim=3
        )

    def forward(self, fields, sides, pokemon):
        return fields, sides, self._concat_pokemon(pokemon)


