import torch
from torch import nn

from state_transformer.categories import (
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
        self.move_embedding = nn.Embedding(len(MOVES) + 1, 64, padding_idx=0)
        self.item_embedding = nn.Embedding(len(ITEMS) + 1, 16, padding_idx=0)
        self.ability_embedding = nn.Embedding(len(ABILITIES) + 1, 16, padding_idx=0)

    def _concat_fields(self, fields):
        return fields

    def _concat_sides(self, sides):
        return self.flatten(sides)

    def _concat_pokemon(self, pokemon):
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
                        pokemon['pokemon_attributes'][:, j][:, i],
                        self.flatten(moves[:, j][:, i]),
                        self.flatten(pokemon['move_attributes'][:, j][:, i])
                    ),
                    dim=1
                )
                for i in range(6)
            ]
            for j in range(2)
        ]

    def forward(self, fields, sides, pokemon):
        fields = self._concat_fields(fields)
        sides = self._concat_sides(sides)
        pokemon = self._concat_pokemon(pokemon)

        # temporary test
        return torch.cat(
            (
                fields,
                sides,
                pokemon[0][0],
                pokemon[0][1],
                pokemon[0][2],
                pokemon[0][3],
                pokemon[0][4],
                pokemon[0][5],
                pokemon[1][0],
                pokemon[1][1],
                pokemon[1][2],
                pokemon[1][3],
                pokemon[1][4],
                pokemon[1][5],
            ),
            dim=1
        )




