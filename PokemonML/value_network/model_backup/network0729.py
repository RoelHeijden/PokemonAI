import torch
from torch import nn
from typing import Dict

from PokemonML.value_network.model.encoder import Encoder


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        # input sizes dependent on the StateTransformer output
        field_size = 21
        side_size = 18
        pokemon_attributes = 72 + (4 * 4)

        # encoding layer
        species_dim = 64
        move_dim = 16
        item_dim = 16
        ability_dim = 16
        self.encoding = Encoder(species_dim, move_dim, item_dim, ability_dim, custom_embeddings=True)

        # pokemon layer
        pkmn_layer_out = 192
        pokemon_layer_in = (species_dim + move_dim * 4 + item_dim + ability_dim) + pokemon_attributes + side_size + field_size
        self.pokemon_layer = PokemonLayer(pokemon_layer_in, pkmn_layer_out)

        # full state layer
        state_layer_in = pkmn_layer_out * 6 * 2
        fc1_out = 1024
        fc2_out = 512
        state_layer_out = 128
        self.state_layer = StateLayer(state_layer_in, fc1_out, fc2_out, state_layer_out)

        # output later
        self.output = OutputLayer(state_layer_out)

    def forward(self, fields: torch.tensor, sides: torch.tensor, pokemon: Dict[str, torch.Tensor]):
        # embed and concatenate pokemon variables
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        # make fields size [batch, 2, 6, field]
        multi_dim_fields = torch.stack(
            (
                fields.unsqueeze(1).repeat(1, 6, 1),
                fields.unsqueeze(1).repeat(1, 6, 1)
            ),
            dim=1
        )

        # make sides size [batch, 2, 6, side]
        multi_dim_sides = sides.unsqueeze(2).repeat(1, 1, 6, 1)

        # pass each concat of (pokemon, side, field) through the pokemon layer
        pokemon_out = self.pokemon_layer(
            torch.cat(
                (
                    pokemon,
                    multi_dim_sides,
                    multi_dim_fields
                ),
                dim=3
            )
        )

        # state layer
        state_out = self.state_layer(torch.flatten(pokemon_out, start_dim=1))

        # output layer
        win_prob = self.output(state_out)

        return win_prob


class PokemonLayer(nn.Module):
    def __init__(self, input_size, fc1_out):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)

        self.drop = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x) -> torch.tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)

        return x


class StateLayer(nn.Module):
    def __init__(self, input_size, fc1_out, fc2_out, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, output_size)

        self.bn0 = nn.BatchNorm1d(input_size)
        self.bn1 = nn.BatchNorm1d(fc1_out)
        self.bn2 = nn.BatchNorm1d(fc2_out)
        self.bn3 = nn.BatchNorm1d(output_size)

        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.1)

        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.bn0(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.bn3(x)

        return x


class OutputLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

