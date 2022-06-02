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
        pokemon_attributes = 73 + (4 * 2)

        # encoding layer
        species_dim = 32
        item_dim = 32
        ability_dim = 32
        move_dim = 32
        self.encoding = Encoder(species_dim, item_dim, ability_dim, move_dim, load_embeddings=True)

        # pokemon layer
        pokemon_in = species_dim + item_dim + ability_dim + move_dim * 4 + pokemon_attributes
        pokemon_out = 192
        self.pokemon_layer = PokemonLayer(pokemon_in, pokemon_out, drop_rate=0.3)

        # full state layer
        state_layer_in = pokemon_out * 12 + side_size * 2 + field_size
        fc1_out = 2048
        fc2_out = 512
        state_out = 128
        self.state_layer = FullStateLayer(state_layer_in, fc1_out, fc2_out, state_out, drop_rate1=0.5, drop_rate2=0.2)

        # output later
        self.output = OutputLayer(state_out)

    def forward(self, fields: torch.tensor, sides: torch.tensor, pokemon: Dict[str, torch.Tensor]):
        # embed and concatenate pokemon variables
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        # pass all pokemon individually through the pokemon layer
        pokemon = self.pokemon_layer(pokemon)

        # pass everything together through the full state layer
        state = self.state_layer(torch.cat(
                (
                    torch.flatten(pokemon, start_dim=1),
                    torch.flatten(sides, start_dim=1),
                    fields
                ),
                dim=1
            )
        )

        # output layer
        win_prob = self.output(state)

        return win_prob


class PokemonLayer(nn.Module):
    def __init__(self, input_size, fc1_out, drop_rate):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.drop = nn.Dropout(p=drop_rate)
        self.relu = nn.ReLU()

    def forward(self, x) -> torch.tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)

        return x


class FullStateLayer(nn.Module):
    def __init__(self, input_size, fc1_out, fc2_out, output_size, drop_rate1, drop_rate2):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, output_size)

        self.bn1 = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(fc1_out)
        self.bn3 = nn.BatchNorm1d(fc2_out)

        self.drop1 = nn.Dropout(p=drop_rate1)
        self.drop2 = nn.Dropout(p=drop_rate2)

        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop1(x)

        x = self.bn2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop2(x)

        x = self.bn3(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.drop2(x)

        return x


class OutputLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor):
        x = self.bn(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

