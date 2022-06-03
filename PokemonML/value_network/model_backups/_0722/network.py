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
        species_dim = 64
        item_dim = 16
        ability_dim = 16
        move_dim = 16
        self.encoding = Encoder(species_dim, item_dim, ability_dim, move_dim, load_embeddings=True)

        # pokemon layer
        pokemon_in = species_dim + item_dim + ability_dim + move_dim * 4 + pokemon_attributes + side_size
        pokemon_out = 192
        self.pokemon_layer = PokemonLayer(pokemon_in, pokemon_out, drop_rate=0.1)

        # # convolutional test
        # channels_in = 6
        # channels_out = 60
        # conv_in = pokemon_out * 2
        # conv_out = channels_in * channels_out
        # self.teams_conv = TeamsConvolution(conv_in, channels_in, channels_out)

        # full state layer
        state_layer_in = pokemon_out * 12 + field_size
        fc1_out = 1024
        fc2_out = 512
        state_out = 128
        self.state_layer = FullStateLayer(state_layer_in, fc1_out, fc2_out, state_out, drop_rate=0.3)

        # output later
        self.output = OutputLayer(state_out)

    def forward(self, fields: torch.tensor, sides: torch.tensor, pokemon: Dict[str, torch.Tensor]):
        # embed and concatenate pokemon variables
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        # make sides size [batch, 2, 6, side]
        multi_dim_sides = sides.unsqueeze(2).repeat(1, 1, 6, 1)

        # pass each concat of (pokemon, side) through the pokemon layer
        pokemon_out = self.pokemon_layer(
            torch.cat(
                (
                    pokemon,
                    multi_dim_sides,
                ),
                dim=3
            )
        )

        # pass everything together through the full state layer
        state_out = self.state_layer(torch.cat(
                (
                    torch.flatten(pokemon_out, start_dim=1),
                    fields
                ),
                dim=1
            )
        )

        # output layer
        win_prob = self.output(state_out)

        return win_prob


class PokemonLayer(nn.Module):
    def __init__(self, input_size, output_size, drop_rate):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.drop = nn.Dropout(p=drop_rate)
        self.relu = nn.ReLU()

    def forward(self, x) -> torch.tensor:
        x = self.fc(x)
        x = self.relu(x)
        x = self.drop(x)

        return x


class TeamsConvolution(nn.Module):
    def __init__(self, input_size, channels_in, channels_out):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=(input_size, 1), stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        p1_side = x[:, 0]
        p2_side = x[:, 1]
        x = torch.stack(
            (
                torch.cat((p1_side, torch.roll(p2_side, 0, dims=1)), dim=2),
                torch.cat((p1_side, torch.roll(p2_side, 1, dims=1)), dim=2),
                torch.cat((p1_side, torch.roll(p2_side, 2, dims=1)), dim=2),
                torch.cat((p1_side, torch.roll(p2_side, 3, dims=1)), dim=2),
                torch.cat((p1_side, torch.roll(p2_side, 4, dims=1)), dim=2),
                torch.cat((p1_side, torch.roll(p2_side, 5, dims=1)), dim=2),
            ),
            dim=3
        )
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)

        return x


class FullStateLayer(nn.Module):
    def __init__(self, input_size, fc1_out, fc2_out, output_size, drop_rate):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, output_size)

        self.bn1 = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(fc1_out)
        self.bn3 = nn.BatchNorm1d(fc2_out)

        self.drop = nn.Dropout(p=drop_rate)
        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.bn2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.bn3(x)
        x = self.fc3(x)
        x = self.relu(x)

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

