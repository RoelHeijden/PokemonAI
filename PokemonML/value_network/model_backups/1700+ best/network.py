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
        pokemon_attributes = 73 + (4 * 3)

        # encoding layer
        species_dim = 16
        item_dim = 16
        ability_dim = 16
        move_dim = 16
        self.encoding = Encoder(species_dim, item_dim, ability_dim, move_dim, load_embeddings=True)

        # input size of a single pokemon
        pokemon_size = species_dim + item_dim + ability_dim + move_dim * 4 + pokemon_attributes

        # conv matchup layer
        out_channels = 16
        matchup_out = out_channels * 6
        self.matchup_layer = MatchupLayer(pokemon_size * 2, out_channels)

        # pokemon layer
        pokemon_in = pokemon_size + matchup_out
        pokemon_out = 192
        self.pokemon_layer = PokemonLayer(pokemon_in, pokemon_out, drop_rate=0.4)

        # full state layer
        state_layer_in = pokemon_out * 12 + side_size * 2 + field_size
        fc1_out = 1024
        fc2_out = 512
        state_out = 64
        self.state_layer = FullStateLayer(state_layer_in, fc1_out, fc2_out, state_out, drop_rate=0.4)

        # output later
        self.output = OutputLayer(state_out)

    def forward(self, fields: torch.tensor, sides: torch.tensor, pokemon: Dict[str, torch.Tensor]):
        # embed and concatenate pokemon variables
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        # matchup layer
        matchups = self.matchup_layer(pokemon)

        # pokemon layer
        pokemon = self.pokemon_layer(torch.cat((pokemon, matchups), dim=3))

        # state layer
        state = self.state_layer(
            torch.cat(
                (
                    torch.flatten(pokemon, start_dim=1),
                    torch.flatten(sides, start_dim=1),
                    fields
                ),
                dim=1
            )
        )

        # output layer
        win_chance = self.output(state)

        return win_chance


class PokemonLayer(nn.Module):
    def __init__(self, input_size, output_size, drop_rate):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.bn2d = nn.BatchNorm2d(2 * 6)

        self.drop = nn.Dropout(p=drop_rate)
        self.relu = nn.ReLU()

    def forward(self, x) -> torch.tensor:
        x = self.fc(x)
        x = self.relu(x)

        # apply batchnorm to each individual pokemon output
        bs, d, h, w = x.shape
        x = x.view(bs, d * h, w).unsqueeze(2)
        x = self.bn2d(x).view(bs, d, h, w)

        x = self.drop(x)

        return x


class MatchupLayer(nn.Module):
    def __init__(self, input_size, out_channels):
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(1, input_size), stride=1)
        self.bn2d = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.tensor:
        p1_side = x[:, 0]
        p2_side = x[:, 1]

        # collect all (p1 pokemon, p2 pokemon) matchups by cyclic shifting p2's side
        matchups = []
        for i in range(x.shape[2]):
            matchups.append(
                torch.cat((p1_side, torch.roll(p2_side, i, dims=1)), dim=2)
            )

        # apply kernel over each of the 36 matchups
        x = torch.cat(matchups, dim=1).unsqueeze(1)
        x = self.conv2d(x)

        x = self.bn2d(x)
        x = self.sigmoid(x)

        x = torch.transpose(x, dim0=1, dim1=2).squeeze(3)
        bs, _, out = x.shape
        x = torch.reshape(x, (bs, 6, 6, out))
        # [bs, 6, 6, out]

        p1_side = torch.reshape(x, (bs, 6, 6 * out))
        # [bs, 6, 6*out]

        p2_side = torch.reshape(
            torch.stack(
                [x[:, :, i] for i in range(6)],
                dim=1
            ),
            (bs, 6, 6 * out)
        )
        # [bs, 6, 6*out]

        x = torch.stack((p1_side, p2_side), dim=1)
        # [bs, 2, 6, 6*out]

        return x


class FullStateLayer(nn.Module):
    def __init__(self, input_size, fc1_out, fc2_out, output_size, drop_rate):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, output_size)

        self.bn1 = nn.BatchNorm1d(fc1_out)
        self.bn2 = nn.BatchNorm1d(fc2_out)
        self.bn3 = nn.BatchNorm1d(output_size)

        self.drop = nn.Dropout(p=drop_rate)
        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.drop(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)

        x = self.drop(x)

        x = self.fc3(x)
        x = self.relu(x)
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

