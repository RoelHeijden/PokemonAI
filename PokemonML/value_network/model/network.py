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
        species_dim = 64
        item_dim = 16
        ability_dim = 16
        move_dim = 16
        self.encoding = Encoder(species_dim, item_dim, ability_dim, move_dim, load_embeddings=True)

        # pokemon layer
        pokemon_in = species_dim + item_dim + ability_dim + move_dim * 4 + pokemon_attributes
        pokemon_out = 128
        self.pokemon_layer = PokemonLayer(pokemon_in, pokemon_out, drop_rate=0.2)

        # team matchup layer
        channels_out = 10
        matchup_in = pokemon_in * 2
        matchup_out = channels_out * 6 * 6
        self.matchup_layer = MatchupLayer(matchup_in, channels_out)

        # full state layer
        state_layer_in = pokemon_out * 12 + side_size * 2 + field_size + matchup_out
        fc1_out = 1024
        fc2_out = 512
        state_out = 128
        self.state_layer = FullStateLayer(state_layer_in, fc1_out, fc2_out, state_out, drop_rate=0.5)

        # output later
        self.output = OutputLayer(state_out)

    def forward(self, fields: torch.tensor, sides: torch.tensor, pokemon: Dict[str, torch.Tensor]):
        # embed and concatenate pokemon variables
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        # matchup layer
        matchup = self.matchup_layer(pokemon)

        # pass each concat of (pokemon, side) through the pokemon layer
        pokemon = self.pokemon_layer(pokemon)

        # pass everything together through the full state layer
        state = self.state_layer(
            torch.cat(
                (
                    torch.flatten(matchup, start_dim=1),
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
        # self.bn = nn.BatchNorm1d(output_size * 12)
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

        # # 1d batchnorm solution
        # bs, d, h, w = pokemon.shape
        # pokemon = pokemon.view(bs, d * h * w)
        # pokemon = self.bn(pokemon)
        # pokemon = pokemon.view(bs, d, h, w)

        return x


class MatchupLayer(nn.Module):
    def __init__(self, input_size, channels_out):
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels=1, out_channels=channels_out, kernel_size=(1, input_size), stride=1)
        self.bn2d = nn.BatchNorm2d(channels_out)
        self.relu = nn.ReLU()

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

        x = self.relu(x)
        x = self.bn2d(x)

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

