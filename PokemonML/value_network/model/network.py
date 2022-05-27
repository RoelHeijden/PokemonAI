import torch
from torch import nn
from typing import Dict

from model.encoder import Encoder


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        # flexible input sizes
        field_size = 21
        side_size = 18
        pokemon_attributes = 72 + (2 * 4)

        # encoding layer
        species_dim = 128
        move_dim = 16
        item_dim = 16
        ability_dim = 16
        self.encoding = Encoder(species_dim, move_dim, item_dim, ability_dim)

        # pokemon layer
        pkmn_layer_out = 128
        pokemon_size = (species_dim + move_dim * 4 + item_dim + ability_dim) + pokemon_attributes
        self.pokemon_layer = PokemonLayer(pokemon_size, pkmn_layer_out)

        # active pokemon layer
        active_layer_out = 64
        self.active_pkmn_layer = ActivePkmnLayer(pkmn_layer_out, active_layer_out)

        # player layer
        player_layer_in = pkmn_layer_out * 6 + side_size
        player_layer_out = 512
        self.player_layer = PlayerLayer(player_layer_in, player_layer_out)

        # state layer
        state_layer_in = (player_layer_out + active_layer_out) * 2 + field_size
        state_layer1_out = 512
        state_layer2_out = 256

        self.state_layer = StateLayer(state_layer_in, state_layer1_out, state_layer2_out)

        # output later
        self.output = OutputLayer(state_layer2_out)

    def forward(self, fields: torch.tensor, sides: torch.tensor, pokemon: Dict[str, torch.Tensor]):

        # embed and concatenate pokemon variables
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        # general pokemon layer
        pokemon = self.pokemon_layer(pokemon)

        # active pokemon layer
        p1_active = self.active_pkmn_layer(pokemon[:, 0][:, 0])
        p2_active = self.active_pkmn_layer(pokemon[:, 1][:, 0])

        # player layer
        player1 = self.player_layer(
            torch.cat(
                (
                    torch.flatten(pokemon[:, 0], start_dim=1),
                    sides[:, 0]
                ),
                dim=1
            )
        )
        player2 = self.player_layer(
            torch.cat(
                (
                    torch.flatten(pokemon[:, 1], start_dim=1),
                    sides[:, 1]
                ),
                dim=1
            )
        )

        # state layer
        state = self.state_layer(
            torch.cat(
                (
                    p1_active,
                    player1,
                    p2_active,
                    player2,
                    fields
                 ),
                dim=1
            )
        )

        # output layer
        win_prob = self.output(state)

        return win_prob


class PokemonLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x) -> torch.tensor:
        x = self.fc(x)
        x = self.relu(x)
        x = self.drop(x)

        return x


class ActivePkmnLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

    def forward(self, x) -> torch.tensor:
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)

        return x


class PlayerLayer(nn.Module):
    def __init__(self, fc_input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(fc_input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

    def forward(self, x) -> torch.tensor:
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)

        return x


class StateLayer(nn.Module):
    def __init__(self, input_size, fc1_out, fc2_out):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)

        self.bn1 = nn.BatchNorm1d(fc1_out)
        self.bn2 = nn.BatchNorm1d(fc2_out)

        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:

        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)

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

