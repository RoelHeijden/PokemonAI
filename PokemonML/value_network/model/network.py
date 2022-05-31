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
        species_dim = 32
        move_dim = 16
        item_dim = 16
        ability_dim = 16
        self.encoding = Encoder(species_dim, move_dim, item_dim, ability_dim, load_embeddings=True)

        # reserve pokemon layer
        reserve_in = (species_dim + move_dim * 4 + item_dim + ability_dim) + pokemon_attributes
        reserve_out = 128
        self.reserve_pokemon_layer = ReservePokemonLayer(reserve_in, reserve_out)

        # active pokemon layer
        active_in = 2 * ((species_dim + move_dim * 4 + item_dim + ability_dim) + pokemon_attributes + side_size) + field_size
        active_out = 512
        self.active_layer = ActiveLayer(active_in, active_out)

        # full state layer
        state_layer_in = (reserve_out * 5 + side_size) * 2 + active_out + field_size
        fc1_out = 1024
        state_out = 512
        self.state_layer = FullStateLayer(state_layer_in, fc1_out, state_out)

        # output later
        self.output = OutputLayer(state_out)

    def forward(self, fields: torch.tensor, sides: torch.tensor, pokemon: Dict[str, torch.Tensor]):
        # embed and concatenate pokemon variables
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        # pass each reserve pokemon individually through a reserve pokemon layer
        reserve_state = self.reserve_pokemon_layer(pokemon[:, :, 1:])

        # pass both active pokemon, side and field conditions together through the active layer
        active_state = self.active_layer(
            torch.cat(
                (
                    torch.flatten(pokemon[:, :, :1], start_dim=1),
                    torch.flatten(sides, start_dim=1),
                    fields
                ),
                dim=1
            )
        )

        # pass everything, including the side and field conditions again, through the full state layer
        state = self.state_layer(
            torch.cat(
                (
                    reserve_state,
                    active_state,
                    torch.flatten(sides, start_dim=1),
                    fields
                ),
                dim=1
            )
        )

        # output layer
        win_prob = self.output(state)

        return win_prob


class ReservePokemonLayer(nn.Module):
    def __init__(self, input_size, fc1_out):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.bn = nn.BatchNorm1d(fc1_out * 10)

        self.lRelu = nn.LeakyReLU(0.011)

    def forward(self, x) -> torch.tensor:
        x = self.fc1(x)
        x = self.lRelu(x)

        x = torch.flatten(x, start_dim=1)
        x = self.bn(x)

        return x


class ActiveLayer(nn.Module):
    def __init__(self, input_size, fc1_out):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.bn = nn.BatchNorm1d(fc1_out)

        self.lRelu = nn.LeakyReLU(0.01)

    def forward(self, x) -> torch.tensor:
        x = self.fc1(x)
        x = self.lRelu(x)
        x = self.bn(x)

        return x


class FullStateLayer(nn.Module):
    def __init__(self, input_size, fc1_out, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, output_size)

        self.bn1 = nn.BatchNorm1d(fc1_out)
        self.bn2 = nn.BatchNorm1d(output_size)

        self.drop = nn.Dropout(0.2)
        self.lRelu = nn.LeakyReLU(0.01)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)
        x = self.lRelu(x)
        x = self.bn1(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.lRelu(x)
        x = self.bn2(x)
        x = self.drop(x)

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

