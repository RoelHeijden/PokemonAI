import torch
from torch import nn
from typing import Dict

from model.encoder import Encoder

"""
1250+, LR=2e-4, gamma=0.9, 128/16/16/16, pokemon: 192 drop(p=.2) no bn, state: 1024 drop(p=.3), 512 drop(p=.3), 256 drop(p=.1) -- epoch: 23, acc: 0.757
1250+:   0.561 | 0.578 | 0.603 | 0.615 | 0.632 | 0.663 | 0.67 | 0.688 | 0.696 | 0.715 | 0.729 | 0.754 | 0.762 | 0.785 | 0.801 | 0.82 | 0.84 | 0.87 | 0.908 | 0.939
1500+: 0.543 | 0.582 | 0.608 | 0.63 | 0.637 | 0.652 | 0.67 | 0.679 | 0.694 | 0.709 | 0.729 | 0.743 | 0.76 | 0.777 | 0.796 | 0.823 | 0.844 | 0.862 | 0.903 | 0.937

"""

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        # flexible input sizes
        field_size = 21
        side_size = 18
        pokemon_attributes = 72 + (4 * 4)

        # encoding layer
        species_dim = 128
        move_dim = 16
        item_dim = 16
        ability_dim = 16
        self.encoding = Encoder(species_dim, move_dim, item_dim, ability_dim)

        # pokemon layer
        pkmn_layer_out = 192
        pokemon_size = (species_dim + move_dim * 4 + item_dim + ability_dim) + pokemon_attributes
        self.pokemon_layer = PokemonLayer(pokemon_size, pkmn_layer_out)

        # full state layer
        state_layer_in = (pkmn_layer_out * 6 + side_size) * 2 + field_size
        fc1_out = 1024
        fc2_out = 512
        state_layer_out = 256
        self.state_layer = StateLayer(state_layer_in, fc1_out, fc2_out, state_layer_out)

        # output later
        self.output = OutputLayer(state_layer_out)

    def forward(self, fields: torch.tensor, sides: torch.tensor, pokemon: Dict[str, torch.Tensor]):
        # embed and concatenate pokemon variables
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        # general pokemon layer
        pokemon = self.pokemon_layer(pokemon)

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
        win_prob = self.output(state)

        return win_prob


class PokemonLayer(nn.Module):
    def __init__(self, input_size, fc1_out):
        super().__init__()
        self.pkmn_size = fc1_out

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.bn1 = nn.BatchNorm1d(fc1_out)

        self.drop1 = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()


    def forward(self, x) -> torch.tensor:
        x = self.fc1(x)
        x = self.relu(x)

        # [batch_size, 2, 6, fc1_out] -> [batch_size, 12, pkmn_size] -> [batch_size, fc1_out, 12]
        # x = torch.reshape(x, (x.size(dim=0), 2 * 6, self.pkmn_size))
        # x = torch.transpose(x, dim0=1, dim1=2)
        # x = self.bn1(x)

        x = self.drop1(x)

        return x


class StateLayer(nn.Module):
    def __init__(self, input_size, fc1_out, fc2_out, fc3_out):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, fc3_out)

        self.bn1 = nn.BatchNorm1d(fc1_out)
        self.bn2 = nn.BatchNorm1d(fc2_out)
        self.bn3 = nn.BatchNorm1d(fc3_out)

        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.drop1(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.drop2(x)

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

