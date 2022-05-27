import torch
from torch import nn
from typing import Dict

from model.encoder import Encoder

"""
BEST: 
ValueNet2 1500+, LR=2e-4, gamma=0.9, 128/16/16/16, 192, 1024, 512, 252, pokemon dropout (p=.2) > bn -- epoch 20: acc=0.755
ValueNet2 1500+, LR=3e-4, gamma=0.95, 128/16/16/16, 192, 1024, 512, 252, pokemon dropout (p=.2) > bn -- ???

---------------------------------------------------------------------------------------------

ValueNet2 1500+, LR=2e-4, gamma=0.9, 32/16/16/16, 128, 2048, 1024, 512 -- epoch 5: acc=0.744
ValueNet2 1500+, LR=2e-4, gamma=0.9, 32/32/16/16, 128, 1024, 512, 256 -- epoch 5: acc=0.739
ValueNet2 1500+, LR=2e-4, gamma=0.85, 8/8/8/8, 128, 1024, 512 -- epoch 7: acc=0.745
ValueNet2 1500+, LR=2e-4, gamma=0.9, 128/16/16/16, 192, 1024, 512, 252, pokemon dropout (p=.2) > bn -- epoch 10: acc=0.751
ValueNet2 1500+, LR=2e-4, gamma=0.9, 128/16/16/16, 192, 1024, 512, 252, 128, pokemon dropout (p=.2) > bn -- epoch 24: acc=0.749
ValueNet2 1500+, LR=2e-4, gamma=0.95, 128/128/16/32, 252, 2048, 1024, 512, 252, pokemon dropout (p=.2) > bn -- epoch 10: acc=0.750
ValueNet2 1500+, LR=2e-4, gamma=0.95, 64/16/16/16, 128, 1024, 512, 252, 128, pokemon dropout (p=.2) > bn -- epoch 10: acc=0.749
ValueNet2 1500+, LR=2e-4, gamma=0.85, 128/16/16/16, 192, 1024, 512, 252, pokemon dropout (p=.2) + bn -- epoch 11: acc=0.753, 
ValueNet2 1500+, LR=3e-4, gamma=0.95, 128/16/16/16, 192, 1024, 512, 252, multiple dropouts (p=.2, p=.3) + bn -- epoch 15: acc=0.748
ValueNet2 1500+, LR=3e-4, gamma=0.95, 128/16/16/16, 192, 1024, 512, 252, multiple dropouts (p=.2, p=.3) > bn -- epoch 14: acc=0.752
"""


class ValueNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)

        # encoding layer
        species_dim = 128
        move_dim = 16
        item_dim = 16
        ability_dim = 16
        self.encoding = Encoder(species_dim, move_dim, item_dim, ability_dim)

        # pokemon layer
        pkmn_layer_out = 192
        pokemon_size = (species_dim + move_dim * 4 + item_dim + ability_dim) + 72 + (4 * 4)
        self.pokemon_layer = PokemonLayer(pokemon_size, pkmn_layer_out)

        # state layer
        field_size = 21
        side_size = 18
        state_size = (pkmn_layer_out * 6 + side_size) * 2 + field_size
        state_layer1_out = 1024
        state_layer2_out = 512
        state_layer3_out = 252
        self.state_layer = StateLayer(state_size, state_layer1_out, state_layer2_out, state_layer3_out)

        # output later
        self.output = OutputLayer(state_layer3_out)

    def forward(self, fields: torch.tensor, sides: torch.tensor, pokemon: Dict[str, torch.Tensor]):

        # embed and concatenate pokemon variables
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        # pokemon layer
        pokemon = self.pokemon_layer(pokemon)

        # concatenate everything
        pokemon = self.flatten(pokemon)
        sides = self.flatten(sides)
        state = torch.cat((pokemon, sides, fields), dim=1)

        # state layer
        state = self.state_layer(state)

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


class StateLayer(nn.Module):
    def __init__(self, input_size, fc1_out, fc2_out, fc3_out):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, fc3_out)

        self.bn1 = nn.BatchNorm1d(fc1_out)
        self.bn2 = nn.BatchNorm1d(fc2_out)
        self.bn3 = nn.BatchNorm1d(fc3_out)

        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:

        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)

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

