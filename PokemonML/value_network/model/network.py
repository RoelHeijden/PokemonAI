import torch
from torch import nn
from typing import Dict, List

from model.encoder import Encoder


class ValueNet(nn.Module):
    """
    Network has a hierarchical structure consisting of 3 parts:

    1. Pokemon layer:
        - for each pokemon's concatenated embeddings: fully connected linear
        - output concatenated with the pokemon's attributes

    2. Team layer:
        - player's pokemon are concatenated as one team
        - for each player's team: max pooling + fully connected linear

    3. State layer:
        - teams, active pokemon, side conditions and field conditions concatenated as game state
        - 2x fully connected linear

    Output: p1 win probability
    """
    def __init__(self, field_size, side_size, pokemon_size):
        super().__init__()

        # encoding layer
        self.encoding = Encoder()

        # pokemon layer
        pkmn_layer_out = 512
        self.pokemon_layer = PokemonLayer(pokemon_size, pkmn_layer_out)

        # team layer
        stride = 2
        kernel_size = 4
        max_pool_out = int(((pkmn_layer_out * 6 - (kernel_size - 1) - 1) / stride) + 1)
        team_layer_out = 1024
        self.team_layer = TeamLayer(stride, kernel_size, max_pool_out, team_layer_out)

        # state layer
        state_size = (team_layer_out + side_size + pkmn_layer_out) * 2 + field_size
        state_layer1_out = 2048
        state_layer2_out = 1024
        self.state_layer = StateLayer(state_size, state_layer1_out, state_layer2_out)

        # output later
        self.output = OutputLayer(state_layer2_out)

    def forward(self, fields: torch.tensor, sides: torch.tensor, pokemon: Dict[str, torch.Tensor]):

        # embed and concatenate pokemon variables
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        # run each of player1's pokemon through the pokemon layer
        p1_active = self.pokemon_layer(pokemon[0][0])
        p1_reserve1 = self.pokemon_layer(pokemon[0][1])
        p1_reserve2 = self.pokemon_layer(pokemon[0][2])
        p1_reserve3 = self.pokemon_layer(pokemon[0][3])
        p1_reserve4 = self.pokemon_layer(pokemon[0][4])
        p1_reserve5 = self.pokemon_layer(pokemon[0][5])

        # run each of player2's pokemon through the pokemon layer
        p2_active = self.pokemon_layer(pokemon[1][0])
        p2_reserve1 = self.pokemon_layer(pokemon[1][1])
        p2_reserve2 = self.pokemon_layer(pokemon[1][2])
        p2_reserve3 = self.pokemon_layer(pokemon[1][3])
        p2_reserve4 = self.pokemon_layer(pokemon[1][4])
        p2_reserve5 = self.pokemon_layer(pokemon[1][5])

        # concatenate player1's team and run it through the team layer
        p1_team = self.team_layer(
            torch.cat(
                (p1_active, p1_reserve1, p1_reserve2, p1_reserve3, p1_reserve4, p1_reserve5),
                dim=1
            )
        )

        # concatenate player2's team and run it through the team layer
        p2_team = self.team_layer(
            torch.cat(
                (p2_active, p2_reserve1, p2_reserve2, p2_reserve3, p2_reserve4, p2_reserve5),
                dim=1
            )
        )

        # concatenate state and run it through the state layer
        state = self.state_layer(
            torch.cat(
                (
                    p1_team,
                    p1_active,
                    sides[:, 0],

                    p2_team,
                    p2_active,
                    sides[:, 1],

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

        self.pokemon_fc = nn.Linear(input_size, output_size)
        self.pokemon_bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

    def forward(self, x) -> torch.tensor:
        x = self.relu(self.pokemon_fc(x))
        x = self.pokemon_bn(x)
        return x


class TeamLayer(nn.Module):
    def __init__(self, stride, kernel_size, max_pool_out, output_size):
        super().__init__()

        self.team_mp = nn.MaxPool1d(kernel_size, stride)
        self.team_fc = nn.Linear(max_pool_out, output_size)
        self.team_bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.squeeze(self.team_mp(x.unsqueeze(0)))
        x = self.relu(self.team_fc(x))
        x = self.team_bn(x)
        return x


class StateLayer(nn.Module):
    def __init__(self, input_size, mid_size, output_size):
        super().__init__()

        self.state_fc1 = nn.Linear(input_size, mid_size)
        self.state_bn1 = nn.BatchNorm1d(mid_size)
        self.state_fc2 = nn.Linear(mid_size, output_size)
        self.state_bn2 = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.relu(self.state_fc1(x))
        x = self.state_bn1(x)
        x = self.relu(self.state_fc2(x))
        x = self.state_bn2(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.out_layer = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor):
        p1_win_chance = self.sigmoid(self.out_layer(x))
        return p1_win_chance

