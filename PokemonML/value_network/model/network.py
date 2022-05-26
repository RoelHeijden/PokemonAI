import torch
from torch import nn
from typing import Dict

from model.encoder import Encoder


class TestNet(nn.Module):
    def __init__(self, field_size, side_size, pokemon_size):
        super().__init__()
        input_size = (pokemon_size * 6 + side_size) * 2 + field_size

        self.encoding = Encoder()

        self.fc1 = nn.Linear(input_size, 2024)
        self.fc2 = nn.Linear(2024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)

        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fields, sides, pokemon):
        fields, sides, pokemon = self.encoding(fields, sides, pokemon)

        x = torch.cat(
            (
                fields,

                sides[:, 0],
                sides[:, 1],

                pokemon[0][0],
                pokemon[0][1],
                pokemon[0][2],
                pokemon[0][3],
                pokemon[0][4],
                pokemon[0][5],

                pokemon[1][0],
                pokemon[1][1],
                pokemon[1][2],
                pokemon[1][3],
                pokemon[1][4],
                pokemon[1][5]
            ),
            dim=1
        )

        x = self.relu(self.drop1(self.fc1(x)))
        x = self.relu(self.drop2(self.fc2(x)))
        x = self.relu(self.drop2(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x


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

    -----------------------------------------------

    6 * 256
    ->

    """
    def __init__(self, field_size, side_size, pokemon_size):
        super().__init__()

        # encoding layer
        self.encoding = Encoder()

        # pokemon layer
        pkmn_layer_out = 128
        self.pokemon_layer = PokemonLayer(pokemon_size, pkmn_layer_out)

        # team layer
        stride = 2
        kernel_size = 4
        max_pool_out = int(((pkmn_layer_out * 6 - (kernel_size - 1) - 1) / stride) + 1)
        team_layer_out = 512
        self.team_layer = TeamLayer(stride, kernel_size, max_pool_out, team_layer_out)

        # state layer
        state_size = (team_layer_out + side_size + pkmn_layer_out) * 2 + field_size
        state_layer1_out = 1024
        state_layer2_out = 512
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

        self.fc = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x) -> torch.tensor:
        x = self.fc(x)
        x = self.drop(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TeamLayer(nn.Module):
    def __init__(self, stride, kernel_size, max_pool_out, output_size):
        super().__init__()

        self.max_pool = nn.MaxPool1d(kernel_size, stride)
        self.fc = nn.Linear(max_pool_out, output_size)
        self.bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # for some reason the max pool wants 3 dimensions instead of [Batch size x Input]
        # Adding a temporary dimension by unsqueezing pre and squeezing post computation
        x = torch.squeeze(self.max_pool(x.unsqueeze(0)), dim=0)

        x = self.fc(x)
        x = self.drop(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StateLayer(nn.Module):
    def __init__(self, input_size, mid_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, mid_size)
        self.fc2 = nn.Linear(mid_size, output_size)
        self.bn1 = nn.BatchNorm1d(mid_size)
        self.bn2 = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.drop(x)
        x = self.bn2(x)
        x = self.relu(x)
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

