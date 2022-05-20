import torch
from torch import nn
import numpy as np

from encoder import Encoder
from encoder import create_embeddings

"""
INPUT LAYER

- Weather type [8]
- Weather count [1]
- Terrain type [5]
- Terrain count [1]
- Trick room [1]
- Trick room count [1]

- For each player:
    - Side conditions [11]
    - Future sight [1]
    - Wish [2]
    - Healing wish [1]
    - Has active [1]
    - N pokemon [1]
    - Active [1]
    - Reserve [5]

    - For each Pokemon
        - Species [1033] -> embedding: [64]
        - Abilities [263] -> embedding: [16]
        - Item [196] -> embedding: [16]
        - Has item [1]
        - Types [18]
        - Stats [6]
        - Level [1]

        - Status conditions: [7]
        - Volatile status [27]
        - Sleep countdown [1]
        - Stat changes [7]
        - Is alive [1]
        - Health [1]
        - N moves [1]
        - Moves [4]

        - For each move:
            - Name [757] -> embedding: [64]
            - Type [18]
            - Move category [3]
            - Base power [1]
            - Max PP [4]

            - Current PP [4]
            - Disabled [1]
            - Used [1]
            - Targets self [1]
            - Priority [1]


Attributes that may be unavailable in pmariglia's state simulation:
    - weather, terrain and trick room count
    - sleep countdown
    - healing wish
    - choicelock in volatile status



---------------------- TO DO ----------------------

Dataloader/Transformer
    1. make category jsons with indices for pokemon, moves, items and abilities 
    2. remove unnecessary variables (rating, etc)
    3. scale data (e.g. stats = stats / 250)
    4. convert to dict[str: tensor]
    5. shuffle moves
    6. move game_states as large batches to files 
    7. split game results from dataset

Encoder/Input
    1. understand the embedding pipeline (what type of dict can dataloader handle?)
    2. How to represent the active pokemon (and the active pokemon attributes). Active always in same slot? (avoid shuffle)

Network/Output
    1. figure out how the output layer should function
    
    
"""


class ValueNet(nn.Module):
    def __init__(self, hidden: torch.nn.Module, hidden_out_size):
        super().__init__()

        self.embeddings = create_embeddings(['species', 'move'], 64)
        self.embeddings.update(create_embeddings(['item', 'ability'], 16))

        self.encoder = Encoder(self.embeddings)

        self.hidden = hidden
        # self.output = output_layer(hidden_out_size)

    def forward(self, field, side1, side2):
        x = self.encoder(field, side1, side2)
        x = self.hidden(x)
        x = self.output(x)
        return x


class Hidden(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x

