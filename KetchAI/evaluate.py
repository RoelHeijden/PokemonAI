import os

import torch

from PokemonML.value_network.model.network import ValueNet
from PokemonML.value_network.data.transformer import StateTransformer


class Evaluate:
    def __init__(self):
        # model parameters path
        model_name = 'epoch_6.pt'
        model_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/relevant_models/'
        model_path = os.path.join(model_folder, model_name)

        # nn input sizes
        pokemon_size = (32 + 16 + 16 + 4 * 32) + 88
        field_size = 21
        side_size = 18

        # evaluation network
        self.model = ValueNet(
            field_size=field_size,
            side_size=side_size,
            pokemon_size=pokemon_size
        )
        self.model.load_state_dict(torch.load(model_path)['model'])

        # state transformer
        self.transform = StateTransformer(shuffle_players=False, shuffle_pokemon=False, shuffle_moves=False)

    def evaluate(self, state):

        # extract state information
        state_dict = state.to_dict()

        # transform state into dict of tensors
        x = self.transform(state_dict)

        # network is hardcoded for batches, so create batch of size 1
        fields = torch.unsqueeze(x['fields'], 0)
        sides = torch.unsqueeze(x['sides'], 0)
        pokemon = {key: torch.unsqueeze(value, 0) for key, value in x['pokemon'].items()}

        # forward pass
        evaluation = self.model(fields, sides, pokemon)

        return evaluation

