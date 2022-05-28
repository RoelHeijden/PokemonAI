import os
import json
import torch

from PokemonML.value_network.model.network import ValueNet
from PokemonML.value_network.data.transformer import StateTransformer


class Evaluate:
    def __init__(self):
        # model parameters path
        model_name = 'network_0759_1250+_epoch_23.pt'
        model_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/relevant_models/'
        model_path = os.path.join(model_folder, model_name)

        # evaluation network
        self.model = ValueNet()
        self.model.load_state_dict(torch.load(model_path)['model'])
        self.model.eval()

        # state transformer
        self.transform = StateTransformer(shuffle_players=False, shuffle_pokemon=False, shuffle_moves=False)

    def evaluate(self, state):
        # extract state information
        state_dict = state.to_dict()

        # StateTransformer checks key 'winner'. Simply set winner to empty
        state_dict['winner'] = ''

        # transform state into dict of tensors
        x = self.transform(state_dict)

        # network is hardcoded for batches, so create batch of size 1
        fields = torch.unsqueeze(x['fields'], 0)
        sides = torch.unsqueeze(x['sides'], 0)
        pokemon = {key: torch.unsqueeze(value, 0) for key, value in x['pokemon'].items()}

        # forward pass
        evaluation = self.model(fields, sides, pokemon).item()

        return evaluation

