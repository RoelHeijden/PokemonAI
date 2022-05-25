import os
import json
import torch

from model.loss import Loss
from data.transformer import StateTransformer


class Tester:
    def __init__(self, model, model_file):
        folder_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/'
        file_path = os.path.join(folder_path, model_file)
        obj = torch.load(file_path)

        self.model = model
        self.model.load_state_dict(obj['model'])

        self.loss_function = Loss('L1')
        self.transform = StateTransformer(shuffle_players=False, shuffle_pokemon=False, shuffle_moves=False)

    def __call__(self):
        self.model.eval()

        path = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_games'

        # collect game files
        files = [os.path.join(path, file_name)
                 for file_name in os.listdir(path)]

        print(f'{len(files)} files found')

        min_game_length = 3

        # iterate over all games
        for file in files:
            with open(file, 'r') as f_in:
                states = json.load(f_in)

                if len(states) < min_game_length:
                    continue

                # get rating and battle ID
                rating = states[0]['average_rating']
                battle_id = states[0]['roomid']

                i = 0
                game_loss = 0.0

                # iterate starting at 1 to avoid team preview states
                for i in range(1, len(states)):

                    # percentage of the game completed at current state
                    percentage_complete = int((i - 1) / (len(states) - 1) * 100)

                    # transform state into dict of tensors
                    state = self.transform(states[i])

                    # network is hardcoded for batches, so create batch of size 1
                    fields = torch.unsqueeze(state['fields'], 0)
                    sides = torch.unsqueeze(state['sides'], 0)
                    pokemon = {key: torch.unsqueeze(value, 0) for key, value in state['pokemon'].items()}

                    # forward pass
                    prediction = self.model(fields, sides, pokemon)

                    # compute loss
                    loss = self.loss_function(prediction, state['result'])
                    game_loss += loss.item()

                    # print(f'{percentage_complete}: {torch.squeeze(prediction).item():.3f}')
                print(f'average loss: {game_loss / (i-1):.3f} -- ID: {battle_id}')

                    # store prediction, label, loss and %completed
                # store game predictions, labels, losses and %completed



