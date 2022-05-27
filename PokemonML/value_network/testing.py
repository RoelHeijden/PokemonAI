import torch
import json
import random
import os
import numpy as np

from model.loss import Loss
from data.transformer import StateTransformer
from data.data_loader import data_loader


class Tester:
    def __init__(self, model, model_file):
        self.test_states_folder = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_states/1500+/'
        self.test_games_folder = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_games/'

        self.transform = StateTransformer(shuffle_players=False, shuffle_pokemon=False, shuffle_moves=False)
        self.data_loader = data_loader(self.test_states_folder, self.transform, batch_size=1)

        self.model = model
        self.model.load_state_dict(torch.load(model_file)['model'])

    def test_states(self):
        self.model.eval()

        n_evaluations = 0
        correct_classifications = 0

        # iterate over all games
        for state in self.data_loader:

            label = torch.squeeze(state['result'])
            fields = state['fields']
            sides = state['sides']
            pokemon = state['pokemon']

            # forward pass
            evaluation = self.model(fields, sides, pokemon)
            n_evaluations += 1

            # get classification
            pred_result = int(round(evaluation.item()))
            actual_result = int(label.item())
            if pred_result == actual_result:
                correct_classifications += 1

            if n_evaluations % 2000 == 0:
                print(f'{n_evaluations} states evaluated')
                print(f'average accuracy: {correct_classifications / n_evaluations:.3f}\n')

    def test_games(self):
        self.model.eval()

        files = sorted([os.path.join(self.test_games_folder, file_name)
                        for file_name in os.listdir(self.test_games_folder)])

        random.shuffle(files)

        correct_preds = np.zeros(100)
        n_predictions = np.zeros(100)

        for n_games, f in enumerate(files, start=1):
            with open(f) as f_in:
                game_states = json.load(f_in)

                evaluations = []

                # iterate states starting at 1 to avoid team preview states
                for i in range(1, len(game_states)):
                    state = game_states[i]

                    # transform state into dict of tensors
                    tensor_dict = self.transform(state)

                    # network is hardcoded for batches, so create batch of size 1
                    fields = torch.unsqueeze(tensor_dict['fields'], 0)
                    sides = torch.unsqueeze(tensor_dict['sides'], 0)
                    pokemon = {key: torch.unsqueeze(value, 0) for key, value in tensor_dict['pokemon'].items()}

                    # forward pass
                    evaluation = self.model(fields, sides, pokemon).item()
                    evaluations.append(round(evaluation, 3))

                    # compare prediction
                    game_result = tensor_dict['result'].item()
                    correct_pred = round(evaluation) == game_result

                    # store correct/incorrect predictions
                    percentage_completed = (i - 1) / (len(game_states) - 1) * 100

                changes = [abs(round(evaluations[i] - evaluations[i+1], 3))
                           for i in range(len(evaluations)-1)]

                if n_games % 100 == 0:
                    print(np.round(correct_preds / n_predictions, 2))
                    print()













