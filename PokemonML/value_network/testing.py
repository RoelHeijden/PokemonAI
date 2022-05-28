import torch
import json
import random
import os
import numpy as np

from data.transformer import StateTransformer
from data.data_loader import data_loader


class Tester:
    def __init__(self, model, model_file):
        self.states_folder = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_states/'
        self.games_folder = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_games/'

        self.transform = StateTransformer(shuffle_players=False, shuffle_pokemon=False, shuffle_moves=False)

        self.model = model
        self.model.load_state_dict(torch.load(model_file)['model'])

    def test_states(self, folder='all'):
        self.model.eval()

        # init data loader
        data_path = os.path.join(self.states_folder, folder)
        dataloader = data_loader(data_path, self.transform, batch_size=1)

        # track amount of evaluations and amount of correct classifications
        n_evaluations = 0
        correct_classifications = 0

        # iterate over all games
        for state in dataloader:

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

            if n_evaluations % 1000 == 0:
                print(f'\raverage accuracy: {correct_classifications / n_evaluations:.3f}', end='')

        print(f'\r{n_evaluations} states evaluated')
        print(f'average accuracy: {correct_classifications / n_evaluations:.3f}\n')

    def test_games(self, folder='all'):
        self.model.eval()

        # init data files
        data_path = os.path.join(self.games_folder, folder)
        files = sorted([os.path.join(data_path, file_name)
                        for file_name in os.listdir(data_path)])

        random.shuffle(files)
        min_game_length = 3

        step_size = 5
        correct_preds = np.zeros(int(100 / step_size))
        n_preds = np.zeros(int(100 / step_size))
        turn_changes = np.zeros(int(100 / step_size))

        for n_games, f in enumerate(files, start=1):
            with open(f) as f_in:
                game_states = json.load(f_in)

                if len(game_states) < min_game_length:
                    continue

                results = []
                evaluations = []
                percentage_completed = []

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

                    # store results each state because the player pov may be shuffled in transformer
                    game_result = tensor_dict['result'].item()
                    results.append(game_result)

                    # store game % completed
                    percentage_completed.append((i - 1) / (len(game_states) - 1) * 100)

                # evenly data
                for i in range(2, 100, step_size):

                    # map the game%completed to 20 indices, representing a percentage range (2, 7, 12, ..., 97)
                    nearest_percentage_index = min(
                        range(len(percentage_completed)),
                        key=lambda j: abs(percentage_completed[j]-i)
                    )

                    # compare evaluation with result
                    result = results[nearest_percentage_index]
                    evaluation = evaluations[nearest_percentage_index]
                    correct_pred = int(round(evaluation) == result)

                    # store results
                    array_idx = int((i - 2) / step_size)
                    correct_preds[array_idx] += correct_pred
                    n_preds[array_idx] += 1

                if n_games % 20 == 0:
                    accuracies = np.round(correct_preds / n_preds, decimals=3)
                    print(f'\rAccuracy per % game played: {" | ".join(str(x) for x in accuracies)}', end='')

        accuracies = np.round(correct_preds / n_preds, decimals=3)
        print(f'\rAccuracy per % game played: {" | ".join(str(x) for x in accuracies)}\n')










