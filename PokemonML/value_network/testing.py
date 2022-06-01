import torch
import torch.nn as nn
import json
import random
import os
import numpy as np
import scipy.spatial.distance as distance

from PokemonML.value_network.data.categories import (
    SPECIES,
    MOVES,
    ITEMS,
    ABILITIES
)
from data.transformer import StateTransformer
from data.data_loader import data_loader


class Tester:
    def __init__(self, model, model_file):
        self.states_folder = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_states/'
        self.games_folder = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_games/'

        self.transform = StateTransformer(shuffle_players=True, shuffle_pokemon=True, shuffle_moves=True)

        self.model = model
        self.model.load_state_dict(torch.load(model_file)['model'])

    def test_states(self, folder):
        self.model.eval()

        # init data loader
        data_path = os.path.join(self.states_folder, folder)
        dataloader = data_loader(data_path, self.transform, batch_size=1, shuffle=False)

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

    def test_games(self, folder):
        self.model.eval()

        # init arrays tracking the data
        self.step_size = 5
        self.n_correct_array = np.zeros(int(100 / self.step_size))
        self.n_evals_array = np.zeros(int(100 / self.step_size))
        self.eval_change_array = np.zeros(int(100 / self.step_size))

        self.n_evals = 0
        self.n_correct = 0
        self.eval_change = 0

        self.sampled_n_correct = 0
        self.sampled_n_evals = 0

        # init data files
        data_path = os.path.join(self.games_folder, folder)
        files = sorted([os.path.join(data_path, file_name)
                        for file_name in os.listdir(data_path)])

        random.shuffle(files)
        min_game_length = 3

        # iterate over and open each game in the test games folder
        for n_games, f in enumerate(files, start=1):
            with open(f) as f_in:
                game_states = json.load(f_in)

                # skip game if too short
                if len(game_states) < min_game_length:
                    continue

                results = []
                evaluations = []
                percentage_completed = []

                # iterate all game states, starting at 1 to avoid team preview states
                for i in range(1, len(game_states)):
                    state = game_states[i]

                    # run state through evaluation network
                    evaluation, game_result = self._evaluate_state(state)

                    # store evaluation
                    evaluations.append(round(evaluation, 3))

                    # store game result each state because the player pov can be shuffled in transformer
                    results.append(game_result)

                    # store game % completed
                    percentage_completed.append((i - 1) / (len(game_states) - 1) * 100)

                    self.n_evals += 1
                    self.n_correct += int(round(evaluation) == game_result)

                # evaluate one random state to measure average accuracy
                # picking one random state per game to avoid longer games skewing the sampling
                random_state = game_states[random.randint(1, len(game_states) - 1)]
                evaluation, game_result = self._evaluate_state(random_state)
                self.sampled_n_evals += 1
                self.sampled_n_correct += int(round(evaluation) == game_result)

                # store data into arrays of size (100 / step_size)
                self._map_to_array(percentage_completed, results, evaluations)

                # plot data
                if n_games % 100 == 0:
                    self._plot_performance(n_games)

    def _map_to_array(self, percentage_completed, results, evaluations):
        # iterate over each array slot
        for i in range(2, 100, self.step_size):

            # map the game%completed to 20 indices, representing a percentage range (2, 7, 12, ..., 97)
            nearest_percentage_index = min(
                range(len(percentage_completed)),
                key=lambda j: abs(percentage_completed[j] - i)
            )

            # compare evaluation with result
            result = results[nearest_percentage_index]
            evaluation = evaluations[nearest_percentage_index]
            correct_pred = int(round(evaluation) == result)

            # store results
            array_idx = int((i - 2) / self.step_size)
            self.n_correct_array[array_idx] += correct_pred
            self.n_evals_array[array_idx] += 1

    def _plot_performance(self, n_games):
        accuracies = np.round(self.n_correct_array / self.n_evals_array, decimals=3)
        print(f'{n_games} games evaluated')
        print(f'Sampled accuracy: {self.sampled_n_correct / self.sampled_n_evals:.3f}')
        print(f'Overall accuracy: {self.n_correct / self.n_evals:.3f}')
        print(f'Accuracy per %completed: {" | ".join(str(x) for x in accuracies)}\n')

    def _evaluate_state(self, state):
        # transform state into dict of tensors
        tensor_dict = self.transform(state)

        # get label
        game_result = int(tensor_dict['result'].item())

        # network is hardcoded for batches, so create batch of size 1
        fields = torch.unsqueeze(tensor_dict['fields'], 0)
        sides = torch.unsqueeze(tensor_dict['sides'], 0)
        pokemon = {key: torch.unsqueeze(value, 0) for key, value in tensor_dict['pokemon'].items()}

        # forward pass
        evaluation = self.model(fields, sides, pokemon).item()

        return evaluation, game_result

    def test_embeddings(self):
        species_weights = self.model.species_embedding.weight
        move_weights = self.model.move_embedding.weight
        item_weights = self.model.item_embedding.weight
        ability_weights = self.model.ability_embedding.weight

        species_embedding = nn.Embedding.from_pretrained(species_weights)
        move_embedding = nn.Embedding.from_pretrained(move_weights)
        item_embedding = nn.Embedding.from_pretrained(item_weights)
        ability_embedding = nn.Embedding.from_pretrained(ability_weights)

        def find_most_similar(target, embedding, category, n=10):
            target_tensor = torch.LongTensor([category.get(target)])
            target_vec = embedding(target_tensor).squeeze()

            comparisons = []

            for cat in category:

                if cat == target:
                    continue

                to_compare = torch.LongTensor([category.get(cat)])
                compare_vec = embedding(to_compare).squeeze()

                score = distance.cosine(target_vec, compare_vec)
                comparisons.append((score, cat))

            comparisons.sort(key=lambda tup: tup[0], reverse=False)

            print(target)
            for i in range(n):
                print(comparisons[i][0], comparisons[i][1])
            print()

        # categories to inspect
        species = 'charizard'
        move = 'flamethrower'
        item = 'choicescarf'
        ability = 'flashfire'

        find_most_similar(species, species_embedding, SPECIES)
        find_most_similar(move, move_embedding, MOVES)
        find_most_similar(item, item_embedding, ITEMS)
        find_most_similar(ability, ability_embedding, ABILITIES)



