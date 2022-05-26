import torch

from model.loss import Loss
from data.transformer import StateTransformer
from data.data_loader import data_loader


class Tester:
    def __init__(self, model, model_file):
        test_states_folder = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_states/1500+/'

        self.transform = StateTransformer(shuffle_players=False, shuffle_pokemon=False, shuffle_moves=False)
        self.data_loader = data_loader(test_states_folder, self.transform, batch_size=1)

        self.model = model
        self.model.load_state_dict(torch.load(model_file)['model'])

        self.loss_function = Loss('L1')

    def __call__(self):
        self.model.eval()

        min_game_length = 3

        total_loss = 0.0
        n_evaluations = 0
        correct_classifications = 0

        # iterate over all games
        for state in self.data_loader:

            label = torch.squeeze(state['result'])
            fields = state['fields']
            sides = state['sides']
            pokemon = state['pokemon']

            # forward pass
            prediction = self.model(fields, sides, pokemon)

            # compute loss
            loss = self.loss_function(prediction, label)
            total_loss += loss.item()
            n_evaluations += 1

            # get classification error
            pred_result = int(round(prediction.item()))
            actual_result = int(label.item())
            if pred_result == actual_result:
                correct_classifications += 1

            if n_evaluations % 2000 == 0:
                print(f'{n_evaluations} states evaluated')
                print(f'average L1 loss: {total_loss / n_evaluations:.3f}')
                print(f'accuracy: {correct_classifications / n_evaluations:.3f}\n')



