import os
import json
import torch


class Tester:
    def __init__(self, model, model_file):
        folder_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/'
        file_path = os.path.join(folder_path, model_file)
        obj = torch.load(file_path)

        self.model = model
        self.model.load_state_dict(obj['model'])

    def __call__(self):
        path = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_games'

        files = [os.path.join(path, file_name)
                 for file_name in os.listdir(path)]

        print(f'{len(files)} files found')

        for file in files:
            with open(file, 'r') as f_in:
                states = json.load(f_in)

                for i, s in enumerate(states):
                    pass
                    # transform state
                    # forward pass
                    # compute loss
                    # store prediction, label, loss and %completed
                # store game predictions, labels, losses and %completed



