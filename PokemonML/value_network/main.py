import os

from model.network import ValueNet
from training import Trainer
from testing import Tester


"""
---------------------- TO DO ----------------------

1. Process more data:
    1. Parse ALL (non-dmax) games
    2. Split train test 90/10, sorted on folders
    3. Create train batches: sample more states (2-6?) per game depending on length, write to multiple files?
    4. Create test batches

2. expand test module with:
    - average eval per %complete (per label)
    - eval distribution per %completed (per label)
    - average eval change per %completed
    - average accuracy per %completed (per rating)
    - eval per turn of a single game

3. find right model parameters with full dataset testing
4. incorporate model into KetchAI
"""


def main():
    mode = 'train'
    # mode = 'test'

    folder = '1250+'

    model_name = 'epoch_17.pt'

    # initialize model
    model = ValueNet()

    # train model
    if mode == 'train':
        trainer = Trainer(model)
        trainer.train(folder)

    # test model
    if mode == 'test':
        model_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/training_dump/'
        tester = Tester(model, os.path.join(model_folder, model_name))
        tester.test_states(folder)
        tester.test_games(folder)


if __name__ == '__main__':
    main()

