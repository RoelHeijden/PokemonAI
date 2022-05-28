import os

from model.network import ValueNet
from model_backup.network_0759 import ValueNet as BestNet
from training import Trainer
from testing import Tester


"""
---------------------- TO DO ----------------------


1. Process more data:
    1. Parse ALL (non-dmax) games
    2. Split train test 90/10, sorted on folders
    3. Create test batches

2. expand test module with:
    - average accuracy per %completed (per rating)
    - average accuracy per %completed (per game length)
    
    - average eval per %complete (per label)
    - eval distribution per %completed (per label)
    
    - average eval change per %completed
    - eval change distribution per %completed

    - eval per %completed of a single sampled game

3. incorporate model into KetchAI

"""


def main():
    # mode = 'train'
    mode = 'test'

    folder = 'old_1250+'

    # model_name = 'training_dump/1250+_epoch_17.pt'
    model_name = 'relevant_models/network_0759_1250+_epoch_23.pt'

    # initialize model
    model = ValueNet()

    # train model
    if mode == 'train':
        trainer = Trainer(model)
        trainer.train(folder)

    # test model
    if mode == 'test':
        model_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/'
        tester = Tester(model, os.path.join(model_folder, model_name))
        # tester.test_states(folder)
        tester.test_games(folder)


if __name__ == '__main__':
    main()

