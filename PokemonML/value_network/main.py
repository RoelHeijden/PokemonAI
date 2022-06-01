import os
import torch

from training import Trainer
from testing import Tester

from model.network import ValueNet


"""
---------------------- TO DO ----------------------

test continued training
create new example state


1. tree search:
    - opp choice lock in state sim get_all_options()
    - prune on probabilities (pass probability to get_score(), check threshold)
        - scale threshold based on n_pokemon alive
        - add crits to threshold (dont branch from crits)
        - add damage rolls to threshold (consider min/max rolls as 12% chance, avg as 75%?)
    - search torch nn pruning

2. model selection:
    - train 1000+, 1100+, 1300+ and 1500+ models
    - test each model on each rating

3. expand test module with:
    - average accuracy per %completed (per rating)
    - average accuracy per %completed (per game length)
    
    - average eval per %complete (per label)
    - eval distribution per %completed (per label)
    
    - average eval change per %completed
    - eval change distribution per %completed

    - eval per %completed of a single sampled game (with notable game moments noted, like critical hits, reads)

4. Finish to_dict() variables for mutator.state


---------------------- TRAINING ----------------------

ValueNet0729 1100+, LR=2e-4, gamma=0.95, 64/16/16/16, Pokemon/side/field: 192 drop(0.1) * 12 bn, state: 1024 drop(0.3), 512 drop(0.3), 128 drop(0.1) -- epoch 30?
    21000 games evaluated
    Sampled accuracy: 0.729
    Overall accuracy: 0.714
    Accuracy per %completed: 0.582 | 0.6 | 0.614 | 0.631 | 0.647 | 0.667 | 0.682 | 0.689 | 0.705 | 0.718 | 0.733 | 0.746 | 0.759 | 0.775 | 0.796 | 0.819 | 0.845 | 0.871 | 0.899 | 0.931
                            
ValueNet0723 1100+, LR=2e-4, gamma=0.90, 64/16/16/16, Pokemon/side/field: 192 drop(0.1) * 12 bn, state: 1024 drop(0.3), 512 drop(0.3), 128 drop(0.1) -- epoch 12
    14900 games evaluated
    Sampled accuracy: 0.723
    Overall accuracy: 0.713
    Accuracy per %completed: 0.576 | 0.597 | 0.615 | 0.629 | 0.635 | 0.654 | 0.674 | 0.685 | 0.702 | 0.716 | 0.73 | 0.746 | 0.763 | 0.782 | 0.803 | 0.823 | 0.845 | 0.873 | 0.905 | 0.931

ValueNet0721 1100+, LR=15e-5, gamma=0.98, 24/24/24train/24train, reserve: 128 drop(0.02), pokemon: 256 drop(0.02), hidden: 1024 drop(0.05), 512 drop(0.05) -- epoch: 13
    14000 games evaluated
    Sampled accuracy: 0.721
    Overall accuracy: 0.707
    Accuracy per %completed: 0.577 | 0.593 | 0.612 | 0.628 | 0.64 | 0.663 | 0.676 | 0.684 | 0.697 | 0.709 | 0.725 | 0.739 | 0.756 | 0.767 | 0.792 | 0.816 | 0.839 | 0.868 | 0.896 | 0.926

ValueNet 1300+, batch size 128, LR=1e-4, gamma=0.98, 64/16/16train/16train, reserve: 192 drop(0.4), pokemon: 256 drop(0.1), hidden: 1024 drop(0.4), 512 drop(0.4):

"""


def main():
    model = ValueNet()
    folder = '1300+'
    model_name = 'training_dump/1300+_epoch_10.pt'

    train(model, folder, model_name, train_new=True)
    # train(model, folder, model_name, train_new=False)
    # test(model, folder, model_name, test_states=True)
    # test(model, folder, model_name, test_games=True)


def train(model, folder, model_name, train_new=True):
    model_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/'
    model_file = os.path.join(model_folder, model_name)

    data_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/training_states/'
    save_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/training_dump/'

    n_epochs = 50
    batch_size = 128
    lr = 1e-4
    lr_decay = 0.98

    trainer = Trainer(
        model=model,
        data_folder=os.path.join(data_folder, folder),
        save_path=save_path,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        lr_decay=lr_decay,
    )

    if train_new:
        trainer.train(run_name=folder, start_epoch=1)

    else:
        checkpoint = torch.load(model_file)
        trainer.model.load_state_dict(checkpoint['model'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

        trainer.train(run_name=folder + '_' + epoch + '_continued', start_epoch=epoch)


def test(model, folder, model_name, test_games=False, test_states=False, test_embeddings=False):
    model_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/'
    model_file = os.path.join(model_folder, model_name)
    tester = Tester(model, model_file)

    if test_states:
        tester.test_states(folder)

    if test_games:
        tester.test_games(folder)

    if test_embeddings:
        tester.test_embeddings()


if __name__ == '__main__':
    main()

