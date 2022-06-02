import os
import torch

from training import Trainer
from testing import Tester

from model.network import ValueNet


"""

ValueNet0729 1100+, LR=2e-4, gamma=0.95, 64/16/16/16, Pokemon/side/field: 192 drop(0.1) * 12 bn, state: 1024 drop(0.3), 512 drop(0.3), 128 drop(0.1) -- epoch 30
    21000 games evaluated
    Sampled accuracy: 0.729
    Overall accuracy: 0.714
    Accuracy per %completed: 0.582 | 0.6 | 0.614 | 0.631 | 0.647 | 0.667 | 0.682 | 0.689 | 0.705 | 0.718 | 0.733 | 0.746 | 0.759 | 0.775 | 0.796 | 0.819 | 0.845 | 0.871 | 0.899 | 0.931
                            
ValueNet0723 1100+, LR=2e-4, gamma=0.90, 64/16/16/16, Pokemon/side/field: 192 drop(0.1) * 12 bn, state: 1024 drop(0.3), 512 drop(0.3), 128 drop(0.1) -- epoch 12
    14900 games evaluated
    Sampled accuracy: 0.723
    Overall accuracy: 0.713
    Accuracy per %completed: 0.576 | 0.597 | 0.615 | 0.629 | 0.635 | 0.654 | 0.674 | 0.685 | 0.702 | 0.716 | 0.73 | 0.746 | 0.763 | 0.782 | 0.803 | 0.823 | 0.845 | 0.873 | 0.905 | 0.931

ValueNet 1100+, batch size 256, LR=2e-4, gamma=0.98, 64/16train/16train/16train, pokemon: 192 drop(0.4), hidden: 1024 drop(0.4), 512 drop(0.4) -- epoch 14:
test states:
    all: 0.727
    1100+: 0.719
    1300+: 0.712
    1500+: 0.707
    1700+: 0.719*


ValueNet 1100+, batch size 256, LR=15e-5, gamma=0.96, 32/32/32train/32train, pokemon: 192 drop(0.3), hidden: 2048 drop(0.5), 512 drop(0.2), 128 drop(0.2)


try cyclic LR??


"""


def main():
    model = ValueNet()
    rating = '1100+'
    folder = 'training_dump'
    model_name = '1100+_epoch_14'

    train(model, rating, folder, model_name, train_new=True)
    # train(model, rating, folder, model_name, train_new=False)
    # test(model, rating, folder, model_name, test_states=True)
    # test(model, rating, folder, model_name, test_games=True)


def train(model, rating, folder, model_name, train_new=True):
    model_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/'
    model_file = os.path.join(model_folder, folder, model_name + '.pt')

    data_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/training_states/'
    save_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/training_dump/'

    n_epochs = 50
    batch_size = 256
    lr = 15e-5
    lr_decay = 0.96
    lr_decay_steps = 1

    trainer = Trainer(
        model=model,
        data_folder=os.path.join(data_folder, rating),
        save_path=save_path,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        lr_decay=lr_decay,
        lr_decay_steps=lr_decay_steps,
        update_every_n_batches=50
    )

    if train_new:
        trainer.train(run_name=rating, start_epoch=1)

    else:
        checkpoint = torch.load(model_file)
        trainer.model.load_state_dict(checkpoint['model'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

        for g in trainer.optimizer.param_groups:
            g['lr'] = lr

        trainer.train(run_name=model_name + '_' + rating, start_epoch=epoch+1)


def test(model, rating, folder, model_name, test_games=False, test_states=False, test_embeddings=False):
    model_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/'
    model_file = os.path.join(model_folder, folder, model_name + '.pt')
    tester = Tester(model, model_file)

    if test_states:
        tester.test_states(rating)

    if test_games:
        tester.test_games(rating)

    if test_embeddings:
        tester.test_embeddings()


if __name__ == '__main__':
    main()

