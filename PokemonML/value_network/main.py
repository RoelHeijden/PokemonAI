import os
import torch

from training import Trainer
from testing import Tester

from model.network import ValueNet


"""
ValueNet 1700+, 256, LR=1e-4, gamma=0.98, 64/16/16/16, pokemon/side: 192, hidden: 1024 drop(0.4), 512 drop(0.4), 128
    Epoch 1 | Train loss: 0.708 | Val loss: 0.690
    Epoch 2 | Train loss: 0.698 | Val loss: 0.689
    Epoch 3 | Train loss: 0.690 | Val loss: 0.687
    Epoch 4 | Train loss: 0.681 | Val loss: 0.674
    Epoch 5 | Train loss: 0.664 | Val loss: 0.658 
    Epoch 6 | Train loss: 0.638 | Val loss: 0.632
    Epoch 7 | Train loss: 0.610 | Val loss: 0.613 
    Epoch 8 | Train loss: 0.591 | Val loss: 0.605 
    Epoch 9 | Train loss: 0.578 | Val loss: 0.600
    Epoch 10 | Train loss: 0.571 | Val loss: 0.597
    Epoch 11 | Train loss: 0.565 | Val loss: 0.599

ValueNet 1700+, 256, LR=1e-4, gamma=0.98, 64/64/64/64, pokemon/side: 192, hidden: 1024 drop(0.4), 512 drop(0.4), 128
    Epoch 1 | Train loss: 0.713 | Val loss: 0.695
    Epoch 2 | Train loss: 0.700 | Val loss: 0.693
    Epoch 3 | Train loss: 0.695 | Val loss: 0.689
    Epoch 4 | Train loss: 0.687 | Val loss: 0.680
    Epoch 5 | Train loss: 0.673 | Val loss: 0.662
    Epoch 6 | Train loss: 0.649 | Val loss: 0.640
    Epoch 7 | Train loss: 0.624 | Val loss: 0.620
    Epoch 8 | Train loss: 0.598 | Val loss: 0.610
    Epoch 9 | Train loss: 0.578 | Val loss: 0.605
    Epoch 10 | Train loss: 0.566 | Val loss: 0.600 
    Epoch 11 | Train loss: 0.557 | Val loss: 0.594
    Epoch 12 | Train loss: 0.549 | Val loss: 0.593
    Epoch 13 | Train loss: 0.544 | Val loss: 0.594
    
ValueNet 1700+, 256, LR=1e-4, gamma=0.98, 64/16/16/16, matchup 4, pokemon/side: 192, hidden: 1024 drop(0.4), 512 drop(0.4), 128
    Epoch 13 | Train loss: 0.555 | Val loss: 0.582


ValueNet 1500+, 256, LR=2e-4, gamma=0.95, 64/16/16/16, matchup 5, pokemon/side: 192, hidden: 1024 drop(0.4), 512 drop(0.4), 128
Baseline:
    Epoch 1 | Train loss: 0.664 | Val loss: 0.596
    Epoch 2 | Train loss: 0.579 | Val loss: 0.577
    Epoch 3 | Train loss: 0.566 | Val loss: 0.572 
    Epoch 4 | Train loss: 0.561 | Val loss: 0.570
    Epoch 5 | Train loss: 0.558 | Val loss: 0.568 
    Epoch 6 | Train loss: 0.555 | Val loss: 0.567
    Epoch 7 | Train loss: 0.552 | Val loss: 0.568
    Epoch 8 | Train loss: 0.549 | Val loss: 0.566 
    Epoch 9 | Train loss: 0.546 | Val loss: 0.564
    Epoch 10 | Train loss: 0.543 | Val loss: 0.564 
    Epoch 11 | Train loss: 0.541 | Val loss: 0.563 X
    Epoch 12 | Train loss: 0.538 | Val loss: 0.566 
    Epoch 13 | Train loss: 0.536 | Val loss: 0.564
    Epoch 14 | Train loss: 0.533 | Val loss: 0.566 


2d pokemon batchnorm, 2d matchup batchnorm, pokemon_out -> matchup_layer
2d pokemon batchnorm, 2d matchup batchnorm, pokemon_in -> matchup_layer
1d pokemon batchnorm, 2d matchup batchnorm, ...




try:
    embedding sizes
    drop rates
    convolution
    
"""


def main():
    model = ValueNet()
    rating = '1500+'
    folder = 'training_dump'
    model_name = '1700+_epoch_13'

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
    lr = 2e-4
    lr_decay = 0.95
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
        update_every_n_batches=20,
        file_size=10000,
        buffer_size=5000,
        num_workers=4,
        shuffle_data=True,
        shuffle_players=True,
        shuffle_pokemon=True,
        shuffle_moves=True,
        save_model=True
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

