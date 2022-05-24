from tqdm import tqdm
import torch
import os
import time
import math

from data.data_loader import data_loader
from model.network import ValueNet, Hidden, Output
from model.loss import Loss
from data.transformer import StateTransformer


"""
---------------------- TO DO ----------------------

1. re-parse games, create test split and create state files

2. check encode len(..) +1 vs len(..) +2
3. change model to where each pokemon is passed individually

4. fix imports?

"""


def main():
    # initialize model
    hidden_layers = Hidden(input_size=3801, output_size=512)
    output_layer = Output(input_size=512)
    model = ValueNet(hidden_layers, output_layer)

    # train model
    train = Trainer(model, save_model=True)
    train()


class Trainer:
    def __init__(self, model, n_epochs=20, batch_size=128, n_workers=4, lr=0.0003, save_model=False):
        self.model = model

        # training settings
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_workers = n_workers

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_function = Loss()

        # data settings
        train_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/all_rated_1200+/training_states/train_test/'
        val_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/all_rated_1200+/training_states/val_test/'

        shuffle_transform = StateTransformer(shuffle_players=True, shuffle_pokemon=True)
        no_shuffle_transform = StateTransformer(shuffle_players=False, shuffle_pokemon=False)

        self.train_samples = sum([10000 for f in os.listdir(train_path)])
        self.validation_samples = sum([10000 for f in os.listdir(val_path)])

        self.train_loader = data_loader(train_path, shuffle_transform, batch_size=batch_size, num_workers=n_workers)
        self.val_loader = data_loader(val_path, no_shuffle_transform, batch_size=batch_size, num_workers=n_workers)

        # misc. settings
        self.running_loss_freq = math.ceil(20_000 / self.batch_size)

        if save_model:
            self.save_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/'
        else:
            self.save_path = None

    def __call__(self):
        self.batch_count = 0

        # start epoch iteration
        start_time = time.time()
        for epoch in range(1, self.n_epochs + 1):
            out_str = ""

            # training loop
            start_epoch_time = time.time()
            epoch_train_loss = self.train()
            out_str += "Epoch {} | Train loss: {:.3f} | ".format(
                epoch, epoch_train_loss
            )
            print(out_str, end="")

            # validating loop
            epoch_val_loss = self.validate()
            out_str += "Val loss: {:.3f} | Epoch time: {:.1f}s".format(
                epoch_val_loss, time.time() - start_epoch_time
            )
            print("\r" + out_str + '\n')

            # save model
            if self.save_path:
                obj = {
                    "epoch": epoch,
                    "optimizer": self.optimizer.state_dict(),
                    "model": self.model.state_dict(),
                }
                torch.save(obj, f"{self.save_path}epoch_{epoch}.pt")

        print('Finished training')
        print("Total time: {:.1f}s".format(time.time() - start_time))

    def train(self):
        self.model.train()

        # init progress bar
        pbar = (
            tqdm(total=self.train_samples) if self.train_samples else tqdm()
        )

        running_loss = 0.0
        epoch_train_loss = 0.0

        i = 0

        # iterate training batches
        for i, sample in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad()
            out, labels = self.forward_pass(sample)
            loss = self.loss_function(out, labels)

            running_loss += loss.item()
            epoch_train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if i % self.running_loss_freq == 0:
                train_loss = running_loss / self.running_loss_freq
                running_loss = 0.0

                pbar.update(self.running_loss_freq * self.batch_size)
                pbar.set_description("Train loss: {:.3f}".format(train_loss))

        if self.train_samples:
            pbar.update(self.train_samples - pbar.n)

        pbar.close()

        self.batch_count += i
        epoch_train_loss = epoch_train_loss / i

        return epoch_train_loss

    def validate(self):
        self.model.eval()
        epoch_val_loss = 0.0
        i = 0

        for i, sample in enumerate(self.val_loader, start=1):
            with torch.no_grad():
                out, labels = self.forward_pass(sample)
                loss = self.loss_function(out, labels)
                epoch_val_loss += loss.item()

        epoch_val_loss = epoch_val_loss / i

        return epoch_val_loss

    def forward_pass(self, sample):
        fields = sample["fields"]
        sides = sample["sides"]
        pokemon = sample["pokemon"]
        result = sample["result"]

        out = self.model(fields, sides, pokemon)
        return out, result


if __name__ == '__main__':
    main()

