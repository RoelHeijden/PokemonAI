from tqdm import tqdm
import torch
import os
import time

from data.data_loader import data_loader
from model.loss import Loss
from data.transformer import StateTransformer


"""
1100+, LR=2e-4, gamma=0.95, 128/16/16/16, pokemon: 192 drop(p=.2), state: 1024 drop(p=.2), 512 drop(p=.2), 256 drop(p=.1)
epoch 3:
    Sampled accuracy: 0.713
    Overall accuracy: 0.700
    Accuracy per %completed: 0.562 | 0.582 | 0.599 | 0.613 | 0.629 | 0.642 | 0.667 | 0.678 | 0.689 | 0.701 | 0.716 | 0.729 | 0.741 | 0.764 | 0.781 | 0.809 | 0.832 | 0.858 | 0.888 | 0.918

1300+, LR=4e-4, gamma=0.90, 128/16/16/16, pokemon: 192 drop(p=.2), state: 4096 drop(p=.2), 1024 drop(p=.2), 256 drop(p=.1) -- epoch:


"""


class Trainer:
    def __init__(self, model, n_epochs=50, batch_size=252, lr=4e-4, lr_gamma=0.90, lr_decay_steps=1, weight_decay=0.001, save_model=True):

        # model
        self.model = model

        # train settings
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, lr_decay_steps, gamma=lr_gamma)

        self.loss_function = Loss()

        # data settings
        self.data_folder = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/training_states/'
        self.file_size = 10000

        self.buffer_size = 30000

        self.shuffle_transform = StateTransformer(shuffle_players=True, shuffle_pokemon=True, shuffle_moves=True)
        self.no_shuffle_transform = StateTransformer(shuffle_players=False, shuffle_pokemon=False, shuffle_moves=False)

        # misc. settings
        self.num_workers = 4
        self.update_every_n_batches = 10

        self.save_model = save_model
        self.save_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/training_dump/'

    def train(self, folder):

        # set data folders
        train_path = os.path.join(self.data_folder, folder, 'train')
        val_path = os.path.join(self.data_folder, folder, 'val')

        n_train_samples = sum([self.file_size for f in os.listdir(train_path)])

        # init data loaders
        train_loader = data_loader(
            train_path,
            self.shuffle_transform,
            self.batch_size,
            shuffle=True,
            buffer_size=self.buffer_size,
            num_workers=self.num_workers)

        val_loader = data_loader(
            val_path,
            self.no_shuffle_transform,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)

        # start epoch iteration
        start_time = time.time()
        for epoch in range(1, self.n_epochs + 1):
            start_epoch_time = time.time()
            out_str = ''

            # training loop
            epoch_train_loss = self.train_loop(train_loader, n_train_samples)
            current_lr = self.optimizer.param_groups[0]['lr']
            out_str += f"Epoch {epoch} | LR: {current_lr:.7f} | Train loss: {epoch_train_loss:.3f} | "
            print(out_str, end="")

            # validating loop
            epoch_val_loss = self.val_loop(val_loader)
            out_str += "Val loss: {:.3f} | Epoch time: {:.1f}s".format(
                epoch_val_loss, time.time() - start_epoch_time
            )
            print("\r" + out_str + '\n')

            # change learning rate each n epochs
            self.scheduler.step()

            # save model
            if self.save_model:
                obj = {
                    "epoch": epoch,
                    "optimizer": self.optimizer.state_dict(),
                    "model": self.model.state_dict(),
                }
                torch.save(obj, f"{self.save_path}{folder}_epoch_{epoch}.pt")

        print('Finished training')
        print("Total time: {:.1f}s".format(time.time() - start_time))

    def train_loop(self, train_loader, n_train_samples):
        self.model.train()

        # init progress bar
        pbar = (
            tqdm(total=n_train_samples) if n_train_samples else tqdm()
        )

        running_loss = 0.0
        epoch_train_loss = 0.0

        i = 0

        # iterate training batches
        for i, sample in enumerate(train_loader, start=1):
            self.optimizer.zero_grad()

            # forward pass
            out, labels = self.forward_pass(sample)
            loss = self.loss_function(out, labels)

            running_loss += loss.item()
            epoch_train_loss += loss.item()

            # backpropagation
            loss.backward()
            self.optimizer.step()

            # update progress bar
            if i % self.update_every_n_batches == 0:
                train_loss = running_loss / self.update_every_n_batches
                running_loss = 0.0

                pbar.update(self.update_every_n_batches * self.batch_size)
                pbar.set_description("Train loss: {:.3f}".format(train_loss))

        # complete progress bar
        if n_train_samples:
            pbar.update(n_train_samples - pbar.n)
        pbar.close()

        epoch_train_loss = epoch_train_loss / i

        return epoch_train_loss

    def val_loop(self, val_loader):
        self.model.eval()
        epoch_val_loss = 0.0
        i = 0

        # iterate batches
        for i, sample in enumerate(val_loader, start=1):
            with torch.no_grad():

                # forward pass
                out, labels = self.forward_pass(sample)
                loss = self.loss_function(out, labels)
                epoch_val_loss += loss.item()

        if i:
            epoch_val_loss = epoch_val_loss / i

        return epoch_val_loss

    def forward_pass(self, sample):
        fields = sample["fields"]
        sides = sample["sides"]
        pokemon = sample["pokemon"]
        result = sample["result"]

        out = self.model(fields, sides, pokemon)
        return out, result

