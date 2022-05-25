from tqdm import tqdm
import torch
import os
import time

from data.data_loader import data_loader
from model.loss import Loss
from data.transformer import StateTransformer


class Trainer:
    def __init__(self,
                 model,
                 n_epochs=100,
                 batch_size=128,
                 update_every=5,
                 lr=1e-4,
                 lr_gamma=0.9,
                 lr_decay_steps=5,
                 num_workers=4,
                 save_model=True
                 ):

        self.model = model

        # training settings
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, lr_decay_steps, gamma=lr_gamma)
        self.loss_function = Loss()

        # data settings
        train_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/training_states/1700+/'
        val_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/training_states/validating/'

        shuffle_transform = StateTransformer(shuffle_players=True, shuffle_pokemon=True, shuffle_moves=True)
        no_shuffle_transform = StateTransformer(shuffle_players=False, shuffle_pokemon=False, shuffle_moves=False)

        file_size = 10000
        last_file_size = 10000  # 7906

        self.train_samples = sum([file_size for f in os.listdir(train_path)]) - (file_size - last_file_size)
        self.validation_samples = sum([file_size for f in os.listdir(val_path)])

        self.train_loader = data_loader(train_path, shuffle_transform, batch_size=batch_size, num_workers=num_workers)
        self.val_loader = data_loader(val_path, no_shuffle_transform, batch_size=batch_size, num_workers=num_workers)

        # misc. settings
        self.update_every_n_batches = update_every

        if save_model:
            self.save_path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/models/'
        else:
            self.save_path = None

    def __call__(self):
        self.batch_count = 0

        # start epoch iteration
        start_time = time.time()
        for epoch in range(1, self.n_epochs + 1):
            start_epoch_time = time.time()
            out_str = ''

            # training loop
            epoch_train_loss = self.train()
            current_lr = self.optimizer.param_groups[0]['lr']
            out_str += f"Epoch {epoch} | LR: {current_lr} | Train loss: {epoch_train_loss:.3f} | "
            # print(out_str, end="")
            #
            # # validating loop
            # epoch_val_loss = self.validate()
            # out_str += "Val loss: {:.3f} | Epoch time: {:.1f}s".format(
            #     epoch_val_loss, time.time() - start_epoch_time
            # )
            print("\r" + out_str + '\n')

            # change learning rate each n epochs
            self.scheduler.step()

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

        # iterate batches
        for i, sample in enumerate(self.val_loader, start=1):
            with torch.no_grad():

                # forward pass
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

