from model.network import ValueNet
from training import Trainer
from testing import Tester


"""
---------------------- TO DO ----------------------

1. set decaying lr

2. expand test module

3. find google cloud gpu details
4. set cuda

5. find right model parameters
6. incorporate model into KetchAI
"""


def main():

    # full input size: ((pokemon_size + pokemon_attributes_size) * 6 + side_attributes_size) * 2 + field_attributes_size
    pokemon_size = (64 + 64 + 32 + 4 * 128) + 88
    field_size = 21
    side_size = 18

    # initialize model
    model = ValueNet(
            field_size=field_size,
            side_size=side_size,
            pokemon_size=pokemon_size
    )

    # # train model
    train = Trainer(model)
    train()

    # test model
    # model_file = 'epoch_30.pt'
    # test = Tester(model, model_file)
    # test()


if __name__ == '__main__':
    main()

