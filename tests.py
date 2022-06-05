import random
import pandas as pd
import numpy as np
import os
import json
import ujson
import csv
import time
import re
import shutil
import torch
import torch.nn as nn
import fasttext
from quantecon.game_theory.lemke_howson import lemke_howson
from quantecon.game_theory.normal_form_game import NormalFormGame


# def to_bimatrix(payoff_matrix):
#     length, width = payoff_matrix.shape
#     bimatrix = np.zeros((length, width, 2))
#     for i in range(length):
#         for j in range(width):
#             bimatrix[i][j][0], bimatrix[i][j][1] = payoff_matrix[i][j], -payoff_matrix[i][j]
#     return bimatrix
#
#
# payoff_matrix = np.array([[2, -5],
#                           [12, 2]])
#
# game = NormalFormGame(to_bimatrix(payoff_matrix))
# user_strategy, opp_strategy = lemke_howson(game, init_pivot=0, max_iter=1000000, capping=None, full_output=False)
#
# print(payoff_matrix.T)
# print()
# print('user:', user_strategy)
# print('opp:', opp_strategy)
# print()
# print("----------------")
#
#
# user = game.players[0]
# values = user.payoff_vector(opp_strategy)
# average_value = sum(values * user_strategy)
#
# print(average_value)
#
# AA = user_strategy[0] * opp_strategy[0] * payoff_matrix[0][0]
# BA = user_strategy[1] * opp_strategy[0] * payoff_matrix[1][0]
# AB = user_strategy[0] * opp_strategy[1] * payoff_matrix[0][1]
# BB = user_strategy[1] * opp_strategy[1] * payoff_matrix[1][1]
#
# average_value2 = AA + BA + AB + BB
# print(average_value2)
#


############################## CONVOLUTION TEST ###############################


pokemon1 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float)
pokemon2 = torch.tensor([1, 2, 3, 4], dtype=torch.float)
pokemon3 = torch.tensor([10, 20, 30, 40], dtype=torch.float)

pokemon4 = torch.tensor([100, 200, 300, 400], dtype=torch.float)
pokemon5 = torch.tensor([1000, 2000, 3000, 5000], dtype=torch.float)
pokemon6 = torch.tensor([10000, 20000, 30000, 50000], dtype=torch.float)

player1 = torch.stack((pokemon1, pokemon2, pokemon3))
player2 = torch.stack((pokemon4, pokemon5, pokemon6))

pokemon = torch.stack((player1, player2)).unsqueeze(0)



def matchup_layer(x):

    conv2d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 8), stride=1)

    p1_side = x[:, 0]
    p2_side = x[:, 1]

    matchups = []
    for i in range(x.shape[2]):
        matchups.append(
            torch.cat((p1_side, torch.roll(p2_side, i, dims=1)), dim=2)
        )

    # apply kernel over each of the 36 matchups
    x = torch.cat(matchups, dim=1).unsqueeze(1)
    x = conv2d(x)

    return x


x = matchup_layer(pokemon)
x = x.squeeze(3)
x = torch.transpose(x, dim0=1, dim1=2)
# [bs, 9, 16]

bs, w, h = x.shape
x = torch.reshape(x, (bs, 3, 3, h))
# [bs, 3, 3, 16]


p1_side = torch.reshape(x, (bs, 3, 3 * h))
# [bs, 3, 48]

p2_side = torch.reshape(
    torch.stack(
        [x[:, :, i] for i in range(3)],
        dim=1
    ),
    (bs, 3, 3 * h)
)


print(p1_side[0, 1, 16:32])
print()
print(p2_side[0, 1, 16:32])


""" 
[bs, 16, 9, 1]
->
[bs, 2, 3, 48]

-------------------------

[bs, 16, 9, 1]
-> squeeze
[bs, 16, 9]
-> transpose
[bs, 9, 16]
1, 1
1, 2
1, 3
2, 1
2, 2
2, 3
3, 1
3, 2
3, 3
-> reshape
[bs, 3, 3, 16]
[bs, p1mons, p2mons, 16]



"""

