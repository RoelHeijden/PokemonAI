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




score = 1

depth = 2


score = abs(score - 0.001 * (depth - 1))

print(score)




