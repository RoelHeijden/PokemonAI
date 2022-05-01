from quantecon.game_theory.mclennan_tourky import mclennan_tourky
from quantecon.game_theory.normal_form_game import NormalFormGame
from quantecon.game_theory.lemke_howson import lemke_howson
import numpy as np


def to_bimatrix(payoff_matrix):
    length, width = payoff_matrix.shape
    bimatrix = np.zeros((length, width, 2))
    for i in range(length):
        for j in range(width):
            bimatrix[i][j][0], bimatrix[i][j][1] = payoff_matrix[i][j], -payoff_matrix[i][j]
    return bimatrix


payoff_matrix = np.array([[10, 0, 100],
                          [10, 60, 40]])

bimatrix = to_bimatrix(payoff_matrix.T)
game = NormalFormGame(bimatrix)
user_strategy, opp_strategy = lemke_howson(game, init_pivot=0, max_iter=1000000, capping=None, full_output=False)

print(payoff_matrix)
print()

print(user_strategy)
print("[")
for i in opp_strategy:
    print(f"{i},")
print("]")
print("\n")



user_1 = 0
user_2 = 0.5
user_3 = 0.5

opp_a = 0.17
opp_b = 0.83

a1 = user_1 * opp_a * 10
a2 = user_2 * opp_a * 0
a3 = user_3 * opp_a * 100

b1 = user_1 * opp_b * 10
b2 = user_2 * opp_b * 60
b3 = user_3 * opp_b * 40

print("Expected value:", a1+a2+a3+b1+b2+b3)




