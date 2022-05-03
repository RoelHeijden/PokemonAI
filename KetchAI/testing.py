from quantecon.game_theory.mclennan_tourky import mclennan_tourky
from quantecon.game_theory.normal_form_game import NormalFormGame
from quantecon.game_theory.lemke_howson import lemke_howson
import numpy as np

# best_move = user_options[np.random.randint(0, len(user_options))]


def to_bimatrix(payoff_matrix):
    length, width = payoff_matrix.shape
    bimatrix = np.zeros((length, width, 2))
    for i in range(length):
        for j in range(width):
            bimatrix[i][j][0], bimatrix[i][j][1] = payoff_matrix[i][j], -payoff_matrix[i][j]
    return bimatrix


payoff_matrix = np.array([[10, 0, 100],
                          [10, 60, 40],
                          [40, 40, 0]])

bimatrix = to_bimatrix(payoff_matrix.T)
game = NormalFormGame(bimatrix)
user_strategy, opp_strategy = lemke_howson(game, init_pivot=0, max_iter=1000000, capping=None, full_output=False)

print(payoff_matrix.T)
print()
print(user_strategy)
print(opp_strategy)
print()
print("----------------")
print()

next_matrix = np.array([[10, 0],
                        [10, 60]])

bimatrix = to_bimatrix(next_matrix.T)
game = NormalFormGame(bimatrix)
user_strategy, opp_strategy = lemke_howson(game, init_pivot=0, max_iter=1000000, capping=None, full_output=False)

print(next_matrix.T)
print()
print(user_strategy)
print(opp_strategy)