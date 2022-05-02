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
                          [10, 60, 40]])

bimatrix = to_bimatrix(payoff_matrix.T)
game = NormalFormGame(bimatrix)
user_strategy, opp_strategy = lemke_howson(game, init_pivot=0, max_iter=1000000, capping=None, full_output=False)

print(user_strategy, opp_strategy)
print()

opp_strategy = [0.3, 0.7]
best_responses = game.players[0].best_response(opp_strategy, tie_breaking=False)

print(best_responses)

user_strategy = [1/len(best_responses) if i in best_responses else 0 for i in range(3)]
print(user_strategy)




