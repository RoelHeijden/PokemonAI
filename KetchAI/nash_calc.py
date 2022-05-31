from quantecon.game_theory.lemke_howson import lemke_howson
from quantecon.game_theory.normal_form_game import NormalFormGame
import numpy as np

from Showdown_Pmariglia import constants


# data required to make a correct switching decision mid turn (caused by switching moves)
# storing them here for now because the Battle class keeps getting reset each turn(?)
PrevOppOptions = []
PrevOppStrategy = []


class NashCalc:
    """ Uses Quantecon.gametheory to calculate Nash equilibrium strategies """
    def __init__(self):
        pass

    def nash_equilibrium_move(self, battle, bimatrix, user_options, opponent_options):
        """ Picks the best move based on a Nash equilibrium strategy. Can be non-deterministic"""
        # checks if a mid-turn switch caused by a switching move needs to be picked
        if battle.force_switch and opponent_options != [constants.DO_NOTHING_MOVE] and battle.opponent.active.hp > 0:
            x, user_strategy, opp_strategy = self.select_mid_turn_switch(bimatrix, user_options, opponent_options)
        else:
            user_strategy, opp_strategy = self.create_nash_equilibrium(bimatrix)

        # set global variables, to potentially be used next turn
        global PrevOppStrategy, PrevOppOptions
        PrevOppStrategy, PrevOppOptions = opp_strategy, opponent_options

        # select move based on calculated strategy
        best_move = np.random.choice(a=user_options, p=user_strategy)
        best_move = best_move.split("-")[0]
        return best_move, user_strategy, opp_strategy

    def create_nash_equilibrium(self, bimatrix):
        """ Generates a move based on a Nash equilibrium strategy """
        # get Nash equilibrium strategies
        game = NormalFormGame(bimatrix)
        user_strategy, opp_strategy = lemke_howson(game, init_pivot=0, max_iter=1000000, capping=None, full_output=False)
        user_strategy = [prob if prob >= 0 else 0 for prob in user_strategy]

        # random move is selected for 'uniform' Nash equilibria
        best_responses = game.players[0].best_response(opp_strategy, tie_breaking=False)
        if np.count_nonzero(user_strategy) == 1 and len(best_responses) > 1:
            response = game.players[0].best_response(opp_strategy, tie_breaking='random')
            user_strategy = np.zeros(len(user_strategy))
            user_strategy[response] = 1

        return user_strategy, opp_strategy

    def select_mid_turn_switch(self, bimatrix, user_options, opp_options):
        """ Decides what strategy to go off when having to select a (faster) switch mid-turn """
        # remove unavailable options from previous opponent_strategy
        opp_strategy = []
        for i, prev_option in enumerate(PrevOppOptions):
            if prev_option in opp_options:
                opp_strategy.append(PrevOppStrategy[i])

        # check whether the previous opponent's strategy translates over to the new opponent's strategy.
        #   if it doesn't: calc new Nash Equilibrium strategy for game state
        #   else: normalize opponent's new strategy and best reacting user strategy
        if sum(opp_strategy) == 0:
            user_strategy, opp_strategy = self.create_nash_equilibrium(bimatrix)
        else:
            game = NormalFormGame(bimatrix)
            opp_strategy = [float(i) / sum(opp_strategy) for i in opp_strategy]
            best_responses = game.players[0].best_response(opp_strategy, tie_breaking=False)
            user_strategy = [1/len(best_responses) if i in best_responses else 0 for i in range(len(user_options))]

        best_move = np.random.choice(a=user_options, p=user_strategy)
        return best_move, user_strategy, opp_strategy

    def display_payoff_matrix(self, payoff_matrix, user_options, opponent_options, user_strategy, opp_strategy):
        """ Displays a payoff matrix. The code may not be pretty but the matrix is """
        class Color:
            PURPLE = '\033[95m'
            CYAN = '\033[96m'
            DARKCYAN = '\033[36m'
            BLUE = '\033[94m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            RED = '\033[91m'
            BOLD = '\033[1m'
            END = '\033[0m'

        line = " | "
        prob_decimals = 3
        longest_str_user = max(user_options, key=len)
        longest_str_opp = max(opponent_options, key=len)

        # print column probabilities
        print("".rjust(prob_decimals + 2 + 1 + len(longest_str_opp) + len(line), ' '), end="")
        for prob in user_strategy:
            prob = round(prob, prob_decimals)
            if prob > 0:
                print(Color.GREEN + format(prob, "." + str(prob_decimals) + "f") + Color.END, end="")
            else:
                print("".rjust(len(str(prob)), " "), end="")
            print("  ".rjust(len(longest_str_user) - len(str(prob)) + len(line), ' '), end="")
        print()

        # print column labels
        print(line.rjust(prob_decimals + 2 + 1 + len(longest_str_opp) + len(line), ' '), end="")
        for label in user_options:
            print(Color.BOLD + label + Color.END, end="")
            print(line.rjust(len(longest_str_user) - len(label) + len(line), ' '), end="")
        print()

        for y, opponent_move_str in enumerate(opponent_options):

            # print row probabilities
            prob = round(opp_strategy[y], prob_decimals)
            if prob > 0:
                print(Color.GREEN + format(prob, "." + str(prob_decimals) + "f") + Color.END, end=" ")
            else:
                print("".rjust(prob_decimals + 2, " "), end=" ")

            # print row labels
            print(Color.BOLD + opponent_move_str + Color.END, end="")
            print(line[0:2].rjust(len(longest_str_opp) - len(opponent_move_str) + len(line[0:2]), ' '), end="")

            # print values
            for x, user_move_str in enumerate(user_options):
                value = payoff_matrix.get((user_move_str, opponent_move_str))
                value = round(value, 3)
                if value < 0:
                    print(Color.BLUE + str(value) + Color.END, end=" ")
                else:
                    print(Color.BLUE + " " + str(value) + Color.END, end="")

                print(line[0:2].rjust(len(longest_str_user) - len(str(value)) + len(line[0:2]), ' '), end="")
            print()