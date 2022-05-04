from .nash_calc import NashCalc
from .turn_simulator import TurnSimulator

from showdown_pmariglia.showdown.battle import Battle
from showdown_pmariglia.showdown.battle_bots.helpers import format_decision
from showdown_pmariglia.showdown.engine.objects import StateMutator


class BattleBot(Battle):
    """ Main AI class

    Currently uses:
    - Tree search (depth=1)
    - Pmariglia's handcrafted evaluation function
    - Nash equilibria
    """
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)
        self.sim = TurnSimulator()
        self.NE = NashCalc()

    def find_best_move(self):
        """ Finds the best move at a given turn """
        # print("\n" + "".rjust(50, "-"), "\033[1mTURN:", self.turn, "".ljust(50, "-"), "\033[0m\n")
        state = self.create_state()
        mutator = StateMutator(state)
        user_options, opponent_options = self.get_all_options()

        # turns switching moves into separate moves: one for each switching option
        user_options, opponent_options = self.sim.get_switch_move_options(user_options, opponent_options, self.user.reserve, self.opponent.reserve)

        # get payoff matrix via tree search and evaluation
        payoff_matrix, bimatrix = self.sim.get_payoff_matrix(mutator, user_options, opponent_options)

        # select move based on a nash equilibrium strategy
        best_move, user_strategy, opp_strategy = self.NE.nash_equilibrium_move(self, bimatrix, user_options, opponent_options)

        # display payoff matrix and strategies
        # self.NE.display_payoff_matrix(payoff_matrix, user_options, opponent_options, user_strategy, opp_strategy)
        # print(f'\nChosen move: \033[1m{best_move}\033[0m')

        return format_decision(self, best_move)





