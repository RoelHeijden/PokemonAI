import numpy as np
import math
from quantecon.game_theory.lemke_howson import lemke_howson
from quantecon.game_theory.mclennan_tourky import mclennan_tourky
from quantecon.game_theory.normal_form_game import NormalFormGame

from showdown_pmariglia import config
from showdown_pmariglia import constants
from showdown_pmariglia.showdown.battle import Battle
from showdown_pmariglia.showdown.battle_bots.helpers import *
from showdown_pmariglia.showdown.engine.objects import *
from showdown_pmariglia.showdown.engine.find_state_instructions import *
from showdown_pmariglia.showdown.engine.damage_calculator import _calculate_damage
from showdown_pmariglia.showdown.engine.evaluate import evaluate
from showdown_pmariglia.data import pokedex




class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)
        self.sim = TurnSimulator()

    def find_best_move(self):
        print("\n" + "".rjust(50, "-"), "\033[1mTURN:", self.turn, "".ljust(50, "-"), "\033[0m\n")
        state = self.create_state()
        mutator = StateMutator(state)
        user_options, opponent_options = self.get_all_options()
        best_move = user_options[np.random.randint(0, len(user_options))]
        user_options, opponent_options = self.get_switch_move_options(user_options, opponent_options)

        payoff_matrix, bimatrix = self.sim.get_payoff_matrix(mutator, user_options, opponent_options)
        game = NormalFormGame(bimatrix)
        user_strategy, opp_strategy = lemke_howson(game, init_pivot=0, max_iter=1000000, capping=None, full_output=False)
        # user_strategy, opp_strategy = mclennan_tourky(game, init=None, epsilon=0.001, max_iter=200, full_output=False)
        user_strategy = [prob if prob >= 0 else 0 for prob in user_strategy]
        self.sim.display_payoff_matrix(payoff_matrix, user_options, opponent_options, user_strategy, opp_strategy)

        best_move = np.random.choice(a=user_options, p=user_strategy)
        best_move = best_move.split("-")[0]

        print(f'\nMove: {best_move}')
        return format_decision(self, best_move)

    def get_switch_move_options(self, user_options, opponent_options):
        """ pairs a switching move with its possible switches, treating each combination as separate move """
        edited_user_options = []
        for move in user_options:
            if move in constants.SWITCH_OUT_MOVES:
                edited_user_options += [move + "-" + pkmn.name for pkmn in self.user.reserve if pkmn.hp > 0]
            else:
                edited_user_options.append(move)

        edited_opponent_options = []
        for move in opponent_options:
            if move in constants.SWITCH_OUT_MOVES:
                edited_opponent_options += [move + "-" + pkmn.name for pkmn in self.opponent.reserve if pkmn.hp > 0]
            else:
                edited_opponent_options.append(move)

        return edited_user_options, edited_opponent_options


class TurnSimulator:
    def __init__(self):
        pass

    def get_payoff_matrix(self, mutator, user_options, opponent_options):
        payoff_matrix = {}
        bimatrix = np.empty((len(user_options), len(opponent_options), 2))

        user_mon = mutator.state.self.active
        opp_mon = mutator.state.opponent.active

        # for each move the bot could use
        for i, user_move_str in enumerate(user_options):
            # for each move the opponent could use
            for j, opponent_move_str in enumerate(opponent_options):

                # if opp_mon.ability is None and mutator.state.turn:
                #     possible_abilities = [ability for _, ability in pokedex[opp_mon.id]['abilities'].items()]
                # else:
                #     possible_abilities = [opp_mon.ability]
                #
                # for opp_ability in possible_abilities:
                #     pass
                # for opp_item, probability in opp_mon.possible_items:
                #     pass

                score = 0
                state_instructions = self.get_all_state_instructions(mutator, user_move_str, opponent_move_str)

                for instructions in state_instructions:
                    mutator.apply(instructions.instructions)
                    t_score = evaluate(mutator.state)
                    score += (t_score * instructions.percentage)
                    mutator.reverse(instructions.instructions)

                payoff_matrix[(user_move_str, opponent_move_str)] = score
                bimatrix[i][j][0], bimatrix[i][j][1] = score, -score

        return payoff_matrix, bimatrix

    def display_payoff_matrix(self, payoff_matrix, user_options, opponent_options, user_strategy, opp_strategy):
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
                value = round(payoff_matrix[(user_move_str, opponent_move_str)], 2)
                if value < 0:
                    print(Color.BLUE + str(value) + Color.END, end=" ")
                else:
                    print(Color.BLUE + " " + str(value) + Color.END, end="")

                print(line[0:2].rjust(len(longest_str_user) - len(str(value)) + len(line[0:2]), ' '), end="")
            print()

    def get_effective_speed_range(self, state, side, user_is_bot):
        """ calculates the effective speed range of a pokemon given their speed stat or speed_range.

        :param state: State
        :param side: Side
        :param user_is_bot: Bool
        :return: StatRange (collections.namedTuple)
        """
        pokemon = side.active
        if user_is_bot:
            boosted_speed = np.array([pokemon.speed, pokemon.speed])
        else:
            boosted_speed = np.array([pokemon.speed_range[0], pokemon.speed_range[1]])
        boosted_speed = boosted_speed * boost_multiplier_lookup[pokemon.speed_boost]

        if 'choicescarf' == side.active.item and user_is_bot:
            boosted_speed *= 1.5

        if state.weather == constants.SUN and pokemon.ability == 'chlorophyll':
            boosted_speed *= 2
        elif state.weather == constants.RAIN and pokemon.ability == 'swiftswim':
            boosted_speed *= 2
        elif state.weather == constants.SAND and pokemon.ability == 'sandrush':
            boosted_speed *= 2
        elif state.weather == constants.HAIL and pokemon.ability == 'slushrush':
            boosted_speed *= 2

        if state.field == constants.ELECTRIC_TERRAIN and pokemon.ability == 'surgesurfer':
            boosted_speed *= 2

        if pokemon.ability == 'unburden' and not pokemon.item:
            boosted_speed *= 2
        elif pokemon.ability == 'quickfeet' and pokemon.status is not None:
            boosted_speed *= 1.5

        if side.side_conditions[constants.TAILWIND]:
            boosted_speed *= 2

        if constants.PARALYZED == pokemon.status and pokemon.ability != 'quickfeet':
            boosted_speed *= 0.5

        return StatRange(min=boosted_speed[0], max=boosted_speed[1])

    def possible_move_order(self, state, user_move, opponent_move):
        """ computes the possible speed orders of the turn. What is calculated:
            - whether the user can possibly outspeed the opponent
            - whether the opponent can possibly outspeed the user

        :param state: State
        :param user_move: Move
        :param opponent_move: Move
        :return: (user_can_outspeed, opp_can_outspeed): (Bool, Bool)
        """
        user_effective_speed_range = self.get_effective_speed_range(state, state.self, user_is_bot=True)
        opponent_effective_speed_range = self.get_effective_speed_range(state, state.opponent, user_is_bot=False)
        user_min, user_max = user_effective_speed_range[0], user_effective_speed_range[1]
        opp_min, opp_max = opponent_effective_speed_range[0], opponent_effective_speed_range[1]

        # factor in a potential choice scarf, given that the pokemon theoretically could hold one
        # NOTE: maybe move this check somewhere else in the code
        if (
            state.opponent.active.item == constants.UNKNOWN_ITEM and
            state.opponent.active.speed_range.max >= math.floor(state.opponent.active.min_speed * 1.5)
        ):
            opp_max = math.floor(opp_max * 1.5)

        # checks whether it's theoretically possible for the user or opponent to outspeed the other
        if user_min > opp_max:
            speed_interaction = True, False
        elif opp_min > user_max:
            speed_interaction = False, True
        else:
            speed_interaction = True, True

        # reverse the speed interaction if Trick room is active, unless both pokemon can already outspeed the other
        if state.trick_room and not (speed_interaction[0] and speed_interaction[1]):
            speed_interaction = (not speed_interaction[0], not speed_interaction[1])

        # both users selected a switch
        if constants.SWITCH_STRING in user_move and constants.SWITCH_STRING in opponent_move:
            return speed_interaction

        # user selected a switch
        elif constants.SWITCH_STRING in user_move:
            if opponent_move[constants.ID] == 'pursuit':
                return False, True
            return True, False

        # opponent selected a switch
        elif constants.SWITCH_STRING in opponent_move:
            if user_move[constants.ID] == 'pursuit':
                return True, False
            return False, True

        user_priority = get_effective_priority(state.self, user_move, state.field)
        opponent_priority = get_effective_priority(state.opponent, opponent_move, state.field)

        if user_priority == opponent_priority:
            return speed_interaction

        if user_priority > opponent_priority:
            return True, False
        else:
            return False, True

    def get_all_state_instructions(self, mutator, user_move_string, opponent_move_string):
        # switching moves are paired with their switch in a string separated by a dash
        user_switch, opp_switch = False, False
        if len(user_move_string.split("-")) == 2:
            user_move_string, user_switch = user_move_string.split("-")
        if len(opponent_move_string.split("-")) == 2:
            opponent_move_string, opp_switch = opponent_move_string.split("-")

        user_move = lookup_move(user_move_string)
        opponent_move = lookup_move(opponent_move_string)

        user_can_outspeed, opp_can_outspeed = self.possible_move_order(mutator.state, user_move, opponent_move)
        all_instructions = []

        # branches the scenario's where the user outspeeds
        if user_can_outspeed:
            instructions = TransposeInstruction(1.0, [], False)
            instructions = self.get_state_instructions_from_move(mutator, user_move, opponent_move, constants.SELF, constants.OPPONENT, True, instructions)
            for instruction in instructions:
                all_instructions += self.get_state_instructions_from_move(mutator, opponent_move, user_move, constants.OPPONENT, constants.SELF, False, instruction)

            # gets the instructions for things like weather damage, status damage, etc..
            if end_of_turn_triggered(user_move_string, opponent_move_string):
                temp_instructions = []
                for instruction_set in all_instructions:
                    temp_instructions += instruction_generator.get_end_of_turn_instructions(mutator, instruction_set, user_move, opponent_move, bot_moves_first=True)
                all_instructions = temp_instructions

        # branches the scenarios where the opponent outspeeds
        if opp_can_outspeed:
            instructions = TransposeInstruction(1.0, [], False)
            instructions = self.get_state_instructions_from_move(mutator, opponent_move, user_move, constants.OPPONENT, constants.SELF, True, instructions)
            for instruction in instructions:
                all_instructions += self.get_state_instructions_from_move(mutator, user_move, opponent_move, constants.SELF, constants.OPPONENT, False, instruction)

            # gets the instructions for things like weather damage, status damage, etc..
            if end_of_turn_triggered(user_move_string, opponent_move_string):
                temp_instructions = []
                for instruction_set in all_instructions:
                    temp_instructions += instruction_generator.get_end_of_turn_instructions(mutator, instruction_set, user_move, opponent_move, bot_moves_first=False)
                all_instructions = temp_instructions

        # if both speed orders are possible: halve all probabilities, so it doesn't sum to 2.0
        if user_can_outspeed and opp_can_outspeed:
            for instruction in all_instructions:
                instruction.percentage *= 0.5

        all_instructions = remove_duplicate_instructions(all_instructions)
        return all_instructions

    def get_state_instructions_from_move(self, mutator, attacking_move, defending_move, attacker, defender, first_move, instructions):
        instructions.frozen = False

        if constants.SWITCH_STRING in attacking_move:
            return instruction_generator.get_instructions_from_switch(mutator, attacker, attacking_move[constants.SWITCH_STRING], instructions)

        # if you are moving second, but you got phased on the first turn, your move will do nothing
        # this can happen if a move with equal priority to a phasing move (generally -6) is used by a slower pokemon and the faster pokemon uses a phasing move
        if not first_move and constants.DRAG in defending_move.get(constants.FLAGS, {}):
            return [instructions]

        mutator.apply(instructions.instructions)
        attacking_side = instruction_generator.get_side_from_state(mutator.state, attacker)
        defending_side = instruction_generator.get_side_from_state(mutator.state, defender)
        attacking_pokemon = attacking_side.active
        defending_pokemon = defending_side.active
        active_weather = mutator.state.weather

        if cannot_use_move(attacking_pokemon, attacking_move):
            attacking_move = lookup_move(constants.DO_NOTHING_MOVE)

        conditions = {
            constants.REFLECT: defending_side.side_conditions[constants.REFLECT],
            constants.LIGHT_SCREEN: defending_side.side_conditions[constants.LIGHT_SCREEN],
            constants.AURORA_VEIL: defending_side.side_conditions[constants.AURORA_VEIL],
            constants.WEATHER: active_weather,
            constants.TERRAIN: mutator.state.field
        }

        if attacking_pokemon.hp == 0:
            # if the attacker is dead, remove the 'flinched' volatile-status if it has it and exit early
            # this triggers if the pokemon moves second but the first attack knocked it out
            instructions = instruction_generator.get_instructions_from_flinched(mutator, attacker, instructions)
            mutator.reverse(instructions.instructions)
            return [instructions]

        attacking_move = update_attacking_move(
            attacking_pokemon,
            defending_pokemon,
            attacking_move,
            defending_move,
            first_move,
            mutator.state.weather,
            mutator.state.field
        )

        instructions = instruction_generator.get_instructions_from_flinched(mutator, attacker, instructions)

        ability_before_move_instructions = ability_before_move(
            attacking_pokemon.ability,
            mutator.state,
            attacker,
            attacking_move,
            attacking_pokemon,
            defending_pokemon
        )
        if ability_before_move_instructions is not None and not instructions.frozen:
            mutator.apply(ability_before_move_instructions)
            instructions.instructions += ability_before_move_instructions

        n_regular_rolls = 0
        damage_amounts = None
        move_status_effect = None
        flinch_accuracy = None
        boosts = None
        boosts_target = None
        boosts_chance = None
        side_condition = None
        hazard_clearing_move = None
        volatile_status = attacking_move.get(constants.VOLATILE_STATUS)

        move_accuracy = min(100, attacking_move[constants.ACCURACY])
        move_status_accuracy = move_accuracy

        move_target = attacking_move[constants.TARGET]
        if move_target == constants.SELF:
            move_status_target = attacker
        else:
            move_status_target = defender

        if attacking_move[constants.ID] in constants.HAZARD_CLEARING_MOVES:
            hazard_clearing_move = attacking_move

        # move is a damaging move
        if attacking_move[constants.CATEGORY] in constants.DAMAGING_CATEGORIES:
            damage_amounts = _calculate_damage(attacking_pokemon, defending_pokemon, attacking_move, conditions=conditions, calc_type=config.damage_calc_type, critical_hit=False)
            n_regular_rolls = len(damage_amounts)

            # checks critical hits
            damage_amounts += _calculate_damage(attacking_pokemon, defending_pokemon, attacking_move, conditions=conditions, calc_type=config.damage_calc_crit_type, critical_hit=True)

            attacking_move_secondary = attacking_move[constants.SECONDARY]
            attacking_move_self = attacking_move.get(constants.SELF)
            if attacking_move_secondary:
                # flinching (iron head)
                if attacking_move_secondary.get(constants.VOLATILE_STATUS) == constants.FLINCH and first_move:
                    flinch_accuracy = attacking_move_secondary.get(constants.CHANCE)

                # secondary status effects (thunderbolt paralyzing)
                elif attacking_move_secondary.get(constants.STATUS) is not None:
                    move_status_effect = attacking_move_secondary[constants.STATUS]
                    move_status_accuracy = attacking_move_secondary[constants.CHANCE]

                # boosts from moves that boost in secondary (charge beam)
                elif attacking_move_secondary.get(constants.SELF) is not None:
                    if constants.BOOSTS in attacking_move_secondary[constants.SELF]:
                        boosts = attacking_move_secondary[constants.SELF][constants.BOOSTS]
                        boosts_target = attacker
                        boosts_chance = attacking_move_secondary[constants.CHANCE]

                # boosts from secondary, but to the defender (crunch)
                elif attacking_move_secondary and attacking_move_secondary.get(constants.BOOSTS) is not None:
                    boosts = attacking_move_secondary[constants.BOOSTS]
                    boosts_target = defender
                    boosts_chance = attacking_move_secondary[constants.CHANCE]

            # boosts from secondary, but it is a guaranteed boost (dracometeor)
            elif attacking_move_self:
                if constants.BOOSTS in attacking_move_self:
                    boosts = attacking_move_self[constants.BOOSTS]
                    boosts_target = attacker
                    boosts_chance = 100

            # guaranteed boosts from a damaging move (none in the moves JSON but items/abilities can cause this)
            elif constants.BOOSTS in attacking_move:
                boosts = attacking_move[constants.BOOSTS]
                boosts_target = attacker if attacking_move[constants.TARGET] in constants.MOVE_TARGET_SELF else defender
                boosts_chance = 100

        # move is a status move
        else:
            move_status_effect = attacking_move.get(constants.STATUS)
            side_condition = attacking_move.get(constants.SIDE_CONDITIONS)

            # boosts from moves that only boost (dragon dance)
            if attacking_move.get(constants.BOOSTS) is not None:
                boosts = attacking_move[constants.BOOSTS]
                boosts_target = attacker if attacking_move[constants.TARGET] == constants.SELF else defender
                boosts_chance = attacking_move[constants.ACCURACY]

        mutator.reverse(instructions.instructions)

        all_instructions = instruction_generator.get_instructions_from_statuses_that_freeze_the_state(mutator, attacker, defender, attacking_move, defending_move, instructions)

        temp_instructions = []
        for instruction_set in all_instructions:
            temp_instructions += instruction_generator.get_instructions_from_move_special_effect(mutator, attacker, attacking_pokemon, defending_pokemon, attacking_move[constants.ID], instruction_set)
        all_instructions = temp_instructions

        if damage_amounts is not None:
            temp_instructions = []
            for instruction_set in all_instructions:
                for i, dmg in enumerate(damage_amounts):
                    these_instructions = copy(instruction_set)
                    if i < n_regular_rolls:
                        these_instructions.update_percentage(1 / n_regular_rolls * 23/24)
                    else:
                        these_instructions.update_percentage(1 / (len(damage_amounts) - n_regular_rolls) * 1/24)
                    temp_instructions += instruction_generator.get_states_from_damage(mutator, defender, dmg, move_accuracy, attacking_move, these_instructions)

            all_instructions = temp_instructions

        if defending_pokemon.ability in constants.ABILITY_AFTER_MOVE:
            temp_instructions = []
            for instruction_set in all_instructions:
                temp_instructions += instruction_generator.get_instructions_from_defenders_ability_after_move(mutator, attacking_move, defending_pokemon.ability, attacking_pokemon, attacker, instruction_set)
            all_instructions = temp_instructions

        if side_condition is not None:
            temp_instructions = []
            for instruction_set in all_instructions:
                temp_instructions += instruction_generator.get_instructions_from_side_conditions(mutator, attacker, move_target, side_condition, instruction_set)
            all_instructions = temp_instructions

        if hazard_clearing_move is not None:
            temp_instructions = []
            for instruction_set in all_instructions:
                temp_instructions += instruction_generator.get_instructions_from_hazard_clearing_moves(mutator, attacker, attacking_move, instruction_set)
            all_instructions = temp_instructions

        if volatile_status is not None:
            temp_instructions = []
            for instruction_set in all_instructions:
                temp_instructions += instruction_generator.get_state_from_volatile_status(mutator, volatile_status, attacker, move_target, first_move, instruction_set)
            all_instructions = temp_instructions

        if move_status_effect is not None:
            temp_instructions = []
            for instruction_set in all_instructions:
                temp_instructions += instruction_generator.get_states_from_status_effects(mutator, move_status_target, move_status_effect, move_status_accuracy, instruction_set)
            all_instructions = temp_instructions

        if boosts is not None:
            temp_instructions = []
            for instruction_set in all_instructions:
                temp_instructions += instruction_generator.get_states_from_boosts(mutator, boosts_target, boosts, boosts_chance, instruction_set)
            all_instructions = temp_instructions

        if attacking_move[constants.ID] in constants.BOOST_RESET_MOVES:
            temp_instructions = []
            for instruction_set in all_instructions:
                temp_instructions += instruction_generator.get_instructions_from_boost_reset_moves(mutator, attacking_move, attacker, instruction_set)
            all_instructions = temp_instructions

        if attacking_move.get(constants.HEAL) is not None:
            temp_instructions = []
            for instruction_set in all_instructions:
                temp_instructions += instruction_generator.get_state_from_attacker_recovery(mutator, attacker, attacking_move, instruction_set)
            all_instructions = temp_instructions

        if flinch_accuracy is not None:
            temp_instructions = []
            for instruction_set in all_instructions:
                temp_instructions += instruction_generator.get_states_from_flinching_moves(defender, flinch_accuracy, first_move, instruction_set)
            all_instructions = temp_instructions

        if constants.DRAG in attacking_move[constants.FLAGS]:
            temp_instructions = []
            for instruction_set in all_instructions:
                temp_instructions += instruction_generator.get_state_from_drag(mutator, attacker, move_target, instruction_set)
            all_instructions = temp_instructions

        if switch_out_move_triggered(attacking_move, damage_amounts):
            temp_instructions = []
            for i in all_instructions:
                best_switch = get_best_switch_pokemon(mutator, i, attacker, attacking_side, defending_move, first_move)
                if best_switch is not None:
                    temp_instructions += instruction_generator.get_instructions_from_switch(mutator, attacker, best_switch, i)
                else:
                    temp_instructions.append(i)
            all_instructions = temp_instructions

        return all_instructions
