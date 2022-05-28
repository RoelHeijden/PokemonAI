import numpy as np
import math
import time

from Showdown_Pmariglia.showdown.engine.objects import *
from Showdown_Pmariglia.showdown.engine.find_state_instructions import *
from Showdown_Pmariglia.showdown.engine.damage_calculator import _calculate_damage
from Showdown_Pmariglia import config
from Showdown_Pmariglia import constants

from KetchAI.evaluate import Evaluate


class TurnSimulator:
    """ Edited version of some of Pmariglia's turn simulation tools """
    def __init__(self):
        self.value_network = Evaluate()

    def get_switch_move_options(self, user_options, opponent_options, user_reserve, opponent_reserve):
        """ Pairs a switching move with its possible switches, treating each combination as separate move """

        # create edited user_options
        edited_user_options = []
        for move in user_options:
            possible_switches = [pkmn.name for pkmn in user_reserve if pkmn.hp > 0]
            if move in constants.SWITCH_OUT_MOVES and len(possible_switches) > 0:
                edited_user_options += [move + "-" + pkmn for pkmn in possible_switches]
            else:
                edited_user_options.append(move)

        # create edited opponent_options
        edited_opponent_options = []
        for move in opponent_options:
            possible_switches = [pkmn.name for pkmn in opponent_reserve if pkmn.hp > 0]
            if move in constants.SWITCH_OUT_MOVES and len(possible_switches) > 0:
                edited_opponent_options += [move + "-" + pkmn for pkmn in possible_switches]
            else:
                edited_opponent_options.append(move)

        return edited_user_options, edited_opponent_options

    def get_payoff_matrix(self, mutator, user_options, opponent_options):
        """ Creates a payoff matrix by simulating the current turn and evaluating all future states """
        # initialize matrices
        user_outspeeds_matrix = np.zeros((len(user_options), len(opponent_options)))
        opp_outspeeds_matrix = np.zeros((len(user_options), len(opponent_options)))
        outspeed_probability_matrix = np.zeros((len(user_options), len(opponent_options)))

        n_simulations = 0
        start_time = time.time()

        # for each move the bot could use
        for i, full_user_move_str in enumerate(user_options):

            # for each move the opponent could use
            for j, full_opponent_move_str in enumerate(opponent_options):

                # switching moves are paired with their switch choice in a string separated by a dash
                user_switch, opp_switch = False, False
                if len(full_user_move_str.split("-")) == 2:
                    user_move_str, user_switch = full_user_move_str.split("-")
                else:
                    user_move_str = full_user_move_str
                if len(full_opponent_move_str.split("-")) == 2:
                    opponent_move_str, opp_switch = full_opponent_move_str.split("-")
                else:
                    opponent_move_str = full_opponent_move_str

                # get possible speed orders
                user_move = lookup_move(user_move_str)
                opponent_move = lookup_move(opponent_move_str)
                user_can_outspeed, opp_can_outspeed = self.possible_move_order(mutator.state, user_move, opponent_move)

                # runs scenario where the user outspeeds, given that it can outspeed
                if user_can_outspeed:
                    state_instructions = self.get_all_state_instructions(mutator, user_move_str, opponent_move_str, user_move, opponent_move, user_switch, opp_switch, user_outspeeds=True)
                    n_simulations += len(state_instructions)

                    score = 0
                    for instructions in state_instructions:
                        mutator.apply(instructions.instructions)
                        t_score = self.value_network.evaluate(mutator.state)
                        score += (t_score * instructions.percentage)
                        mutator.reverse(instructions.instructions)
                    user_outspeeds_matrix[i][j] = score

                # runs scenario where the opponent outspeeds, given that it can outspeed
                if opp_can_outspeed:
                    state_instructions = self.get_all_state_instructions(mutator, user_move_str, opponent_move_str, user_move, opponent_move, user_switch, opp_switch, user_outspeeds=False)
                    n_simulations += len(state_instructions)

                    score = 0
                    for instructions in state_instructions:
                        mutator.apply(instructions.instructions)
                        t_score = self.value_network.evaluate(mutator.state)
                        score += (t_score * instructions.percentage)
                        mutator.reverse(instructions.instructions)
                    opp_outspeeds_matrix[i][j] = score

                # set the probabilities for the probability matrix. Will eventually be based on a model
                if user_can_outspeed and opp_can_outspeed:
                    outspeed_probability_matrix[i][j] = 0.5
                elif user_can_outspeed:
                    outspeed_probability_matrix[i][j] = 1
                else:
                    outspeed_probability_matrix[i][j] = 0

        # save search time
        search_time = time.time() - start_time

        # create weighted matrices
        user_outspeeds_matrix = np.multiply(user_outspeeds_matrix, outspeed_probability_matrix)
        opp_outspeeds_matrix = np.multiply(opp_outspeeds_matrix, 1-outspeed_probability_matrix)

        # users's switch-move when the opponent outspeeds
        # for each opposing move, every switch-move value = max([switch-move-switches])
        user_switching_moves = list(set([move.split("-")[0] for move in user_options if len(move.split("-")) == 2]))
        for switching_move in user_switching_moves:
            new_values = np.array([-float('inf') for _ in range(len(opponent_options))])
            rows_to_replace = []

            # fill new_values with the highest values and remove the switching-move rows
            for i, full_user_move_str in enumerate(user_options):
                if full_user_move_str.startswith(switching_move):
                    new_values = np.maximum(new_values, opp_outspeeds_matrix[i])
                    rows_to_replace.append(i)

            # replace old values with the new values
            for idx in rows_to_replace:
                opp_outspeeds_matrix[idx] = new_values

        # opponent's switch-move when the user outspeeds
        # for each of the user's move, every switch-move value = min([switch-move-switches])
        opp_switching_moves = list(set([move.split("-")[0] for move in opponent_options if len(move.split("-")) == 2]))
        for switching_move in opp_switching_moves:
            new_values = np.array([float('inf') for _ in range(len(user_options))])
            columns_to_replace = []

            # fill new_values with the lowest values and remove the switching-move columns
            for j, full_opp_move_str in enumerate(opponent_options):
                if full_opp_move_str.startswith(switching_move):
                    new_values = np.minimum(new_values, user_outspeeds_matrix[:, j])
                    columns_to_replace.append(j)

            # replace old values with the new values
            for idx in columns_to_replace:
                user_outspeeds_matrix[:, idx] = new_values

        # create full payoff dict, and a bimatrix to be used for NE calculation
        full_weighted_matrix = opp_outspeeds_matrix + user_outspeeds_matrix
        payoff_matrix = {}
        bimatrix = np.zeros((len(user_options), len(opponent_options), 2))
        for i, full_user_move_str in enumerate(user_options):
            for j, full_opponent_move_str in enumerate(opponent_options):
                score = full_weighted_matrix[i][j]
                payoff_matrix[(full_user_move_str, full_opponent_move_str)] = score
                bimatrix[i][j][0], bimatrix[i][j][1] = score, -score

        return payoff_matrix, bimatrix, n_simulations, search_time

    def get_effective_speed_range(self, state, side, user_is_bot):
        """ Calculates the effective speed range of a pokemon given their speed stat or speed_range.

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
        """ Computes the possible speed orders of the turn. What is calculated:
            - whether the user can possibly outspeed the opponent
            - whether the opponent can possibly outspeed the user

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

    def get_all_state_instructions(self, mutator,  user_move_str, opponent_move_str, user_move, opponent_move, user_switch, opp_switch, user_outspeeds):
        """ Gets all possible state instructions for a given speed order """
        all_instructions = []

        # branches the scenarios where the user outspeeds
        if user_outspeeds:
            instructions = TransposeInstruction(1.0, [], False)
            instructions = self.get_state_instructions_from_move(mutator, user_move, opponent_move, constants.SELF, constants.OPPONENT, True, instructions, user_switch)
            for instruction in instructions:
                all_instructions += self.get_state_instructions_from_move(mutator, opponent_move, user_move, constants.OPPONENT, constants.SELF, False, instruction, opp_switch)

            # gets the instructions for things like weather damage, status damage, etc..
            if end_of_turn_triggered(user_move_str, opponent_move_str):
                temp_instructions = []
                for instruction_set in all_instructions:
                    temp_instructions += instruction_generator.get_end_of_turn_instructions(mutator, instruction_set, user_move, opponent_move, bot_moves_first=True)
                all_instructions = temp_instructions

        # branches the scenarios where the opponent outspeeds
        else:
            instructions = TransposeInstruction(1.0, [], False)
            instructions = self.get_state_instructions_from_move(mutator, opponent_move, user_move, constants.OPPONENT, constants.SELF, True, instructions, opp_switch)
            for instruction in instructions:
                all_instructions += self.get_state_instructions_from_move(mutator, user_move, opponent_move, constants.SELF, constants.OPPONENT, False, instruction, user_switch)

            # gets the instructions for things like weather damage, status damage, etc..
            if end_of_turn_triggered(user_move_str, opponent_move_str):
                temp_instructions = []
                for instruction_set in all_instructions:
                    temp_instructions += instruction_generator.get_end_of_turn_instructions(mutator, instruction_set, user_move, opponent_move, bot_moves_first=False)
                all_instructions = temp_instructions

        all_instructions = remove_duplicate_instructions(all_instructions)

        return all_instructions

    def get_state_instructions_from_move(self, mutator, attacking_move, defending_move, attacker, defender, first_move, instructions, switching_move_switch):
        """ Generates all state instructions from a combination of moves """
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
            if config.check_critical_hits:
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

        if config.check_critical_hits:
            crit_chance = 1/24
        else:
            crit_chance = 0

        if damage_amounts is not None:
            temp_instructions = []
            for instruction_set in all_instructions:
                for i, dmg in enumerate(damage_amounts):
                    these_instructions = copy(instruction_set)
                    if i < n_regular_rolls:
                        these_instructions.update_percentage(1 / n_regular_rolls * (1 - crit_chance))
                    else:
                        these_instructions.update_percentage(1 / (len(damage_amounts) - n_regular_rolls) * crit_chance)
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
            for inst in all_instructions:
                if switching_move_switch:
                    temp_instructions += instruction_generator.get_instructions_from_switch(mutator, attacker, switching_move_switch, inst)
                else:
                    temp_instructions.append(inst)
            all_instructions = temp_instructions

        return all_instructions

