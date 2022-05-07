import numpy as np

import constants
from .damage_calculator import pokemon_type_indicies, damage_multipication_array
from .helpers import normalize_name


class Scoring:
    POKEMON_ALIVE_STATIC = 75
    POKEMON_HP = 100  # 100 points for 100% hp, 0 points for 0% hp. This is in addition to being alive
    POKEMON_HIDDEN = 10
    BOARD_POSITION_FACTOR = 5  # score += 5 * (offensive + defensive type advantages)
    POKEMON_BOOSTS = {
        constants.ATTACK: 15,
        constants.DEFENSE: 15,
        constants.SPECIAL_ATTACK: 15,
        constants.SPECIAL_DEFENSE: 15,
        constants.SPEED: 25,
        constants.ACCURACY: 3,
        constants.EVASION: 3
    }

    POKEMON_BOOST_DIMINISHING_RETURNS = {
        -6: -3.3,
        -5: -3.15,
        -4: -3,
        -3: -2.5,
        -2: -2,
        -1: -1,
        0: 0,
        1: 1,
        2: 2,
        3: 2.5,
        4: 3,
        5: 3.15,
        6: 3.30,
    }

    POKEMON_STATIC_STATUSES = {
        constants.FROZEN: -40,
        constants.SLEEP: -25,
        constants.PARALYZED: -25,
        constants.TOXIC: -30,
        constants.POISON: -10,
        None: 0
    }

    @staticmethod
    def BURN(burn_multiplier):
        return -25*burn_multiplier

    POKEMON_VOLATILE_STATUSES = {
        constants.LEECH_SEED: -30,
        constants.SUBSTITUTE: 40,
        constants.CONFUSION: -20
    }

    STATIC_SCORED_SIDE_CONDITIONS = {
        constants.REFLECT: 20,
        constants.STICKY_WEB: -25,
        constants.LIGHT_SCREEN: 20,
        constants.AURORA_VEIL: 40,
        constants.SAFEGUARD: 5,
        constants.TAILWIND: 7,
    }

    POKEMON_COUNT_SCORED_SIDE_CONDITIONS = {
        constants.STEALTH_ROCK: -10,
        constants.SPIKES: -7,
        constants.TOXIC_SPIKES: -7,
    }


def evaluate_pokemon(pkmn):
    score = 0
    if pkmn.hp <= 0:
        return score

    score += Scoring.POKEMON_ALIVE_STATIC
    score += Scoring.POKEMON_HP * (float(pkmn.hp) / pkmn.maxhp)

    # boosts have diminishing returns
    score += Scoring.POKEMON_BOOST_DIMINISHING_RETURNS[pkmn.attack_boost] * Scoring.POKEMON_BOOSTS[constants.ATTACK]
    score += Scoring.POKEMON_BOOST_DIMINISHING_RETURNS[pkmn.defense_boost] * Scoring.POKEMON_BOOSTS[constants.DEFENSE]
    score += Scoring.POKEMON_BOOST_DIMINISHING_RETURNS[pkmn.special_attack_boost] * Scoring.POKEMON_BOOSTS[constants.SPECIAL_ATTACK]
    score += Scoring.POKEMON_BOOST_DIMINISHING_RETURNS[pkmn.special_defense_boost] * Scoring.POKEMON_BOOSTS[constants.SPECIAL_DEFENSE]
    score += Scoring.POKEMON_BOOST_DIMINISHING_RETURNS[pkmn.speed_boost] * Scoring.POKEMON_BOOSTS[constants.SPEED]
    score += Scoring.POKEMON_BOOST_DIMINISHING_RETURNS[pkmn.accuracy_boost] * Scoring.POKEMON_BOOSTS[constants.ACCURACY]
    score += Scoring.POKEMON_BOOST_DIMINISHING_RETURNS[pkmn.evasion_boost] * Scoring.POKEMON_BOOSTS[constants.EVASION]

    try:
        score += Scoring.POKEMON_STATIC_STATUSES[pkmn.status]
    except KeyError:
        # KeyError only happens when the status is BURN
        score += Scoring.BURN(pkmn.burn_multiplier)

    for vol_stat in pkmn.volatile_status:
        try:
            score += Scoring.POKEMON_VOLATILE_STATUSES[vol_stat]
        except KeyError:
            pass

    return round(score)


def evaluate(state):
    score = 0

    number_of_opponent_reserve_revealed = len(state.opponent.reserve) + 1
    bot_alive_reserve_count = len([p.hp for p in state.self.reserve.values() if p.hp > 0])
    opponent_alive_reserves_count = len([p for p in state.opponent.reserve.values() if p.hp > 0]) + (6 - number_of_opponent_reserve_revealed)

    # evaluate the bot's pokemon
    score += evaluate_pokemon(state.self.active)
    for pkmn in state.self.reserve.values():
        this_pkmn_score = evaluate_pokemon(pkmn)
        score += this_pkmn_score

    # evaluate the opponent's visible pokemon
    score -= evaluate_pokemon(state.opponent.active)
    for pkmn in state.opponent.reserve.values():
        this_pkmn_score = evaluate_pokemon(pkmn)
        score -= this_pkmn_score

    # evaluate the side-conditions for the bot
    for condition, count in state.self.side_conditions.items():
        if condition in Scoring.STATIC_SCORED_SIDE_CONDITIONS:
            score += count * Scoring.STATIC_SCORED_SIDE_CONDITIONS[condition]
        elif condition in Scoring.POKEMON_COUNT_SCORED_SIDE_CONDITIONS:
            score += count * Scoring.POKEMON_COUNT_SCORED_SIDE_CONDITIONS[condition] * bot_alive_reserve_count

    # evaluate the side-conditions for the opponent
    for condition, count in state.opponent.side_conditions.items():
        if condition in Scoring.STATIC_SCORED_SIDE_CONDITIONS:
            score -= count * Scoring.STATIC_SCORED_SIDE_CONDITIONS[condition]
        elif condition in Scoring.POKEMON_COUNT_SCORED_SIDE_CONDITIONS:
            score -= count * Scoring.POKEMON_COUNT_SCORED_SIDE_CONDITIONS[condition] * opponent_alive_reserves_count

    # evaluate board position based on pokemon typing
    score += Scoring.BOARD_POSITION_FACTOR * evaluate_board_position(state)

    return int(score)


def evaluate_board_position(state):
    """Returns a value between -4 and 4 depending on the offensive/defensive type advantages"""
    user_mon = state.self.active
    opp_mon = state.opponent.active

    user_types = [normalize_name(t) for t in user_mon.types]
    opp_types = [normalize_name(t) for t in opp_mon.types]

    # get the factors of hits dealth
    dealth = []
    for i, user_type in enumerate(user_types):
        dealth_per_type = []
        for j, opp_type in enumerate(opp_types):
            opp_type_idx = pokemon_type_indicies[opp_type]
            user_type_idx = pokemon_type_indicies[user_type]
            dealth_per_type.append(damage_multipication_array[user_type_idx][opp_type_idx])
        dealth.append(np.prod(dealth_per_type, dtype=float))
    offensive_value = max(dealth)

    if offensive_value == 0.25:
        offensive_value = -2
    elif offensive_value == 0.5:
        offensive_value = -1
    elif offensive_value == 1.0:
        offensive_value = 0
    elif offensive_value == 2.0:
        offensive_value = 1
    elif offensive_value == 4.0:
        offensive_value = 2

    # get the factors of hits taken
    taken = []
    for j, opp_type in enumerate(opp_types):
        taken_per_type = []
        for i, user_type in enumerate(user_types):
            opp_type_idx = pokemon_type_indicies[opp_type]
            user_type_idx = pokemon_type_indicies[user_type]
            taken_per_type.append(damage_multipication_array[opp_type_idx][user_type_idx])
        taken.append(np.prod(taken_per_type, dtype=float))
    defensive_value = max(taken)

    if defensive_value == 0.25:
        defensive_value = 2
    elif defensive_value == 0.5:
        defensive_value = 1
    elif defensive_value == 1.0:
        defensive_value = 0
    elif defensive_value == 2.0:
        defensive_value = -1
    elif defensive_value == 4.0:
        defensive_value = -2

    return defensive_value + offensive_value


