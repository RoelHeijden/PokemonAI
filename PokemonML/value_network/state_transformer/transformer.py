import random
import torch
from typing import Dict, Any

from state_transformer.categories import (
    SPECIES,
    ITEMS,
    ABILITIES,
    MOVES,
    WEATHERS,
    TERRAINS,
    TYPES,
    STATUS,
    MOVE_CATEGORIES,
    VOLATILE_STATUS,
    SIDE_CONDITIONS,
)

from state_transformer.lookups import (
    MOVE_LOOKUP,
    FORM_LOOKUP,
    VOLATILES_TO_IGNORE,
    INVULNERABLE_STAGES,
    VULNERABLE_STAGES
)

import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


class Transformer:
    def __init__(self):
        pass

    def __call__(self, state: Dict[str, Any]) -> Dict[str, torch.tensor]:
        """ transforms the game state information to a tensor """
        out = dict()

        out['result'] = self._result(state['winner'])
        out['fields'] = self._fields(state)
        out['p1'] = self._side(state['p1'])
        out['p2'] = self._side(state['p2'])

        return out

    def _result(self, winner: str) -> torch.tensor:
        """
        Result:
            1 for a win
           -1 for a loss
            0 for a (rare) tie
        """

        if winner:
            result = 1 if winner == 'p1' else -1
        else:
            result = 0

        return torch.tensor(result, dtype=torch.int)

    def _fields(self, state: Dict[str, Any]) -> torch.tensor:
        """
        weather: weather type, turn count
        terrain: terrain type, turn count
        trick_room: rick room, turn count
        """

        # one-hot encoded weather
        weather = [0] * len(WEATHERS)
        weather_pos = WEATHERS.get(state['weather'])
        if weather_pos is not None:
            weather[weather_pos] = 1
        else:
            logging.debug(f'weather "{state["weather"]}" does not exist in weathers.json')

        # one-hot encoded terrain
        terrain = [0] * len(TERRAINS)
        terrain_pos = TERRAINS.get(state['terrain'])
        if terrain_pos is not None:
            terrain[terrain_pos] = 1
        else:
            logging.debug(f'terrain "{state["terrain"]}" does not exist in terrain.json')

        # 1 if trick room is active, 0 otherwise
        trick_room = int(state['trick_room'])

        # weather turn count
        weather_count = state['weather_count']

        # terrain turn count
        terrain_count = state['terrain_count']

        # n turns trick room has been active
        trick_room_count = state['trick_room_count']

        return torch.tensor(
            weather +
            terrain +
            [trick_room, weather_count, terrain_count, trick_room_count],
            dtype=torch.float
        )

    def _side(self, side: Dict[str, Any]) -> Dict[str, torch.tensor]:
        """
        TBD
        """

        out = {}

        # one-hot encode side conditions
        side_conditions = []
        for s_con in SIDE_CONDITIONS:
            if s_con in side['side_conditions']:
                count = side['side_conditions'].get(s_con)
                side_conditions.append(count)
            else:
                side_conditions.append(0)

        # two wish variables: [turn, amount]
        wish = [side['wish']['countdown'], side['wish']['hp_amount']]

        # future sight turn count
        future_sight = side['future_sight']['countdown']

        # 1 if the Healing wish/Lunar dance effect is incoming, 0 otherwise
        healing_wish = int(side['healing_wish'])

        # 1 if side's active is trapped, 0 otherwise
        trapped = int(side['trapped'])

        # 1 if the side's active pokemon is alive, 0 otherwise
        has_active = int(not side['active']['fainted'])

        # n amount of pokemon the player has
        n_pokemon = (len(side['reserve']) + 1) / 6

        out['player_state'] = torch.tensor(
            side_conditions +
            wish +
            [future_sight, healing_wish, trapped, has_active, n_pokemon],
            dtype=torch.float
        )

        # the player's active pokemon -- fainted or alive
        out['active'] = self._pokemon(side['active'])

        # the player's reserve pokemon -- fainted or alive
        out['reserve'] = [self._pokemon(pkmn) for pkmn in side['reserve']]

        # create empty pokemon when the player has less than 6 pokemon
        for _ in range(5 - len(side['reserve'])):
            out['reserve'].append(
                {
                    'species': torch.tensor(0, dtype=torch.float),
                    'ability': torch.tensor(0, dtype=torch.float),
                    'item': torch.tensor(0, dtype=torch.float),
                    # 'moves': torch.tensor(0, dtype=torch.float),
                    'attributes': torch.tensor(
                        [0] * len(out['active']['attributes']),
                        dtype=torch.float
                    )
                }
            )

        return out

    def _pokemon(self, pokemon: Dict[str, Any]) -> Dict[str, torch.tensor]:
        """
        TBD
        """
        # revert aesthetic forms back to original
        name = pokemon['name']
        if FORM_LOOKUP.get(name):
            name = FORM_LOOKUP.get(name)

        # species
        species = SPECIES.get(name)
        if species is None:
            species = 0
            logging.debug(f'pokemon "{pokemon["name"]}" does not exist in species.json')

        # ability
        ability = ABILITIES.get(pokemon['ability'])
        if ability is None:
            ability = 0
            logging.debug(f'ability "{pokemon["ability"]}" does not exist in ability.json')

        # item
        item = ITEMS.get(pokemon['item'])
        if item is None:
            if pokemon['item'] == "none":
                item = 0
            else:
                item = ITEMS.get("USELESS_ITEM")

        # one-hot encode types
        types = []
        for t in TYPES:
            if t in pokemon['types']:
                types.append(1)
            else:
                types.append(0)

        # pokemon stats
        stats = [
            stat / 250 for stat in
            [
                pokemon['stats']['hp'],
                pokemon['stats']['attack'],
                pokemon['stats']['defense'],
                pokemon['stats']['special-attack'],
                pokemon['stats']['special-defense'],
                pokemon['stats']['speed'],
            ]
        ]

        # pokemon stat boosts/drops
        stat_changes = [
            change / 6 for change in
            [
                pokemon['stat_changes']['attack'],
                pokemon['stat_changes']['defense'],
                pokemon['stat_changes']['special-attack'],
                pokemon['stat_changes']['special-defense'],
                pokemon['stat_changes']['speed'],
                pokemon['stat_changes']['accuracy'],
                pokemon['stat_changes']['evasion']
            ]
        ]

        # pokemon level
        level = pokemon['level'] / 100

        # pokemon health range
        health = int(pokemon['hp'] / pokemon['maxhp'])

        # 1 if still alive, 0 if fainted
        is_alive = int(not pokemon['fainted'])

        # the amount of turns the pokemon may stay asleep
        sleep_countdown = pokemon['sleep_countdown'] / 3

        # one-hot encode status conditions
        status = [0] * (len(STATUS) + 1)
        status_pos = STATUS.get(pokemon['status'])
        if status_pos is None:
            status_pos = 0
        status[status_pos] = 1

        # one-hot encode volatile_status
        volatile_status = [0] * len(VOLATILE_STATUS)
        for v in pokemon['volatile_status']:

            if VOLATILES_TO_IGNORE.get(v):
                continue
            if VULNERABLE_STAGES.get(v):
                v = 'vulnerablestage'
            if INVULNERABLE_STAGES.get(v):
                v = 'invulnerablestage'

            index = VOLATILE_STATUS.get(v)

            if index is not None:
                volatile_status[index] = 1
            else:
                logging.debug(f'volatile_status "{v}" does not exist in '
                              f'volatile_status.json and volatiles_to_ignore.json')

        # amount of moves the pokemon has, range 1-4
        n_moves = len(pokemon['moves'])

        # pokemon's moves
        # moves = [self._move(move) for move in pokemon['moves']]

        # # account for a pokemon having less than 4 moves
        # if n_moves[0] < 4:
        #     moves += [0] * int(len(moves) / n_moves[0]) * (4 - n_moves[0])
        # elif n_moves[0] <= 0:
        #     raise ValueError(f'Pokemon {pokemon["name"]} has no moves')

        return {
            'species': torch.tensor(species, dtype=torch.float),
            'ability': torch.tensor(ability, dtype=torch.float),
            'item': torch.tensor(item, dtype=torch.float),
            # 'moves': moves,
            'attributes': torch.tensor(
                types +
                stats +
                stat_changes +
                status +
                volatile_status +
                [level, health, is_alive, sleep_countdown, n_moves],
                dtype=torch.float)
        }


    def _move(self, move: Dict[str, Any]) -> Dict[str, torch.tensor]:
        move_name = move['name']

        # one-hot-encode moves
        moves = [0] * len(self.move_positions)
        move_index = self.move_positions.get(move_name)
        if move_index is not None:
            moves[move_index] = 1
        else:
            logging.debug(f'move "{move_name}" does not exist in moves.json')

        # one-hot-encode typing
        typing = [0] * len(self.types_positions)
        typing_index = self.types_positions.get(self.move_lookup[move_name]['type'].lower())
        if typing_index is not None:
            typing[typing_index] = 1
        else:
            logging.debug(f'type "{self.move_lookup[move_name]["type"]}" does not exist in types.json')

        # one-hot-encode move category
        move_category = [0] * len(self.move_category_positions)
        category_index = self.move_category_positions.get(self.move_lookup[move_name]['category'])
        if category_index is not None:
            move_category[category_index] = 1
        else:
            logging.debug(f'category "{self.move_lookup[move_name]["category"]}" does not exist in move_categories.json')

        # move base power
        base_power = [self.move_lookup[move_name]['basePower']]

        # move accuracy
        accuracy = [self.move_lookup[move_name]['accuracy']]
        if accuracy[0] == 1:
            accuracy[0] = 100

        # n multi hits of the move: [min_hits, max_hits]
        multi_hits = [1, 1]
        if self.move_lookup[move_name].get('multihit'):
            multi_hits = self.move_lookup[move_name]['multihit']
            if type(multi_hits) != list:
                multi_hits = [multi_hits, multi_hits]

        # current move pp
        current_pp = [move['pp']]

        # maximum move pp
        max_pp = [int(self.move_lookup[move_name]['pp'] * 1.6)]

        # [1] is move targets the user, [0] otherwise
        target_self = [int(self.move_lookup[move_name]['target'] == 'self')]

        # [1] if move can't be used this turn, [0] otherwise
        disabled = [int(move['disabled'])]

        # [1] if the move has been used previously, [0] otherwise
        used = [int(current_pp < max_pp)]

        # move priority level
        priority = [self.move_lookup[move_name]['priority']]

        return moves + typing + move_category + base_power + accuracy + multi_hits + current_pp + max_pp + \
            target_self + disabled + used + priority



