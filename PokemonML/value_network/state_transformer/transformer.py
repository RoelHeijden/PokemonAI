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
    def __init__(self, shuffle_players=True, shuffle_pokemon=False, shuffle_moves=False):
        self.shuffle_players = shuffle_players
        self.shuffle_pokemon = shuffle_pokemon
        self.shuffle_moves = shuffle_moves

        # field scaling
        self.turn_count_scaling = 5

        # side scaling
        self.wish_scaling = 200
        self.n_pokemon_scaling = 6

        # pokemon scaling
        self.stat_scaling = 250
        self.stat_change_scaling = 3
        self.level_scaling = 100
        self.n_moves_scaling = 4
        self.sleep_count_scaling = 3

        # move scaling
        self.pp_scaling = 16
        self.bp_scaling = 100
        self.acc_scaling = 100
        self.priority_scaling = 3
        self.multihit_scaling = 3

    def __call__(self, state: Dict[str, Any]) -> Dict[str, torch.tensor]:
        """ transforms the game state information to a tensor """
        if self.shuffle_players:
            if random.random() > 0.5:
                self.p1 = "p2"
                self.p2 = "p1"
            else:
                self.p1 = "p1"
                self.p2 = "p2"

        # import json
        # print(json.dumps(state, indent=3))

        out = dict()

        out['result'] = self._get_result(state['winner'])
        out['fields'] = self._transform_field(state)
        out['p1_attributes'] = self._transform_side(state[self.p1])
        out['p2_attributes'] = self._transform_side(state[self.p2])
        out['pokemon'] = self._transform_pokemon(state)

        return out

    def _get_result(self, winner: str) -> torch.tensor:
        """
        Result:
            1 for a win
           -1 for a loss
            0 for a (rare) tie
        """

        if winner:
            result = 1 if winner == self.p1 else -1
        else:
            result = 0

        return torch.tensor(result, dtype=torch.int)

    def _transform_field(self, state: Dict[str, Any]) -> torch.tensor:
        """
        Weather
        Terrain
        Trick_room
        Magic_room
        Gravity
        Weather_count
        Terrain_count
        Trick_room_count
        Magic_room_count
        Gravity_count
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

        # 1 if magic room is active, 0 otherwise
        magic_room = int(state['magic_room'])

        # 1 if gravity is active, 0 otherwise
        gravity = int(state['gravity'])

        # weather turn count
        weather_count = state['weather_count'] / self.turn_count_scaling

        # terrain turn count
        terrain_count = state['terrain_count'] / self.turn_count_scaling

        # n turns trick room has been active
        trick_room_count = state['trick_room_count'] / self.turn_count_scaling

        # n turns magic room has been active
        magic_room_count = state['magic_room_count'] / self.turn_count_scaling

        # n turns gravity has been active
        gravity_count = state['gravity_count'] / self.turn_count_scaling

        return torch.tensor(
            weather +
            terrain +
            [
                trick_room,
                magic_room,
                gravity,
                weather_count,
                terrain_count,
                trick_room_count,
                magic_room_count,
                gravity_count
            ],
            dtype=torch.float
        )

    def _transform_side(self, side: Dict[str, Any]) -> torch.tensor:
        """
        Side_conditions
        Wish
        Future_sight
        Healing_wish
        Trapped
        Has_active
        N_pokemon
        """

        # one-hot encode side conditions
        side_conditions = []
        for s_con in SIDE_CONDITIONS:
            if s_con in side['side_conditions']:
                count = side['side_conditions'].get(s_con)
                side_conditions.append(count)
            else:
                side_conditions.append(0)

        # two wish variables: [turn, amount]
        wish = [side['wish']['countdown'], side['wish']['hp_amount'] / self.wish_scaling]

        # future sight turn count
        future_sight = side['future_sight']['countdown']

        # 1 if the Healing wish/Lunar dance effect is incoming, 0 otherwise
        healing_wish = int(side['healing_wish'])

        # 1 if side's active is trapped, 0 otherwise
        trapped = int(side['trapped'])

        # 1 if the side's active pokemon is alive, 0 otherwise
        has_active = int(not side['active']['fainted'])

        # n amount of pokemon the player has
        n_pokemon = (len(side['reserve']) + 1) / self.n_pokemon_scaling

        player_attributes = side_conditions + wish + [future_sight, healing_wish, trapped, has_active, n_pokemon]
        return torch.tensor(player_attributes, dtype=torch.float)

    def _transform_pokemon(self, state: Dict[str, Any]) -> Dict[str, torch.tensor]:
        """
        Species
        Items
        Abilities
        Moves
        Pokemon attributes
        """
        out = {}

        # shuffle reserve pokemon positions
        if self.shuffle_pokemon:
            random.shuffle(state[self.p1]['reserve'])
            random.shuffle(state[self.p2]['reserve'])

        p1_team = [state[self.p1]['active']] + state[self.p1]['reserve']
        p2_team = [state[self.p2]['active']] + state[self.p2]['reserve']

        # shuffle move positions
        if self.shuffle_moves:
            for pokemon in p1_team + p1_team:
                random.shuffle(pokemon['moves'])

        species = []
        items = []
        abilities = []
        pokemon_attributes = []

        moves = []
        move_attributes = []

        # for each player, collect the pokemon data
        for team in [p1_team, p2_team]:
            team_size = len(team)

            # pokemon species
            pkmn_names = [
                FORM_LOOKUP.get(name) if FORM_LOOKUP.get(name) else name
                for name in [pokemon['name'] for pokemon in team]
            ]
            species.append(
                [
                    SPECIES[pkmn_names[i]] if i < team_size else 0
                    for i in range(6)
                ]
            )

            # pokemon items
            item_names = [
                name if ITEMS.get(name) else "USELESS_ITEM"
                for name in [pokemon['item'] for pokemon in team]
            ]
            items.append(
                [
                    ITEMS[item_names[i]] if i < team_size else 0
                    for i in range(6)
                ]
            )

            # pokemon abilites
            ability_names = [pokemon['ability'] for pokemon in team]
            abilities.append(
                [
                    ABILITIES[ability_names[i]] if i < team_size else 0
                    for i in range(6)
                ]
            )

            # pokemon attributes
            pokemon_attributes.append(
                [
                    self._pokemon_attributes(team[i])
                    if i < team_size
                    else self._pokemon_attributes(team[0], return_zeros=True)
                    for i in range(6)
                ]
            )

            # pokemon moves
            moves.append(
                [
                    [
                        MOVES[team[i]["moves"][j]['name']]
                        if j < len(team[i]["moves"])
                        else 0
                        for j in range(4)
                    ]
                    if i < team_size
                    else [0, 0, 0, 0]
                    for i in range(6)
                ]
            )

            # move attributes
            move_attributes.append(
                [
                    [
                        self._move_attributes(team[i]["moves"][j])
                        if j < len(team[i]["moves"])
                        else self._move_attributes(team[0]["moves"][0], return_zeros=True)
                        for j in range(4)
                    ]
                    if i < team_size
                    else [self._move_attributes(team[0]["moves"][0], return_zeros=True)] * 4
                    for i in range(6)
                ]
            )

        out['species'] = torch.tensor(species, dtype=torch.long)
        out['items'] = torch.tensor(items, dtype=torch.long)
        out['abilites'] = torch.tensor(abilities, dtype=torch.long)
        out['moves'] = torch.tensor(moves, dtype=torch.long)

        out['move_attributes'] = torch.tensor(move_attributes, dtype=torch.float)
        out['pokemon_attributes'] = torch.tensor(pokemon_attributes, dtype=torch.float)

        return out

    def _pokemon_attributes(self, pokemon, return_zeros=False) -> list:
        """
        Types
        Stats
        Stat_changes
        Level
        n_moves
        Health
        Is_alive
        Sleep_countdown
        Status
        Volatile_status
        """

        # one-hot encode pokemon types
        types = []
        for t in TYPES:
            if t in pokemon['types']:
                types.append(1)
            else:
                types.append(0)

        # pokemon stats
        stats = [
            stat / self.stat_scaling for stat in
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
            change / self.stat_change_scaling for change in
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
        level = pokemon['level'] / self.level_scaling

        # n pokemon moves
        n_moves = len(pokemon['moves']) / self.n_moves_scaling

        # pokemon health range
        health = pokemon['hp'] / pokemon['maxhp']

        # pokemon is alive
        is_alive = int(not pokemon['fainted'])

        # n turns pokemon may stay asleep
        sleep_countdown = pokemon['sleep_countdown'] / self.sleep_count_scaling

        # one-hot encode status conditions
        status = []
        for s in STATUS:
            if s == pokemon['status']:
                status.append(1)
            else:
                status.append(0)

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

        out = [
            level,
            n_moves,
            health,
            is_alive,
            sleep_countdown,
        ]
        out.extend(types)
        out.extend(stats)
        out.extend(stat_changes)
        out.extend(status)
        out.extend(volatile_status)

        if return_zeros:
            return [0] * len(out)

        return out

    def _move_attributes(self, move, return_zeros=False) -> list:
        """
        disabled
        pp
        max_pp
        base_power
        accuracy
        priority
        is_used
        target_self
        min_hits
        max_hits
        category
        typing
        """
        name = move['name']

        # move is disabled
        disabled = int(move['disabled'])

        # current move pp
        pp = move['pp'] / self.pp_scaling

        # max move pp
        max_pp = int(MOVE_LOOKUP[name]['pp'] * 1.6) / self.pp_scaling

        # move base power
        base_power = MOVE_LOOKUP[name]['basePower'] / self.bp_scaling

        # move accuracy
        accuracy = (MOVE_LOOKUP[name]['accuracy'] if MOVE_LOOKUP[name]['accuracy'] != 1 else 100) / self.acc_scaling

        # move priority
        priority = MOVE_LOOKUP[name]['priority'] / self.priority_scaling

        # move has been used
        is_used = int(pp < max_pp)

        # move targets the user
        target_self = int(MOVE_LOOKUP[name]['target'] == 'self')

        # min and max amount of hits the move may consist of
        multi_hits = MOVE_LOOKUP[name].get('multihit')
        if not multi_hits:
            multi_hits = [1, 1]
        elif type(multi_hits) != list:
            multi_hits = [multi_hits, multi_hits]
        min_hits, max_hits = multi_hits[0] / self.multihit_scaling, multi_hits[1] / self.multihit_scaling

        # # one-hot encode move category type
        # move_category = []
        # for c in MOVE_CATEGORIES:
        #     if c == MOVE_LOOKUP[name]['category']:
        #         move_category.append(1)
        #     else:
        #         move_category.append(0)
        #
        # # one-hot encode move typing
        # typing = []
        # for t in TYPES:
        #     if t == MOVE_LOOKUP[name]['type']:
        #         typing.append(1)
        #     else:
        #         typing.append(0)

        out = [
            disabled,
            pp,
            max_pp,
            base_power,
            accuracy,
            priority,
            is_used,
            target_self,
            min_hits,
            max_hits,
        ]
        # out.extend(move_category)
        # out.extend(typing)

        if return_zeros:
            return [0] * len(out)

        return out

