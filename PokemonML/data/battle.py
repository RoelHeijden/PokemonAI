from collections import defaultdict
from collections import namedtuple

from Showdown_Pmariglia import constants
from Showdown_Pmariglia.data import all_move_json
from Showdown_Pmariglia.data import pokedex
from Showdown_Pmariglia.showdown.engine.helpers import normalize_name
from Showdown_Pmariglia.showdown.engine.helpers import calculate_stats


LastUsedMove = namedtuple('LastUsedMove', ['pokemon_name', 'move', 'turn'])


class Battle:

    def __init__(self, battle_tag):
        self.battle_tag = battle_tag
        self.user = Battler()
        self.opponent = Battler()
        self.weather = None
        self.field = None
        self.trick_room = False

        self.turn = False

        self.battle_type = None
        self.generation = None
        self.time_remaining = None

    def initialize_team_preview(self, p1_team, p2_team):
        for pkmn_info in p1_team:

            evs = pkmn_info['evs']
            ivs = pkmn_info['ivs']

            pkmn = Pokemon(
                pkmn_info['species'],
                pkmn_info['level'],
                pkmn_info.get('gender'),
                pkmn_info['nature'],
                (evs['hp'], evs['atk'], evs['def'], evs['spa'], evs['spd'], evs['spe']),
                (ivs['hp'], ivs['atk'], ivs['def'], ivs['spa'], ivs['spd'], ivs['spe']),
                pkmn_info['ability'],
                pkmn_info['moves'],
                pkmn_info['item']
            )
            self.user.reserve.append(pkmn)

        for pkmn_info in p2_team:

            evs = pkmn_info['evs']
            ivs = pkmn_info['ivs']

            pkmn = Pokemon(
                pkmn_info['species'],
                pkmn_info['level'],
                pkmn_info.get('gender'),
                pkmn_info['nature'],
                (evs['hp'], evs['atk'], evs['def'], evs['spa'], evs['spd'], evs['spe']),
                (ivs['hp'], ivs['atk'], ivs['def'], ivs['spa'], ivs['spd'], ivs['spe']),
                pkmn_info['ability'],
                pkmn_info['moves'],
                pkmn_info['item']
            )
            self.opponent.reserve.append(pkmn)

    def to_dict(self):
        return {
            'turn': self.turn,
            'p1': self.user.to_dict(),
            'p2': self.opponent.to_dict(),
            'weather': self.weather,
            'field': self.field,
            'trick_room': self.trick_room,
        }

    def extract_game_state(self):
        return {
            'turn': self.turn,
            'p1': self.user.to_game_state(),
            'p2': self.opponent.to_game_state(),
            'weather': self.weather,
            'field': self.field,
            'trick_room': self.trick_room,
        }


class Battler:

    def __init__(self):
        self.active = None
        self.reserve = []
        self.side_conditions = defaultdict(lambda: 0)

        self.name = None
        self.trapped = False
        self.wish = (0, 0)
        self.future_sight = (0, 0)

        self.account_name = None

        self.last_used_move = LastUsedMove('', '', 0)

    def lock_active_pkmn_first_turn_moves(self):
        # disable firstimpression and fakeout if the last_used_move was not a switch
        if self.last_used_move.pokemon_name == self.active.name:
            for m in self.active.moves:
                if m.name in constants.FIRST_TURN_MOVES:
                    m.disabled = True

    def lock_active_pkmn_status_moves_if_active_has_assaultvest(self):
        if self.active.item == 'assaultvest':
            for m in self.active.moves:
                if all_move_json[m.name][constants.CATEGORY] == constants.STATUS:
                    m.disabled = True

    def choice_lock_moves(self):
        # if the active pokemon has a choice item and their last used move was by this pokemon -> lock their other moves
        if self.active.item in constants.CHOICE_ITEMS and self.last_used_move.pokemon_name == self.active.name:
            for m in self.active.moves:
                if m.name != self.last_used_move.move:
                    m.disabled = True

    def taunt_lock_moves(self):
        if constants.TAUNT in self.active.volatile_statuses:
            for m in self.active.moves:
                if all_move_json[m.name][constants.CATEGORY] == constants.STATUS:
                    m.disabled = True

    def lock_moves(self):
        self.choice_lock_moves()
        self.lock_active_pkmn_status_moves_if_active_has_assaultvest()
        self.lock_active_pkmn_first_turn_moves()
        self.taunt_lock_moves()

    def get_switches(self):
        if self.trapped:
            return []

        switches = []
        for pkmn in self.reserve:
            if pkmn.hp > 0:
                switches.append("{} {}".format(constants.SWITCH_STRING, pkmn.name))
        return switches

    def check_if_trapped(self, battle):
        """ sets the 'trapped' flag"""
        pkmn = self.active
        opp_pkmn = battle.user.active

        # check via items/abilities
        if pkmn.item == 'shedshell' or 'ghost' in pkmn.types or pkmn.ability == 'shadowtag':
            self.trapped = False
            return
        elif constants.PARTIALLY_TRAPPED in pkmn.volatile_statuses:
            self.trapped = True
            return
        elif constants.TRAPPED in pkmn.volatile_statuses:
            self.trapped = True
            return
        elif opp_pkmn.ability == 'shadowtag':
            self.trapped = True
            return
        elif opp_pkmn.ability == 'magnetpull' and 'steel' in pkmn.types:
            self.trapped = True
            return
        elif opp_pkmn.ability == 'arenatrap' and pkmn.is_grounded():
            self.trapped = True
            return
        else:
            self.trapped = False

    def to_dict(self):
        return {
            'active': self.active.to_dict() if self.active else None,
            'reserve': [p.to_dict() for p in self.reserve],
            'side_conditions': dict(self.side_conditions),
            'trapped': self.trapped,
            'last_used_move': {
                'user': self.last_used_move.pokemon_name,
                'move': self.last_used_move.move,
                'turn': self.last_used_move.turn
            },
            'wish': {'countdown': self.wish[0], 'hp_amount': int(self.wish[1])},
            'future_sight': {'countdown': self.future_sight[0], 'user': self.future_sight[1]},
        }

    def to_game_state(self):
        return {
            'active': self.active.to_game_state_active() if self.active else None,
            'reserve': {p.name: p.to_game_state_reserve() for p in self.reserve},
            'side_conditions': dict(self.side_conditions),
            'trapped': self.trapped,
            'last_used_move': {
                'user': self.last_used_move.pokemon_name,
                'move': self.last_used_move.move,
                'turn': self.last_used_move.turn
            },
            'wish': {'countdown': self.wish[0], 'hp_amount': int(self.wish[1])},
            'future_sight': {'countdown': self.future_sight[0], 'user': self.future_sight[1]},
        }


class Pokemon:
    def __init__(self, name: str, level: int, gender: str, nature: str, evs: tuple, ivs: tuple, ability: str, moves: list, item: str):
        self.name = normalize_name(name)
        self.base_name = self.name
        self.level = level
        self.gender = gender
        self.nature = normalize_name(nature)
        self.evs = evs
        self.ivs = ivs
        self.ability = ability
        self.moves = [self.add_move(move) for move in moves]
        self.item = item

        try:
            self.base_stats = pokedex[self.name][constants.BASESTATS]
        except KeyError:
            self.name = [k for k in pokedex if self.name.startswith(k)][0]
            self.base_stats = pokedex[self.name][constants.BASESTATS]

        self.types = pokedex[self.name][constants.TYPES]
        self.stats = calculate_stats(self.base_stats, self.level, ivs=ivs, evs=evs, nature=self.nature)

        self.max_hp = self.stats.get(constants.HITPOINTS)
        self.hp = self.max_hp
        if self.name == 'shedinja':
            self.max_hp = 1
            self.hp = 1

        self.fainted = False
        self.status = None
        self.volatile_statuses = []

        self.boosts = {'attack': 0, 'defense': 0, 'special-attack': 0, 'special-defense': 0,
                       'speed': 0, 'accuracy': 0, 'evasion': 0}

        self.can_mega_evo = False
        self.can_ultra_burst = False
        self.can_dynamax = False
        self.is_mega = False

        self.can_have_assaultvest = True
        self.can_have_choice_item = True
        self.can_not_have_band = False
        self.can_not_have_specs = False
        self.can_have_life_orb = True
        self.can_have_heavydutyboots = True

    def is_grounded(self):
        if self.item == 'ironball':
            return True
        if 'flying' in self.types or self.ability == 'levitate' or self.item == 'airballoon':
            return False
        return True

    def is_alive(self):
        return self.hp > 0

    def add_move(self, move_name: str):
        try:
            new_move = Move(move_name)
            return new_move
        except KeyError:
            return None

    def get_move(self, move_name: str):
        for m in self.moves:
            if m.name == normalize_name(move_name):
                return m
        return None

    def forced_move(self):
        if "phantomforce" in self.volatile_statuses:
            return "phantomforce"
        elif "shadowforce" in self.volatile_statuses:
            return "shadowforce"
        elif "dive" in self.volatile_statuses:
            return "dive"
        elif "dig" in self.volatile_statuses:
            return "dig"
        elif "bounce" in self.volatile_statuses:
            return "bounce"
        elif "fly" in self.volatile_statuses:
            return "fly"
        else:
            return None

    def to_dict(self):
        return {
            'name': self.name,
            'base_name': self.base_name,
            'level': self.level,
            'gender': self.gender,
            'types': self.types,
            'hp': int(self.hp),
            'max_hp': self.max_hp,
            'ability': self.ability,
            'item': self.item,
            'base_stats': self.base_stats,
            'stats': self.stats,
            'nature': self.nature,
            'evs': {stat: self.evs[i] for i, stat in enumerate(self.stats)},
            'ivs': {stat: self.ivs[i] for i, stat in enumerate(self.stats)},
            'stat_changes': self.boosts,
            'status_condition': self.status,
            'volatile_status': list(set(self.volatile_statuses)),
            'moves': [m.to_dict() for m in self.moves],
            'fainted': self.fainted,
        }

    def to_game_state_active(self):
        return {
            'name': self.name,
            'types': self.types,
            'hp': int(self.hp),
            'max_hp': self.max_hp,
            'ability': self.ability,
            'item': self.item,
            'stats': self.stats,
            'stat_changes': self.boosts,
            'status_condition': self.status,
            'volatile_status': list(set(self.volatile_statuses)),
            'moves': [m.to_game_state() for m in self.moves],
            'fainted': self.fainted,
        }

    def to_game_state_reserve(self):
        return {
            'name': self.name,
            'hp': int(self.hp),
            'max_hp': self.max_hp,
            'ability': self.ability,
            'item': self.item,
            'status_condition': self.status,
            'moves': [{'name': m.name, 'pp': m.current_pp} for m in self.moves],
            'fainted': self.fainted,
        }

    def __eq__(self, other):
        return self.name == other.name and self.level == other.level

    def __repr__(self):
        return "{}, level {}".format(self.name, self.level)


class Move:
    def __init__(self, name):
        name = normalize_name(name)
        if constants.HIDDEN_POWER in name and not name.endswith(constants.HIDDEN_POWER_ACTIVE_MOVE_BASE_DAMAGE_STRING):
            name = "{}{}".format(name, constants.HIDDEN_POWER_ACTIVE_MOVE_BASE_DAMAGE_STRING)
        move_json = all_move_json[name]

        self.name = name
        self.base_power = move_json['basePower']
        self.accuracy = move_json['accuracy']
        self.category = move_json['category']
        self.type = move_json['type']
        self.max_pp = int(move_json.get(constants.PP) * 1.6)

        self.disabled = False
        self.can_z = False
        self.current_pp = self.max_pp

    def to_dict(self):
        return {
            "name": self.name,
            "base_power": self.base_power,
            "accuracy": self.accuracy,
            "category": self.category,
            "type": self.type,
            "current_pp": self.current_pp,
            "max_pp": self.max_pp,
            "disabled": self.disabled,
        }

    def to_game_state(self):
        return {
            "name": self.name,
            "current_pp": self.current_pp,
            "disabled": self.disabled,
        }

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return "{}".format(self.name)
