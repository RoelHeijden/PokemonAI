import json
import os
import numpy as np
import csv
import time
import ujson
import math


"""
STATES
    Collect: 
        - possible status conditions
        - possible volatile statuses
        - possible side conditions
        - pokemon species (check nickname bug?)
        - weathers (check if names are right)

    Check:
        - how moves like toxic, taunt, encore, disable, lightscreen and volatiles are represented (do they have turn counts?).
        - sleep representation after Rest
        - how layers of spikes, etc are represented
        - if Protect being used is represented in volatiles

    Fix:
        - represent choice-lock disabled moves in end-of-turn state: if a pokemon get's a KO with a choice item, 
          the choice lock is set, but moves are not set to disabled until the next state (the start of next turn)
          
          solution: if choice-lock, each moves not moves['last_used'] = Disabled


CSV    
    Create:
        - train/test split (90%/10%)
        - test split with three categories:
            - turn 1
            - mid-game
            - endgame 
    
    Check:
        - that reverse_state() is correct
        - that the header is correct
"""


def main():
    converter = Converter()

    path_in = 'test_games'

    path_out = os.getcwd()
    file_name = 'training_states.csv'
    file_out = os.path.join(path_out, file_name)

    headers = converter.create_header()
    write_csv_headers(file_out, headers)

    convert_all_games(converter, path_in, file_out)


def write_csv_headers(file, headers):
    """ writes multiple headers to csv file """
    with open(file, 'w') as f_out:
        writer = csv.writer(f_out)
        for header in headers:
            writer.writerow(header)
        f_out.close()


def convert_all_games(converter, path_in, file_out, min_game_length=6):
    """ converts each game into one or two number-converted game states """

    print("starting\n")

    # collect file names
    files = sorted([os.path.join(path_in, file_name)
                    for file_name in os.listdir(path_in)])

    print(f'{len(files)} files found\n')

    start_time = time.time()
    n_states = 0
    n_games = 0

    games_too_short = {}

    # open each game_states file
    for f in files:
        with open(f) as f_in, open(file_out, 'a') as f_out:

            all_states = json.load(f_in)

            # skip game if game doesn't last long enough
            if len(all_states) < min_game_length:
                games_too_short[f] = len(all_states)
                continue

            # select two random states to write
            states = pick_random_states(all_states, min_turns_apart=math.floor(min_game_length/2))

            # write states to file
            for s in states:
                input_array = converter.convert_state(s)
                np.savetxt(f_out, input_array, delimiter=",")

                n_states += 1

            n_games += 1

            if n_games % 10000 == 0:
                print(f'{n_games} games processed')
                print(f'{n_states} states converted')
                print(f'runtime: {round(time.time() - start_time, 1)}s\n')

    print("finished")
    print(f'{n_games} games processed')
    print(f'{n_states} states converted')
    print(f'runtime: {round(time.time() - start_time, 1)}s\n')

    print(f"Games skipped because they lasted shorter than {min_game_length} turns")
    print(ujson.dumps(games_too_short))


def pick_random_states(all_states, min_turns_apart=1):
    """ selects two random non-preview states from a game_state list,
        one for each of the players POV and at least n turns apart """

    state_idx1 = np.random.randint(1, len(all_states) - min_turns_apart)
    state_idx2 = np.random.randint(state_idx1 + min_turns_apart, len(all_states))

    state1 = all_states[state_idx1]
    state2 = all_states[state_idx2]

    return [state1, reverse_pov(state2)]


def reverse_pov(state):
    """ flips the POV of a state """
    if state['winner'] == 'p1':
        state['winner'] = 'p2'
    elif state['winner'] == 'p2':
        state['winner'] = 'p1'
    else:
        raise ValueError(f'State winner cannot be {state["winner"]}')

    hold_my_beer = state['state']['p1']
    state['state']['p1'] = state['state']['p2']
    state['state']['p2'] = hold_my_beer

    hold_my_beer = state['p1_move']
    state['p1_move'] = state['p2_move']
    state['p2_move'] = hold_my_beer

    hold_my_beer = state['p1rating']
    state['p1rating'] = state['p2rating']
    state['p2rating'] = hold_my_beer

    return state


class Converter:
    def __init__(self):
        self.pokemon_list = self.init_category('pokemon.json')
        self.item_list = self.init_category('items.json')
        self.ability_list = self.init_category('abilities.json')
        self.move_list = self.init_category('moves.json')
        self.weather_list = self.init_category('weathers.json')
        self.terrain_list = self.init_category('terrains.json')
        self.type_list = self.init_category('types.json')
        self.status_list = self.init_category('status.json')
        self.move_category_list = self.init_category('move_categories.json')
        self.volatile_status_list = []  # TBD
        self.side_condition_list = []  # TBD

        self.move_lookup = json.load(open('lookups/move_lookup.json'))

    @staticmethod
    def init_category(file_name, relative_path='categories'):
        path = os.path.join(relative_path, file_name)
        file = open(path, 'r')
        return np.asarray(list(json.load(file)))

    def convert_state(self, game_state):
        """ convert state information to an array numbers """

        p1_win = np.asarray([1 if game_state['winner'] == 'p1' else -1])
        p1_move = np.asarray([game_state['p1_move']])
        p2_move = np.asarray([game_state['p2_move']])
        p1_rating = np.asarray([game_state['p1rating']])
        p2_rating = np.asarray([game_state['p2rating']])
        avg_rating = np.asarray([game_state['average_rating']])
        rated_battle = np.asarray([game_state['rated_battle']])
        room_id = np.asarray([game_state['roomid']])
        turn = np.asarray([game_state['turn']])

        fields = self.convert_fields(game_state['state'])
        player1 = self.convert_side(game_state['state']['p1'])
        player2 = self.convert_side(game_state['state']['p2'])

        return np.concatenate(p1_win, p1_move, p2_move, p1_rating, p2_rating, avg_rating,
                              rated_battle, room_id, turn, fields, player1, player2)

    def convert_fields(self, state):
        # one-hot-encode weather
        weather = np.zeros(len(self.weather_list))
        weather[np.where(self.weather_list == state['weather'])] = 1

        weather_count = np.asarray([state['weather_count']])

        # one-hot-encode terrain
        terrain = np.zeros(len(self.terrain_list))
        terrain[np.where(self.terrain_list == state['terrain'])] = 1

        terrain_count = np.asarray([state['terrain_count']])

        trick_room = np.asarray([int(state['trick_room'])])

        trick_room_count = np.asarray([state['trick_room_count']])

        return np.concatenate(weather, weather_count, terrain, terrain_count, trick_room, trick_room_count)

    def convert_side(self, side):
        # one-hot-encode side conditions
        side_conditions = np.zeros(len(self.side_condition_list))
        for v in side['side_conditions']:
            side_conditions[np.where(self.side_condition_list == v)] = 1

        wish = np.asarray(side['wish'])

        future_sight = np.asarray([side['future_sight'][0]])

        # if active pokemon is knocked out, reserve[0] is the (fainted) active pokemon
        if side['active']:
            active = self.convert_pokemon(side['active'], is_active=True)
            reserve = np.concatenate([self.convert_pokemon(pkmn) for pkmn in side['reserve']])
        else:
            active = self.convert_pokemon(side['reserve'][0], is_active=False)
            reserve = np.concatenate([self.convert_pokemon(pkmn) for pkmn in side['reserve'][1:]])

        return np.concatenate(side_conditions, wish, future_sight, active, reserve)

    def convert_pokemon(self, pokemon, is_active=False):
        # one-hot-encode species
        species = np.zeros(len(self.pokemon_list))
        species[np.where(self.pokemon_list == pokemon['id'])] = 1

        # one-hot-encode ability
        ability = np.zeros(len(self.ability_list))
        ability[np.where(self.ability_list == pokemon['ability'])] = 1

        # one-hot-encode types
        types = np.zeros(len(self.type_list))
        for t in pokemon['types']:
            types[np.where(self.type_list == t)] = 1

        # one-hot-encode item
        item = np.zeros(len(self.item_list))
        item[np.where(self.item_list == pokemon['item'])] = 1

        has_item = np.asarray([int(pokemon['item'] != "")])

        active = np.asarray([int(is_active)])

        level = np.asarray([pokemon['level']])

        stats = np.asarray([
            pokemon['maxhp'],
            pokemon['attack'],
            pokemon['defense'],
            pokemon['special_attack'],
            pokemon['special_defense'],
            pokemon['speed'],
        ])

        stat_changes = np.asarray([
            pokemon['attack_boost'],
            pokemon['defense_boost'],
            pokemon['special_attack_boost'],
            pokemon['special_defense_boost'],
            pokemon['speed_boost'],
            pokemon['accuracy_boost'],
            pokemon['evasion_boost']
        ])

        health = np.asarray([int(pokemon['hp'] / pokemon['maxhp'] * 100)])

        fainted = np.asarray([int(pokemon['status'] == 'fnt')])

        # one-hot-encode status conditions
        status = np.zeros(len(self.status_list))
        status[np.where(self.status_list == pokemon['status'])] = 1

        # one-hot-encode volatile_status
        volatile_status = np.zeros(len(self.volatile_status_list))
        for v in pokemon['volatile_status']:
            volatile_status[np.where(self.volatile_status_list == v)] = 1

        first_turn_out = np.asarray([int(pokemon['first_turn_out'])])

        moves = np.concatenate([self.convert_move(move) for move in pokemon['moves']])

        return np.concatenate(species, ability, types, item, has_item, active, level, stats,
                              stat_changes, health, fainted, status, volatile_status, first_turn_out, moves)

    def convert_move(self, move):
        move_name = move['id']

        # one-hot-encode moves
        moves = np.zeros(len(self.move_list))
        moves[np.where(self.move_list == move_name)] = 1

        # one-hot-encode typing
        typing = np.zeros(len(self.type_list))
        typing[np.where(self.type_list == self.move_lookup[move_name]['type'])] = 1

        # one-hot-encode move category
        move_category = np.zeros(len(self.move_category_list))
        move_category[np.where(self.move_category_list == self.move_lookup[move_name]['category'])] = 1

        base_power = np.asarray([self.move_lookup[move_name]['basePower']])

        current_pp = np.asarray([move['pp']])

        max_pp = np.asarray(['maxpp'])

        target_self = np.asarray([int(move['target'] == 'self')])

        disabled = np.asarray([int(move['disabled'])])

        last_used_move = np.asarray([int(move['last_used_move'])])

        used = np.asarray([int(move['used'])])

        priority = np.asarray([self.move_lookup[move_name]['priority']])

        return np.concatenate(moves, typing, move_category, base_power, current_pp, max_pp,
                              target_self, disabled, last_used_move, used, priority)

    def create_header(self):
        """ returns a 4*m header """
        move_header = (
                ['move' for _ in self.move_list] +
                ['type' for _ in self.type_list] +
                ['move category' for _ in self.move_category_list] +
                ['base power'] +
                ['current pp'] +
                ['max pp'] +
                ['target self'] +
                ['disabled'] +
                ['last used move'] +
                ['used'] +
                ['priority']
        )

        pokemon_header = (
                ['name' for _ in self.pokemon_list] +
                ['ability' for _ in self.ability_list] +
                ['type' for _ in self.type_list] +
                ['item' for _ in self.item_list] +
                ['has item'] +
                ['is active'] +
                ['level'] +
                ['stats'] * 6 +
                ['stat changes'] * 7 +
                ['health'] +
                ['fainted'] +
                ['status' for _ in self.status_list] +
                ['volatile status' for _ in self.volatile_status_list] +
                ['first turn out'] +
                ['move1' for _ in move_header] +
                ['move2' for _ in move_header] +
                ['move3' for _ in move_header] +
                ['move4' for _ in move_header]
        )

        player_header = (
                ['side condition' for _ in self.side_condition_list] +
                ['wish'] * 2 +
                ['future sight'] +
                ['active pokemon' for _ in pokemon_header] +
                ['reserve pokemon1' for _ in pokemon_header] +
                ['reserve pokemon2' for _ in pokemon_header] +
                ['reserve pokemon3' for _ in pokemon_header] +
                ['reserve pokemon4' for _ in pokemon_header] +
                ['reserve pokemon5' for _ in pokemon_header]
        )

        game_header = (
                ['p1 win'] +
                ['p1_move'] +
                ['p2_move'] +
                ['p1 rating'] +
                ['p2 rating'] +
                ['avg rating'] +
                ['rated battle'] +
                ['room id'] +
                ['turn'] +
                ['weather' for _ in self.weather_list] +
                ['weather count'] +
                ['terrain' for _ in self.terrain_list] +
                ['terrain count'] +
                ['trick room'] +
                ['trick room count'] +
                ['p1' for _ in player_header] +
                ['p2' for _ in player_header]
        )

        first_header = game_header

        second_header = (
                game_header[:game_header.index('p1')] +
                player_header +
                player_header
        )

        third_header = (
                game_header[:game_header.index('p1')] +
                (
                        player_header[:player_header.index('active pokemon')] +
                        pokemon_header * 6
                ) * 2
        )

        fourth_header = (
                game_header[:game_header.index('p1')] +
                (
                        player_header[:player_header.index('active pokemon')] +
                        (
                                pokemon_header[:pokemon_header.index('move1')] +
                                move_header * 4
                        ) * 6
                ) * 2
        )

        return [first_header, second_header, third_header, fourth_header]


if __name__ == "__main__":
    main()









