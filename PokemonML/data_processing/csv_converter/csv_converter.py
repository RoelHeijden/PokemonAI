import pandas as pd
import ujson
import json
import os
import numpy as np
import csv
import time


"""
STATES
    Check:
        - presence of weird nicknames under 'species' in p1team/p2team jsons
        - how moves like taunt, encore, disable, lightscreen and volatiles are represented.
        - sleep representation after Rest
        - fainted status is represented as status['fnt']
    
    add:
        - counters for: toxic, sleep, weather, terrain, trickroom, encore, side_conditions, volatiles, etc.
        - current turn 
        - first_turn_out
        - last_used_move
        - player ratings
        
    Fix:
        - represent choice-lock disabled moves in end-of-turn state


CSV
    Use:
        - only states with 2 active Pokemon
        - two states per game (min 10 turns apart)
    
    Create:
        - train/test split (90%/10%)
        - test split with three categories:
            - turn 1
            - mid-game
            - endgame 
"""


def main():
    converter = Converter()

    path_in = 'test_games'

    path_out = os.getcwd()
    file_name = 'training_states.csv'
    file_out = os.path.join(path_out, file_name)

    headers = converter.create_header()
    write_csv_headers(file_out, headers)

    convert_games(converter, path_in, file_out)


def write_csv_headers(file, headers):
    """ writes multiple headers to csv file """
    with open(file, 'w') as f_out:
        writer = csv.writer(f_out)
        for header in headers:
            writer.writerow(header)
        f_out.close()


def convert_games(converter, path_in, file_out):
    """ converts each game into one or two number-converted game states """

    print("starting\n")

    # collect file names
    files = sorted([os.path.join(path_in, file_name)
                    for file_name in os.listdir(path_in)])

    print(f'{len(files)} files found\n')

    start_time = time.time()
    n_states = 0
    n_games = 0

    # open each game_states file
    for f in files:
        with open(f) as f_in, open(file_out, 'a') as f_out:

            # select states to write
            all_states = json.load(f_in)
            states = pick_random_states(all_states)

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
    print(f'runtime: {round(time.time() - start_time, 1)}s ')


def pick_random_states(all_states, max_in_between=6, min_game_length=15):
    """ selects one or two random states from a game_state list, depending on the game length """

    # extract two states if game has 20 or more turns
    if len(all_states) > min_game_length:

        # iterate until two random states are found where no active pokemon are None
        while True:
            turns_in_between = np.random.randint(max_in_between, len(all_states))
            state1_idx = np.random.randint(0, len(all_states) - turns_in_between)
            state2_idx = state1_idx + turns_in_between
            state1 = all_states[state1_idx]
            state2 = all_states[state2_idx]

            if state1['state']['p1']['active'] and state1['state']['p2']['active'] and \
               state2['state']['p1']['active'] and state2['state']['p2']['active']:
                return [state1, state2]

    else:
        # iterate until a random state is found where no active pokemon are None
        while True:
            state_idx = np.random.randint(0, len(all_states))
            state = all_states[state_idx]

            if state['state']['p1']['active'] and state['state']['p2']['active']:
                return [state]


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

    @staticmethod
    def init_category(file_name, relative_path='categories'):
        path = os.path.join(relative_path, file_name)
        file = open(path, 'r')
        return list(json.load(file))

    def convert_state(self, game_state):
        """ convert state information to an array numbers """

        p1_win = np.asarray([1 if game_state['winner'] == 'p1' else -1])
        turn = np.asarray([game_state['turn']])
        fields = self.convert_fields(game_state['state'])
        player1 = self.convert_side(game_state['state']['p1'])
        player2 = self.convert_side(game_state['state']['p2'])

        return p1_win + turn + fields + player1 + player2

    def convert_fields(self, state):
        weather = np.asarray([])
        terrain = np.asarray([])
        trick_room = np.asarray([int(state['trick_room'])])

        return []

    def convert_side(self, side):
        side_conditions = np.asarray([])
        wish = np.asarray([])
        future_sight = np.asarray([])
        active = self.convert_pokemon(side['active'])
        reserve = np.concatenate([self.convert_pokemon(pkmn) for pkmn in side['reserve']])

        return []

    def convert_pokemon(self, pokemon):
        species = []
        ability = []
        types = []
        item = []
        stats = []

        max_hp = [pokemon['maxhp']]
        current_hp = [pokemon['hp']]

        stat_changes = []
        status = []
        volatile_status = []

        fainted = [int(pokemon['status'] == 'fnt')]

        moves = [self.convert_move(move) for move in pokemon['moves']]
        last_used_move = []

        return []

    def convert_move(self, move):
        move = np.asarray([])
        typing = np.asarray([])
        move_category = np.asarray([])
        base_power = np.asarray([])
        current_pp = np.asarray([])
        max_pp = np.asarray([])
        disabled = np.asarray([])

        return []

    def create_header(self):
        """ returns a 4*m header """
        move_header = (
                ['move' for _ in self.move_list] +
                ['type' for _ in self.type_list] +
                ['move category' for _ in self.move_category_list] +
                ['base power'] +
                ['current pp'] +
                ['max pp'] +
                ['disabled']
        )

        pokemon_header = (
                ['name' for _ in self.pokemon_list] +
                ['type' for _ in self.type_list] +
                ['item' for _ in self.item_list] +
                ['ability' for _ in self.ability_list] +
                ['stats'] * 6 +
                ['max hp'] +
                ['current hp'] +
                ['fainted'] +
                ['stat changes'] * 7 +
                ['last used move'] * 4 +
                ['status' for _ in self.status_list] +
                ['volatile status' for _ in self.volatile_status_list] +
                ['move1' for _ in move_header] +
                ['move2' for _ in move_header] +
                ['move3' for _ in move_header] +
                ['move4' for _ in move_header]
        )

        player_header = (
                ['player rating'] +
                ['player move'] +
                ['side condition' for _ in self.side_condition_list] +
                ['wish'] * 2 +
                ['future sight'] * 2 +
                ['active pokemon' for _ in pokemon_header] +
                ['reserve pokemon1' for _ in pokemon_header] +
                ['reserve pokemon2' for _ in pokemon_header] +
                ['reserve pokemon3' for _ in pokemon_header] +
                ['reserve pokemon4' for _ in pokemon_header] +
                ['reserve pokemon5' for _ in pokemon_header]
        )

        game_header = (
                ['p1_win'] +
                ['turn'] +
                ['weather' for _ in self.weather_list] +
                ['terrain' for _ in self.terrain_list] +
                ['trick room'] +
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









