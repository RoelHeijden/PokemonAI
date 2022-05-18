import json
import os
import numpy as np
import csv
import time
import ujson
import math
import logging


logging.basicConfig(level=logging.DEBUG, format='%(message)s')


"""

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
        

######################################################################################

CSV implements:
    - pokemon forms
    - volatile status 2-turn move check
    
    - sleep countdown
    - counts for toxic, spikes and tspikes
    - perish counter
    - header lengths for side_conditions and volatile_status
    
    - remove first_turn_out
    - remove last_used_move


13. test states




"""


def main():
    converter = Converter()

    path_in = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/python/PokemonML/data/granted_data_testing/processed_data'

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


def convert_all_games(converter, path_in, file_out, min_game_length=3):
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
    ignore_list = json.load(open('games_to_ignore.json', 'r'))

    # open each game_states file
    for f in files:
        with open(f) as f_in, open(file_out, 'a') as f_out:

            all_states = json.load(f_in)

            # skip game if in the ignore list
            if ignore_list.get(str(all_states[0]['roomid'])):
                continue

            # skip game if game doesn't last long enough
            if len(all_states) < min_game_length:
                games_too_short[f] = len(all_states)
                continue

            # select two random states to write
            states = pick_random_states(all_states, min_turns_apart=math.floor(min_game_length/3))

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

    print(f"{len(games_too_short)} games skipped because they lasted shorter than {min_game_length} turns")
    print(ujson.dumps(games_too_short))


def pick_random_states(all_states, min_turns_apart=1):
    """ selects two random non-preview states from a game_state list,
        one for each of the players POV and at least n turns apart """

    # rolling from (1, ..) in order to skip the team_preview state
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
        # tie
        pass

    hold_my_beer = state['p1']
    state['p1'] = state['p2']
    state['p2'] = hold_my_beer

    hold_my_beer = state['p1rating']
    state['p1rating'] = state['p2rating']
    state['p2rating'] = hold_my_beer

    return state


class Converter:
    def __init__(self):
        self.pkmn_positions = self.init_category('pokemon.json')
        self.item_positions = self.init_category('items.json')
        self.ability_positions = self.init_category('abilities.json')
        self.move_positions = self.init_category('moves.json')
        self.weather_positions = self.init_category('weathers.json')
        self.terrain_positions = self.init_category('terrains.json')
        self.types_positions = self.init_category('types.json')
        self.status_positions = self.init_category('status.json')
        self.move_category_positions = self.init_category('move_categories.json')
        self.volatile_status_positions = {}  # TBD
        self.side_condition_positions = {}  # TBD

        self.move_lookup = json.load(open('lookups/move_lookup.json'))
        self.volatiles_to_ignore = json.load(open('lookups/volatiles_to_ignore.json'))

    @staticmethod
    def init_category(file_name, relative_path='categories'):
        """ opens a category file and converts it to a dictionary with the position index as value """
        path = os.path.join(relative_path, file_name)
        file = open(path, 'r')
        data = {key: i for i, key in enumerate(json.load(file))}
        return data

    def convert_state(self, game_state):
        """ convert state information to an array numbers """

        # [1] for a win, [-1] for a loss, [0] for a (rare) tie
        if game_state['winner']:
            p1_win = [1 if game_state['winner'] == 'p1' else -1]
        else:
            p1_win = [0]

        p1_rating = [game_state['p1rating']]
        p2_rating = [game_state['p2rating']]
        avg_rating = [game_state['average_rating']]
        rated_battle = [game_state['rated_battle']]
        room_id = [game_state['roomid']]
        turn = [game_state['turn']]

        fields = self.convert_fields(game_state)
        player1 = self.convert_side(game_state['p1'])
        player2 = self.convert_side(game_state['p2'])

        return np.asarray(p1_win + p1_rating + p2_rating + avg_rating +
                          rated_battle + room_id + turn + fields + player1 + player2)

    def convert_fields(self, state):
        # one-hot-encode weather
        weather = [0] * len(self.weather_positions)
        weather_index = self.weather_positions.get(state['weather'])
        if weather_index is not None:
            weather[weather_index] = 1
        else:
            logging.debug(f'weather "{state["weather"]}" does not exist in weathers.json')

        # n turns the weather has been active
        weather_count = [state['weather_count']]

        # one-hot-encode terrain
        terrain = [0] * len(self.terrain_positions)
        terrain_index = self.terrain_positions.get(state['terrain'])
        if terrain_index is not None:
            terrain[terrain_index] = 1
        else:
            logging.debug(f'terrain "{state["terrain"]}" does not exist in terrains.json')

        # n turns the terrain has been active
        terrain_count = [state['terrain_count']]

        # [1] if trick room is active, [0] otherwise
        trick_room = [int(state['trick_room'])]

        # n turns the trick room has been active
        trick_room_count = [state['trick_room_count']]

        return weather + weather_count + terrain + terrain_count + trick_room + trick_room_count

    def convert_side(self, side):
        # one-hot-encode side conditions
        side_conditions = [0] * len(self.side_condition_positions)
        for side_condition in side['side_conditions']:
            index = self.side_condition_positions.get(side_condition)
            if index is not None:
                side_conditions[index] = 1
            else:
                logging.debug(f'side condition "{side_condition}" not in side_conditions.json')

        # two wish variables: [turn, amount]
        wish = [side['wish']['countdown'], side['wish']['hp_amount']]

        # one future sight variable: [turn]
        future_sight = [side['future_sight']['countdown']]

        # [1] if the Healing wish/Lunar dance effect is incoming, [0] otherwise
        healing_wish = [int(side['healing_wish'])]

        # player's pokemon is trapped
        trapped = [int(side['trapped'])]

        # [1] if the player's active pokemon is alive, [0] otherwise
        has_active = [int(not side['active']['fainted'])]

        # the player's active pokemon -- fainted or alive
        active = self.convert_pokemon(side['active'])

        # the player's reserve pokemon -- fainted or alive
        reserve = [val for pkmn in [self.convert_pokemon(pkmn) for pkmn in side['reserve']] for val in pkmn]

        return side_conditions + wish + future_sight + healing_wish + trapped + has_active + active + reserve

    def convert_pokemon(self, pokemon):
        # one-hot-encode species
        species = [0] * len(self.pkmn_positions)
        species_index = self.pkmn_positions.get(pokemon['name'])
        if species_index is not None:
            species[species_index] = 1
        else:
            logging.debug(f'pokemon "{pokemon["id"]}" does not exist in pokemon.json')

        # one-hot-encode ability
        ability = [0] * len(self.ability_positions)
        ability_index = self.ability_positions.get(pokemon['ability'])
        if ability_index is not None:
            ability[ability_index] = 1
        else:
            logging.debug(f'ability "{pokemon["ability"]}" does not exist in ability.json')

        # one-hot-encode types
        types = [0] * len(self.types_positions)
        for t in pokemon['types']:
            index = self.types_positions.get(t)
            if index is not None:
                types[index] = 1
            else:
                logging.debug(f'type "{t}" does not exist in types.json')

        # one-hot-encode item
        item = [0] * len(self.item_positions)
        item_index = self.item_positions.get(pokemon['item'])
        if item_index is not None:
            item[item_index] = 1
        elif pokemon['item'] != "none":
            logging.debug(f'item "{pokemon["item"]}" does not exist in items.json')

        # [1] if has item, [0] if has no item
        has_item = [int(pokemon['item'] != "none")]

        # pokemon level
        level = [pokemon['level']]

        # pokemon stats
        stats = [
            pokemon['stats']['hp'],
            pokemon['stats']['attack'],
            pokemon['stats']['defense'],
            pokemon['stats']['special-attack'],
            pokemon['stats']['special-defense'],
            pokemon['stats']['speed'],
        ]

        # pokemon stat boosts/drops
        stat_changes = [
            pokemon['stat_changes']['attack'],
            pokemon['stat_changes']['defense'],
            pokemon['stat_changes']['special-attack'],
            pokemon['stat_changes']['special-defense'],
            pokemon['stat_changes']['speed'],
            pokemon['stat_changes']['accuracy'],
            pokemon['stat_changes']['evasion']
        ]

        # pokemon hp range 0-100
        health = [int(pokemon['hp'] / pokemon['maxhp'] * 100)]

        # [1] if pokemon fainted, [0] if still alive
        fainted = [int(pokemon['fainted'])]

        # one-hot-encode status conditions
        status = [0] * len(self.status_positions)
        status_index = self.status_positions.get(pokemon['status'])
        if status_index is not None:
            status[status_index] = 1
        elif pokemon['status'] != "fnt":
            logging.debug(f'status "{pokemon["status"]}" does not exist in status.json')

        # one-hot-encode volatile_status
        volatile_status = [0] * len(self.volatile_status_positions)
        for v in pokemon['volatile_status']:
            index = self.volatile_status_positions.get(v)
            if index is not None:
                volatile_status[index] = 1
            elif not self.volatiles_to_ignore.get(v):
                logging.debug(f'volatile_status "{v}" does not exist in volatile_status.json')

        # [1] if its the pokemon's first turn out, [0] otherwise
        first_turn_out = [int(pokemon['first_turn_out'])]

        # amount of moves the pokemon has, range 1-4
        n_moves = [len(pokemon['moves'])]

        # pokemon's moves
        moves = [val for move in [self.convert_move(move) for move in pokemon['moves']] for val in move]

        # account for a pokemon having less than 4 moves
        if n_moves[0] < 4:
            moves += [0] * int(len(moves) / n_moves[0]) * (4 - n_moves[0])
        elif n_moves[0] <= 0:
            raise ValueError(f'Pokemon {pokemon["name"]} has no moves')

        return species + ability + types + item + has_item + level + stats + stat_changes + \
            health + fainted + status + volatile_status + first_turn_out + n_moves + moves

    def convert_move(self, move):
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

        # current move pp
        current_pp = [move['pp']]

        # maximum move pp
        max_pp = [int(self.move_lookup[move_name]['pp'] * 1.6)]

        # [1] is move targets the user, [0] otherwise
        target_self = [int(self.move_lookup[move_name]['target'] == 'self')]

        # [1] if move can't be used this turn, [0] otherwise
        disabled = [int(move['disabled'])]

        # [1] if move was the last used move by this pokemon, [0] otherwise
        last_used_move = [int(move['last_used_move'])]

        # [1] if the move has been used previously, [0] otherwise
        used = [int(current_pp < max_pp)]

        # move priority level
        priority = [self.move_lookup[move_name]['priority']]

        return moves + typing + move_category + base_power + accuracy + current_pp + max_pp + \
            target_self + disabled + last_used_move + used + priority

    def create_header(self):
        """ returns a 4*m header """
        move_header = (
                ['move' for _ in self.move_positions] +
                ['type' for _ in self.types_positions] +
                ['move category' for _ in self.move_category_positions] +
                ['base power'] +
                ['accuracy'] +
                ['current pp'] +
                ['max pp'] +
                ['target self'] +
                ['disabled'] +
                ['last used move'] +
                ['used'] +
                ['priority']
        )

        pokemon_header = (
                ['name' for _ in self.pkmn_positions] +
                ['ability' for _ in self.ability_positions] +
                ['type' for _ in self.types_positions] +
                ['item' for _ in self.item_positions] +
                ['has item'] +
                ['level'] +
                ['stats'] * 6 +
                ['stat changes'] * 7 +
                ['health'] +
                ['fainted'] +
                ['status' for _ in self.status_positions] +
                ['volatile status' for _ in self.volatile_status_positions] +
                ['first turn out'] +
                ['n_moves'] +
                ['move1' for _ in move_header] +
                ['move2' for _ in move_header] +
                ['move3' for _ in move_header] +
                ['move4' for _ in move_header]
        )

        player_header = (
                ['side condition' for _ in self.side_condition_positions] +
                ['wish'] * 2 +
                ['future sight'] +
                ['healing wish'] +
                ['is trapped'] +
                ['has active'] +
                ['active pokemon' for _ in pokemon_header] +
                ['reserve pokemon1' for _ in pokemon_header] +
                ['reserve pokemon2' for _ in pokemon_header] +
                ['reserve pokemon3' for _ in pokemon_header] +
                ['reserve pokemon4' for _ in pokemon_header] +
                ['reserve pokemon5' for _ in pokemon_header]
        )

        game_header = (
                ['p1 win'] +
                ['p1 rating'] +
                ['p2 rating'] +
                ['avg rating'] +
                ['rated battle'] +
                ['room id'] +
                ['turn'] +
                ['weather' for _ in self.weather_positions] +
                ['weather count'] +
                ['terrain' for _ in self.terrain_positions] +
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









