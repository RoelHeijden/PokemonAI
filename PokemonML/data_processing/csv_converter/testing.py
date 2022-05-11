import pandas as pd
import numpy as np
import os
import json
import ujson
import csv


####################################################################################################


def init_category(file_name, relative_path='categories'):
    path = os.path.join(relative_path, file_name)
    file = open(path, 'r')
    return list(json.load(file))


pokemon_list = init_category('pokemon.json')
item_list = init_category('items.json')
ability_list = init_category('abilities.json')
move_list = init_category('moves.json')
weather_list = init_category('weathers.json')
terrain_list = init_category('terrains.json')
type_list = init_category('types.json')
status_list = init_category('status.json')
move_category_list = init_category('move_categories.json')
volatile_status_list = []  # TBD
side_condition_list = []  # TBD


####################################################################################################


top_atts = ['p1_win', 'turn', 'weather', 'terrain', 'trick room', 'p1', 'p2']

players_atts = ['player rating', 'player move', 'side conditions', 'wish', 'future sight' 'active pokemon',
                  'reserve pokemon1', 'reserve pokemon2', 'reserve pokemon3', 'reserve pokemon4', 'reserve pokemon5']

pokemon_atts = ['name', 'item' 'ability', 'stats', 'max hp', 'current hp', 'fainted',
                  'stat changes', 'moves', 'last used move', 'types', 'status', 'volatile status']

moves_atts = ['move', 'type', 'category', 'base power', 'current pp', 'max pp', 'disabled']


####################################################################################################


move_header = (
    ['move' for _ in move_list] +
    ['type' for _ in type_list] +
    ['move category' for _ in move_category_list] +
    ['base power'] +
    ['current pp'] +
    ['max pp'] +
    ['disabled']
)

pokemon_header = (
    ['name' for _ in pokemon_list] +
    ['type' for _ in type_list] +
    ['item' for _ in item_list] +
    ['ability' for _ in ability_list] +
    ['stats'] * 6 +
    ['max hp'] +
    ['current hp'] +
    ['fainted'] +
    ['stat changes'] * 7 +
    ['last used move'] * 4 +
    ['status' for _ in status_list] +
    ['volatile status' for _ in volatile_status_list] +
    ['move1' for _ in move_header] +
    ['move2' for _ in move_header] +
    ['move3' for _ in move_header] +
    ['move4' for _ in move_header]
)

player_header = (
    ['player rating'] +
    ['player move'] +
    ['side condition' for _ in side_condition_list] +
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
    ['weather' for _ in weather_list] +
    ['terrain' for _ in terrain_list] +
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


####################################################################################################


# f_out = open('test_csv.csv', 'a')
#
# writer = csv.writer(f_out)
# writer.writerow(first_header)
# writer.writerow(second_header)
# writer.writerow(third_header)
# writer.writerow(fourth_header)
#
#
# test_data = np.asarray([np.zeros(len(first_header))])
# np.savetxt(f_out, test_data, delimiter=",")
#
#
# file = pd.read_csv('test_csv.csv', header=[0, 1, 2, 3])
#
# print(file['p2']['reserve pokemon5']['move4']['current pp'])

