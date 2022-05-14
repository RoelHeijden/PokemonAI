import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import json
import ujson
import csv
import time


####################################################################################################


def init_category(file_name, relative_path='categories'):
    path = os.path.join(relative_path, file_name)
    file = open(path, 'r')
    data = {key: i for i, key in enumerate(json.load(file))}
    return data


# pokemon_list = init_category('pokemon.json')
# item_list = init_category('items.json')
# ability_list = init_category('abilities.json')
# move_list = init_category('moves.json')
# weather_list = init_category('weathers.json')
# terrain_list = init_category('terrains.json')
# type_list = init_category('types.json')
# status_list = init_category('status.json')
# move_category_list = init_category('move_categories.json')
# volatile_status_list = []  # TBD
# side_condition_list = []  # TBD


####################################################################################################


# adds the currently unavailable attributes
def fill_new_attributes(state):
    state['p1rating'] = 1069
    state['p2rating'] = 1042
    state['average_rating'] = 1051
    state['rated_battle'] = True
    state['roomid'] = 1
    state['turn'] = 2

    state['state']['weather_count'] = 1
    state['state']['terrain_count'] = 1
    state['state']['trick_room_count'] = 0

    for key in state['state']:
        if key == 'p1' or key == 'p2':
            if state['state'][key]['active']:
                state['state'][key]['active']['first_turn_out'] = False
                for j in range(len(state['state'][key]['active']['moves'])):
                    state['state'][key]['active']['moves'][j]['last_used_move'] = False

            for i in range(len(state['state'][key]['reserve'])):

                state['state'][key]['reserve'][i]['first_turn_out'] = False
                for j in range(len(state['state'][key]['reserve'][i]['moves'])):
                    state['state'][key]['reserve'][i]['moves'][j]['last_used_move'] = False

    return state


# go test each test state
def test_test_states():
    from csv_converter import Converter
    cv = Converter()

    files = sorted([os.path.join('test_states', file_name)
                    for file_name in os.listdir('test_states')])

    for i, f in enumerate(files):
        with open(f, 'r') as f_in:

            state = json.load(f_in)
            state = fill_new_attributes(state)

            start_time = time.time()
            output = cv.convert_state(state)
            end_time = time.time() - start_time

            print("state", i)
            print("output length:", len(output))
            print(f'{round(end_time, 5)}s')
            print()


test_test_states()



