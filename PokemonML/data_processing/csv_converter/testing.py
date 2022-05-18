import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import json
import ujson
import csv
import time
import re
import shutil


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


# go test each test state
def test_test_states():
    from csv_converter import Converter
    cv = Converter()

    headers = cv.create_header()

    files = sorted([os.path.join('test_states', file_name)
                    for file_name in os.listdir('test_states')])

    tic = time.time()

    for i, f in enumerate(files):
        with open(f, 'r') as f_in:

            state = json.load(f_in)

            start_time = time.time()
            output = cv.convert_state(state)
            end_time = time.time() - start_time

            print("state", f.strip('.txt')[-2:].strip('_'))
            print("turn:", state['turn'])
            print("output length:", len(output))
            print(f'{round(end_time, 8)}s')
            print()

    toc = time.time()

    print(f'header size: {len(headers[0])}\n')
    print("-----------------------------------------------")
    print(len(files), "states converted")
    print("Total runtime:", round(toc - tic, 4))


# test_test_states()


####################################################################################################


# folder = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/python/PokemonML/data/granted_data_testing/processed_data"
#
# files = []
# for file_name in os.listdir(folder):
#     files.append(os.path.join(folder, file_name))
#
#
# file = json.load(open(files[0]))
#
# for i, s in enumerate(file):
#     if i == 0:
#         continue
#     open('test_states/test_state_' + str(i) + '.txt', 'w').write(json.dumps(s, indent=4))


####################################################################################################


# from csv_converter import Converter
# cv = Converter()
#
# file = 'test_states/test_state_20.txt'
# test_state = json.load(open(file))
#
# headers = cv.create_header()
# input_array = cv.convert_state(test_state)
#
# df = pd.DataFrame([input_array], columns=headers)
#
# sub_df = df['p2']['side_conditions']
#
# for i in range(sub_df.size):
#     value = int(sub_df.iloc[:, i])
#     print(i, value)



np.rando


