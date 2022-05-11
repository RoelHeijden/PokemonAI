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
    return np.asarray(list(json.load(file)))


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

