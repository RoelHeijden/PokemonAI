import pandas as pd
import numpy as np
import os
import json
import ujson
import csv
import time
import re
import shutil


path_in = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/all_rated_1200+/training_states_large/'

files = [
    os.path.join(path_in, file_name)
    for file_name in os.listdir(path_in)
]
print(len(files), "files found")

i = 0
states = []
for f in files:
    with open(f, 'r') as f_in:
        for line in f_in:
            state = ujson.loads(line)
            states.append(state)
            i += 1

            if i % 1000 == 0:
                print(f'{i} states opened')

print(json.dumps(states[17001], indent=3))
















