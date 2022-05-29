import random

import pandas as pd
import numpy as np
import os
import json
import ujson
import csv
import time
import re
import shutil
import torch
import torch.nn as nn

"""
C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/training_games\game-states-gen8ou-4364493.json
C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/training_games\game-states-gen8ou-4369812.json
C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/training_games\game-states-gen8ou-4605078.json
C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/training_games\game-states-gen8ou-5424967.json
"""

path_in = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/training_games'

files = sorted([os.path.join(path_in, file_name)
                for file_name in os.listdir(path_in)])

print(f'{len(files)} files found')
# 977594

i = 0
for file in files:
    if os.stat(file).st_size == 0:
        # os.remove(file)
        print(file)
        i += 1

print(i)

