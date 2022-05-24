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


def inspect_states():
    path = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/all_rated_1200+/test_games'

    files = [os.path.join(path, file_name)
             for file_name in os.listdir(path)]

    print(f'{len(files)} files found')

    random.shuffle(files)

    for file in files[:10000]:
        with open(file, 'r') as f_in:
            states = json.load(f_in)

            for i, s in enumerate(states):
                if s['p1']['wish']['countdown'] > 1 or s['p2']['wish']['countdown'] > 1:
                    print(json.dumps(s, indent=3))
                    break
            else:
                continue
            break


