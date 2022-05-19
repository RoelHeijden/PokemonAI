import pandas as pd
import numpy as np
import os
import json
import ujson
import csv
import time
import re
import shutil

path = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/all_rated_1200+/training_states/'

name = 'gen8ou-100012-state-22.json'

file = json.load(open(os.path.join(path, name), 'r'))

print(json.dumps(file, indent=3))





















