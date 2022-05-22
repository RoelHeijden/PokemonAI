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

s = time.time()

for i in range(100000):
    item = torch.tensor([[1, 2, 3, 4, 5, 6],
                         [1, 2, 3, 4, 5, 6]])

    species = torch.tensor([[1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6]])

    total = torch.stack((item, species))
    # total = torch.cat((item, species))


print(time.time() - s)







