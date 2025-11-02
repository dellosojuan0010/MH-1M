import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import csv



# Itera sobre cada linha do array com um índice
for i in tqdm(range(10000000)):
    print(i)
