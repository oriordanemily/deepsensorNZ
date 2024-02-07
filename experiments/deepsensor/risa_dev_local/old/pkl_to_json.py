#%%
import logging
logging.captureWarnings(True)
import os
import time
import importlib
import json
import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.downscaler.validate import ValidateV1
from nzdownscale.dataprocess import utils

#%% 

dir = 'models/downscaling/metadata'

file_name = 'hr_1_model_1705857990'

path = f'{dir}/{file_name}.pkl'
md = utils.open_pickle(path)

with open(f'{dir}/json/{file_name}.json', 'w') as json_file:
    json.dump(md, json_file)


with open(f'{dir}/yaml/{file_name}.yaml', 'w') as yaml_file:
    yaml.dump(md, yaml_file)


#%% 
    

# # Define a custom constructor for Python tuples
# def tuple_constructor(loader, node):
#     return tuple(loader.construct_sequence(node))

# # Add the custom constructor to the PyYAML loader
# yaml.Loader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

with open(f'{dir}/yaml/{file_name}.yaml', 'r') as yaml_file:
    # Load data from the YAML file
    data = yaml.load(yaml_file, Loader=yaml.FullLoader)