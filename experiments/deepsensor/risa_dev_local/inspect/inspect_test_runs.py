#%%
import logging
logging.captureWarnings(True)
import os
import time
import importlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.downscaler.validate import ValidateV1
from nzdownscale.dataprocess import utils

#%% 
# Load trained model
# ------------------------------------------

train_metadata_path = 'models/downscaling/metadata/test_model_1705608588.pkl'
model_path = 'models/downscaling/test_model_1705608588.pt'

validate = ValidateV1(
training_metadata_path=train_metadata_path)
validate.load_model(load_model_path=model_path)

metadata = validate.get_metadata()
print(metadata)
model = validate.model

#%%
# Inspect trained model
# ------------------------------------------
# ! bug

validate.plot_example_prediction()
validate.emily_plots()
