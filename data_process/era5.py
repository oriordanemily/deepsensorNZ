#%% 

import os

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% 

class ProcessERA5:
    def __init__(self) -> None:
        self.path = 'data/ftp.bodekerscientific.com/Greg/ForRisa2'
        self.names = {
            'precipitation': {
                'folder': 'total_precipitation_hourly',
                #'var': 'precipitation'
            },
            'temperature': {
                'folder': '2m_temperature',
                #'var': 'dry_bulb'
            },
        }


#%% 
