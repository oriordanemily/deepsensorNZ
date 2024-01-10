import numpy as np

VARIABLE_OPTIONS = ['temperature', 'precipitation']

DATA_PATHS = {
    'ERA5': {
        'parent': 'data/ERA5-land',
    },    
    'topography': {
        'parent': 'data/topography',
        'file': 'data/topography/nz_elevation_200m.nc',
    },    
    'stations': {
        # 'parent': 'data/stations',
        'parent': '/mnt/datasets/NationalClimateDatabase/NetCDFFilesByVariableAndSite/Hourly',
    },
}

VAR_STATIONS = {
            'precipitation': {
                'subdir': 'Precipitation',
                'var_name': 'precipitation'
            },
            'temperature': {
                'subdir': 'ScreenObs',
                'var_name': 'dry_bulb'
            },
}

VAR_ERA5 = {
            'precipitation': {
                'subdir': 'total_precipitation_hourly',
                'var_name': 'precipitation',
            },
            'temperature': {
                'subdir': '2m_temperature',
                'var_name': 't2m',
            },
}


PLOT_EXTENT = {
            'all': {
                'minlon': 165,
                'maxlon': 179,
                'minlat': -48,
                'maxlat': -34,
            },
            'north': {
                'minlon': 171,
                'maxlon': 179,
                'minlat': -42,
                'maxlat': -34,
            },
        }

LOCATION_LATLON = {
    'alexandra': np.array([-45.2479, 169.2844]),
    'arthurs_pass': np.array([-42.9402, 171.5620]),
    'aoraki': np.array([-43.5950, 170.1418]),
    'auckland': np.array([-36.8509, 174.7645]),
    'christchurch': np.array([-43.5320, 172.6306]),
    'dunedin': np.array([-45.8795, 170.5006]),
    'gisborne': np.array([-38.6641, 178.0228]),
    'greymouth': np.array([-42.4614, 171.1985]),
    'hamilton': np.array([-37.7826, 175.2528]),
    'invercargill': np.array([-46.4179, 168.3615]),
    'napier': np.array([-39.4823, 176.9192]),
    'nelson': np.array([-41.2985, 173.2444]),
    'new_plymouth': np.array([-39.0572, 174.0794]),
    'palmerston_north': np.array([-40.3545, 175.6097]),
    'rotorua': np.array([-38.1446, 176.2378]),
    'ruatoria': np.array([-37.8898, 178.3191]),
    'taupo': np.array([-38.6843, 176.0704]),
    'tauranga': np.array([-37.6870, 176.1654]),
    'te_anau': np.array([-45.4144, 167.7180]),
    'timaru': np.array([-44.3904, 171.2373]),
    'twizel': np.array([-44.2615, 170.0876]),
    'wellington': np.array([-41.2924, 174.7787]),
    'whangarei': np.array([-35.7275, 174.3166]),
}
