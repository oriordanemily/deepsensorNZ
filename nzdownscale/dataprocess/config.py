VARIABLE_OPTIONS = ['temperature', 'precipitation']

DATA_PATHS = {
    'ERA5': {
        'parent': 'data/ERA5-land',
    },    
    'topography': {
        'parent': 'data/topography',
        'file': 'data/topography/nz_elevation_100m.nc',
    },    
    'stations': {
        'parent': 'data/stations',
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
