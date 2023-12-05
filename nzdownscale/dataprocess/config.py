VARIABLE_OPTIONS = ['temperature', 'precipitation']

DATA_PATHS = {
    'ERA5': 'data/ERA5-land',
    'topography': 'data/topography',
    'stations': 'data/stations',
}

DIR_STATIONS = {
            'precipitation': {
                'subdir': 'Precipitation',
                'var': 'precipitation'
            },
            'temperature': {
                'subdir': 'ScreenObs',
                'var': 'dry_bulb'
            },
        }

DIR_ERA5 = {
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
