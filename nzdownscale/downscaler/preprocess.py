import logging

from sklearn import base
from sympy import stationary_points
logging.captureWarnings(True)
from cycler import V
from typing_extensions import Literal
from tqdm import tqdm
import warnings
from time import time
import pickle
import os

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cf
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors

from deepsensor.data.processor import DataProcessor
from deepsensor.data.utils import construct_x1x2_ds
from deepsensor.data import construct_circ_time_ds
from nzdownscale.dataprocess import era5, wrf, stations, topography, utils, config
from nzdownscale.dataprocess.config import LOCATION_LATLON, PLOT_EXTENT
from nzdownscale.dataprocess.config_local import DATA_PATHS


class PreprocessForDownscaling:

    def __init__(self,
                 base='era5',
                 training_years=None,
                 validation_years=None,
                 training_fpaths=None,
                 validation_fpaths=None,
                 variable='temperature',
                 use_daily_data=True,
                 validation = False,
                 area=None,
                 context_variables=[]
                 ) -> None:
        
        """
        use_daily_data: if True, era5 and station data will be converted to daily data
        """
        
        self.var = variable
        self.base = base

        if self.base == 'era5':
            self.use_daily_data = use_daily_data
            self.training_years = training_years
            self.validation_years = validation_years

            if validation:
                self.years = validation_years
            else:
                if training_years is None or validation_years is None:
                    self.years = None
                else:
                    self.years = list(set(training_years + validation_years))

        elif self.base == 'wrf':
            self.use_daily_data = False
            self.training_fpaths = training_fpaths
            self.validation_fpaths = validation_fpaths
            if validation:
                self.all_paths = validation_fpaths
            else:
                self.all_paths = training_fpaths + validation_fpaths
            self.years = None

        self.area = area
        
        if context_variables == []:
            context_variables = [self.var]
        self.context_variables = context_variables 

        self.dataprocess = utils.DataProcess()

        self.process_top = topography.ProcessTopography()
        if base=='era5':
            self.process_era = era5.ProcessERA5()
        elif base=='wrf':
            self.process_wrf = wrf.ProcessWRF()
        self.process_stations = stations.ProcessStations()
        self.nzplot = utils.PlotData()

        self.station_metadata_all = None
        self.station_metadata = None

        self.ds_elev = None
        self.ds_base = None
        self.da_base = None

        self.data_processor = None
        self.aux_ds = None
        self.base_ds = None
        self.highres_aux_ds = None
        self.landmask_ds = None
        self.station_raw_df = None
        self.station_df = None

        self._ds_elev_hr = None

        # self._check_args()

    
    def _check_args(self):
        # if self.use_daily_data is False:
        #     raise NotImplementedError
        # if self.var == 'precipitation':
        #     raise NotImplementedError
        pass

    
    def run_processing_sequence(self,
        topography_highres_coarsen_factor,
        topography_lowres_coarsen_factor,
        era5_coarsen_factor=1,
        include_time_of_year=True,
        include_landmask=False,
        data_processor_dict=None,
        save_data_processor_dict=None,
        remove_stations=[None],
        station_as_context=False
        ):
        
        self.load_topography()
        highres_aux_raw_ds, aux_raw_ds = self.preprocess_topography(topography_highres_coarsen_factor, topography_lowres_coarsen_factor)
        self.highres_aux_raw_ds, self.aux_raw_ds = highres_aux_raw_ds, aux_raw_ds

        if self.base == 'era5':
            self.load_era5()
        elif self.base == 'wrf':
            self.load_wrf()
        self.load_stations()
        
        if self.base == 'era5':
            base_raw_ds = self.preprocess_era5(coarsen_factor=era5_coarsen_factor)
        elif self.base == 'wrf':
            base_raw_ds = self.preprocess_wrf()
        station_raw_df = self.preprocess_stations(remove_stations=remove_stations, fill_missing=True)

        if include_landmask:
            landmask_raw_ds = self.load_landmask()
        else:
            landmask_raw_ds = None

        if data_processor_dict == None:
            data_processor_dict = self.process_all_for_training(
                base_raw_ds=base_raw_ds, 
                highres_aux_raw_ds=highres_aux_raw_ds, 
                aux_raw_ds=aux_raw_ds, 
                station_raw_df=station_raw_df,
                landmask_raw_ds=landmask_raw_ds,
                include_time_of_year=include_time_of_year,
                save=save_data_processor_dict,
                station_as_context=station_as_context
                )
            self.data_processor = data_processor_dict['data_processor']
            self.aux_ds = data_processor_dict['aux_ds']
            self.highres_aux_ds = data_processor_dict['highres_aux_ds']
            self.landmask_ds = data_processor_dict['landmask_ds']
        else:
            self.data_processor = data_processor_dict['data_processor']
            self.aux_ds = data_processor_dict['aux_ds']
            self.highres_aux_ds = data_processor_dict['highres_aux_ds']
            self.landmask_ds = data_processor_dict['landmask_ds']

        print(f'Processing {self.base} and stations')
        self.base_ds = self.data_processor(base_raw_ds)
        if include_time_of_year:
            self.base_ds = self.add_time_of_year(self.base_ds)

        station_raw_df = station_raw_df.rename({station_raw_df.columns[0]: f'{self.var}_station'}, axis=1)
        station_raw_df = station_raw_df.drop(columns=['station_name'])
        self.station_df = self.data_processor(station_raw_df)
        self.station_as_context = station_as_context

    def load_topography(self):
        print('Loading topography...')
        self.ds_elev = self.process_top.open_ds()
        if self.area is not None:
            minlon = PLOT_EXTENT[self.area]['minlon']
            maxlon = PLOT_EXTENT[self.area]['maxlon']
            minlat = PLOT_EXTENT[self.area]['minlat']
            maxlat = PLOT_EXTENT[self.area]['maxlat']
            self.ds_elev = self.ds_elev.sel(latitude=slice(minlat, maxlat), longitude=slice(minlon, maxlon))

    
    def load_era5(self):
        print('Loading era5...')
        self.base_ds = self.process_era.load_ds(self.var, self.years)
        for variable in self.context_variables:
            if variable != self.var:
                da = self.process_era.load_ds(variable, self.years)
                self.base_ds = xr.merge([self.base_ds, da])
        

    def load_wrf(self):
        print('Loading wrf...')
        base_ds = self.process_wrf.load_ds(filenames=self.all_paths,
                                            context_variables = self.context_variables)
        self.base_ds = self.process_wrf.regrid_to_topo(base_ds,
                                                    self.aux_raw_ds)
        times = self.base_ds.time.values
        self.years = np.unique([t.year for t in pd.to_datetime(times)])

    def load_stations(self, use_cache=False):
        print('Loading stations...')

        if use_cache:
            if "cache" not in DATA_PATHS.keys():
                raise ValueError("Please set 'cache' path in DATA_PATHS dict e.g. 'cache':'data/.datacache' (recommended) or set use_cache=False ")
            savedir = f'{DATA_PATHS["cache"]}/station_data'
            filepath = f'{savedir}/station_metadata_all.pkl'
            if os.path.exists(filepath):
                print(f"Loading station metadata from cache: {filepath}, set use_cache=False if you want to manually load them.")
                self.station_metadata_all = utils.open_pickle(filepath)
            else:
                self.station_metadata_all = self.process_stations.get_metadata_df(self.var)
                os.makedirs(savedir, exist_ok=True)
                utils.save_pickle(self.station_metadata_all, filepath)
        else:
            self.station_metadata_all = self.process_stations.get_metadata_df(self.var)


    def preprocess_topography(self, 
                              highres_coarsen_factor=30,
                              lowres_coarsen_factor=10,
                              ):
        """ Gets self.highres_aux_raw_ds, self.aux_raw_ds """

        self.topography_highres_coarsen_factor = highres_coarsen_factor
        self.topography_lowres_coarsen_factor = lowres_coarsen_factor
        
        assert self.ds_elev is not None, "Run load_topography() first"

        # Get highres topography
        ds_elev_highres = self._get_highres_topography(self.ds_elev, highres_coarsen_factor)
        highres_aux_raw_ds = self._compute_tpi(ds_elev_highres)
        
        # Get lowres topography 
        aux_raw_ds = self._get_lowres_topography(ds_elev_highres, lowres_coarsen_factor)

        highres_aux_raw_ds['elevation_diff'] = self._compute_topo_difference(highres_aux_raw_ds, aux_raw_ds)

        self.highres_aux_raw_ds = highres_aux_raw_ds
        self.aux_raw_ds = aux_raw_ds # does this need to include coarsened TPI?
        return self.highres_aux_raw_ds, self.aux_raw_ds

    def preprocess_wrf(self,):
        
        assert self.base_ds is not None, "Run load_wrf() first"
        assert self.highres_aux_raw_ds is not None, "Run preprocess_topography() first"

        # Coarsening not implemented for WRF

        # Trim to topography extent
        ds_base_trimmed = self._trim_ds(self.base_ds, self.highres_aux_raw_ds)
        self.base_raw_ds = ds_base_trimmed

        return self.base_raw_ds

    def preprocess_era5(self,
                        coarsen_factor=10,
                        ):
        """ Gets self.era5_raw_ds """

        assert self.base_ds is not None, "Run load_era5() first"
        assert self.highres_aux_raw_ds is not None, "Run preprocess_topography() first"

        # Convert hourly to daily data
        if self.use_daily_data:
            ds_era = self._convert_era5_to_daily(self.base_ds)
        else:
            ds_era = self.base_ds

        # Coarsen
        self.era5_coarsen_factor = coarsen_factor
        self.ds_era_coarse = self._coarsen_era5(ds_era, self.era5_coarsen_factor)

        # Trim to topography extent
        ds_era_trimmed = self._trim_ds(self.ds_era_coarse, self.highres_aux_raw_ds)
        
        self.base_raw_ds = ds_era_trimmed
        return self.base_raw_ds


    def preprocess_stations(self, remove_stations=['None'], fill_missing=False):
        """ Gets self.station_raw_df """

        assert self.station_metadata_all is not None, "Run load_stations() first"
        
        self.station_metadata = self._filter_stations(self.station_metadata_all, remove_stations=remove_stations)
        self.station_raw_df = self._get_station_raw_df(self.station_metadata, fill_missing=fill_missing)
        
        return self.station_raw_df 
        

    def _get_highres_topography(self,
                               ds_elev,
                               coarsen_factor=30,
                               plot=False,
                               fillna=True,
                               ):
        
        process_top = self.process_top

        # Topography = 0.002 degrees (~200m)
        _ds_elev_hr = process_top.coarsen_da(ds_elev, coarsen_factor)
        if fillna:
            #fill all nan values with 0 to avoid training error
            ds_elev_highres = _ds_elev_hr.fillna(0)
        else:
            ds_elev_highres = _ds_elev_hr
        self._ds_elev_hr = _ds_elev_hr

        if plot:
            da_elev_highres = process_top.ds_to_da(ds_elev_highres)
            latres_topo = self.dataprocess.resolution(ds_elev_highres, 'latitude')
            lonres_topo = self.dataprocess.resolution(ds_elev_highres, 'longitude')
            ax = self.nzplot.nz_map_with_coastlines()
            da_elev_highres.plot()
            str_coarsened = 'Coarsened ' if coarsen_factor != 1 else ''
            plt.title(f'{str_coarsened}topography\n'
                    f'Lat res: {latres_topo:.4f} degrees, '
                    f'lon res: {lonres_topo:.4f} degrees')
            plt.show()

        
        self.ds_elev_highres = ds_elev_highres
        return ds_elev_highres
    
    
    def _get_lowres_topography(self,
                       ds_elev_highres,
                       coarsen_factor=10,
                       plot=False,
                       ):
        
        #coarsen_factor = 10 # int(latres_era/latres_topo) #changed this to match era5 resolution
        aux_raw_ds = self.process_top.coarsen_da(ds_elev_highres, coarsen_factor)

        if plot:
            aux_raw_da = self.process_top.ds_to_da(aux_raw_ds)
            latres = self.dataprocess.resolution(aux_raw_da, 'latitude')
            lonres = self.dataprocess.resolution(aux_raw_da, 'longitude')
            ax = self.nzplot.nz_map_with_coastlines()
            aux_raw_da.plot()
            plt.title(f'Low-res topography\nLat res: {latres:.4f} degrees, lon res: {lonres:.4f} degrees')
            plt.show()

        return aux_raw_ds


    def _compute_tpi(self, 
                    ds_elev_highres,
                    plot=False,
                    ):
         
        # TPI helps us distinguish topo features, e.g. hilltop, valley, ridge...

        highres_aux_raw_ds = ds_elev_highres

        # Calculate the lat and lon resolutions in the elevation dataset
        # Here we assume the elevation is on a regular grid,
        # so the first difference is equal to all others.
        # This may not be a fair assumption... 
        coord_names = list(highres_aux_raw_ds.dims)
        resolutions = np.array(
            [np.abs(np.diff(highres_aux_raw_ds.coords[coord].values)[0]) 
            for coord in coord_names])

        for window_size in [.1, .05, .025]:
            smoothed_elev_da = highres_aux_raw_ds['elevation'].copy(deep=True)

            # Compute gaussian filter scale in terms of grid cells
            scales = window_size / resolutions

            smoothed_elev_da.data = gaussian_filter(smoothed_elev_da.data, 
                                                    sigma=scales, 
                                                    mode='constant', 
                                                    cval=0)

            TPI_da = highres_aux_raw_ds['elevation'] - smoothed_elev_da
            highres_aux_raw_ds[f"TPI_{window_size}"] = TPI_da

            if plot:
                ax = self.nzplot.nz_map_with_coastlines()
                TPI_da.plot(ax=ax)
                ax.add_feature(cf.BORDERS)
                ax.coastlines()
                ax.set_title(f'TPI with window size {window_size}, scales = {scales}')
                plt.show()

        return highres_aux_raw_ds
    
    def _compute_topo_difference(self, ds_elev_highres, ds_elev_lowres): 
        # Compute the difference between highres and lowres topography
        # This can be used as an additional auxiliary input to the model
        # Use NN interpolation to not adjust LR data too much (i.e. no smoothing/averaging)
        lr_interp = ds_elev_lowres['elevation'].interp_like(ds_elev_highres['elevation'], method='nearest')
        topo_diff = ds_elev_highres['elevation'] - lr_interp

        # some of the boundary is na - replace with 0
        topo_diff = topo_diff.fillna(0)
        return topo_diff
    

    def _convert_era5_to_daily(self, da_era):
        if self.var == 'precipitation':
            function = 'sum'
        else:
            function = 'mean'
        da = self.process_era.convert_hourly_to_daily(da_era, function)
        return da
    

    def _coarsen_era5(self,
                    da_era,
                    coarsen_factor_era=10,
                    plot=False,
                    ):
        
        process_era = self.process_era
        
        # note that this was just coarsened to make the model fit into memory 
        # may want to change this down the line

        # ERA5 = 0.1 degrees (~10km)
        # coarsen_factor_era = 10
        if coarsen_factor_era == 1:
            da_era_coarse = da_era
        else:
            da_era_coarse = process_era.coarsen_da(da_era, coarsen_factor_era)

        # da_era_coarse = da_era_coarse.fillna(0)

        if plot:
            latres_era = self.dataprocess.resolution(da_era_coarse, 'latitude')
            lonres_era = self.dataprocess.resolution(da_era_coarse, 'longitude')
            ax = self.nzplot.nz_map_with_coastlines()
            da_era_coarse.isel(time=0).plot()
            str_coarsened = 'Coarsened ' if coarsen_factor_era != 1 else ' '
            plt.title(f'{str_coarsened}ERA5\n'
                    f'Lat res: {latres_era:.4f} degrees, '
                    f'lon res: {lonres_era:.4f} degrees')
            plt.plot()

        return da_era_coarse


    def _trim_ds(self,
                  ds,
                  highres_aux_raw_ds,
                  plot=False,
                  ):
        # Slice base data to elevation data's spatial extent
        top_min_lat = highres_aux_raw_ds['latitude'].min()
        top_max_lat = highres_aux_raw_ds['latitude'].max()
        top_min_lon = highres_aux_raw_ds['longitude'].min()
        top_max_lon = highres_aux_raw_ds['longitude'].max()
        
        if self.base == 'era5':
            lat_slice = slice(top_max_lat, top_min_lat)
        elif self.base == 'wrf':
            lat_slice = slice(top_min_lat, top_max_lat)
        base_raw_ds = ds.sel(
            latitude=lat_slice,
            longitude=slice(top_min_lon, top_max_lon))

        if plot:
            ax = self.nzplot.nz_map_with_coastlines()
            base_raw_ds.isel(time=0).plot()
            ax.set_title(f'{self.base} with topography extent');

        return base_raw_ds


    def _filter_stations(self, df_station_metadata, remove_stations=[None]):
        """ filter stations by years used in settings """

        # this was only using stations that have data across all years
        # have changed to use stations covering any part of the years specified
        # check this with Risa

        df = df_station_metadata
        years = self.years
        area = self.area

        if remove_stations != [None]:
            from nzdownscale.dataprocess.config import STATION_LATLON
            for station in remove_stations:
                station_no = STATION_LATLON[station]['station_no']
                df = df[df['station_id'] != str(station_no)]

        # ! Eventually change this so that stations only partially available are still used
        # df_filtered_years = df[(df['start_year']<years[-1]) & (df['end_year']>=years[0])]
        condition = df.apply(lambda row: any(year in range(row['start_year'], row['end_year'] + 1) for year in years), axis=1)
        df_filtered_years = df[condition]

        if area is not None:
            df_filtered_area = df_filtered_years[(df_filtered_years['lon'] > PLOT_EXTENT[area]['minlon']) & (df_filtered_years['lon'] < PLOT_EXTENT[area]['maxlon']) & (df_filtered_years['lat'] > PLOT_EXTENT[area]['minlat']) & (df_filtered_years['lat'] < PLOT_EXTENT[area]['maxlat'])]
        else:
            df_filtered_area = df_filtered_years

        df_station_metadata_filtered = df_filtered_area

        print(f'Number of stations used: {len(df_station_metadata_filtered)}')
        self.station_metadata_filtered = df_station_metadata_filtered

        return self.station_metadata_filtered
    

    def _get_station_raw_df(self, df_station_metadata, fill_missing=True):

        var = self.var
        years = self.years

        station_paths = list(df_station_metadata.index)

        df_list = []
        for path in tqdm(station_paths, desc='Loading filtered stations'):
            df = self.process_stations.load_station_df(path, var, daily=self.use_daily_data)
            df_list.append(df)
        # print('Concatenating station data...')
        df = pd.concat(df_list)
        # station_raw_df = df.reset_index().set_index(['time', 
        #                                             'latitude', 
        #                                             'longitude']).sort_index()

        ### filter years
        station_raw_df_ = df.reset_index()
        station_raw_df_ = station_raw_df_[(station_raw_df_['time']>=str(years[0])) &
                                (station_raw_df_['time']<=f'{str(years[-1])}-12-31')]
        station_raw_df = station_raw_df_.set_index(['time', 
                                                    'latitude', 
                                                    'longitude']).sort_index()
        
        if fill_missing:
            station_raw_df = self._fill_missing_stations_with_nans(station_raw_df)

            # Fill NaNs with nearest neighbours
            # station_raw_df = station_raw_df.groupby('time').apply(self.fill_missing_values).reset_index(drop=True)
            station_raw_df.set_index(['time', 'latitude', 'longitude'], inplace=True)

            # self.station_raw_df = station_raw_df
            counts_per_date = station_raw_df.groupby(level='time').size()
            assert (counts_per_date == counts_per_date[0]).all(), "Number of stations per date is not consistent"
        return station_raw_df

    
    def _fill_missing_stations_with_nans(self, station_raw_df):
        """Returns a dataframe with all combinations of times, latitudes and longitudes
         Those without real data in station_raw_df will be filled in with nans"""
        
         # get all times in station_raw_df
        times_df = pd.DataFrame({'time': station_raw_df.index.get_level_values('time').unique()})

        # get all latitudes and longitudes in station_raw_df
        lat_lon_df = pd.DataFrame(list(station_raw_df.index.droplevel(0).unique()), columns=['latitude', 'longitude'])
        
        # create a df with all combinations of times, latitudes and longitudes
        all_combinations_df = pd.merge(times_df.assign(key=1), lat_lon_df.assign(key=1), on='key').drop('key', axis=1)

        station_raw_df_reset = station_raw_df.reset_index()
        # merge all_combinations_df with station_raw_df to fill in missing values
        complete_df = pd.merge(all_combinations_df, station_raw_df_reset, on=['time', 'latitude', 'longitude'], how='left')
        complete_df = complete_df.drop_duplicates(subset=['time', 'latitude', 'longitude'], keep='first')

        #convert to datetime
        complete_df['time'] = pd.to_datetime(complete_df['time'])

        return complete_df

    def __str__(self):
        s = "PreprocessForDownscaling with data_processor:\n"
        s = s + self.data_processor.__str__
        return s

    def adjust_duplicates(self, df, increment=0.00001):
        # Create a key column for identifying duplicates based on time, latitude, and longitude
        df['key'] = df['time'].astype(str) + '_' + df['latitude'].astype(str) + '_' + df['longitude'].astype(str)
        
        # Find all keys with duplicates
        duplicate_keys = df['key'].value_counts()
        duplicate_keys = duplicate_keys[duplicate_keys > 1]
        
        # Loop through all duplicate keys and increment latitudes
        for key, count in tqdm(duplicate_keys.items(), total=len(duplicate_keys), desc='Adjusting duplicates'):
            indices = df[df['key'] == key].index
            for i, idx in enumerate(indices):
                df.at[idx, 'latitude'] += i * increment
        
        # Drop the temporary key column
        df.drop(columns=['key'], inplace=True)
        return df
    
    def fill_missing_values(self, group):
        # Filter out rows where 'dry_bulb' is not NaN
        with_data = group.dropna(subset=['dry_bulb'])
        without_data = group[group['dry_bulb'].isna()]
        
        # Check if there are missing values and valid data to train on
        if without_data.empty or with_data.empty:
            return group

        # Train NearestNeighbors on available data
        neigh = NearestNeighbors(n_neighbors=1, metric='haversine')
        neigh.fit(np.radians(with_data[['latitude', 'longitude']].values))

        # Find nearest neighbors for NaN 'dry_bulb' entries
        _, indices = neigh.kneighbors(np.radians(without_data[['latitude', 'longitude']].values))

        # Impute missing 'dry_bulb' using the values from the nearest neighbors
        without_data['dry_bulb'] = with_data.iloc[indices.flatten()]['dry_bulb'].values

        # Combine the results back together
        return pd.concat([with_data, without_data])


    def load_landmask(self):
        """ Land mask data array, same resolution as high res topography """ 
        assert self._ds_elev_hr is not None, "_get_highres_topography() must be run first"

        _da_elev = self.process_top.ds_to_da(self._ds_elev_hr)
        da_landmask = xr.where(np.isnan(_da_elev), 0, 1)
        da_landmask.name = 'landmask'
        # da_landmask variable name convention: land_mask_raw_ds
        return da_landmask

    
    def add_time_of_year(self, ds):
        """ 
        Add cos_D and sin_D to output dataset to be used as context set, original ds must have the time coordinate. Info: https://alan-turing-institute.github.io/deepsensor/user-guide/convnp.html
        """
        if self.use_daily_data:
            freq='D'
            dates = pd.date_range(ds.time.values.min(), ds.time.values.max(), freq=freq)
        else:
            freq='H'
            dates = pd.DatetimeIndex(ds.time.values)
        doy_ds = construct_circ_time_ds(dates, freq=freq)
        
        # ds = xr.Dataset({
        #     da.name: da,
        #     'cos_D': doy_ds["cos_D"], 
        #     'sin_D': doy_ds["sin_D"],
        #     })
        ds[f"cos_{freq}"] = doy_ds[f"cos_{freq}"]
        ds[f"sin_{freq}"] = doy_ds[f"sin_{freq}"]
        return ds


    def process_all_for_training(self,
                    base_raw_ds,
                    aux_raw_ds,
                    highres_aux_raw_ds,
                    station_raw_df,
                    landmask_raw_ds=None,
                    include_time_of_year=False,
                    test_norm=False,
                    # data_processor_dict=None,  # ?
                    save=None,
                    station_as_context=False
                    ):
        """
        Creates DataProcessor:
        Gets processed data for deepsensor input
        Normalises all data and add necessary dims
        """
        start = time()
        # if data_processor_dict is None:
        print('Creating DataProcessor...')
        data_processor = DataProcessor(
            x1_name="latitude", 
            x1_map=(highres_aux_raw_ds["latitude"].min(), 
                    highres_aux_raw_ds["latitude"].max()),
            x2_name="longitude", 
            x2_map=(highres_aux_raw_ds["longitude"].min(), 
                    highres_aux_raw_ds["longitude"].max()),
            )
        print('DataProcessor created in', time()-start, 'seconds')

        # Compute normalisation parameters
        start = time()
        print('Computing normalisation parameters...')

        assert_computed = False
        # If hourly data, take a random hour from each day and produce the normalization parameters from that
        if self.use_daily_data == False:
            _ = data_processor(utils.random_hour_subset_xr(base_raw_ds))
            assert_computed = True
    
        base_ds = base_raw_ds.copy()
        for var in base_raw_ds.data_vars:

            # TODO ! IF min < 0, x[x<0] = 0
            if var == 'precipitation':
                base_ds[var] = data_processor(base_raw_ds[var], 
                                                  method='positive_semidefinite')
            else:
                base_ds[var] = data_processor(base_raw_ds[var], 
                                                  method='mean_std',
                                                  assert_computed=assert_computed)
        # base_ds = data_processor(base_raw_ds, assert_computed=assert_computed)

        # if station_raw_df.columns[0] == self.var:
            # Rename the df variable so it doesn't clash with the base variable
        
        station_raw_df = station_raw_df.rename({station_raw_df.columns[0]: f'{self.var}_station'}, axis=1)
        station_raw_df = station_raw_df.drop(columns=['station_name'])

        if self.var == 'precipitation':
            # TODO ! IF min < 0, x[x<0] = 0
            station_df = data_processor(station_raw_df, 
                                        method='positive_semidefinite') 
        else:
            station_df = data_processor(station_raw_df, 
                                        method='mean_std')
            
        # base_ds, station_df = data_processor([base_raw_ds, station_raw_df]) #meanstd
        aux_ds, highres_aux_ds = data_processor([aux_raw_ds, highres_aux_raw_ds], method="min_max") #minmax
        landmask_ds = data_processor(landmask_raw_ds, method='min_max') if landmask_raw_ds is not None else None
        print(data_processor)
        print('Normalisation parameters computed in', time()-start, 'seconds')

        # Normalisation test (optional)
        if test_norm: 
            self.test_normalisation(data_processor, base_ds, aux_ds, highres_aux_ds, station_df, base_raw_ds, aux_raw_ds, highres_aux_raw_ds, station_raw_df)

        start = time()
        # Generate auxilary datasets with additional data
        print('Generating auxiliary datasets...')
        aux_ds = self.add_coordinates(aux_ds)
        if include_time_of_year:
            base_ds = self.add_time_of_year(base_ds)
        print('Auxiliary datasets generated in', time()-start, 'seconds')
    
        data_processor_dict = {}
        data_processor_dict['data_processor'] = data_processor
        # if self.var == 'precipitation':
            # data_processor_dict['min_val'] = min_val
        data_processor_dict['aux_raw_ds'] = aux_raw_ds
        data_processor_dict['aux_ds'] = aux_ds
        # data_processor_dict['highres_aux_raw_ds'] = highres_aux_raw_ds
        # data_processor_dict['base_ds'] = base_ds
        data_processor_dict['highres_aux_ds'] = highres_aux_ds
        # data_processor_dict['station_df'] = station_df
        data_processor_dict['landmask_ds'] = landmask_ds
        data_processor_dict['station_as_context'] = station_as_context
        if save != None:
            data_processor_dict_fpath = save
            print(f'Saving data_processor_dict to {data_processor_dict_fpath}')
            with open(data_processor_dict_fpath, 'wb+') as f:
                pickle.dump(data_processor_dict, f)
         
        return data_processor_dict
        # self.data_processor = data_processor
        # self.aux_ds = aux_ds
        # self.base_ds = base_ds
        # self.highres_aux_ds = highres_aux_ds
        # self.station_df = station_df
        # self.landmask_ds = landmask_ds


    def test_normalisation(self, data_processor, base_ds, aux_ds, highres_aux_ds, station_df, base_raw_ds, aux_raw_ds, highres_aux_raw_ds, station_raw_df):

        for ds, raw_ds, ds_name in zip([base_ds, aux_ds, highres_aux_ds], 
                    [base_raw_ds, aux_raw_ds, highres_aux_raw_ds], 
                    ['base', 'Topography', 'Topography (high res)']):
            ds_unnormalised = data_processor.unnormalise(ds)
            xr.testing.assert_allclose(raw_ds, ds_unnormalised, atol=1e-3)
            print(f"Unnormalised {ds_name} matches raw data")

        station_df_unnormalised = data_processor.unnormalise(station_df)
        pd.testing.assert_frame_equal(station_raw_df, station_df_unnormalised)
        print(f"Unnormalised station_df matches raw data")


    def add_coordinates(self, ds):
        """
        Generate auxiliary dataset of x1/x2 coordinates to break translation equivariance in the model's CNN to enable learning non-stationarity
        """
        x1x2_ds = construct_x1x2_ds(ds)
        ds['x1_arr'] = x1x2_ds['x1_arr']
        ds['x2_arr'] = x1x2_ds['x2_arr']
        return ds


    def get_processed_output_dict(self, validation=False):

        if self.base == 'era5':
            date_info = {
                    'validation_years': self.validation_years,
                    'use_daily_data': self.use_daily_data,
                }
            if not validation:
                date_info['training_years'] = self.training_years

        elif self.base == 'wrf':
            date_info = {
                    'validation_paths': self.validation_fpaths
                }
            if not validation:
                date_info['training_paths'] = self.training_fpaths


        data_settings = {
            'var': self.var,
            'base': self.base,
            'topography_highres_coarsen_factor': self.topography_highres_coarsen_factor,
            'topography_lowres_coarsen_factor': self.topography_lowres_coarsen_factor,
            'resolutions': self._get_resolutions_dict(),
            'area': self.area,
            'context_variables': self.context_variables,
        }
        if self.base == 'era5':
            data_settings['era5_coarsen_factor'] = self.era5_coarsen_factor

        processed_output_dict = {
            'data_processor': self.data_processor,
            'base_ds': self.base_ds,
            'highres_aux_ds': self.highres_aux_ds,
            'aux_ds': self.aux_ds,
            'landmask_ds': self.landmask_ds,
            'station_df': self.station_df,
            'station_as_context': self.station_as_context,

            'station_raw_df': self.station_raw_df,
            'base_raw_ds': self.base_raw_ds,
            
            'data_settings': data_settings,
            'date_info': date_info,
        }
        
        self.processed_output_dict = processed_output_dict

        return processed_output_dict


    def plot_dataset(self,
                     type: Literal['base', 'top_highres', 'top_lowres'],
                     area=None,
                     with_stations=True):
        """
        Plot heatmap of processed data (input for model training)
        type options: ['base', 'top_highres', 'top_lowres']
        """
        if area is None:
            area = self.area

        if type == 'top_highres':
            ds = self.highres_aux_raw_ds
            assert ds is not None
            da_plot = self.process_top.ds_to_da(ds)
        elif type == 'top_lowres':
            ds = self.aux_raw_ds
            assert ds is not None
            da_plot = self.process_top.ds_to_da(ds)
        elif type == 'base':
            ds = self.base_raw_ds
            assert ds is not None
            da_plot = ds.isel(time=0)
        else:
            raise ValueError(f"type={type} not recognised, choose from ['base', 'top_highres', 'top_lowres']")

        ax = self.nzplot.nz_map_with_coastlines(area)
        da_plot.plot()
        if with_stations:
            assert self.station_metadata is not None, 'Run process_stations() first or set with_stations=False'
            ax = self.process_stations.plot_stations(self.station_metadata, ax)

        res = self._lat_lon_dict(ds)

        plt.title(f'{type}\n Lat res: {res["lat"]:.4f} degrees, '
                    f'lon res: {res["lon"]:.4f} degrees')
        plt.show()


    def _get_resolutions_dict(self):
        resolutions = {
            'topography_high_res': self._lat_lon_dict(self.highres_aux_raw_ds),
            'topography_low_res': self._lat_lon_dict(self.aux_raw_ds),
            self.base: self._lat_lon_dict(self.base_raw_ds),
        }
        self.resolutions = resolutions
        return resolutions


    def _lat_lon_dict(self, ds):
        return {
            'lat': self.dataprocess.resolution(ds, 'latitude'),
            'lon': self.dataprocess.resolution(ds, 'longitude'),
            }
    

    def print_resolutions(self):
        resolutions = self._get_resolutions_dict()
        print(f"Topography highres:\n  lon={resolutions['topography_high_res']['lon']:.4f}, lat={resolutions['topography_high_res']['lat']:.4f}", 
              f"\nTopography lowres:\n  lon={resolutions['topography_low_res']['lon']:.4f}, lat={resolutions['topography_low_res']['lat']:.4f}",
              f"\n{self.base}:\n  lon={resolutions[self.base]['lon']:.4f}, lat={resolutions[self.base]['lat']:.4f} ")
        if resolutions['topography_high_res']['lon'] > resolutions[self.base]['lon'] or resolutions['topography_high_res']['lat'] > resolutions[self.base]['lat']:
            warnings.warn(f"highres topography resolution is higher than {self.base} resolution", UserWarning)
        if resolutions['topography_high_res']['lon'] > resolutions['topography_low_res']['lon'] or resolutions['topography_high_res']['lat'] > resolutions['topography_low_res']['lon']:
            warnings.warn("lowres topography resolution is higher than highres topography resolution", UserWarning)

    def load_data_processor_dict(self, fpath):
        with open(fpath, 'rb') as f:
            data_processor_dict = pickle.load(f)
        return data_processor_dict

# class PreprocessForDownscalingERA5(PreprocessForDownscaling):
#     def __init__(self, var, training_years, validation_years, context_variables=['None'], 
#                  use_daily_data=False, base='era5', validation=False,):
#         super().__init__(var, training_years, validation_years, context_variables, 
#                          use_daily_data, base, validation)
        
#         self.process_era = era5.ProcessERA5()
#         self.process_stations = stations.ProcessStations()
#         self.nzplot = utils.PlotData()

#         if validation:
#             self.years = validation_years
#         else:
#             if training_years is None or validation_years is None:
#                 self.years = None
#             else:
#                 self.years = list(set(training_years + validation_years))

