import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


class DataProcess:
    def __init__(self) -> None:
        pass


    def open_da(self, 
                da_file:str,
                ) -> xr.DataArray:
        #return rioxarray.open_rasterio(da_file)
        return xr.open_dataarray(da_file)

    
    def mask_da(self, 
                da: xr.DataArray, 
                mask_value:float=-1e30,
                ) -> xr.DataArray:
        """ set to None "no data" points below mask_value (e.g. -1e30) """
        return da.where(da > mask_value).squeeze()


    def coarsen_da(self, 
                    da:xr.DataArray, 
                    coarsen_by:int, 
                    boundary:str='exact',
                    ):
        """
        https://stackoverflow.com/questions/53886153/resample-xarray-object-to-lower-resolution-spatially
        """
        return da.coarsen(lon=coarsen_by, boundary=boundary).mean().coarsen(lat=coarsen_by, boundary=boundary).mean().squeeze()


    def rename_xarray_coords(self,
                             da,
                             rename_dict:dict,
                             ):
        return da.rename(rename_dict)


    def save_nc(self,
                da, 
                name:str,
                ):
        da.to_netcdf(name)