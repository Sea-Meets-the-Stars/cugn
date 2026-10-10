""" Code kindly provided by Michael Dalsin  """
import numpy as np
import xarray as xr
import pandas as pd
import glob
import platform

""" globals """
#tgt_path = get_path('tgt-data')
cruiseshare_path = get_path('cruiseshare')


def get_paths():
    if platform.system() == 'Darwin':
        paths = {
            'cruiseshare': '/Volumes/cruiseshare',
            'tgt-data': '/Volumes/tgt-data/TN441B',
            'Shore': '/Volumes/Shore',
            'Ship': '/Volumes/Ship'
        }
    elif platform.system() == 'Windows':
        paths = {
            'cruiseshare': 'C:/cruiseshare',
            'tgt-data': 'C:/tgt_data/TN441B',
            'Shore': 'C:/Shore',
            'Ship': 'C:/Ship'
        }
    elif platform.system() == 'Linux':
        paths = {
            'cruiseshare': '/run/user/1000/gvfs/smb-share:server=10.43.20.20,share=cruiseshare',
            'tgt-data': None,
            'Shore': None,
            'Ship': None,
        }
    else:
        raise Exception('Unsupported platform')
    return paths


def get_path(name:str):
    paths = get_paths()
    return paths[name]

""" Data Loaders """

def load_adcp(adcp_name):
    # check input
    if adcp_name not in ['wh300', 'os75nb']:
        raise ValueError(f'Unknown adcp name: {adcp_name}')
    fname = f'{tgt_path}/adcp/proc/{adcp_name}/contour/{adcp_name}.nc'
    adcp = xr.load_dataset(fname)
    return adcp


def load_ship(tlims=None):
    def preprocess(ds):
        ds = ds.sortby('time')
        ds = ds.drop_duplicates('time')
        # drop dat with t<2020
        ds = ds.sel(time=ds.time.dt.year > 2020)
        return ds

    fname = f'{cruiseshare_path}/Data/ship/ship.2025*.nc'
    # files = glob.glob(fname)
    # files.sort()
    # files = files[1:]
    # ship = xr.open_mfdataset(files, combine='by_coords', preprocess=preprocess).load() 
    with xr.open_mfdataset(fname, combine='by_coords', preprocess=preprocess) as ship:
        if tlims is not None:
            ship = ship.sel(time=slice(*tlims))
            loaded_ship = ship.load()
            ship.close()
    return loaded_ship

def ship_gradient(ds,varName):
    ds = ds.where(~np.isnan(ds[varName]),drop=True)

    diffTime = ds.time[1::]-np.diff(ds.time)/2

    dq = np.diff(ds[varName])
    dq = np.interp(ds.time,diffTime,dq)

    dt = np.diff(ds.time)*1e-9 # convert to seconds
    dt = np.interp(ds.time,diffTime,dt.astype(float))

    dr = ds.sog*dt

    grad = dq/dr

    return grad

""" ADCP Plotters """

def adcpcolor(adcp, axs, xvar='time', **kwargs):
    y = adcp['depth']

    # check dimensions for transpose
    if adcp['u'].shape[0] == len(adcp[xvar]):
        u = adcp['u']
        v = adcp['v']
    else:
        u = adcp['u'].T
        v = adcp['v'].T

    x = adcp[xvar]
    (x, _) = np.meshgrid(x, adcp['depth_cell'], indexing='ij')
    
    pc = axs[0].pcolor(x, y, u, **kwargs)
    pc = axs[1].pcolor(x, y, v, **kwargs)
    return pc


