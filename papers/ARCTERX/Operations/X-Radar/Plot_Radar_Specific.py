## Get and plot latest radar data

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import pandas as pd
import glob
import cmocean.cm as cmo
import numpy as np
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import platform


## Define functions 

def coriolis(lat):
	"""
	f = coriolis frequency (rad/sec)
	ft = coriolis frequency(cycle/day)
	T = coriolis period (hours/cycle)
	"""
	omega = (2*np.pi)/86400
	latrad = lat*np.pi/180
	f = 2*omega*np.sin(latrad)
	T = ((1/f)*2*np.pi)/60/60 # hours
	ft = 1/(T/24)
	return f,ft,T

def calculate_vorticity(xg,yg,uu,vv,latmn):
    '''
    Calculates vorticity and Rossby Number using center differencing.
    Input:
    ------
    xg = x-coordinate grid in meters (2-d, ndarray)
    yg = y-coordinate grid in meters (2-d, ndarray)
    uu = u-velocity
    vv = v-velocity
    latmn = latitude to be used to calculate coriolis frequency

    Output:
    ------
    vort = vorticity sec^-1
    Ro = Rossby number (vort/f)
    '''
    fcor = coriolis(latmn)
    du = np.gradient(uu,axis=-2)
    dv = np.gradient(vv,axis=-1)
    dy = np.gradient(yg,axis=-2)
    dx = np.gradient(xg,axis=-1)
    vort = dv/dx - du/dy
    Ro = vort / fcor[0]
    return vort,Ro

def calculate_divergence(xg,yg,uu,vv):
	'''
	Calculates divergence using center differencing.
	Input:
	------
	xg = x-coordinate grid in meters (2-d, ndarray)
	yg = y-coordinate grid in meters (2-d, ndarray)
	uu = u-velocity
	vv = v-velocity

	Output:
	------
	div = divergence sec^-1
	'''
	du = np.gradient(uu,axis=-1)
	dv = np.gradient(vv,axis=-2)
	dy = np.gradient(yg,axis=-2)
	dx = np.gradient(xg,axis=-1)
	div = du/dx + dv/dy
	return div

def calculate_strain(xg,yg,uu,vv):
	'''
	Calculates strain using center differencing.
	Input:
	------
	xg = x-coordinate grid in meters (2-d, ndarray)
	yg = y-coordinate grid in meters (2-d, ndarray)
	uu = u-velocity
	vv = v-velocity

	Output:
	------
	strain = strain rate sec^-1
	'''
	du = np.gradient(uu,axis=-1)
	dv = np.gradient(vv,axis=-2)
	dy = np.gradient(yg,axis=-2)
	dx = np.gradient(xg,axis=-1)

	strain = np.sqrt((du/dx-dv/dy)**2+(dv/dx+du/dy)**2)
	return strain

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
    else:
        raise Exception('Unsupported platform')
    return paths


def get_path(name):
    paths = get_paths()
    return paths[name]
cruiseshare_path = get_path('cruiseshare')
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

def load_doppler(tlims=None, lon_min=-np.inf, lon_max=np.inf, lat_min=-np.inf, lat_max=np.inf):
    """ Get files within time limits """
    if tlims is None:
        fname = '%s/Data/doppler_radar/currents/tn_*_currents.nc' % get_path('cruiseshare')
        files = glob.glob(fname)
        files.sort()
    elif tlims == 'latest':
        files = [get_path('cruiseshare') + '/Data/doppler_radar/latest/tn_latest_currents.nc']
    else:
        files = get_doppler_files(tlims[0], tlims[1])

    """ Regrid doppler data to a common grid """
    # point to data

    time = np.array([], dtype='datetime64[ns]')
    lat = np.array([])
    lon = np.array([])
    u = np.array([])
    v = np.array([])
    flag = np.array([])
    u_err = np.array([])
    v_err = np.array([])
    
    for file in files:
        # print('loading %s' % file)
        with xr.open_dataset(file) as ds:
            # filter by lon, lat
            ds = ds.where((ds['longitude'] > lon_min) & (ds['longitude'] < lon_max) &(ds['latitude'] > lat_min) & (ds['latitude'] < lat_max),drop=True)
            # append values
            time = np.append(time, ds['time'].values)
            lat = np.append(lat, ds['latitude'].values)
            lon = np.append(lon, ds['longitude'].values)
            u = np.append(u, ds['eastward_sea_water_velocity'].values)
            v = np.append(v, ds['northward_sea_water_velocity'].values)
            flag = np.append(flag, ds['measurement_quality'].values)
            u_err = np.append(u_err,ds['eastward_sea_water_velocity_standard_error'].values)
            v_err = np.append(v_err,ds['northward_sea_water_velocity_standard_error'].values)
    
    # flag values where standard error is greater than 10% of the component measurement
    flag[np.where(u_err > 0.3*u)] = 1
    flag[np.where(v_err > 0.3*v)] = 1

    # create grid based on unique lat, lon, time
    utime = np.unique(time)
    ulat = np.unique(lat)
    ulon = np.unique(lon)

    utime = np.array([pd.Timestamp(tt).timestamp() for tt in utime])
    t_grid, lat_grid, lon_grid = np.meshgrid(utime, ulat, ulon, indexing='ij')

     # # curvilinear x-y grid in km
    xGrid = 111.320*1e3*np.multiply(lon_grid,np.cos(np.deg2rad(lat_grid)))
    yGrid = 110.574*1e3*lat_grid

    ug = np.zeros_like(t_grid)*np.nan
    vg = np.zeros_like(t_grid)*np.nan
    flagg = np.zeros_like(t_grid)*np.nan
    u_errg = np.zeros_like(t_grid)*np.nan
    v_errg = np.zeros_like(t_grid)*np.nan

    for ii, (tt, ll, la) in enumerate(zip(time, lon, lat)):
        i = np.where(pd.Timestamp(tt).timestamp() == utime)[0][0]
        j = np.where(ulat == la)[0][0]
        k = np.where(ulon == ll)[0][0]
        ug[i, j, k] = u[ii]
        vg[i, j, k] = v[ii]
        flagg[i,j,k] = flag[ii]
        u_errg[i, j, k] = u_err[ii]
        v_errg[i, j, k] = v_err[ii]

    data_vars = {
        'u': (('time', 'lat', 'lon'), ug,{'units':'m/s','description':'eastward velocity'}),
        'v': (('time', 'lat', 'lon'), vg,{'units':'m/s','description':'northward velocity'}),
        'flag': (('time', 'lat', 'lon'), flagg,{'description':r'0=good, 1=bad, measurements with error greater than 10% of error are flagged'}),
        'x': (('time','lat', 'lon'), xGrid,{'units':'m','description':'curvilinear x-y grid referenced at (0 E, 0 N)'}),
        'y': (('time','lat', 'lon'), yGrid,{'units':'m','description':'curvilinear x-y grid referenced at (0 E, 0 N)'}),
        'u_err': (('time', 'lat', 'lon'), u_errg,{'units':'m/s','description':'standard error of u'}),
        'v_err': (('time', 'lat', 'lon'), v_errg,{'units':'m/s','description':'standard error of v'}),
    }
    coords = {'time': pd.to_datetime(utime, unit='s'), 'lat': ulat, 'lon': ulon}
    ds = xr.Dataset(data_vars, coords=coords)
    return ds


def get_doppler_files(t0, t1):
    # load doppler
    fname = '%s/Data/doppler_radar/currents/tn_*_currents.nc' % get_path('cruiseshare')
    files = glob.glob(fname)
    files.sort()

    fgood = []
    day_start = t0.floor('H')
    day_end = t1.ceil('H')
    for f in files:
        split = f.split('_')
        date = pd.to_datetime(split[2], format='%Y-%m-%d-%H')
        if date >= day_start and date <= day_end:
            fgood.append(f)
    return fgood



## Load Doppler data 

t0 = pd.Timestamp('2025-02-13T00:00:00')
t1 = pd.Timestamp('2025-02-13T23:59:59')

#ds = load_doppler(tlims='latest')
ds = load_doppler(tlims=[t0,t1])

# Load ship data
tom = load_ship(tlims=[ds.time.min(),ds.time.max()])


# Calculate vorticity, divergence, strain

vort = np.zeros_like(ds.u)*np.nan
div = np.zeros_like(ds.u)*np.nan
strain = np.zeros_like(ds.u)*np.nan

for i,t in enumerate(ds.time.values):
    vort[i,:,:],_ = calculate_vorticity(ds.x.sel(time=t),ds.y.sel(time=t),ds.u.sel(time=t),ds.v.sel(time=t),latmn=np.nanmean(ds.lat.values)) 
    div[i,:,:] = calculate_divergence(ds.x.sel(time=t),ds.y.sel(time=t),ds.u.sel(time=t),ds.v.sel(time=t))
    strain[i,:,:] = calculate_strain(ds.x.sel(time=t),ds.y.sel(time=t),ds.u.sel(time=t),ds.v.sel(time=t))

# Broadcast f for normalization
f,_,_ = coriolis(ds.lat.values)
broadcasted_f = f.reshape(1,len(f),1)
broadcasted_f = np.broadcast_to(broadcasted_f,vort.shape)
f = broadcasted_f

# add data to ds
ds['vorticity'] = (('time','lat','lon'),vort,{'units':'s^-1','description':'Relative vorticity'})
ds['divergence'] = (('time','lat','lon'),div,{'units':'s^-1','description':'Divergence'})
ds['strain'] = (('time','lat','lon'),strain,{'units':'s^-1','description':'Strain rate'})
ds['f'] = (('time','lat','lon'),f,{'units':'s^-1','description':'Coriolis parameter. Does not change with time or lon but is broadcasted to the same shape as the other fields for convenience.'})

## Update netcdfs on cruiseshare
ds_timeMean  = ds.mean(dim='time')
ds_timeMean.to_netcdf('/Volumes/cruiseshare/Situational_awareness/XBand_Doppler/netcdf/latest_timeMean.nc',mode='w')

ds_trackMean = ds.mean(dim=('lat','lon'))
radarTime = ds_trackMean.time.values.astype('float')
shipTime = tom.time.values.astype('float')
ship_lat = np.interp(radarTime,shipTime,tom.lat)
ship_lon = np.interp(radarTime,shipTime,tom.lon)
ds_trackMean['ship_lat'] = ('time',ship_lat)
ds_trackMean['ship_lon'] = ('time',ship_lon)
ds_trackMean.to_netcdf('/Volumes/cruiseshare/Situational_awareness/XBand_Doppler/netcdf/latest_trackMean.nc',mode='w')

## MAKE PLOTS
ds = ds_timeMean

## Projections onto cross and along-front velcoity

theta = 43.8 # degrees
cross_front = [np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta))]
along_front = [np.cos(np.deg2rad(theta-90)),np.sin(np.deg2rad(theta-90))]

vel_cross = ds.u*cross_front[0] + ds.v*cross_front[1]
vel_along = ds.u*along_front[0] + ds.v*along_front[1]

# main figure
mainFig, mainAx = plt.subplots(2,2,figsize=(8,8))

vortIm = mainAx[0,0].pcolormesh(ds.lon,ds.lat,ds.vorticity/ds.f,cmap=cmo.curl,vmin=-2,vmax=2)
mainFig.colorbar(vortIm,ax=mainAx[0,0],label='$\zeta/f$') 
tempPlot = mainAx[0,0].scatter(tom.lon,tom.lat,c=tom.temperatureInlet,s=10,cmap=cmo.thermal)
mainFig.colorbar(tempPlot,ax=mainAx[0,0],label='Temperature (C)',location='bottom')


divIm = mainAx[1,0].pcolormesh(ds.lon,ds.lat,ds.divergence/ds.f,cmap=cmo.balance,vmin=-2,vmax=2)
mainFig.colorbar(divIm,ax=mainAx[1,0],label='$\delta/f$')
saltPlot = mainAx[1,0].scatter(tom.lon,tom.lat,c=tom.salinity,s=10,cmap=cmo.haline)
mainFig.colorbar(saltPlot,ax=mainAx[1,0],label='Salinity (psu)',location='bottom')

crossIm = mainAx[0,1].pcolormesh(ds.lon,ds.lat,ds.u,cmap=cmo.balance,vmin=-0.5,vmax=0.5) # *cross_front[0] + ds.v*cross_front[1]
mainFig.colorbar(crossIm,ax=mainAx[0,1],label='Eastward velocity (m/s)')
timePlot = mainAx[0,1].scatter(tom.lon,tom.lat,c=mdates.date2num(tom.time),s=10,cmap='jet')


alongIm = mainAx[1,1].pcolormesh(ds.lon,ds.lat,ds.v,cmap=cmo.balance,vmin=-0.5,vmax=0.5) # ds.u*along_front[0] + ds.v*along_front[1]
mainFig.colorbar(alongIm,ax=mainAx[1,1],label='Northward velocity (m/s)')


plt.show()
