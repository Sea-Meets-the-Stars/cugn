# Code for generating Structure Functions (SF) of second and third order.
from tqdm.notebook import tqdm
import numpy as np
import xarray as xr
from dask import delayed, compute  # <-- Importing the necessary Dask modules
from dask.diagnostics import ProgressBar

from IPython import embed

def SF2_3(ull, utt):
    ''' Estimates 2nd and 3rd order structure function
        
        Input:
        
        ull, utt: Longitudinal and transversal structure function components''' 
    
    SF2 = ull**2 + utt**2
    SF3 = ull*SF2
    return SF2, SF3

def SF2_3_ul(ull):
    ''' Estimates 2nd and 3rd order structure function
    for the longitudinal component only.
        
        Input:
        
        ull, utt: Longitudinal and transversal structure function components''' 
    
    SF2 = ull**2 #+ utt**2
    SF3 = ull*SF2
    return SF2, SF3

def uv_ult(dU, dV, dx, dy):
    ''' Rotates velocity difference vector to longitudinal and transversal components (Balhwada 2022)
        
        Input:
        
        dU, dV: Zonal and meridional velocity difference component
        dx, dy: Zona and meridional distance vector '''     
    
    norm = (dx**2 + dy**2)**(1/2)
    rx = dx/norm
    ry = dy/norm
    utt = rx*dV - ry*dU # transversal: (u,v) dot [(0, 0, 1) x (dx, dy, 0)]
    ull = rx*dU + ry*dV # longitudinal: (u,v) dot (dx, dy)
    return ull, utt

def dist_ll2xy(xdata, ydata, kwargs, grid):
    ''' Calculates distance between pairs
    
        Input:
        xdata, ydata: X and Y vectors (xarray)
        kwargs: optional kwargs for shifting matrix 
        grid: "m" refers to metric, "deg" refers to lat/lon'''
    

    lon1 = xdata.shift(**kwargs)
    lon0 = xdata
    lat1 = ydata.shift(**kwargs)
    lat0 = ydata
    
    if grid == "m":
        X = abs(lon0 - lon1)
        Y = abs(lat0 - lat1)
    elif grid =="deg":
        # Transforms from lon/lat to meters
        X = abs(lon0 - lon1)*np.cos(0.5*(lon1 + lon0) * np.pi/180.)* 111321
        Y = abs(lat0 - lat1)*111321
    return (X**2 + Y**2)**(1/2)

def distx_dlon(xdata, ydata, kwargs, grid):
    ''' Calculates X distance difference between pairs
    
        Input:
        xdata, ydata: X and Y vectors (xarray)
        kwargs: optional kwargs for shifting matrix 
        grid: "m" refers to metric, "deg" refers to lat/lon'''
    
    lon1 = xdata.shift(**kwargs)
    lon0 = xdata
    lat1 = ydata.shift(**kwargs)
    lat0 = ydata
    
    if grid == "m":
        #dlon = (lon0 - lon1)  # Original
        dlon = (lon1 - lon0)  # JXP
        return dlon
    elif grid == "deg":
        #dlon = (lon0 - lon1)*np.cos(0.5*(lat1 + lat0)*np.pi/180.)  # Original
        dlon = (lon1 - lon0)*np.cos(0.5*(lat1 + lat0)*np.pi/180.)
        return (np.cos(np.pi/180*ydata) * np.pi/180.*6371e3*dlon)


def disty_dlat(ydata, kwargs, grid):
    ''' Calculates Y distance between pairs
    
        Input:
        xdata, ydata: X and Y vectors (xarray)
        kwargs: optional kwargs for shifting matrix 
        grid: "m" refers to metric, "deg" refers to lat/lon'''
    
    if grid =="m":
        dlat = (ydata.shift(**kwargs) - ydata)
    elif grid =="deg":
        dlat = (ydata.shift(**kwargs) - ydata)
        dlat = np.pi/180.* 6371e3 * dlat
    return dlat

def diff_vel(udata, vdata, kwargs):
    ''' Calculates velocity difference between pairs
    
        Input:
        udata, vdata: U and V velocity components (xarray)
        kwargs: optional kwargs for shifting matrix '''
    
    dU = udata.shift(**kwargs) - udata
    dV = vdata.shift(**kwargs) - vdata
    dU = dU
    dV = dV
    return dU, dV

def timescale(sf2, dr):
    ''' Calculates decorrelation time scale from 2nd-order structure function
    
        Input:
        sf2: Second-order structure function
        dr: Separation distance bins '''
    
    return dr/sf2**(1/2)

def rossby_r(sf2, dr, fcor):
    ''' Calculates rossby number from 2nd-order structure function
    
        Input:
        sf2: Second-order structure function
        dr: Separation distance bins 
        fcor: Coriolis frequency'''    
    
    return sf2**(1/2)/(dr*fcor)

@delayed
def process_dcorr(dcorr1, dcorr2, Udata, Vdata, Xg, Yg, grid,
                  reduce_xy=False):
    """Compute separation distance and velocity components for one grid shift (Dask delayed).

    Args:
        dcorr1: Number of grid shifts in the x direction.
        dcorr2: Number of grid shifts in the y direction.
        Udata: Zonal velocity xarray DataArray.
        Vdata: Meridional velocity xarray DataArray.
        Xg: Broadcasted x-coordinate DataArray.
        Yg: Broadcasted y-coordinate DataArray.
        grid: Coordinate system: 'm' for metric, 'deg' for lat/lon.
        reduce_xy: If True, compute SF powers and average over (x, y)
            inside this task so the returned arrays have dims (time,)
            instead of (x, y, time). Shrinks the downstream dask graph
            dramatically for large regions / large maxcorr.

    Returns:
        tuple:
            If reduce_xy is False: (dr, ull, utt) with dims (x, y, time).
            If reduce_xy is True: (dr, ull, utt, sf2, sf3) already
            spatially averaged, with dims (time,). sf2 = <ull**2>,
            sf3 = <ull**3> (powers taken BEFORE the mean).
    """
    kwargs = {'x': dcorr1, 'y': dcorr2}
    dU, dV = diff_vel(Udata, Vdata, kwargs)
    dy = disty_dlat(Yg, kwargs, grid)
    dx = distx_dlon(Xg, Yg, kwargs, grid)
    dr = (dx**2 + dy**2)**(1/2)

    ull, utt = uv_ult(dU, dV, dx, dy)

    if reduce_xy:
        # Take powers BEFORE the spatial mean so we get <ull^n>,
        # not (<ull>)^n.
        sf2 = ull**2
        sf3 = ull * sf2
        dims = ('x', 'y')
        return (dr.mean(dim=dims, skipna=True),
                ull.mean(dim=dims, skipna=True),
                utt.mean(dim=dims, skipna=True),
                sf2.mean(dim=dims, skipna=True),
                sf3.mean(dim=dims, skipna=True))

    return dr, ull, utt

def calculateSF_2(Uds, maxcorr, shiftdim, grid, reduce_xy=False):
    '''Calculates structure functions without ensemble averaging.
    Input:
        Uds: Xarray dataset with uvel and vvel as the zonal and meridional velocity components,
             and X and Y as the zonal and meridional distances, and time as the time.
        maxcorr: Maximum number of shifts.
        shiftdim: Dimensions to shift.
        grid: 'm' when is in meters and 'deg' when is lat/lon
        reduce_xy: If True, take the (x, y) spatial mean inside each
            per-shift task (with SF powers taken first). Output dims
            become (time, dcorr) and du2/du3 are included. Big win for
            peak memory and dask graph size on large regions.

    Output:
        SF_dist: Xarray dataset.
            reduce_xy=False: dims (x, y, time, dcorr), variables ulls, utts, dr.
            reduce_xy=True:  dims (time, dcorr), variables ulls, utts, dr, du2, du3.'''

    output_vars = {}

    Nt = int(len(Uds.time))

    Udata = Uds.u.chunk({'x': 256, 'y': 256, 'time': Nt})
    Vdata = Uds.v.chunk({'x': 256, 'y': 256, 'time': Nt})

    # Precompute meshgrid once
    Xg, Yg, tg = xr.broadcast(Uds.x, Uds.y, Uds.time)

    # Parallelize the loops with Dask
    tasks = []
    for dcorr1 in range(0, maxcorr):
        #for dcorr2 in range(0, maxcorr):
            #if dcorr1 + dcorr2 != 0:
        for dcorr2 in range(-(maxcorr-1), maxcorr):
            if (dcorr1>0) or ((dcorr1 == 0) & (dcorr2 > 0)):
                tasks.append(process_dcorr(
                    dcorr1, dcorr2, Udata, Vdata, Xg, Yg, grid,
                    reduce_xy=reduce_xy))

    # Compute the tasks in parallel using Dask
    with ProgressBar():
        results = compute(*tasks)  # <-- Use compute from Dask

    # Unpack results from Dask's delayed objects
    if reduce_xy:
        dr_list, ull_list, utt_list, sf2_list, sf3_list = zip(*results)
    else:
        dr_list, ull_list, utt_list = zip(*results)

    # Add progress bar on the concatenation (if needed)
    with tqdm(total=len(dr_list), desc="Concatenating results", position=2) as pbar:
        dr = xr.concat(dr_list, dim='dcorr')
        ull = xr.concat(ull_list, dim='dcorr')
        utt = xr.concat(utt_list, dim='dcorr')
        if reduce_xy:
            sf2 = xr.concat(sf2_list, dim='dcorr')
            sf3 = xr.concat(sf3_list, dim='dcorr')
        pbar.update(len(dr_list))  # Update progress after concatenation is done

    output_vars['dr'] = dr
    output_vars['dr'].name = '$r$'
    output_vars['dr'].attrs['long_name'] = 'Separation distance $r$'
    output_vars['dr'].attrs['units'] = 'meters'

    output_vars['ulls'] = ull
    output_vars['ulls'].name = '$\\delta u_L(\\mathbf{s}, \\mathbf{r}, t)$'
    output_vars['ulls'].attrs['long_name'] = 'Longitudinal velocity fluctuation component'
    output_vars['ulls'].attrs['units'] = 'm/s'

    output_vars['utts'] = utt
    output_vars['utts'].name = '$\\delta u_T(\\mathbf{s}, \\mathbf{r}, t)$'
    output_vars['utts'].attrs['long_name'] = 'Transversal velocity fluctuation component'
    output_vars['utts'].attrs['units'] = 'm/s'

    if reduce_xy:
        output_vars['du2'] = sf2
        output_vars['du2'].attrs['long_name'] = (
            'Samples of second-order structure function (spatial mean)')
        output_vars['du2'].attrs['units'] = 'm^2/s^2'

        output_vars['du3'] = sf3
        output_vars['du3'].attrs['long_name'] = (
            'Samples of third-order structure function (spatial mean)')
        output_vars['du3'].attrs['units'] = 'm^3/s^3'
    
#     output_vars['du1'] = ull + utt
#     output_vars['du1'].name = '$\\delta u1(\\mathbf{s}, \\mathbf{r}, t)$'
#     output_vars['du1'].attrs['long_name'] = 'Samples of first-order structure function'
#     output_vars['du1'].attrs['units'] = 'm/s'
    
#     output_vars['du2'] = sf2
#     output_vars['du2'].name = '$\\delta u2(\\mathbf{s}, \\mathbf{r}, t)$'
#     output_vars['du2'].attrs['long_name'] = 'Samples of second-order structure function'
#     output_vars['du2'].attrs['units'] = 'm^2/s^2'
    
#     output_vars['du3'] = sf3
#     output_vars['du3'].name = '$\\delta u2(\\mathbf{s}, \\mathbf{r}, t)$'
#     output_vars['du3'].attrs['long_name'] = 'Samples of third-order structure function'
#     output_vars['du3'].attrs['units'] = 'm^3/s^3'

    # Create the final xarray dataset
    SFdist = xr.Dataset(output_vars)
    
    return SFdist

# def calculateSF(Uds, maxcorr, shiftdim, stackdim, grid):
#     '''Calculates structure functions without ensemble averaging.
    
#        Input:
#        Uds: Xarray dataset with uvel and vvel as the zonal and meridional velocity components, 
#             and X and Y as the zonal and meridional distances, and time as the time.'''
    
#     SFdist = xr.Dataset()
#     output_vars = {}
#     dr_list, sf2_list, sf3_list = [], [], []
#     ull_list, utt_list = [], []
    
#     Udata = Uds.u
#     Vdata = Uds.v
    
#     # meshgrid of lon,lat
#     Xg, Yg, tg = xr.broadcast(Uds.x, Uds.y, Uds.time) 
    
#     # Shifting data
#     for dcorr1 in tqdm(range(0, maxcorr)): #creates kwargs
#         for dcorr2 in range(0, maxcorr):
#             if dcorr1+dcorr2 !=0:
#                 kwargs = {}
# #             for idx in shiftdim:
# #                 print(idx)
#                 kwargs['x'] = dcorr1
#                 kwargs['y'] = dcorr2
        
# #                 print(kwargs)
#                 # Calculates velocity differences and distance
#                 dU, dV = diff_vel(Udata, Vdata, kwargs)
#                 dy = disty_dlat(Yg, kwargs, grid)
#                 dx = distx_dlon(Xg, Yg, kwargs, grid)
#                 dr = (dx**2 + dy**2)**(1/2)
        
#                 ull, utt = uv_ult(dU, dV, dx, dy) #converts to longitudinal and transversal component
#                 sf2, sf3 = SF2_3(ull, utt) # second and third order SF
    
#                 dr_list.append(dr)        
#                 ull_list.append(ull)
#                 utt_list.append(utt)
#                 sf2_list.append(sf2)
#                 sf3_list.append(sf3)
    
#     # Concatenate data 
#     start = time.time()
#     dr = xr.concat(dr_list, dim='dcorr').stack(r=stackdim)
#     end = time.time()
#     print(f"Execution time: {end - start} seconds")
#     ull = xr.concat(ull_list, dim='dcorr').stack(r=stackdim)
#     utt = xr.concat(utt_list, dim='dcorr').stack(r=stackdim)
#     StFc2 = xr.concat(sf2_list, dim='dcorr').stack(r=stackdim)
#     StFc3 = xr.concat(sf3_list, dim='dcorr').stack(r=stackdim)
    
#     # Drops Nans for total
#     dr = dr.dropna(dim="r", how="all")
#     StFc2 = StFc2.dropna(dim="r", how="all")
#     StFc3 = StFc3.dropna(dim="r", how="all")
#     ull = ull.dropna(dim="r", how="all")
#     utt = utt.dropna(dim="r", how="all")

#     # Output variables in data array 
#     output_vars['D2s'] = StFc2.compute()
#     output_vars['D3s'] = StFc3.compute()
#     output_vars['ulls'] = ull.compute()
#     output_vars['utts'] = utt.compute()
#     output_vars['utt2'] = utt**2
#     output_vars['ull2'] = ull**2
#     SFdist = SFdist.assign(**output_vars)
#     SFdist['dr'] = dr.compute()
    
#     SFdist = SFdist.reset_index('r')
#     SFdist = SFdist.assign_coords(r=('r', SFdist.dr.isel(time=0).values))
    
    
#     # Binned means and std dev
#     #meanSF = averageSF(SFdist, rbins)
#     #stdSF = stddevSF(SFdist, rbins)
    
#     #return SFdist, meanSF, stdSF #returns xarrays
#     return SFdist


def dult_mean_orientation(data, rbins, mid_rbins):
    '''Averages samples of structure function for all orientatios (i.e., dcorr).
    
       Input:
       data: Xarray dataset with variables "ulls", "utts", "du2", "du3", and dimensions
             "time", and "dcorr" 
       rbins: Separation distance bins to average
       mid_rbins: Mid point between rbins'''
    
    # Group by bins along 'dr' using 'mid_rbins' as the coordinate labels
    grouped = data.groupby_bins('dr', rbins, labels=mid_rbins)

    # Manually compute the mean over the grouped data
    means = []
    dr_means = []  # List to store mean of 'dr' for each time slice
    bin_labels = []  # Track which bins have data
    for group in grouped:
        bin_value, group_data = group
        means.append(group_data.mean(dim='dcorr').compute())  # Mean over the 'dcorr' dimension for each group
        bin_labels.append(float(bin_value))

        # Compute the mean of 'dr' (Mean of the binned dr values)
        # Ensure that dr_means is an xarray DataArray with mid_rbins as a coordinate
        dr_means.append(xr.DataArray([bin_value.mean()], coords=[('mid_rbins', [bin_value.mean()])]))

    # Concatenate the results along the 'mid_rbins' dimension
    mean_result = xr.concat(means, dim='mid_rbins')

    # Concatenate the mean of 'dr' along the 'mid_rbins' dimension
    dr_mean_result = xr.concat(dr_means, dim='mid_rbins')

    # Assign actual bin-center values as coordinate, drop dr (tracked separately),
    # then reindex to full mid_rbins so all bins are present (NaN for empty bins)
    mean_result = mean_result.drop_vars('dr')
    mean_result = mean_result.assign_coords(mid_rbins=bin_labels)
    mean_result = mean_result.reindex(mid_rbins=mid_rbins)
    dr_mean_result = dr_mean_result.reindex(mid_rbins=mid_rbins)

    return mean_result, dr_mean_result

def process_SF_samples(fs, rbins, mid_rbins):
    """Average structure function samples over orientation for each time step.

    For each time step, bins the SF data by separation distance and averages
    over all orientations (dcorr dimension), then concatenates across time.

    Args:
        fs: xarray Dataset with variables ulls, utts, du2, du3 and dimensions
            time and dcorr.
        rbins: 1D array of separation distance bin edges (meters).
        mid_rbins: 1D array of bin center values.

    Returns:
        xarray.Dataset: Orientation-averaged SF dataset with dimensions
            (time, mid_rbins) and variables ulls, utts, du2, du3, dr.
    """
    # Initialize an empty list to store the results for each time step
    all_results = []
    dr_results = []

    # Parallelize over the time dimension using Dask
    for ii in tqdm(range(len(fs.time))):  # You can modify the slicing here if needed
        # Isolate the data for the current time step and chunk along 'dcorr'
        data_slice = fs.isel(time=ii).chunk({'dcorr': 3000})  # Adjust chunk size if necessary

        # Compute the mean for the current time slice (this can be parallelized with Dask)
        mall, dr_mean = dult_mean_orientation(data_slice, rbins, mid_rbins)

        # Append the result to the list for time dimension concatenation
        all_results.append(mall)
        dr_results.append(dr_mean)

    # Concatenate all results along the 'time' dimension after the loop
    SFaver = xr.concat(all_results, dim='time')

    # Concatenate the mean of 'dr' along the 'time' dimension after the loop
    draver = xr.concat(dr_results, dim='time')

    # Set the 'mid_rbins' as a coordinate in the final result
    SFaver.coords['mid_rbins'] = ('mid_rbins', mid_rbins)
    draver.coords['mid_rbins'] = ('mid_rbins', mid_rbins)

    # Now assign the final_result0['du2'], ['du3'], ['ulls'], and ['utts'] with the new dimensions
    SFaver['du2'].name = 'SF2'
    SFaver['du2'].attrs['long_name'] = '$\\delta u2$ averaged for all orientations'
    SFaver['du2'].attrs['units'] = 'm^2/s^2'

    SFaver['du3'].name = 'SF3'
    SFaver['du3'].attrs['long_name'] = '$\\delta u3$ averaged for all orientations'
    SFaver['du3'].attrs['units'] = 'm^3/s^3'

    SFaver['ulls'].name = 'du_{ll}'
    SFaver['ulls'].attrs['long_name'] = '$\\delta u_L$ averaged for all orientations'
    SFaver['ulls'].attrs['units'] = 'm/s'

    SFaver['utts'].name = 'du_{tt}'
    SFaver['utts'].attrs['long_name'] = '$\\delta u_T$ averaged for all orientations'
    SFaver['utts'].attrs['units'] = 'm/s'

    # Add the mean of 'dr' as a new variable
    SFaver['dr'] = draver
    SFaver['dr'].name = 'Mean Separation distance'
    SFaver['dr'].attrs['long_name'] = 'Mean Separation distance for each time slice'
    SFaver['dr'].attrs['units'] = 'm'

    # Adding dataset-wide attributes (attributes for the entire dataset)
    SFaver.attrs['description'] = 'Structure Function Dataset'
    SFaver.attrs['Model'] = 'Two-layer QG Turbulence'
    
    # Sorts in mid_rbins and in 'time'
    SFaver = SFaver.sortby('time').sortby('mid_rbins')

    # Return the final result with computed means and attributes
    return SFaver

# def SFbin_xr(SFdist, rbins):
#     '''Calculates averages over all orientations and positions ('x', 'y') of structure functions.
    
#        Input:
#        SFdist: Xarray dataset calculated from calculateSF, 
#        rbins: Vector with separation distance'''

#     # rbins = np.linspace(0, 2e5, 25)
#     mid_rbins = 0.5*(rbins[0:-1] + rbins[1:])

#     # Estimates structure function
#     d3s_mean = np.zeros((len(SFdist.time.values), len(mid_rbins)))
#     d2s_mean = np.copy(d3s_mean)*0.
#     ul_mean = np.copy(d3s_mean)*0.
#     ut_mean = np.copy(d3s_mean)*0.
#     ul2_mean = np.copy(d3s_mean)*0.
#     ut2_mean = np.copy(d3s_mean)*0.
#     r_mean = np.copy(d3s_mean)*0.
#     nobs = np.copy(d3s_mean)*0.

#     d3s_std = np.copy(d3s_mean)*0.
#     d2s_std = np.copy(d2s_mean)*0.
#     ul_std = np.copy(ul_mean)*0.
#     ut_std = np.copy(ut_mean)*0.
#     ul2_std = np.copy(ul2_mean)*0.
#     ut2_std = np.copy(ut2_mean)*0.
#     r_std = np.copy(r_mean)*0.

#     for ii in tqdm(np.arange(0, len(SFdist.time))):
#         groupD2s = SFdist.isel(time=ii).groupby_bins('dr', rbins, labels=mid_rbins)
#         mall = groupD2s.mean(...)
#         nobs_SF = groupD2s.count(...)
        
#         # Creates np arrays
#         d3s_mean[ii, :] = mall.D3s.values
#         d2s_mean[ii, :] = mall.D2s.values
#         ul_mean[ii, :] = mall.ulls.values
#         ut_mean[ii, :] = mall.utts.values
#         ul2_mean[ii, :] = mall.ull2.values
#         ut2_mean[ii, :] = mall.utt2.values
#         r_mean[ii, :] = mall.dr.values
#         nobs[ii, :] = nobs_SF.D2s.values
    
#         sall = groupD2s.std(...)
#         d3s_std[ii, :] = sall.D3s.values
#         d2s_std[ii, :] = sall.D2s.values
#         ul_std[ii, :] = sall.ulls.values
#         ut_std[ii, :] = sall.utts.values
#         ul2_std[ii, :] = sall.ull2.values
#         ut2_std[ii, :] = sall.utt2.values
#         r_std[ii, :] = sall.dr.values
        
#     # Saves variables in Xarray dataset
#     mSF1 = xr.Dataset(data_vars=dict(D2s=(["time", "rbins"], d2s_mean),
#                                      D3s=(["time", "rbins"], d3s_mean),
#                                      ulls=(["time", "rbins"], ul_mean),
#                                      utts=(["time", "rbins"], ut_mean),
#                                      ull2=(["time", "rbins"], ul2_mean),
#                                      utt2=(["time", "rbins"], ut2_mean),
#                                      dr=(["time", "rbins"], r_mean),
#                                      nobs=(["time", "rbins"], nobs)),
#                       coords=dict(time=SFdist.time, rbins=mid_rbins),
#                       attrs=dict(description="Structure functions as a function of separation distance and time"),)
    
#     sSF1 = xr.Dataset(data_vars=dict(D2s=(["time", "rbins"], d2s_std),
#                                      D3s=(["time", "rbins"], d3s_std),
#                                      ulls=(["time", "rbins"], ul_std),
#                                      utts=(["time", "rbins"], ut_std),
#                                      ull2=(["time", "rbins"], ul2_std),
#                                      utt2=(["time", "rbins"], ut2_std),
#                                      dr=(["time", "rbins"], r_std)),
#                       coords=dict(time=SFdist.time, rbins=mid_rbins),
#                      attrs=dict(description="Std. Dev. Structure functions as a function of separation distance and time"),)
    
#     # Name and attributes
#     mSF1.D2s.name = 'SF2'
#     mSF1.D2s.attrs['long_name'] = 'Second Order Structure Function'
#     mSF1.D2s.attrs['units'] = 'm^2/s^2'
#     mSF1.D3s.name = 'SF3'
#     mSF1.D3s.attrs['long_name'] = 'Third Order Structure Function'
#     mSF1.D3s.attrs['units'] = 'm^3/s^3'
#     mSF1.ulls.name = 'u_{ll}'
#     mSF1.ulls.attrs['long_name'] = 'Averaged longitudinal velocity component'
#     mSF1.ulls.attrs['units'] = 'm/s'
#     mSF1.utts.name = 'u_{tt}'
#     mSF1.utts.attrs['long_name'] = 'Averaged transversal velocity component'
#     mSF1.utts.attrs['units'] = 'm/s'
#     mSF1.utt2.name = 'u^2_{tt}'
#     mSF1.utt2.attrs['long_name'] = 'Second Order Structure Function transveral velocity component'
#     mSF1.utt2.attrs['units'] = 'm^2/s^2'
#     mSF1.ull2.name = 'u^2_{ll}'
#     mSF1.ull2.attrs['long_name'] = 'Second Order Structure Function longitudinal velocity component'
#     mSF1.ull2.attrs['units'] = 'm^2/s^2'
#     mSF1.dr.name = 'r'
#     mSF1.dr.attrs['long_name'] = 'Mean Separation distance'
#     mSF1.dr.attrs['units'] = 'm'
#     mSF1.nobs.name = 'nobs'
#     mSF1.nobs.attrs['long_name'] = 'Number of observations'



#     # Name and attributes
#     sSF1.D2s.name = 'SF2'
#     sSF1.D2s.attrs['long_name'] = 'Std. Dev. Second Order Structure Function'
#     sSF1.D2s.attrs['units'] = 'm^2/s^2'
#     sSF1.D3s.name = 'SF3'
#     sSF1.D3s.attrs['long_name'] = 'Std. Dev. Third Order Structure Function'
#     sSF1.D3s.attrs['units'] = 'm^3/s^3'
#     sSF1.ulls.name = 'u_{ll}'
#     sSF1.ulls.attrs['long_name'] = 'Std. Dev. Averaged longitudinal velocity component'
#     sSF1.ulls.attrs['units'] = 'm/s'
#     sSF1.utts.name = 'u_{tt}'
#     sSF1.utts.attrs['long_name'] = 'Std. Dev. Averaged transversal velocity component'
#     sSF1.utts.attrs['units'] = 'm/s'
#     sSF1.utt2.name = 'u^2_{tt}'
#     sSF1.utt2.attrs['long_name'] = 'Std. Dev. Second Order Structure Function transveral velocity component'
#     sSF1.utt2.attrs['units'] = 'm^2/s^2'
#     sSF1.ull2.name = 'u^2_{ll}'
#     sSF1.ull2.attrs['long_name'] = 'Std. Dev. Second Order Structure Function longitudinal velocity component'
#     sSF1.ull2.attrs['units'] = 'm^2/s^2'
#     sSF1.dr.name = 'r'
#     sSF1.dr.attrs['long_name'] = 'Std. Dev. Mean Separation distance'
#     sSF1.dr.attrs['units'] = 'm'
    
#     return mSF1, sSF1

# def timescale(sf2, dr):
#     ''' Calculates decorrelation time scale from 2nd-order structure function
    
#         Input:
#         sf2: Second-order structure function
#         dr: Separation distance bins '''
#     return dr/sf2**(1/2)

# def rossby_r(sf2, dr, fcor):
#     ''' Calculates rossby number from 2nd-order structure function
    
#         Input:
#         sf2: Second-order structure function
#         dr: Separation distance bins 
#         fcor: Coriolis frequency'''    
    
#     return sf2**(1/2)/(dr*fcor)