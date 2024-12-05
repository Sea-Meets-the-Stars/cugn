""" Code to pilot gliders in a survey pattern """

import numpy as np
import math

from IPython import embed

def irregularsurvey(side, duration, numgliders, speed, divetime):
    side = side * 1000
    if numgliders % 2 != 0:
        raise ValueError('numgliders must be even')
    
    lineloc = np.logspace(
        np.log10(divetime/3600*1000), 
        np.log10(side-divetime/3600*1000), 
        numgliders//2)
    
    ctd = {'x': [], 'y': [], 'time': [], 'missid': []}
    n0 = 0
    
    for n in range(1, numgliders + 1):
        if n <= numgliders // 2:
            xwp0 = lineloc[n-1]
            xwp = xwp0
            ywp0 = np.random.rand() * side
            twp0 = 0
            ywp = side / 2 * (np.sign(np.random.rand() - 0.5) + 1)
        else:
            ywp0 = lineloc[n - numgliders//2 - 1]
            ywp = ywp0
            xwp0 = np.random.rand() * side
            twp0 = 0
            xwp = side / 2 * (np.sign(np.random.rand() - 0.5) + 1)
        
        ctd['x'].append(xwp0)
        ctd['y'].append(ywp0)
        ctd['time'].append(twp0)
        ctd['missid'].append(n)
        
        while twp0 < duration:
            lengthsection = np.sqrt((xwp - xwp0)**2 + (ywp - ywp0)**2)
            timesection = lengthsection / speed
            ndives = round(timesection / divetime)
            heading = math.atan2(ywp - ywp0, xwp - xwp0)
            xdive = speed * divetime * math.cos(heading)
            ydive = speed * divetime * math.sin(heading)
            
            xx = np.linspace(xwp0 + xdive, xwp0 + ndives * xdive, ndives)
            yy = np.linspace(ywp0 + ydive, ywp0 + ndives * ydive, ndives)
            tt = np.linspace(twp0 + divetime, twp0 + ndives * divetime, ndives)
            
            ctd['x'].extend(xx)
            ctd['y'].extend(yy)
            ctd['time'].extend(tt)
            ctd['missid'].extend([n] * len(xx))
            
            xwp0 = ctd['x'][-1]
            ywp0 = ctd['y'][-1]
            twp0 = ctd['time'][-1]
            
            if xwp == side:
                xwp = 0
            elif xwp == 0:
                xwp = side
            
            if ywp == side:
                ywp = 0
            elif ywp == 0:
                ywp = side
        
        n0 = len(ctd['missid'])
    
    # trim time values longer than duration
    jj = [i for i, t in enumerate(ctd['time']) if t <= duration]
    
    ctd['time'] = [ctd['time'][j] for j in jj]
    ctd['x'] = [ctd['x'][j] / 1000 for j in jj]  # output km
    ctd['y'] = [ctd['y'][j] / 1000 for j in jj]  # output km
    ctd['missid'] = [ctd['missid'][j] for j in jj]
    
    embed(header='irregularsurvey 78')
    return ctd


def randomsurvey(side, duration, numgliders, speed, divetime):
    side = side * 1000
    ctd = {'x': [], 'y': [], 'time': [], 'missid': []}
    n0 = 0
    
    for n in range(1, numgliders + 1):
        xwp0 = side / 2
        ywp0 = side / 2
        twp0 = 0
        ctd['x'].append(xwp0)
        ctd['y'].append(ywp0)
        ctd['time'].append(twp0)
        ctd['missid'].append(n)
        
        while twp0 < duration:
            xwp = np.random.rand() * side
            ywp = np.random.rand() * side
            lengthsection = np.sqrt((xwp - xwp0)**2 + (ywp - ywp0)**2)
            timesection = lengthsection / speed
            ndives = round(timesection / divetime)
            heading = math.atan2(ywp - ywp0, xwp - xwp0)
            xdive = speed * divetime * math.cos(heading)
            ydive = speed * divetime * math.sin(heading)
            
            xx = np.linspace(xwp0 + xdive, xwp0 + ndives * xdive, ndives)
            yy = np.linspace(ywp0 + ydive, ywp0 + ndives * ydive, ndives)
            tt = np.linspace(twp0 + divetime, twp0 + ndives * divetime, ndives)
            
            ctd['x'].extend(xx)
            ctd['y'].extend(yy)
            ctd['time'].extend(tt)
            ctd['missid'].extend([n] * len(xx))
            
            xwp0 = ctd['x'][-1]
            ywp0 = ctd['y'][-1]
            twp0 = ctd['time'][-1]
        
        n0 = len(ctd['missid'])
    
    # trim time values longer than duration
    jj = [i for i, t in enumerate(ctd['time']) if t <= duration]
    
    ctd['time'] = [ctd['time'][j] for j in jj]
    ctd['x'] = [ctd['x'][j] / 1000 for j in jj]  # output km
    ctd['y'] = [ctd['y'][j] / 1000 for j in jj]  # output km
    ctd['missid'] = [ctd['missid'][j] for j in jj]
    
    return ctd

import matplotlib.pyplot as plt
import numpy as np

def plot(ctd, dtimemax, numbins):
    x = np.array(ctd['x'])
    y = np.array(ctd['y'])
    time = np.array(ctd['time'])

    # Calculate deltas
    dx = x[:, np.newaxis] - x
    dy = y[:, np.newaxis] - y
    dtime = time[:, np.newaxis] - time
    dr = np.sqrt(dx**2 + dy**2)

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot locations of profiles
    for nn in range(1, max(ctd['missid']) + 1):
        mask = np.array(ctd['missid']) == nn
        ax1.plot(np.array(ctd['x'])[mask], np.array(ctd['y'])[mask], '-x')
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (km)')
    ax1.set_ylabel('y (km)')

    # Plot histogram of separations (log scale)
    jj = (dtime > 0) & (dtime < dtimemax)
    dr_filtered = dr[jj]
    embed(header='plot 68')
    bin_edges = np.logspace(np.log10(dr_filtered.min()), np.log10(dr_filtered.max()), numbins)
    
    ax2.hist(dr_filtered, bins=bin_edges)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\Delta r$ (km)')
    ax2.set_ylabel('Count')

    # Plot histogram of separations (linear scale)
    bin_edges_linear = np.linspace(dr_filtered.min(), dr_filtered.max(), numbins)
    ax3.hist(dr_filtered, bins=bin_edges_linear)
    ax3.set_xlabel(r'$\Delta r$ (km)')
    ax3.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    return {'binr': bin_edges}

# Example usage:
# Assuming you have a ctd dictionary from the previous functions
# sf = randomplots(ctd, dtimemax=24, numbins=50)

if __name__ == '__main__':

    # Random
    ctdrand=randomsurvey(100, 
                         duration=60*86400, # seconds?
                         numgliders=8, 
                         speed=0.25, # m/s
                         divetime=6*3600) # seconds
    sf = plot(ctdrand, 
              dtimemax=10*3600, # 10 hours 
              numbins=50)

    # Irregular
    