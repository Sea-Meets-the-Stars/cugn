import os
import glob
import numpy as np

from profiler import gliderdata
from profiler import floatdata
from profiler import vmpdata
from profiler import triaxusdata
from profiler.specific import em_apex
from profiler.specific import altos
from profiler import profilers_io

dataset = 'ARCTERX-Leg2'
apath = os.path.join(os.getenv('OS_ARCTERX'), '2025_IOP')

def load_vmp():
    print("Loading VMP")
    datafile = os.path.join(apath, 'VMP/combo.nc')
    vmp = vmpdata.VMPData.from_binned_file(datafile, 'cusack', 
                                       dataset, in_field=True,
                                       missid=20000)
    return [vmp]

def load_seagliders():
    print("Loading Seagliders")
    datafiles = glob.glob(os.path.join(apath, 'gliders/Seagliders/sg*_level2.nc'))
    seagliders = []
    for datafile in datafiles:
        print("Loading: ", datafile)
        s = gliderdata.SeagliderData.from_binned_file(
            datafile, 'seaglider', dataset, in_field=True,
            extra_dict={'adcp_on': False})
        seagliders.append(s)
    return seagliders

def load_slocum():
    print("Loading Slocum")
    datafile = os.path.join(apath, 'gliders/slocum/osu685.l3.nc')
    pData = gliderdata.SlocumData.from_binned_file(
        datafile, 'slocum', dataset, missid=60000, in_field=True)
    return [pData]

def load_sprays():
    print("Loading Sprays")
    datafiles = glob.glob(os.path.join(apath, 'gliders/spray/*.mat'))
    sprays = []
    for datafile in datafiles:
        s = gliderdata.SprayData.from_binned_file(
            datafile, 'idg', dataset, in_field=True,
            extra_dict={'adcp_on': False})
        sprays.append(s)
    return sprays

def load_solos():
    print("Loading Solos")
    solos = []
    datafiles = glob.glob(os.path.join(apath, 'Floats/Solo/*.mat'))
        #'/home/xavier/Projects/Oceanography/data/ARCTERX/Floats/Solo/*.mat')
    for datafile in datafiles:
        solo = floatdata.SoloData.from_binned_file(
            datafile, 'idg', dataset, in_field=True)
        # Add in random second
        solo.time += np.random.uniform(0, 1, size=solo.Nprof)
        solos.append(solo)
    # 
    return solos

def load_flips():
    print("Loading Flips")
    flips = []
    #datafiles = glob.glob('/home/xavier/Projects/Oceanography/data/ARCTERX/Floats/Flip/*.mat')
    datafiles = glob.glob(os.path.join(apath, 'Floats/Flip/*.mat'))
    for datafile in datafiles:
        flip = floatdata.FlipData.from_binned_file(
            datafile, 'idg', dataset, in_field=True)
        # Random time
        flip.time += np.random.uniform(0, 1, size=flip.Nprof)
        flips.append(flip)
    # 
    return flips

def load_apexes():
    print("Loading Apexes")
    emapexs = []
    #dfiles = glob.glob('/home/xavier/Projects/Oceanography/data/ARCTERX/Floats/EM_Apex/EMApex_data_*.mat')
    dfiles = glob.glob(os.path.join(apath, 'Floats/EM_Apex/EMApex_data_*.mat'))
    for dfile in dfiles:
        pData = em_apex.load_emapex_infield(dfile, dataset, binme=True, add_vel=False)
        #  
        emapexs += pData
    return emapexs

def load_altos():
    print("Loading Altos")
    dfile = os.path.join(apath, 'Floats/Alto/tn441_alto_gridded.mat')
    my_altos = altos.load_infield(dfile, dataset, missid_offset=100000)
    return my_altos

def load_triaxes():
    print("Loading Triaxes")
    tris = []
    #datafiles = glob.glob('/home/xavier/Projects/Oceanography/data/ARCTERX/Triaxus/CTD*.mat')
    datafiles = glob.glob(os.path.join(apath, 'Triaxus/CTD*.mat'))
    for datafile in datafiles:
        missid = int(datafile.split('/')[-1].split('_')[1].split('.')[0]) + 50000
        #embed(header='load_triaxes: 79')
        triaxus = triaxusdata.TriaxusData.from_binned_file(datafile, 'triaxus', 
                                       dataset, in_field=True,
                                       missid=missid)
        tris.append(triaxus)
    # 
    return tris


def load_by_asset(assets:list):
    # Generate pairs
    profilers = []
    for asset in assets:
        if asset == 'Spray':
            profilers += load_sprays()
        elif asset == 'Solo':
            profilers += load_solos()
        elif asset == 'Flip':
            profilers += load_flips()
        elif asset == 'Alto':
            profilers += load_altos()
        elif asset == 'EMApex':
            profilers += load_apexes()
        elif asset == 'VMP':
            profilers += load_vmp()
        elif asset == 'Triaxus':
            profilers += load_triaxes()
        elif asset == 'Slocum':
            profilers += load_slocum()
        elif asset == 'Seaglider':
            profilers += load_seagliders()
        else:
            raise IOError(f"Bad asset! {asset}")
    # Return
    return profilers