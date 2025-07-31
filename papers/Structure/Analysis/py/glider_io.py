import os
import glob

import numpy as np

from profiler.loading.pymatreader import pymatreader
from profiler.gliderdata import SprayData

def load_dataset(dataset:str):
    """
    Load a dataset based on the provided dataset name.

    Parameters:
        dataset (str): The name of the dataset to load.

    Returns:
        list: List of profilerdata.ProfilerData objects

    Raises:
        ValueError: If the provided dataset is not supported.
    """

    if dataset == 'Calypso2019':
        dfile = os.path.join(
            os.getenv('OS_SPRAY'), 'Calypso', 'calypso2019_ctd.mat')
    elif dataset == 'Calypso2022':
        dfile = os.path.join(
            os.getenv('OS_SPRAY'), 'Calypso', 'calypso2022_ctd.mat')
    elif dataset == 'ARCTERX-2023':
        dfile = os.path.join(
            os.getenv('OS_DATA'), 'ARCTERX', '2023_IOP', 'arcterx_ctd.mat')
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    # Loop on mission IDs
    d = pymatreader.read_mat(dfile)
    mission_ids = np.unique(d['ctd']['missid'])
    pDatas = []
    for missid in mission_ids:
        sData =  SprayData.from_binned_file(dfile, 'idg', dataset, in_field=False,
                                            missid=missid)

        if dataset == 'Calypso2022':
            # Survey specific cuts
            maxt = np.max(sData.time)
            mint = np.min(sData.time)
            goodt = sData.time < (maxt - 12*24*3600)
            goodt &= (sData.time > (mint + 3*24*3600))
            #embed(header='Cutting data 59')
            sData = sData.profile_subset(np.where(goodt)[0], init=False)
        # Good velocity data
        sData = sData.cut_on_good_velocity(init=False)
        # Save
        pDatas.append(sData)

    return pDatas
