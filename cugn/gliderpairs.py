""" Simple Class to Analyze Pairs of Gliders """
import numpy as np
import datetime
import getpass

from cugn import gliderdata
from cugn import utils as cugn_utils

from IPython import embed

def load_Sndict(filename:str):
    Sn_dict = cugn_utils.loadjson(filename)
    # Convert lists to numpy arrays
    for key in Sn_dict.keys():
        if isinstance(Sn_dict[key], list):
            Sn_dict[key] = np.array(Sn_dict[key])

    # Return
    return Sn_dict

class GliderPairs:

    idx0:np.ndarray = None
    """ 
    Index array for the first glider of the pair
    """

    idx1:np.ndarray = None
    """ 
    Index array for the second glider of the pair
    """

    dtime:np.ndarray = None
    """
    Time difference between the two gliders [hours]
    """

    iz:int = None
    """
    Index of the vertical level
    """

    def __init__(self, gdata:gliderdata.GliderData,
                 max_dist:float=None, max_time:float=None,
                 from_scratch:bool=True, avoid_same_glider:bool=True):

        # Inputs
        self.gdata = gdata
        self.max_dist = max_dist
        self.max_time = max_time
        self.avoid_same_glider = avoid_same_glider

        # Separations
        self.r = None
        self.rx = None
        self.ry = None
        self.rxN = None
        self.ryN = None

        # Velocity
        self.umag = None
        self.du = None
        self.dv = None
        self.duL = None

        # Other variables
        self.dS = None
        self.dT = None

        self.generate_pairs(max_dist=max_dist, max_time=max_time,
                            from_scratch=from_scratch,
                            avoid_same_glider=avoid_same_glider)


    def add_meta(self, sdict:dict):
        """
        Add metadata to the GliderPairs object.

        Args:
            sdict (dict): A dictionary containing analysis output
        """
        sdict['config']['max_dist'] = self.max_dist
        sdict['config']['max_time'] = self.max_time
        sdict['config']['avoid_self'] = self.avoid_same_glider
        sdict['config']['dataset'] = self.gdata.dataset
        if self.iz is not None:
            sdict['config']['iz'] = self.iz
        # Add creation date
        sdict['config']['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add created by
        sdict['config']['created_by'] = getpass.getuser()

    def generate_pairs(self, max_dist:float=None, max_time:float=None,
                       from_scratch:bool=True, avoid_same_glider:bool=True):
        """
        Generate pairs of gliders that are within max_dist and max_time.

        Args:
            max_dist (float, optional): Maximum distance between gliders. Defaults to None.
            max_time (float, optional): Maximum time difference between gliders. Defaults to None.
            from_scratch (bool, optional): Whether to generate pairs from scratch. Defaults to True.
            avoid_same_glider (bool, optional): Whether to avoid pairing the same glider with itself. Defaults to True.

        Raises:
            ValueError: If neither max_dist nor max_time is specified.
            ValueError: If from_scratch is set to False.

        Notes:
            - If max_dist is not specified, only time-based pairing will be performed.
            - If max_time is not specified, only distance-based pairing will be performed.
        """
        if max_dist is None and max_time is None:
            raise ValueError("Must specify either max_dist or max_time")

        if not from_scratch:
            raise ValueError("Only from_scratch=True is supported")

        # Time
        if max_time is not None:
            t = self.gdata.time
            dt = np.zeros((t.size, t.size))
            for kk in range(t.size):
                dt[kk] = (t[kk] - t)/3600.

            # Cut
            t = self.gdata.time
            dt = np.zeros((t.size, t.size))
            for kk in range(t.size):
                dt[kk] = (t[kk] - t)/3600.  # hours

            # Insert a negative sign (trust me)
            dt = -1.*dt

            # Restrcit to the positive values to avoid double counting
            pos = dt > 0.
            tcut = dt < max_time

            cut = tcut & pos
            idx = np.where(tcut & pos)
            # Parse
            self.idx0 = idx[0]
            self.idx1 = idx[1]

        # Avoid using the same glider for any pairs
        if avoid_same_glider:
            keep = self.data('missid', 0) != self.data('missid', 1)
            # Parse
            self.idx0 = self.idx0[keep]
            self.idx1 = self.idx1[keep]

        # Calculate standard stats
        self.update()

        # Checks
        if avoid_same_glider:
            assert not np.any(self.data('missid', 0) == self.data('missid', 1))

    def data(self, key:str, ipair:int, iz:int=None):
        if ipair == 2:
            idx = np.unique(np.concatenate((self.idx0, self.idx1)))
        elif ipair in [0,1]:
            idx = self.idx0 if ipair == 0 else self.idx1
        else:
            raise ValueError("Bad ipair")
        # Return
        if iz is None:
            return getattr(self.gdata, key)[idx]
        else:
            return getattr(self.gdata, key)[iz][idx]
        
    def update(self):
        """
        Update the glider pairs.

        This method calculates the separation between two gliders, generates the r vector,
        and calculates the time difference between the two gliders.

        Parameters:
            None

        """
        # Separations
        d0 = self.data('dist', 0)
        d1 = self.data('dist', 1)
        #
        o0 = self.data('offset', 0)
        o1 = self.data('offset', 1)

        # Separation
        self.r = np.sqrt((d0-d1)**2 + (o0-o1)**2)

        # Generate the r vector
        self.rx = d1-d0
        self.ry = o1-o0
        self.rxN = (d1-d0)/self.r
        self.ryN = (o1-o0)/self.r

        # Time
        t0 = self.data('time', 0)
        t1 = self.data('time', 1)
        self.dtime = (t1-t0)/3600.

    def calc_delta(self, iz:int, variables:str):

        self.iz = iz

        # Velocity
        u0 = self.data('udopacross', 0, iz)
        u1 = self.data('udopacross', 1, iz)
        v0 = self.data('udopalong', 0, iz)
        v1 = self.data('udopalong', 1, iz)

        self.umag = np.sqrt((u1-u0)**2 + (v1-v0)**2)
        self.du = u1-u0
        self.dv = v1-v0

        self.duL = self.rxN*self.du + self.ryN*self.dv

        # Other
        if 'dS' in variables:
            s0 = self.data('s', 0, iz)
            s1 = self.data('s', 1, iz)
            self.dS = s1-s0

        # Other
        if 'dT' in variables:
            t0 = self.data('t', 0, iz)
            t1 = self.data('t', 1, iz)
            self.dT = t1-t0


    def calc_Sn(self, variables:str):

        self.variables = variables
        # duL
        if variables == 'duLduLduL':
            self.S1 = self.duL
            self.S2 = self.duL**2
            self.S3 = self.duL**3
            self.dlbls = ['duL', 'duL**2', variables]
        elif variables == 'duLdSdS':
            self.S1 = self.duL
            self.S2 = self.dS**2
            self.S3 = self.duL*self.dS*self.dS
            self.dlbls = ['duL', 'dS**2', variables]
        elif variables == 'duLdTdT':
            self.S1 = self.duL
            self.S2 = self.dT**2
            self.S3 = self.duL*self.dT*self.dT
            self.dlbls = ['duL', 'dT**2', variables]
        else: 
            raise ValueError("Bad variables")

    def calc_Sn_vs_r(self, rbins:np.ndarray, nboot:int=None):
        """
        Calculate S1, S2, S3 vs r in bins.

        Parameters:
            rbins (np.ndarray): Bin edges

        Returns:
            np.ndarray: Binned S3 values
        """
        N = []
        avg_S1 = []
        med_S1 = []
        std_S1 = []
        err_S1 = []
        avg_S2 = []
        std_S2 = []
        err_S2 = []
        avg_S3 = []
        std_S3 = []
        err_S3 = []

        avg_r = []
        for ss in range(rbins.size-1):
            in_r = (self.r > rbins[ss]) & (self.r <= rbins[ss+1])
            #
            N.append(np.sum(in_r))

            # Bootstrap?
            if nboot is not None:
                # Bootstrap samples
                r_idx = np.where(in_r)[0]
                in_r = np.random.choice(
                    r_idx, size=(nboot,N[-1]), replace=True)
            
            # Stats
            avg_r.append(np.nanmean(self.r[in_r]))
            avg_S1.append(np.nanmean(self.S1[in_r])) 
            med_S1.append(np.nanmedian(self.S1[in_r])) 
            std_S1.append(np.nanstd(self.S1[in_r])) 
            avg_S2.append(np.nanmean(self.S2[in_r])) 
            std_S2.append(np.nanstd(self.S2[in_r])) 
            avg_S3.append(np.nanmean(self.S3[in_r])) 
            std_S3.append(np.nanstd(self.S3[in_r])) 

            if nboot is not None:
                err_S1.append(np.nanstd(np.nanmean(self.S1[in_r], axis=1)))
                err_S2.append(np.nanstd(np.nanmean(self.S2[in_r], axis=1)))
                err_S3.append(np.nanstd(np.nanmean(self.S3[in_r], axis=1)))
                #embed(header='calc_Sn_vs_r 223')
            else:
                err_S1.append(np.nanstd(self.S1[in_r])/np.sqrt(np.sum(np.isfinite(self.S1[in_r])))) 
                err_S2.append(np.nanstd(self.S2[in_r])/np.sqrt(np.sum(np.isfinite(self.S2[in_r])))) 
                err_S3.append(np.nanstd(self.S3[in_r])/np.sqrt(np.sum(np.isfinite(self.S3[in_r])))) 

        # generate a dict
        out_dict = {}
        out_dict['config'] = {}
        out_dict['config']['variables'] = self.variables 
        out_dict['config']['N'] = np.array(N)
        out_dict['r'] = np.array(avg_r)

        out_dict['S1_'+f'{self.dlbls[0]}'] = np.array(avg_S1)
        out_dict['std_S1_'+f'{self.dlbls[0]}'] = np.array(std_S1)
        out_dict['med_S1_'+f'{self.dlbls[0]}'] = np.array(med_S1)
        out_dict['err_S1_'+f'{self.dlbls[0]}'] = np.array(err_S1)
        out_dict['S2_'+f'{self.dlbls[1]}'] = np.array(avg_S2)
        out_dict['std_S2_'+f'{self.dlbls[1]}'] = np.array(std_S2)
        out_dict['err_S2_'+f'{self.dlbls[1]}'] = np.array(err_S2)
        out_dict['S3_'+f'{self.dlbls[2]}'] = np.array(avg_S3)
        out_dict['std_S3_'+f'{self.dlbls[2]}'] = np.array(std_S3)
        out_dict['err_S3_'+f'{self.dlbls[2]}'] = np.array(err_S3)

        # Return
        return out_dict

    def calc_corr_Sn(self, Sn_dict:dict):

        # Init
        Sn_dict['S2corr_'+f'{self.dlbls[1]}'] = Sn_dict['S2_'+f'{self.dlbls[1]}'].copy()
        Sn_dict['S3corr_'+f'{self.dlbls[2]}'] = Sn_dict['S3_'+f'{self.dlbls[2]}'].copy()

        # Correct me
        for ibin in range(Sn_dict['r'].size):
            Sn_dict['S2corr_'+f'{self.dlbls[1]}'][ibin] -= Sn_dict['S1_'+f'{self.dlbls[0]}'][ibin]**2
            Sn_dict['S3corr_'+f'{self.dlbls[2]}'][ibin] -= 3.*Sn_dict['S1_'+f'{self.dlbls[0]}'][ibin]*Sn_dict['S2_'+f'{self.dlbls[1]}'][ibin] \
                + 2.*Sn_dict['S1_'+f'{self.dlbls[0]}'][ibin]**3
