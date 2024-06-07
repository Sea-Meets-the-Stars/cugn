""" Simple Class to Analyze Pairs of Gliders """
import numpy as np

from cugn import gliderdata

from IPython import embed

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
    Time difference between the two gliders
    """

    def __init__(self, gdata:gliderdata.GliderData,
                 max_dist:float=None, max_time:float=None,
                 from_scratch:bool=True):
        self.gdata = gdata

        # Separations
        self.r = None
        self.rx = None
        self.ry = None
        self.rxN = None
        self.ryN = None

        self.generate_pairs(max_dist=max_dist, max_time=max_time,
                            from_scratch=from_scratch)


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

    def data(self, key:str, ipair:int):
        if ipair == 2:
            idx = np.unique(np.concatenate((self.idx0, self.idx1)))
        elif ipair in [0,1]:
            idx = self.idx0 if ipair == 0 else self.idx1
        else:
            raise ValueError("Bad ipair")
        # Return
        return getattr(self.gdata, key)[idx]
        
    def update(self):

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
        self.dtime = t1-t0