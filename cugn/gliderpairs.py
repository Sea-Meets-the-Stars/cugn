""" Simple Class to Analyze Pairs of Gliders """
import numpy as np

from cugn import gliderdata

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

    def __init__(self, gdata:gliderdata.GliderData):
        self.gdata = gdata


    def find_pairs(self, max_dist:float=None, max_time:float=None,
                   from_scratch:bool=True):
        """
        Find pairs of gliders that are within max_dist and max_time
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

            idx = np.where(tcut & pos)
            # Parse
            self.idx0 = idx[0]
            self.idx1 = idx[1]
            self.dtime = dt[idx]
