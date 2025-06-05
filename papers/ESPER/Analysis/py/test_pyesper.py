
import numpy as np
import pandas
from importlib.resources import files
from importlib import reload

from PyESPER import lir as Py_lir

from cugn import io as cugn_io

from IPython import embed


line = '90.0'
items = cugn_io.load_up(line)
grid_extrem = items[0]
ds = items[1]
times = items[2]
grid_tbl = items[3]


iz = 5  # depth = 50m

T = ds.temperature[iz,:]
S = ds.salinity[iz,:]


keep = np.isfinite(T.values) & np.isfinite(S.values)


# Format
OutputCoordinates = {}
PredictorMeasurements = {}

OutputCoordinates.update({"longitude": ds.lon.values[keep].tolist(), 
                          "latitude": ds.lat.values[keep].tolist(), 
                          "depth": [5. + iz*10]*np.sum(keep)})

PredictorMeasurements.update({"salinity": S[keep].values.tolist(),
                              "temperature": T[keep].values.tolist(),
                             })


# Dates
dates = pandas.to_datetime(ds.time[keep])

# Convert to decimal years
decimal_years = dates.year + (dates.dayofyear - 1) / 365.25


# Estimated dates
EstDates = decimal_years.tolist()

pyESPER_path = files('PyESPER').joinpath('../')


EstimatesLIR, UncertaintiesLIR, CoefficientsLIR = Py_lir.PyESPER_LIR(
    ['oxygen'], pyESPER_path, OutputCoordinates, PredictorMeasurements, 
    EstDates=EstDates, Equations=[8])