import os
import xarray
import numpy

def readfiles(path):
    dataset = xarray.open_mfdataset(
        os.path.join(path, '*_radiance.nc')
    ).items()
    
    ret = []
    for name, data in dataset:
        data = data.values[:, ~numpy.isnan(data.values).any(axis=0)]
        ret.append((name, data))
        
    return numpy.array(ret)