import numpy
import xarray
import os

def _radiance2reflectance(
    radiance,
    solar_flux,
    angles
):
    return numpy.pi * radiance / solar_flux / numpy.cos(numpy.radians(angles))

def get_reflectance(path):
    instrument_data = xarray.open_dataset(
        os.path.join(path, 'instrument_data.nc')
    )
    solar_flux     = instrument_data['solar_flux'].values.T
    detector_index = instrument_data['detector_index'].values[
        :, ~numpy.isnan(instrument_data['detector_index'].values).any(axis=0)
    ].astype(int)
    
    tie_geometries = xarray.open_dataset(
        os.path.join(path, 'tie_geometries.nc')
    )
    sza = tie_geometries['SZA']
    angles = numpy.zeros((*detector_index.shape, 1))
    for j in range(angles.shape[1]):
        angles[:, j, 0] = sza[:, j // 64]
        
    radiance_data = xarray.open_mfdataset(
        os.path.join(path, '*_radiance.nc')
    ).items()
    
    radiance = []
    for _, data in radiance_data:
        radiance.append(data.values[:, ~numpy.isnan(data.values).any(axis=0)])
    
    radiance = numpy.stack(radiance, axis=2)
    return _radiance2reflectance(
        radiance, solar_flux[detector_index], angles
    ).astype(numpy.float32)