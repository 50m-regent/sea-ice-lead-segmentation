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
    
def normalize(x):
    return (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x))

def patch_image(image, size):
    image = image[:image.shape[0] // size * size, :image.shape[1] // size * size]
    
    image = numpy.split(image, range(size, image.shape[0], size))
    image = numpy.stack(image)
    
    image = numpy.split(image, range(size, image.shape[2], size), axis = 2)
    image = numpy.stack(image)

    return image.transpose(1, 0, *range(2, len(image.shape)))

def load_labelled_data(size):
    assert size in (1, 3)
        
    with numpy.load(f'data/{size}x{size}_training_set.npz') as f:
        train_x = f['x'].astype(numpy.float32)
        train_y = f['y'].astype(numpy.int64)
    
    with numpy.load(f'data/{size}x{size}_test_set.npz') as f:
        test_x = f['x'].astype(numpy.float32)
        test_y = f['y'].astype(numpy.int64)    
    
    return (train_x, train_y), (test_x, test_y)

def load_unlabelled_data(size):
    assert size in (1, 3)
    
    with numpy.load(f'data/{size}x{size}_unlabelled_set.npz') as f:
        x = f['x'].astype(numpy.float32)
        
    return x