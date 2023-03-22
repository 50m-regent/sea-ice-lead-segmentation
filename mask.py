import cv2
import numpy
import os

from utils import patch_image

def create_edge_mask(file, patch_size):
    edges   = cv2.imread(file)[:, :, 0]
    patches = patch_image(edges, patch_size)
    
    mask = []
    for row in patches:
        _mask_row = []
        for patch in row:
            _mask_row.append(numpy.min(patch) != numpy.max(patch))
            
        mask.append(_mask_row)
        
    mask = numpy.asarray(mask)
    return mask
    
def apply_mask(file, mask, patch_size):
    image   = cv2.imread(file)[:, :, 0]
    patches = patch_image(image, patch_size)
    
    assert patches.shape[:2] == mask.shape
    
    for h in range(patches.shape[0]):
        for w in range(patches.shape[1]):
            patches[h, w] = patches[h, w] if mask[h, w] else numpy.zeros(patches.shape[2:])
            
    image = numpy.concatenate(patches, axis = 1)
    image = numpy.concatenate(image, axis = 1)
    
    return image

def main(mask_path, image_path, save_path, save_name, patch_size):
    mask  = create_edge_mask(mask_path, patch_size)
    image = apply_mask(image_path, mask, patch_size)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    cv2.imwrite(f'{save_path}/{save_name}.png', image)
    
if __name__ == '__main__':
    patch_size = 128
    
    main(
        'rollout/edge_detection/1_96_112.png',
        'rollout/vit/1.png',
        'rollout/mask',
        '1',
        patch_size
    )
    main(
        'rollout/edge_detection/2_80_112.png',
        'rollout/vit/2.png',
        'rollout/mask',
        '2',
        patch_size
    )