import cv2
import numpy

reflectance1 = cv2.imread('reflectance/1/1.png')
kmeans1 = 255 - cv2.imread('rollout/k-means_reflectance/1/8/2.png')
vit1 = cv2.imread('rollout/vit/1.png')
edge1 = cv2.imread('rollout/edge_detection/1_0_96.png')
label_spreading1 = cv2.imread('rollout/label_spreading_map/1.png')
random_forest1 = cv2.imread('rollout/rfc_map/1_1x1.png')

reflectance2 = cv2.imread('reflectance/2/1.png')
kmeans2 = 255 - cv2.imread('rollout/k-means_reflectance/2/8/2.png')
vit2 = cv2.imread('rollout/vit/2.png')
edge2 = cv2.imread('rollout/edge_detection/2_0_96.png')
label_spreading2 = cv2.imread('rollout/label_spreading_map/2.png')
random_forest2 = cv2.imread('rollout/rfc_map/2_1x1.png')

reflectance1 = cv2.resize(
    reflectance1,
    (2048, int(reflectance1.shape[0] * 2048 / reflectance1.shape[1]))
)
kmeans1 = cv2.resize(
    kmeans1,
    (2048, int(kmeans1.shape[0] * 2048 / kmeans1.shape[1]))
)
vit1 = cv2.resize(
    vit1,
    (2048, int(vit1.shape[0] * 2048 / vit1.shape[1]))
)
edge1 = cv2.resize(
    edge1,
    (2048, int(edge1.shape[0] * 2048 / edge1.shape[1]))
)
label_spreading1 = cv2.resize(
    label_spreading1,
    (2048, int(label_spreading1.shape[0] * 2048 / label_spreading1.shape[1]))
)
random_forest1 = cv2.resize(
    random_forest1,
    (2048, int(random_forest1.shape[0] * 2048 / random_forest1.shape[1]))
)

reflectance2 = cv2.resize(
    reflectance2,
    (2048, int(reflectance2.shape[0] * 2048 / reflectance2.shape[1]))
)
kmeans2 = cv2.resize(
    kmeans2,
    (2048, int(kmeans2.shape[0] * 2048 / kmeans2.shape[1]))
)
vit2 = cv2.resize(
    vit2,
    (2048, int(vit2.shape[0] * 2048 / vit2.shape[1]))
)
edge2 = cv2.resize(
    edge2,
    (2048, int(edge2.shape[0] * 2048 / edge2.shape[1]))
)
label_spreading2 = cv2.resize(
    label_spreading2,
    (2048, int(label_spreading2.shape[0] * 2048 / label_spreading2.shape[1]))
)
random_forest2 = cv2.resize(
    random_forest2,
    (2048, int(random_forest2.shape[0] * 2048 / random_forest2.shape[1]))
)

image1 = numpy.concatenate([
    numpy.concatenate([reflectance1, vit1, edge1], axis = 1),
    numpy.concatenate([kmeans1, label_spreading1, random_forest1], axis = 1)
])
image2 = numpy.concatenate([
    numpy.concatenate([reflectance2, vit2, edge2], axis = 1),
    numpy.concatenate([kmeans2, label_spreading2, random_forest2], axis = 1)
])

cv2.imwrite('doc/rollouts1.png', image1)
cv2.imwrite('doc/rollouts2.png', image2)