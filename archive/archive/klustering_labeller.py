import numpy
from sklearn.cluster import KMeans
from utils.load_data import load_train_data

if __name__ == '__main__':
    n_clusters = 8
    
    train_x = load_train_data()
    
    kmeans = KMeans(n_clusters, n_init='auto', verbose=True)
    prediction = kmeans.fit_predict(train_x)
    
    brightest_clusters = list(numpy.argsort(kmeans.cluster_centers_, axis=0)[:, 0][-4:])
    darkest_cluster    = numpy.argmin(numpy.linalg.norm(kmeans.cluster_centers_, axis=1))
    
    x0 = train_x[numpy.where([pred in brightest_clusters for pred in prediction])[0]]
    y0 = numpy.zeros((x0.shape[0],))
    x1 = train_x[numpy.where(prediction == darkest_cluster)[0]]
    y1 = numpy.ones((x1.shape[0],))
    
    x = numpy.concatenate([x0, x1])
    y = numpy.concatenate([y0, y1])
    
    print(x0.shape)
    print(x1.shape)
    print(x.shape)
    print(y.shape)
    
    numpy.savez_compressed(
        f'data/klustering_labelled.npz',
        x=x.astype(numpy.float32),
        y=y.astype(numpy.float32)
    )
    