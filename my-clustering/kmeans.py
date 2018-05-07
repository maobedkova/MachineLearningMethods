import numpy as np
import pandas as pd
from time import time
from sklearn import metrics


class KMeans:
    def predict(self, data, centroids):
        data = np.array(data)
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clusters = np.argmin(distances, axis=0)
        inertia = np.sum(distances)
        return clusters, inertia

    def cluster(self, data, k, iters=10):
        data = np.array(data)
        copied_data = data.copy()
        np.random.shuffle(copied_data)
        centroids = copied_data[:k]
        for i in range(iters):
            # closest centroids to data points
            clusters, _ = self.predict(data, centroids)
            # new centroids
            new_centroids = np.array([data[clusters == k].mean(axis=0) for k in range(centroids.shape[0])])
            centroids = new_centroids
        return centroids, clusters


def read_data(path):
    df = pd.read_csv(path, header=None, sep='\t')
    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    return data, labels


def summary(data, labels, preds, time, inertia):
    n_samples, n_features = data.shape

    print("n_samples %d, \t n_features %d"
          % (n_samples, n_features))

    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

    def bench_k_means(time, inertia, preds, data):

        print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
              % ("k-means", time, inertia,
                 metrics.homogeneity_score(labels, preds),
                 metrics.completeness_score(labels, preds),
                 metrics.v_measure_score(labels, preds),
                 metrics.adjusted_rand_score(labels, preds),
                 metrics.adjusted_mutual_info_score(labels, preds),
                 metrics.silhouette_score(data, preds,
                                          metric='euclidean',
                                          sample_size=100)))

    bench_k_means(time, inertia, preds, data)


def call_kmeans(train_data, test_data, test_labels):
    km = KMeans()
    t0 = time()
    centroids, clusters = km.cluster(train_data, 24)
    time_elapsed = time() - t0
    clusters, inertia = km.predict(test_data, centroids)
    summary(test_data, test_labels, clusters, time_elapsed, inertia)


if __name__ == "__main__":
    path_test = "pamap_easy.test.txt"
    path_train = "pamap_easy.train.txt"

    test_data, test_labels = read_data(path_test)
    train_data, train_labels = read_data(path_train)

    call_kmeans(train_data, test_data, test_labels)
