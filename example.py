import numpy as np
import matplotlib.pyplot as plt

from dbscan import DBSCAN

if __name__ == '__main__':
    epsilon = 0.5
    min_pts = 2
    points = np.array(
        [(np.cos(x), np.sin(x)) for x in np.linspace(0, 2 * np.pi, 100)] +
        [(2 * np.cos(x), 2 * np.sin(x)) for x in np.linspace(0, 2 * np.pi, 100)] +
        [(3 * np.cos(x), 3 * np.sin(x))
         for x in np.linspace(0, 2 * np.pi, 100)]
    )

    def euclidean_distance_2d(x, y):
        return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1])**2)

    dbscan = DBSCAN(euclidean_distance_2d, epsilon, min_pts)
    clusters = dbscan.cluster(points)

    for points in clusters.values():
        pt_cluster = np.array(points)
        plt.scatter(pt_cluster[:, 0], pt_cluster[:, 1])
    plt.show()
