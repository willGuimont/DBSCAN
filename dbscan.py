import numpy as np


class DBSCAN:
    """DBCAN clusterer
    Density-based spatial clustering of applications with noise

    https://en.wikipedia.org/wiki/DBSCAN
    """
    UNDEFINED = -2
    NOISE = -1

    class __ClusterPoint:
        def __init__(self, point, cluster):
            self.point = point
            self.cluster = cluster

    def __init__(self, dist_func, epsilon, min_neighbors):
        """Create a DBSCAN clusterer

        dist_function -- function that takes two points and returns the distance between them (e.g. euclidean distance)
        epsilon -- min distance to considere points neighbor
        min_neighbors -- minimum number of points to consider a bunch of points a cluster
        """
        self.dist_func = dist_func
        self.epsilon = epsilon
        self.min_neighbors = min_neighbors

    def cluster(self, points):
        """Split the point per cluster

        points -- Data points
        return dict {cluster: points}
        """
        clustered = self.__dbscan(points)
        splitted = self.__split_by_cluster(clustered)
        return splitted

    def __dbscan(self, points):
        """Perform DBSCAN clustering on points

        points -- Data points
        returns list of point
        """
        point_class_list = [self.__ClusterPoint(
            x, self.UNDEFINED) for x in points]
        current_cluster = 0

        for current_point in point_class_list:
            if current_point.cluster != self.UNDEFINED:
                continue

            neighbors = self.__range_query(point_class_list, current_point)

            if len(neighbors) < self.min_neighbors:
                current_point.cluster = self.NOISE
                continue

            current_point.cluster = current_cluster
            current_cluster += 1

            if current_point in neighbors:
                neighbors.remove(current_point)

            for q_label in neighbors:
                if q_label.cluster == self.NOISE:
                    q_label.cluster = current_cluster
                elif q_label.cluster != self.UNDEFINED:
                    continue

                q_label.cluster = current_cluster
                q_neighbors = self.__range_query(point_class_list, q_label)

                if len(q_neighbors) >= self.min_neighbors:
                    neighbors.extend(q_neighbors)

        return point_class_list

    def __range_query(self, db, current_point):
        """Return points in epsilon range

        db -- all points and their class
        point -- current point
        """
        def is_point_in_range(other_point): return self.dist_func(
            current_point.point, other_point.point) <= self.epsilon

        return list(filter(is_point_in_range, db))

    def __split_by_cluster(self, clustered):
        """Split the returned points of __dbscan into a dictionnary

        points -- List of __ClusterPoint
        return {cluster_number: [points]}
        """
        splitted = {}
        for p in clustered:
            current_cluster = p.cluster
            if current_cluster not in splitted:
                splitted[current_cluster] = []
            splitted[current_cluster].append(p.point)

        return splitted
