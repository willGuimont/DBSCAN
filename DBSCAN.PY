import numpy as np
import matplotlib.pyplot as plt

# DBSCAN algorithm
def DBSCAN(points, dist_func, eps, min_pts):
    # db = [[pt, class], [pt, class], [pt, class], ...]
    db = [[x, -2] for x in points]  # -2 = undefined
    c = 0
    for p_label in db:
        if p_label[1] != -2:
            continue
        neighbors = range_query(db, dist_func, p_label, eps)
        if len(neighbors) < min_pts:
            p_label[1] = -1  # noise
            continue
        c += 1
        p_label[1] = c
        if p_label[0] in neighbors:
            neighbors.remove(p_label[0])
        for q_label in neighbors:
            if q_label[1] == -1:
                q_label[1] = c
            elif q_label[1] != -2:
                continue
            q_label[1] = c
            q_neighbors = range_query(db, dist_func, q_label, eps)
            if len(q_neighbors) >= min_pts:
                neighbors.extend(q_neighbors)
    return db


def range_query(db, dist_func, p, eps):
    out = []
    for q in db:
        if dist_func(p[0], q[0]) <= eps:
            out.append(q)
    return out
	
	
if __name__ == '__main__':
	epsilon = 0.5
	min_pts = 2
	values = np.array(
	    [(np.cos(x), np.sin(x)) for x in np.linspace(0, 2 * np.pi, 100)] + 
        [(2 * np.cos(x), 2 * np.sin(x)) for x in np.linspace(0, 2 * np.pi, 100)] +
		[(3 * np.cos(x), 3 * np.sin(x)) for x in np.linspace(0, 2 * np.pi, 100)]
		)
	db = DBSCAN(values, lambda x, y: (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2, epsilon, min_pts)
	db = np.array(db)
	# Split by label
	splitted = []
	for c in np.unique(db[:, 1]):
	    splitted.append((c, [x[0] for x in db if x[1] == c]))
	# Display with different colors
	for s in splitted:
		# Don't draw noise
		if s[0] == -1: 
			continue
		a = np.array(s[1])
		plt.scatter(a[:, 0], a[:, 1])
	plt.show()

