import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[indices]

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    centroids = [data[np.random.choice(range(data.shape[0]))]]
    for _ in range(1, k):
        dist_sq = np.zeros(len(data))
        for i in range(len(data)):
            x = data[i]
            min_dist_sq = np.inf
            for c in centroids:
                dist_sq_i = np.inner(c - x, c - x)
                if dist_sq_i < min_dist_sq:
                    min_dist_sq = dist_sq_i
            dist_sq[i] = min_dist_sq

        probs = dist_sq / dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()

        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break

        centroids.append(data[i])
    return np.array(centroids)

def assign_to_cluster(data, centroid):
    # TODO find the closest cluster for each data point
    cluster_assignments = np.zeros(data.shape[0], dtype=int)
    for i in range(data.shape[0]):
        distances = np.sqrt(np.sum((data[i] - centroid) ** 2, axis=1))
        cluster_assignments[i] = np.argmin(distances)
    return cluster_assignments

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    new_centroids = []
    for i in range(assignments.max() + 1):
        assigned_data = data[assignments == i]
        new_centroid = assigned_data.mean(axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

