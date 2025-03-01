import numpy as np
import os

def make_classification(d, n, u=5, threshold=None, random_state=1, debug=False):
    """
    This function generates a set of points in a
    'd' dimensional space. Each point will lie in
    the boundary [-u, u] in each direction of the space.

    This function returns the data points and labels in a single
    2D array.

    d: the dimension of the space
    
    n: the number of data points
    
    u: the range in which the data points will be generated.
    Should be no greater than 5.

    threshold: the maximum distance such that if |np.dot(a^T, x)| >= threshold,
    the point is kept in the dataset. Otherwise it is removed.

    random_state: the seed for generating random points

    debug: if true, prints information for debugging
    """

    if debug:
        print(f'Generating data for d={d}, n={n}, u={u}')
    # create a random number generator based on
    # the given random_state
    rng = np.random.default_rng(random_state)
    # create a random normal vector to define a hyperplane
    normal_vector = rng.standard_normal(d)
    # uses the L2 norm by default
    normal_vector_norm = np.linalg.norm(normal_vector)

    # if a threshold wasn't given, use 1 / ||normal_vector|| as the threshold
    if threshold == None:
        threshold = 1 / normal_vector_norm

    valid_points = []

    while(len(valid_points) < n):
        # print(len(valid_points))
        num_needed = n - len(valid_points)
        if debug:
            print(f'Num needed: {num_needed}')
        new_points = rng.uniform(-u, u, size=(num_needed, d))
        # computing the distances between each point and our line
        distances = np.dot(new_points, normal_vector) / normal_vector_norm

        # removing points on the line
        points_near_line = np.abs(distances) > threshold
        valid_points.extend(new_points[points_near_line])
    
    data_points = np.array(valid_points[:n])

    # generating the labels for each point
    labels = np.where(np.dot(data_points, normal_vector) < 0, -1, 1)

    # appending the labels to the data points
    data = np.column_stack((data_points, labels))

    return data

def main():
    random_state = 1

    # generating data to show how each model scales with n = 1000 and d increasing
    ds = [10, 50, 100, 500]
    n = 1000
    u = 5

    if not os.path.isdir('./data'):
        os.mkdir('./data')

    for d in ds:
        data = make_classification(d, n, u, random_state=random_state)
        np.savetxt(f"./data/data_d{d}_n{n}_u{u}.csv", data, delimiter=",")


    # generating data to show how each model scales with d = 50 and n increasing
    d = 50
    ns = [100, 500, 1000, 10_000]
    u = 5

    for n in ns:
        data = make_classification(d, n, u, random_state=random_state)
        np.savetxt(f"./data/data_d{d}_n{n}_u{u}.csv", data, delimiter=",")


if __name__ == "__main__":
    main()
