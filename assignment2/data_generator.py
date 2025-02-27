import numpy as np
import matplotlib.pyplot as plt


def make_classification(d, n, u=5, threshold=None, random_state=1):
    """
    This function generates a set of points in a
    'd' dimensional real space. Each point will lie in
    the boundary [-u, u] in each direction of the space.

    this functions returns the data points and labels in a single
    2D array, and it returns the normal vector used to define the
    hyperplane.

    d: the dimension of the space
    
    n: the number of data points
    
    u: the range in which the data points will be generated.
    Should be no greater than 5.

    threshold: the maximum distance such that if |np.dot(a^T, x)| >= threshold,
    the point is kept in the dataset. Otherwise it is removed.

    random_state: the seed for generating random points
    """
    # create a random number generator based on
    # the given random_state
    rng = np.random.default_rng(random_state)
    # create a random normal vector to define a hyperplane
    normal_vector = rng.standard_normal(d)
    # uses the L2 norm by default
    normal_vector_norm = np.linalg.norm(normal_vector)

    # normalize the vector
    normal_vector /= normal_vector_norm

    # threshold for removing points within the margin
    if threshold == None:
        threshold = u / 5
    

    # uniform set of joint angles
    data_points = rng.uniform(
        low=-u,  # lower bound
        high=u,  # upper bound
        # Generate a (n x d) array
        size=(n, d)
    )

    valid_points = []


    # while loop to fill in the data for removed points
    while(len(valid_points) < n):
        # print(len(valid_points))
        num_needed = n - len(valid_points)
        new_points = rng.uniform(-u, u, size=(num_needed, d))
        # computing the distances between each point and our line
        distances = np.dot(new_points, normal_vector) / normal_vector_norm
        # print(distances)

        # removing points on the line
        points_near_line = np.abs(distances) > threshold
        valid_points.extend(new_points[points_near_line])
    
    data_points = np.array(valid_points[:n])

    # generating the labels for each point
    labels = np.where(np.dot(data_points, normal_vector) < 0, -1, 1)

    # appending the labels to the data points
    data = np.column_stack((data_points, labels))

    return (data, normal_vector)

