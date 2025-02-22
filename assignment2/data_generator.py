import numpy as np
import matplotlib.pyplot as plt


def make_classification(d, n, u=5, threshold=0.01, random_state=1):
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
    normal_vector = rng.random(d)
    normal_vector /= np.linalg.norm(normal_vector)

    # uniform set of joint angles
    data_points = rng.uniform(
        low=-u,  # lower bound
        high=u,  # upper bound
        # Generate a (n x d) array
        size=(n, d)
    )

    # computing the inner products
    inner_products = np.dot(data_points, normal_vector)

    # removing points on the line
    points_near_line = np.abs(inner_products) >= threshold
    data_points = data_points[points_near_line]
    inner_products = inner_products[points_near_line]
    n = inner_products.shape[0]

    # generating the labels for each point
    labels = np.where(inner_products < 0, -1, 1)

    # appending the labels to the data points
    data = np.hstack((data_points, labels.reshape(n, 1)))

    return (data, normal_vector)


d = 2
u = 1
treshold = 0.05
n = 1000

data_points, normal_vector = make_classification(d=d, n=n, u=u, threshold=0.15)
print(data_points[:5])
print(data_points.shape)

# plotting line and data points
x1_vals = np.linspace(-u, u, 100)
# solving for x2 values x2 = (-a1 / a2) * x1
x2_vals = - (normal_vector[0] / normal_vector[1]) * x1_vals

plt.plot(data_points[:, 0], data_points[:, 1], 'xg')
plt.plot(x1_vals, x2_vals, '-b')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
