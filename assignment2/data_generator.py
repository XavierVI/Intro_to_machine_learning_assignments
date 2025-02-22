import numpy as np
import matplotlib.pyplot as plt


def make_classification(d, n, u=5, random_state=1):
    """
    This function generates a set of points in a
    'd' dimensional real space. Each point will lie in
    the boundary [-u, u] in each direction of the space.

    d: the dimension of the space
    
    n: the number of data points
    
    u: the range in which the data points will be generated.
    Should be no greater than 5.

    random_state: the seed for generating random points
    """
    # create a random number generator based on
    # the given random_state
    rng = np.random.default_rng(random_state)
    # create a random normal vector to define a hyperplane
    normal_vector = rng.random(d)
    
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
    points_near_line = np.abs(inner_products) >= 0.00001
    data_points = data_points[points_near_line]
    inner_products = inner_products[points_near_line]
    n = inner_products.shape[0]
    
    # generating the labels for each point
    labels = np.where(inner_products < 0, -1, 1)

    # appending the labels to the data points
    data_points = np.hstack((data_points, labels.reshape(n, 1)))

    return (data_points, normal_vector)



data_points, normal_vector = make_classification(2, 100)
print(data_points)
print(data_points.shape[0])

x1_coords = np.linspace(-5, 5, 100)
x2_coodrs = np.linspace(-5, 5, 100)
x_coords = np.column_stack((x1_coords, x2_coodrs))
line = np.dot(x_coords, normal_vector)

plt.plot(data_points[:, 0], data_points[:, 1], 'xr')
plt.plot(x_coords, line, '-b')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([-5.5, 5.5])
plt.ylim([-5.5, 5.5])
plt.show()
