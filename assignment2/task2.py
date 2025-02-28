from data_generator import make_classification
import  matplotlib.pyplot as plt

d = 2
n = 100
u = 5

data = make_classification(d, n, u, random_state=1)

plt.plot(data[:, 0], data[:, 1], 'xb')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.title('Plot of generated data')
plt.show()