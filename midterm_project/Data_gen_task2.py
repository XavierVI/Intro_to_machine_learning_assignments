import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def trainspliter(perc):
    return 1 if random.random() < perc else 0

def plotout(X, Y, tt):
    print(len(X))
    for i in range(len(X)):
        if tt[i] == 1:
            plt.scatter(X[i], Y[i], c = 'red')
        else:
            plt.scatter(X[i], Y[i], c = 'blue')

    plt.title("Input Datagram, with y = .8 sin(x-1)")
    plt.xlabel("random input")
    plt.ylabel("determined output")
    plt.show()
    return 0

def generate(num_samp, domain_range, percent): # percent must be [0, 1]
    input_x_array = np.random.uniform(domain_range[0], domain_range[1], num_samp)
    output_y_array = []
    testortrain = []
    for i in range(num_samp):
        x = input_x_array[i]
        y = 0.8 * np.sin(x - 1)
        testortrain.append(trainspliter(percent))
        output_y_array.append(y)
    for i in range(num_samp):
        dict = {'input_x': input_x_array, 'output_y': output_y_array, 'label': testortrain}
        df = pd.DataFrame(dict)
        df.to_csv('Task2Data.csv', index=False)
    plotout(input_x_array, output_y_array, tt = testortrain)
    return 0

generate(100, [-3,3], .2)

