from task1 import LinearSCV
from sklearn.model_selection import train_test_split
import os
import time
import numpy as np


"""
Task 3 instructions:
- Investigate the scalability of the LinearSVC class you have implemented.

- You may consider the datasets of the combinations of the following scales:
d = 10, 50, 100, 500, 1000 and
n = 500, 1000, 5000, 10000, 100000.

- Please feel free to adjust the scales according to your computers' configurations,
however the time costs should be obviously different.

- Make sure that you use the same dataset for each combination.
This can be controlled by using the same random seed (see textbook).
"""

random_state = 1
model = LinearSCV(random_state=random_state)
test_size = 0.3
data_dir = os.path.join(os.curdir, 'data')
table = []

for file in os.listdir(data_dir):
    # loading the dataset and splitting the data using scikit learn
    file_path = os.path.join(data_dir, file)
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    d = X.shape[1]
    n = X.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # training the model and timing the training time
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    model_time_cost = end - start

    # making predictions and calculating the accuracy for each model
    model_predictions = model.predict(X_test)
    model_accuracy = np.mean(model_predictions == y_test) * 100

    table.append([d, n, model_time_cost, model_accuracy])


# sort the table
table = sorted(table, key=lambda x: (x[0], x[1]))

# printing out the data as a table
for row in table:
    print(f'{row[0]} & {row[1]} & {row[2]:<.3f} & {row[3]:<.2f} \\\\')





