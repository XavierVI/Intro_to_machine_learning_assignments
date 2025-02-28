from task1 import LinearSCV
from data_generator import make_classification
from sklearn.model_selection import train_test_split

d = 10
n = 100
u = 5
test_size = 0.3

model = LinearSCV(random_state=1)
data = make_classification(d, n, u, random_state=1)

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)


model.fit(X_train, y_train)

print(y_test == model.predict(X_test))

