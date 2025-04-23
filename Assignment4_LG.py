
import re
import torchvision as tv
import torch
import tarfile
import pyprind
import pandas as pd
import os
import sys
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from torch.xpu import device




class FNN(nn.Module):
    def __init__(self, input_feature_size):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.layer1 = nn.Linear(input_feature_size, 2048)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(2048, 64)
        self.act2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer3 = nn.Linear(64, 64)
        self.act3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(64, 2)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.Sigmoid(self.output(x))
        return x


#start of Task 1
np.random.seed(0)
torch.manual_seed(1)
count = CountVectorizer()
tfidf = TfidfTransformer(use_idf=True, norm = 'l2',smooth_idf=True)
np.set_printoptions(precision =2)
movie_dir_path = '/Users/liamg/PycharmProjects/PythonProject/movieReviewdataset/'




def load_movie_reviews(csv_file, dataset_size, max_features):
    """
    returns the data as two tensors
    """
    # read the csv file path
    df = pd.read_csv(csv_file)
    # shrink the dataset
    df = df[:dataset_size]

    # creating the feature vector
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=lambda text: text.split(),
        stop_words='english',
        max_features=max_features,
    )
    labels = torch.tensor(df['sentiment'].values)
    features = tfidf_vectorizer.fit_transform(df['review'])
    features = torch.tensor(features.toarray(), dtype=torch.float32)
    return features, labels

def get_test_accuracy(model, test_loader,device):
    correct = 0
    total = 0
    with torch.no_grad():
        for (X, y) in test_loader:
            X, y = X.to(device), y.to(device)
            X = X.float()
            predictions = model(X)
            # compute the accuracy
            _, y_hat = torch.max(predictions.data, dim=1)
            if torch.equal(y_hat, y):
                correct += 1
            total += y.size(0)

    return correct / total



review_array, labels = load_movie_reviews('./movie_data.csv', 10000, 20000 )
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")
X_train, X_test, y_train, y_test = train_test_split(review_array, labels, test_size = 0.3, random_state = 0)
#end of Task 1
#start of Task 2
bs = 8
joint_train = TensorDataset(X_train, y_train)
joint_test = TensorDataset(X_test, y_test)
test_loader = DataLoader(joint_test, batch_size=bs, shuffle=True, drop_last=False)
train_loader = DataLoader(joint_train, batch_size=bs, shuffle=True, drop_last=False)

def model_train(model, train_load, test_load, n_epochs):
    model = model.to(device)
    accuracies = []
    epoch_acc = []
    L = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=.00001, weight_decay=1e-5  )

    model.train()
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        for (x, y) in train_load:
            x, y = x.to(device), y.to(device)
            x = x.float()
            output = net(x)
            loss = L(output, y)
            loss.backward()
            optimizer.step()
            net.zero_grad()

        curr_accuracy = float(get_test_accuracy(model, test_load, device))
        print(f"accuracy in epoch {epoch} = {curr_accuracy}")
        epoch_acc.append(curr_accuracy)
    # evaluate accuracy after training
    model.eval()
    for (x, y) in test_load:
        x, y = x.to(device), y.to(device)
        x = x.float()
        y_pred = model(x)
        y_pred = y_pred.float()
        if torch.equal(y_pred, y):
            accuracies.append(1)
        else:
            accuracies.append(0)


    return accuracies, epoch_acc

net = FNN(20000)
acc_arr, epoch_accuracy = model_train(net,train_loader,test_loader,20)
acc_arr = np.array(acc_arr)
epoch_accuracy = np.array(epoch_accuracy)
for i in range(len(epoch_accuracy)):
    print('Epoch {}: Accuracy {}'.format(i + 1, epoch_accuracy[i]))
print(f"Accuracy after eval:{np.mean(acc_arr)}")


''' This Code block was used to extract the original compressed data set and get a csv created so each time you run the program, you dont ahve to wait for decompression
with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
    tar.extractall()

basepath = 'aclImdb'

labels = {'pos':1, 'neg':0}
pbar = pyprind.ProgBar(50000, stream=sys.stdout)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath,s,l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path,file),'r', encoding = 'utf-8') as infile:
                txt = infile.read()
            df = pd.concat([df, pd.DataFrame([[txt, labels[l]]])], ignore_index=True)
            pbar.update()w
df.columns = ['review', 'sentiment']
df = df.reindex(np.random.permutation(df.index))

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + " ".join(emoticons).replace('-', ''))
    return text

def tokenizer(text):
    return text.split()

df = pd.read_csv('movie_data.csv', encoding = 'utf-8')
print(df.head())
df['review'] = df['review'].apply(preprocessor)
df.to_csv('movie_data.csv', index=False, encoding ='utf-8')
'''
'''
loaded_train_x = DataLoader(train_tensor_x, batch_size=bs, shuffle=True, drop_last=False)
loaded_test_x = DataLoader(test_tensor_x, batch_size=bs, shuffle=True, drop_last=False)
loaded_test_y = DataLoader(test_tensor_y, batch_size=bs, shuffle=True, drop_last=False)
loaded_train_y = DataLoader(train_tensor_y, batch_size=bs, shuffle=True, drop_last=False)

joint_train = TensorDataset(loaded_train_x, loaded_train_y)
joint_test = TensorDataset(loaded_test_x, loaded_test_y)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(104083, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 64)
        self.fc5 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = torch.nn.functional.softmax(self.fc5(x))
        return x

'''
'''
df = pd.read_csv('movie_data.csv', encoding = 'utf-8')
df = df.rename(columns={"0": 'review',"1":'sentiment'})
review_array = df['review'].to_numpy()
labels = df['sentiment'].to_numpy()
tfidf_data = tfidf.fit_transform(count.fit_transform(review_array))
#tfidf_label = tfidf.fit_transform(labels).toarray()

train_tensor_x = torch.from_numpy(X_train)
test_tensor_x = torch.from_numpy(X_test)
train_tensor_y = torch.from_numpy(y_train)
test_tensor_y = torch.from_numpy(y_test)
X_train = X_train.to(device)
y_train = y_train.to(device)
'''
'''

net = FNN(20000)
net = net.to(device)
optimizer = optim.Adam(
        net.parameters(),
        lr=.000001,
        weight_decay=1e-6  # L2 regularization
        )


L = torch.nn.CrossEntropyLoss()
for epoch in range(50):
    #print(f"Staring epoch: {epoch}")
    for (x,y) in train_loader:
        x,y = x.to(device), y.to(device)
        x = x.float()
        output = net(x)
        loss = L(output, y)
        loss.backward()
        optimizer.step()
        net.zero_grad()

    print(f"accuracy: {get_test_accuracy(net, test_loader,device)} in epoch {epoch}")
    #print(f"Ending epoch: {epoch}")

'''



