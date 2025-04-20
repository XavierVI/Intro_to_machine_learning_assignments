
import re
import torchvision as tv
import torch
import tarfile
import pyprind
import pandas as pd
import os
import sys
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

#start of Task 1
np.random.seed(0)
torch.manual_seed(1)
count = CountVectorizer()
tfidf = TfidfTransformer(use_idf=True, norm = 'l2',smooth_idf=True)
np.set_printoptions(precision =2)
movie_dir_path = '/Users/liamg/PycharmProjects/PythonProject/movieReviewdataset/'
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

df = pd.read_csv('movie_data.csv', encoding = 'utf-8')
df = df.rename(columns={"0": 'review',"1":'sentiment'})
review_array = df['review'].to_numpy()
labels = df['sentiment'].to_numpy()

tfidf_data = tfidf.fit_transform(count.fit_transform(review_array)).toarray()
#tfidf_label = tfidf.fit_transform(labels).toarray()

X_train, X_test, y_train, y_test = train_test_split(tfidf_data, labels, test_size = 0.3, random_state = 0)
#end of Task 1
#start of Task 2

train_tensor_x = torch.from_numpy(X_train)
test_tensor_x = torch.from_numpy(X_test)
train_tensor_y = torch.from_numpy(y_train)
test_tensor_y = torch.from_numpy(y_test)
bs = 64

joint_train = TensorDataset(train_tensor_x, train_tensor_y)
joint_test = TensorDataset(test_tensor_x, test_tensor_y)

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
test_loader = DataLoader(joint_test, batch_size=bs, shuffle=True, drop_last=False)
train_loader = DataLoader(joint_train, batch_size=bs, shuffle=True, drop_last=False)


net = torch.nn.Sequential(
    torch.nn.Linear(104083, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2),
    torch.nn.Softmax(dim=1)
)



optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)
L = torch.nn.CrossEntropyLoss()
for epoch in range(10):
    print(f"Staring epoch: {epoch}")
    for (x,y) in test_loader:

        x = x.float()
        output = net.forward(x.view(-1,104083))
        loss = L(output, y)
        loss.backward()
        optimizer.step()
        net.zero_grad()

    print(f"Ending epoch: {epoch}")

torch.onnx.export(
net, # model to export
(train_loader,), # inputs of the model,
"FNN_clf.onnx", # filename of the ONNX model
input_names=["input"], # Rename inputs for the ONNX model
dynamo=True # True or False to select the exporter to use
)