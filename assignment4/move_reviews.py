from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import pandas as pd

import os
import sys
import pyprind

import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


def build_csv():
    basepath = 'aclImdb'
    labels = {'pos': 1, 'neg': 0}
    pbar = pyprind.ProgBar(50_000, stream=sys.stdout)
    data = []

    for dataset in ('test', 'train'):
        for sentiment in ('pos', 'neg'):
            path = os.path.join(basepath, dataset, sentiment)
            
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                    txt = infile.read()
                # The append method for dataframe is deprecated and is not used
                # anymore
                data.append([txt, labels[sentiment]])
                pbar.update()

    df = pd.DataFrame(data, columns=['review', 'sentiment'])

    # shuffle the dataset using the sample method
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    
    #### remove HTML tags and emoticons
    df['review'] = df['review'].apply(preprocessor)

    #### save data to a csv
    df.to_csv('movie_data.csv', index=False, encoding='utf-8')


def load_movie_reviews(csv_file, dataset_size, max_features):
    """
    returns the data as two tensors
    """
    # read the csv file path
    df = pd.read_csv(csv_file)
    # shrink the dataset
    df = df[:dataset_size]

    porter = PorterStemmer()

    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]

    # creating the feature vector
    tfidf_vectorizer = TfidfVectorizer(
        # tokenizer=lambda text: text.split(),
        tokenizer=tokenizer_porter,
        stop_words=stopwords.words('english'),
        max_features=max_features,
        ngram_range=(1, 2)
    )
    # labels = torch.tensor(df['sentiment'].values)
    X = tfidf_vectorizer.fit_transform(df['review'])
    y = df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=1,
        shuffle=True,
        stratify=y
    )

    return X_train.toarray(), X_test.toarray(), y_train, y_test


if __name__ == '__main__':
    build_csv()



