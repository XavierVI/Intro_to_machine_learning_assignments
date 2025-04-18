import pandas as pd
import os
import sys
import pyprind

import re



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



build_csv()



