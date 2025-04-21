from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def tokenizer(text):
    return text.split()


df = pd.read_csv('movie_data.csv')
print(df.info())

X_train = df.loc[:25_000, 'review'].values
y_train = df.loc[:25_000, 'sentiment'].values
X_test = df.loc[25_000:, 'review'].values
y_test = df.loc[25_000:, 'sentiment'].values

tfidf = TfidfVectorizer(
    strip_accents=None,
    lowercase=False,
    preprocessor=None
)

small_param_grid = [
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf': [False],
        'vect__norm': [None],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
]

lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(solver='liblinear'))
])

gs_lr_tfidf = GridSearchCV(
    lr_tfidf,
    small_param_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1,
    refit=True
)

gs_lr_tfidf.fit(X_train, y_train)

print(f'Time cost: {gs_lr_tfidf.refit_time_}')
print(f'Best score: {gs_lr_tfidf.best_score_:.3f}')
clf = gs_lr_tfidf.best_estimator_
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')
