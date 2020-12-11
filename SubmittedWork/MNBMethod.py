import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import SparsePCA, PCA


# pie graph for data
def sample_class_show(y, savepath='res.png'):
    target_stats = Counter(y)
    labels = list(target_stats.keys())
    sizes = list(target_stats.values())
    explode = tuple([0.1] * len(target_stats))
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True, autopct='%1.1f%%')
    ax.axis('equal')
    plt.savefig(savepath)


# data input
df = pd.read_csv('training.csv')
df_test = pd.read_csv('test.csv')
# df = df.drop_duplicates(subset=['article_words', 'topic'])
# sample_class_show(df['topic'], savepath='sample1.png')

# dict for topics
dict_topic = {'ARTS CULTURE ENTERTAINMENT': 1, 'BIOGRAPHIES PERSONALITIES PEOPLE': 2,
              'DEFENCE': 3, 'DOMESTIC MARKETS': 4, 'FOREX MARKETS': 5,
              'HEALTH': 6, 'MONEY MARKETS': 7, 'SCIENCE AND TECHNOLOGY': 8,
              'SHARE LISTINGS': 9, 'SPORTS': 10, 'IRRELEVANT': 0}
df.topic = df.topic.map(dict_topic)
df_test.topic = df_test.topic.map(dict_topic)
print('ratio:')
print(sorted(Counter(df_test['topic']).items()))


# oversample
oversample_8 = df.loc[df['topic'] == 8].sample(n=200, replace=True)
oversample = df.loc[df['topic'].isin([1, 2, 3, 4, 6, 9])].sample(n=1497, replace=True)
df = pd.concat([df, oversample_8, oversample])

# undersample
rest_df = df.loc[df['topic'] != 0]
undersample_0 = df.loc[df['topic'] == 0].sample(n=3000, random_state=42)
df = pd.concat([rest_df, undersample_0])
# print('ratio after undersample and oversample:')
# print(sorted(Counter(df['topic']).items()))

# train development split
# shuffle the data
# df = df.sample(frac=1, random_state=4)
X_train, X_dev, y_train, y_dev = train_test_split(df['article_words'],
                                                  df['topic'],
                                                  test_size=0.2,
                                                  stratify=df['topic'])
# imbalance_label = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
# print(imbalance_label)
# print('ratio on train set:')
# print(sorted(Counter(y_train).items()))

# X_train = df['article_words']
# y_train = df['topic']
X_test = df_test['article_words']
y_test = df_test['topic']
# create features(use word count vectors as feature)
# vectorizer = CountVectorizer()
vectorizer = CountVectorizer(max_df=1.0, ngram_range=(1, 2))
features_train = vectorizer.fit_transform(X_train)
features_test = vectorizer.transform(X_test)
features_dev = vectorizer.transform(X_dev)
# X_train_counts = vectorizer.fit_transform(X_train)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# features_train = X_train_tfidf
# features_dev = vectorizer.transform(X_dev)


labels_train = y_train
print(features_train.shape)
labels_test = y_test
labels_dev = y_dev
# print(features_dev.shape)

# mnb
mnb = MultinomialNB(alpha=0.0001, class_prior=None, fit_prior=True)
# mnb = MultinomialNB()
mnb.fit(features_train, labels_train)
mnb_pred = mnb.predict(features_test)
# mnb_pred = mnb.predict(features_train)
accuracy = cross_val_score(mnb, features_train, labels_train,
                           scoring='accuracy', cv=5)
print("Accuracy = ", accuracy.mean())
print("The training accuracy is: ")
print(accuracy_score(labels_train, mnb.predict(features_train)))
print("The development accuracy is: ")
print(accuracy_score(labels_dev, mnb.predict(features_dev)))
# Classification report
print("Classification report")
print(classification_report(labels_test, mnb_pred))

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])
parameters = {
    'vect__max_df': (1.0, 10, 20, 50),
    'vect__max_features': (None, 5000, 10000),
    'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001),
    'vect__ngram_range': ((1, 1), (1, 2))
    # 'vect__min_df': (1, 2, 5)
}


# cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)
# grid_search = GridSearchCV(pipeline, parameters, n_jobs=1)
# grid_search.fit(X_train, y_train)
# print(grid_search.best_score_)
# print(grid_search.best_params_)


# LogisticRegression
# lr = LogisticRegression()
# create parameter grid
# C = [float(x) for x in np.linspace(start=0.6, stop=1, num=10)]
# multi_class = ['multinomial']
# solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
# class_weight = ['balanced', None]
# penalty = ['l2']
# pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', LogisticRegression())
# ])
# parameters = {
#     'vect__max_features': (None, 5000, 10000),
#     'clf__penalty': penalty,
#     'clf__solver': solver,
#     'clf__multi_class': multi_class,
#     'clf__class_weight': class_weight,
#     'vect__ngram_range': ((1, 1), (1, 2))
#     # 'vect__min_df': (1, 2, 5)
# }
# gridSearch = GridSearchCV(pipeline, parameters, n_jobs=1)
# gridSearch.fit(X_train, y_train)
# print(gridSearch.best_params_)
# print(gridSearch.best_score_)
# scores = cross_val_score(lr, features_dev, labels_dev, cv=5, scoring='accuracy')
# print('development set accuracy:', np.mean(scores))

def suggest(model, data):
    pass


if __name__ == '__main__':

    df_test = pd.read_csv('test.csv')
    df_test.topic = df_test.topic.map(dict_topic)
    X_test = df_test['article_words']
    y_test = df_test['topic']
    features_test = vectorizer.transform(X_test)
    prob_matrix = mnb.predict_log_proba(features_test)
    # print(prob_matrix[:, 5])
    # print(np.argmax(prob_matrix, axis=1))
    # print(mnb.predict(features_train[:10]))
    suggestion = []
    # recommendation
    for i in range(1, 11):
        category = 5
        top_k = 10
        if Counter(df_test['topic'])[i] < top_k:
            # a = df_test.loc[df_test['topic'] == i].index.tolist()
            # b = [i+9501 for i in a]
            top_k_idx = prob_matrix[:, i].argsort()[::-1][:Counter(df_test['topic'])[i]]
            top_k_idx = [i + 9501 for i in top_k_idx]
            suggestion.append([i, top_k_idx])
            continue
        top_k_idx = prob_matrix[:, i].argsort()[::-1][:top_k]
        top_k_idx = [i+9501 for i in top_k_idx]
        suggestion.append([i, top_k_idx])
    #     compute precision recall f1score
    for i in range(0, 10):
        print(suggestion[i])
        dframe = [i-9501 for i in suggestion[i][1]]
        for j in dframe:
            if df_test.iloc[j].loc['topic'] == i+1:
                # TP =
                pass
        print(dframe)
    # print(prob_matrix[:, 5].argsort()[::-1])
    # print(top_k_idx)
    # print(df_test.iloc[top_k_idx])
    # print(df_test.iloc[[1,2,3,4]])
    print(prob_matrix.shape)
    # print(np.argmax(prob_matrix, axis=1), np.argmax(prob_matrix, axis=1).shape)
