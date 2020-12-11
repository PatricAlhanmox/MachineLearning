import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, ShuffleSplit
import matplotlib.pyplot as plt


def load_dataText(data):
    df_data = pd.DataFrame(data, index=[x for x in range(0, 9501)],
                           columns=list(['article_number', 'article_words', 'topic']))
    newData = df_data.drop(columns=['article_number'])
    res = newData.drop_duplicates(None, 'first', False)
    result = res.dropna()
    return result


def createBagWords(data):
    count = CountVectorizer()
    dataX = count.fit_transform(data['article_words'].to_numpy())
    X_train = dataX[:9001]
    X_test = dataX[9001:]
    return X_train, X_test


def createBagWordsTest(data):
    count = CountVectorizer()
    dataX = count.fit_transform(data['article_words'].to_numpy())
    return dataX


def takeYvalue(data):
    yData = data['topic'].tolist()
    for column in range(len(yData)):
        if yData[column] == 'IRRELEVANT':
            yData[column] = '0'
        elif yData[column] == 'FOREX MARKETS':
            yData[column] = '1'
        elif yData[column] == 'ARTS CULTURE ENTERTAINMENT':
            yData[column] = '2'
        elif yData[column] == 'BIOGRAPHIES PERSONALITIES PEOPLE':
            yData[column] = '3'
        elif yData[column] == 'DEFENCE':
            yData[column] = '4'
        elif yData[column] == 'DOMESTIC MARKETS':
            yData[column] = '5'
        elif yData[column] == 'HEALTH':
            yData[column] = '6'
        elif yData[column] == 'SCIENCE AND TECHNOLOGY':
            yData[column] = '7'
        elif yData[column] == 'SHARE LISTINGS':
            yData[column] = '8'
        elif yData[column] == 'SPORTS':
            yData[column] = '9'
        elif yData[column] == 'MONEY MARKETS':
            yData[column] = '10'
    return yData


# Adding weight to each of the sample
def weight_adding(yData):
    a = np.bincount(yData)
    aa = 1 / a
    class_weight = 'balanced'
    classes = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    w = compute_class_weight(class_weight, classes, yData)
    d = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}
    for key, value in d.items():
        i = int(key)
        d[key] = w.tolist()[i]
    return d


def model_basic(train_x, train_y, test_x, test_y):
    result = DecisionTreeClassifier()
    # Performing training
    result.fit(train_x, train_y)
    pred_y_train = result.predict(train_x)
    score = accuracy_score(train_y, pred_y_train)
    print("For Part A: the accuracy score for training set is", round(score, 2))
    pred_y = result.predict(test_x)
    score = accuracy_score(test_y, pred_y)
    print("For Part A: the accuracy score for test set is", round(score, 2))


def decision_tree_defualt(maxDepth, minimpurity, train_x, train_y):
    # set the lists to store the result
    accuracyscore = []
    precisionscore = []
    recallscore = []
    f1scoreMicro = []
    f1scoreMacro = []

    # iterated for the min_samples_leaf
    for i in range(1, 7):
        Tree = DecisionTreeClassifier(max_depth=maxDepth, min_samples_leaf=i, min_impurity_decrease=minimpurity)
        Tree.fit(train_x, train_y)

        train_y_pred = Tree.predict(train_x)
        y_score = Tree.predict_proba(train_x)

        score = accuracy_score(train_y, train_y_pred)
        accuracyscore.append(round(score, 2))

        ps = precision_score(train_y, train_y_pred, average='macro')
        precisionscore.append(round(ps, 2))

        rs = recall_score(train_y, train_y_pred, average='macro')
        recallscore.append(round(rs, 2))

        f1Micro = f1_score(train_y, train_y_pred, average='micro')
        f1scoreMicro.append(round(f1Micro, 2))

        f1Macro = f1_score(train_y, train_y_pred, average='macro')
        f1scoreMacro.append(round(f1Macro, 2))

    accuracy = cross_val_score(Tree, train_x, train_y, scoring='accuracy', cv=5)
    print("Accuracy = ", accuracy.mean())

    print("For tuning parameters: the accuracy score for train set is: ")
    print(accuracyscore)
    print("For tuning parameters: the precision score for train set is: ")
    print(precisionscore)
    print("For tuning parameters: the recall score for train set is: ")
    print(recallscore)
    print("For tuning parameters: the f1Micro score for train set is: ")
    print(f1scoreMicro)
    print("For tuning parameters: the f1Macro score for train set is: ")
    print(f1scoreMacro)
    print("For tuning parameters: the tees score for train set is: ")

    print(classification_report(train_y, train_y_pred))
    return y_score


def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator of k folds
    cv = KFold(K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print("mean scores=", round(100 * scores.mean(), 5), '%')


def pram(X, Y):
    thresholds = np.linspace(0, 0.2, 50)
    depths = np.arange(1, 11)
    # print(clf.cv_results_, clf.best_estimator_, clf.best_params_)
    cv_set = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)
    param_grid = {'min_impurity_decrease': thresholds, 'max_depth': depths}
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring='accuracy', cv=cv_set, verbose=1)
    clf.fit(X, Y)
    return clf.best_params_


def construct_yData(yData):
    D = {}
    for i in range(len(yData)):
        if yData[i] not in D:
            D[yData[i]] = [i]
        else:
            D[yData[i]].append(i)
    return D


def pre_processed(XData):
    L = []
    for x in range(len(XData)):
        k = max(XData[x])
        L.append(k)
    return L


def process_predict(XData, yData):
    d = []
    for i in range(0, 11):
        L = yData[str(i)]
        new = []
        for j in L:
            new.append(XData[j])
        keys = L
        values = new
        dictionary = dict(zip(keys, values))
        d.append(dictionary)
    return d


import operator


def suggestion(d):
    for i in range(len(d)):
        if len(d[i]) <= 10:
            print(str(i) + ' : ')
            print(d[i].keys())
        else:
            new = dict(sorted(d[i].items(), key=operator.itemgetter(1), reverse=True)[:10])
            print(str(i) + ' : ')
            print(new.keys())


if __name__ == '__main__':
    # load up the data and reforge it
    df = pd.read_csv("training.csv")
    dfTest = pd.read_csv("test.csv")

    datas = load_dataText(df)
    dataTest = load_dataText(dfTest)

    # Filte out the TRAIN_X, TEST_X, TRAIN_Y, TEST_Y
    A, B = createBagWords(datas)
    yData = takeYvalue(datas)
    C = yData[:9001]
    D = yData[9001:]

    AT = createBagWordsTest(dataTest)
    CT = takeYvalue(dataTest)
    # balancedC = weight_adding(C)
    # balancedD = weight_adding(D)

    # Final report
    C1 = pram(A, C)
    D1 = pram(B, D)

    testPra = pram(AT, CT)

    MN = decision_tree_defualt(C1['max_depth'], C1['min_impurity_decrease'], A, C)
    decision_tree_defualt(D1['max_depth'], D1['min_impurity_decrease'], B, D)
    OP = decision_tree_defualt(testPra['max_depth'], testPra['min_impurity_decrease'], AT, CT)
    # evaluate_cross_validation(MN, A, C, 10)

    # Give the final suggestions
    FBI1 = pre_processed(MN.tolist())
    KGB1 = construct_yData(C)
    perf = process_predict(FBI1, KGB1)
    suggestion(perf)

    FBI2 = pre_processed(OP.tolist())
    KGB2 = construct_yData(CT)
    perfx = process_predict(FBI2, KGB2)
    suggestion(perfx)

'''
    #Plotting precisionscore,recallscore vs Iterations
    threshold = []
    for i in range(2, 10):
        threshold.append(i)
    plt.plot(threshold, precisionscore[:-1], "b", label="Precisions")
    plt.plot(threshold, recallscore[:-1], "g", label="Recall")
    plt.plot(threshold, accuracyscore[:-1], "r", label="Accuracy")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1]) 
    plt.show()
'''     