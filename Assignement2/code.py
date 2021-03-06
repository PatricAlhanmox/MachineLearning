import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelBinarizer

n_sample = 0
n_missed = 0

def to_int(data, index):
    i = 0
    D = {}
    data_new = []
    for row in data:
        if row[index] not in D:
            D[row[index]] = i
            i += 1
        row[index] = D[row[index]]
        data_new.append(row)
    return data_new

def char_trans(data, index,strList,V):
    str_list = []
    #countryName = pd.dataFrame({'country':[]})
    for row in data:
        for i in range(V):
            if row[index] == strList[i]:
                row[index] = i
    return data

def char_length(data,index):
    strList = []
    for row in data:
        if row[index] not in strList:
            strList.append(row[index])
    L = len(strList)
    return L, strList
    
def deal_label(label):
    for i in range(len(label)):
        if label[i] == '>50K':
            label[i] = 1
        else:
            label[i] = 0
    return label

def find_intervals(data, index, n_splite):
    min = 999999999999
    max = 0
    for row in data:
        val = int(row[index])
        if val > max:
            max = val
        if val < min:
            min = val
    inter_length = (max-min)/n_splite
    new_data = []
    for row in data:
        val = int(row[index])
        i = 0
        while i*inter_length+min < val:
            i += 1
        row[index] = str(i)
        new_data.append(row)
    return new_data, min, inter_length


if '__main__' == __name__:
    sample = []
    label = []
    with open('adult.data') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            n_sample += 1
            row_cleaned = []

            miss_val = False
            for attr in row:
                attr = attr.strip()
                if attr == "?":
                    n_missed += 1
                    miss_val = True
                    break
                row_cleaned.append(attr)
            if miss_val:
                continue
            if row_cleaned == []:
                continue
            sample.append(row_cleaned[:-1])
            label.append(row_cleaned[-1])
    
    l = [0, 2, 4, -2, -3, -4]
    for index in l:
        sample, _, _ = find_intervals(sample, index, 10)    
    
    for i in range(13):
        new_sample = to_int(sample, i)

    V, strList = char_length(new_sample,-1)
    new_sample = char_trans(new_sample, -1, strList, V)
    new_label = deal_label(label)
    
    x_test = new_sample[:len(new_sample) // 10]
    y_test = label[:len(new_label) // 10]

    x_train = new_sample[len(new_sample) // 10:]
    y_train = label[len(new_label) // 10:]
    
    clf = tree.DecisionTreeClassifier()
    tree.plot_tree(clf.fit(x_train[0:30], y_train[0:30])) 
    plt.show()
