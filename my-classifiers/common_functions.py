import pandas as pd


def accuracy(true, predicted):
    correct = 0
    for i in range(len(true)):
        if true[i] == predicted[i]:
            correct += 1
    print ("Accuracy:", correct / float(len(true)) * 100.0)


def read_data(f):
    print('Processing document', f)
    dataset = pd.read_csv(f, sep=",", header=None)
    df = pd.DataFrame(dataset)
    label = df.iloc[:,-1].map({'other': 0, 'tiny_polygon': 1,
                               ' >50K': 0, ' <=50K': 1, ' >50K.': 0, ' <=50K.': 1})
    df = df.iloc[:, :-1]
    return df, label


def data_manip(train, test):
    whole = pd.concat([train, test])
    whole = pd.get_dummies(whole.iloc[:, :-1])
    new_train = whole.iloc[:len(train.index), :]
    new_test = whole.iloc[len(train.index):, :]
    return new_train, new_test


if __name__ == "__main__":
    pass
