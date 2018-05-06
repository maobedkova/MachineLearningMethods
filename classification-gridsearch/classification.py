import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


def read_data(path):
    df = pd.read_csv(path, compression='gzip', header=None, sep='\t')
    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    return data, labels


def cross_validation(data, labels):
    kernels = ["rbf", "linear"]#, "poly"] unstable
    for k in kernels:
        clf = SVC(kernel=k, random_state=42)
        print(k, np.mean(cross_val_score(clf, data, labels, cv=10)))
    # linear 0.93389752924293
    # rbf 0.9437270540557712


def heatmap(data, labels):
    clf = SVC(kernel="rbf", random_state=42)

    C_params = [500, 1000, 2000, 5000]
    G_params = [0.75, 0.5, 0.1, 0.05]
    params = {"C": C_params, "gamma": G_params}
    gs = GridSearchCV(clf, params)
    gs.fit(data, labels)
    scores = gs.cv_results_["mean_test_score"].reshape(len(C_params), len(G_params))

    # Borrowed code
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('gama')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(G_params)), G_params)
    plt.yticks(np.arange(len(C_params)), C_params)
    plt.title('Grid Search')
    plt.show()


def grid_search(data, labels, test_data, test_labels):
    train_data, dev_data, train_labels, dev_labels = train_test_split(data, labels,
                                                                      test_size=0.3, random_state=42)

    clf = SVC(kernel="rbf", random_state=42)

    C_params = [500, 1000, 2000, 5000]
    G_params = [0.75, 0.5, 0.1, 0.05]
    params = [
        {'C': C_params, 'gamma': G_params},
    ]
    gs = GridSearchCV(clf, params)
    gs.fit(train_data, train_labels)
    print("Best:", gs.best_score_, gs.best_params_)
    dev_preds = gs.predict(dev_data)
    print("Accuracy dev set:", accuracy_score(dev_labels, dev_preds))
    test_preds = gs.predict(test_data)
    print("Accuracy test set:", accuracy_score(test_labels, test_preds))


if __name__ == "__main__":
    path_test = "./pamap-easy/test.txt.gz"
    path_train = "./pamap-easy/train.txt.gz"

    test_data, test_labels = read_data(path_test)
    train_data, train_labels = read_data(path_train)

    cross_validation(train_data, train_labels)

    heatmap(train_data, train_labels)

    grid_search(train_data, train_labels, test_data, test_labels)

