import pandas as pd
import os

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def retrieve_data(path):
    for d, dirs, files in os.walk(path):
        if "test.txt.gz" and "train.txt.gz" in files:
            yield d


def read_data(path):
    df = pd.read_csv(path, compression='gzip', header=None, sep=',')
    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    print(data, labels)
    return data, labels


def find_params(clf, params, data, labels, test_data, test_labels):
    train_data, dev_data, train_labels, dev_labels = train_test_split(data, labels,
                                                                      test_size=0.3, random_state=42)
    print(clf)
    gs = GridSearchCV(clf, params)
    gs.fit(train_data, train_labels)
    print("Best params:",  gs.best_params_)
    print("Accuracy train set:", gs.best_score_)
    dev_preds = gs.predict(dev_data)
    print("Accuracy dev set:", accuracy_score(dev_labels, dev_preds))
    test_preds = gs.predict(test_data)
    print("Accuracy test set:", accuracy_score(test_labels, test_preds))


def classifiers(train_data, train_labels, test_data, test_labels):
    mnb = MultinomialNB()
    A_params = [0.0, 0.25, 0.5, 0.75, 1.0]
    mnb_params = [{"alpha": A_params, "fit_prior": [True, False]}, ]

    svc = SVC(kernel="rbf", random_state=42)
    G_params = [1.0, 0.75, 0.5, 0.1, 0.05]
    svc_params = [{"C": range(25, 1000, 200), "gamma": G_params}, ]

    dt = DecisionTreeClassifier(random_state=42)
    dt_params = [{"min_samples_split": range(10, 100, 10), "max_depth": range(1, 20, 2)}, ]

    knn = KNeighborsClassifier(algorithm="auto")
    N_params = [2, 3, 5, 7, 10, 20]
    knn_params = [{"n_neighbors": N_params}, ]

    clfs = [svc, mnb, dt, knn]
    params = [svc_params, mnb_params, dt_params, knn_params]
    for i in range(0, len(params)):
        find_params(clfs[i], params[i], train_data, train_labels, test_data, test_labels)
        print("-" * 20)


if __name__ == "__main__":
    path = "./data/"
    path2test = "/test.txt.gz"
    path2train = "/train.txt.gz"

    # for path2dataset in retrieve_data(path):
    #     if path2dataset == "./data/cinlp-twitter" or \
    #             path2dataset == "./data/connect-4-raw" or \
    #                 path2dataset == "./data/music-genre-classification" or \
    #                     path2dataset == "./data/car-evaluation":
    #         continue
    path2dataset = "./data/connect-4-interpreted"
    print(path2dataset)
    test_data, test_labels = read_data(path2dataset + path2test)
    train_data, train_labels = read_data(path2dataset + path2train)
    classifiers(train_data, train_labels, test_data, test_labels)
    print("=" * 30)

# cinilp-twitter: remove some columns, T/F->1/0
# credit-card-fraud: negative numbers -> normalize
# music-genre-classification: some features are categorical
