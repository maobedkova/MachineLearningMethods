import numpy as np
import pandas as pd
# import pickle
import gzip
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer


def lemmatize(word):
    return WordNetLemmatizer().lemmatize(word)


def tokenize(line):
    return line.split()


def vectorize(data, filename, to_fit):
    if to_fit:
        tf_idf.fit(data.Tweet)
    vec_repr = tf_idf.transform(data.Tweet)

    return vec_repr

    # with open(filename + '_tf_idf.pickle', 'wb') as f:
    #     pickle.dump(vec_repr, f, pickle.HIGHEST_PROTOCOL)


def union(data_vec, new_features, tf_idf):
    data_vec = pd.DataFrame(data_vec.toarray(), columns=tf_idf.get_feature_names())
    res = pd.concat([data_vec, new_features], axis=1)
    return res



if __name__ == "__main__":
    tf_idf = TfidfVectorizer(encoding="utf-8", lowercase=True, tokenizer=tokenize,
                             analyzer=lemmatize, stop_words=None)

    data = pd.read_csv("train_test.csv", sep=",")

    vectorize(data, "train_test", True)

    test = data.ix[:int(np.floor(data.iloc[:,0].count() * 0.3)), :].reset_index(drop=True)
    train = data.ix[int(np.ceil(data.iloc[:,0].count() * 0.3)):, :].reset_index(drop=True)
    train_labels = train["Annotation"]
    test_labels = test["Annotation"]
    train.drop(["Annotation", "Unnamed: 0"], axis=1, inplace=True)
    test.drop(["Annotation", "Unnamed: 0"], axis=1, inplace=True)

    vec_test = vectorize(test, "test", False)
    vec_train = vectorize(train, "train", False)

    test_all = union(vec_test, test, tf_idf)
    train_all = union(vec_train, train, tf_idf)

    test_all["Annotation"] = test_labels
    train_all["Annotation"] = train_labels

    test_all.to_csv("test_tmp.txt")
    train_all.to_csv("train_tmp.txt")

    with gzip.open('test.txt.gz', 'wb') as w:
        with open("test_tmp.txt") as f:
            test_file = f.read()
            w.write(test_file)
    with gzip.open('train.txt.gz', 'wb') as w:
        with open("train_tmp.txt") as f:
            train_file = f.read()
            w.write(train_file)






