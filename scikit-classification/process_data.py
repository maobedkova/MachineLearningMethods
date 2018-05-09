import os
import pandas as pd


def retrieve_data(path):
    for d, dirs, files in os.walk(path):
        if "test.txt.gz" and "train.txt.gz" in files:
            yield d


def cinlp(path):
    df = pd.read_csv(path, compression="gzip", header=0, sep=',')
    # print(df)
    df.drop(["Unnamed: 0", "Tweet", "lemmas"], axis=1, inplace=True)
    df.drop([0], axis=0, inplace=True)
    df = df * 1
    # print(df)
    df.to_csv(path, compression="gzip", index=False, header=False)


def music(path):
    df = pd.read_csv(path, compression="gzip", header=0, sep=',')
    # print(df)
    df.drop(df.columns[[1, 2, 5, 7, 8, 11, 13, 15, 18, 21]], axis=1, inplace=True)
    df.dropna(axis=1, how="any", inplace=True)
    # print(df)
    df.to_csv(path, compression="gzip", index=False, header=False)


if __name__ == "__main__":
    path = "./data/"
    path2test = "/test.txt.gz"
    path2train = "/train.txt.gz"

    for path2dataset in retrieve_data(path):
        # if path2dataset == "./data/cinlp-twitter":
        #     cinlp(path2dataset + path2test)
        #     cinlp(path2dataset + path2train)
        if path2dataset == "./data/music-genre-classification":
            music(path2dataset + path2test)
            music(path2dataset + path2train)
