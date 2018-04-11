import math
import pandas as pd
from sklearn import preprocessing


def read_data(f):
    print('Processing document', f)
    dataset = pd.read_csv(f, sep="\t", header=None)
    df = pd.DataFrame(dataset)
    if f.startswith("pragueestateprices_"):
        label = df.iloc[:, -2]
        df = df.iloc[:, :-2]
    else:
        label = df.iloc[:, -1]
        df = df.iloc[:, :-1]
    return df, label


def data_manip(train, test):
    whole = pd.concat([train, test])
    whole = pd.get_dummies(whole, columns=[0, 1, 2, 3, 5, 6, 7])
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(whole.values)
    df = pd.DataFrame(x_scaled)
    new_train = df.iloc[:len(train.index), :]
    new_test = df.iloc[len(train.index):, :]
    return new_train, new_test


def accuracy(true, predicted):
    correct = 0
    for i in range(len(true)):
        if true[i] - predicted[i] < 1.0 or true[i] - predicted[i] < -1.0:
            correct += 1
    print ("Accuracy:", correct / float(len(true)) * 100.0)


def mse(true, predicted):
    sum_error = 0.0
    for i in range(len(true)):
        error = predicted[i] - true[i]
        sum_error += (error ** 2)
    mean_sq_error = float(sum_error) / len(true)
    print("MSE:", mean_sq_error, "RMSE:", math.sqrt(mean_sq_error))


class SimpleLinearRegression:
    def train(self, train_data, train_labels):
        train_data["label"] = train_labels
        mean = train_data.mean()
        variance = train_data.var()
        covariance = train_data.cov()
        coef_1 = covariance.ix[0, 1] / variance.ix[0, 0]
        coef_0 = mean.ix[1, 0] - coef_1 * mean.ix[0, 0]
        return [coef_0, coef_1]

    def predict(self, test, coeffs):
        test["labels"] = test.apply(lambda y: coeffs[0] + coeffs[1] * y)
        return test["labels"]

class MultiLinearregression:
    def predict(self, row, coeffs):
        y = coeffs[0]
        for i in range(0, len(row) - 1):
            y += coeffs[i + 1] * row.ix[i, ]
        return y

    def sgd(self, train, labels, l_rate, n_epoch):
        coeffs = [0.0 for i in range(len(train.columns))]
        for epoch in range(n_epoch):
            sum_error = 0
            for id, row in train.iterrows():
                y = self.predict(row, coeffs)
                error = y - labels.ix[id, ]
                sum_error += error ** 2
                coeffs[0] = coeffs[0] - l_rate * error
                for i in range(len(row) - 1):
                    coeffs[i + 1] = coeffs[i + 1] - l_rate * error * row[i]
        return coeffs

    def predict_data(self, test, coeffs):
        preds = [self.predict(row, coeffs) for id, row in test.iterrows()]
        return preds


def simple_linear_regression(train_data, train_labels, test_data, test_labels):
    slr = SimpleLinearRegression()
    coeffs = slr.train(train_data, train_labels)
    preds = slr.predict(test_data, coeffs)
    accuracy(test_labels, preds)
    mse(test_labels, preds)

def multi_linear_regression(train_data, train_labels, test_data, test_labels, l_rate, n_epochs):
    mlr = MultiLinearregression()
    coeffs = mlr.sgd(train_data, train_labels, l_rate, n_epochs)
    preds = mlr.predict_data(test_data, coeffs)
    mse(test_labels, preds)


if __name__ == "__main__":
    train_1, train_lb_1 = read_data("artificial_2x_train.tsv")
    test_1, test_lb_1 = read_data("artificial_2x_test.tsv")
    simple_linear_regression(train_1, train_lb_1, test_1, test_lb_1)
    multi_linear_regression(train_1, train_lb_1, test_1, test_lb_1, 0.0003, 100)
    print('=' * 50)
    train_2, train_lb_2 = read_data("pragueestateprices_train.tsv")
    test_2, test_lb_2 = read_data("pragueestateprices_test.tsv")
    train_2, test_2 = data_manip(train_2, test_2)
    multi_linear_regression(train_2, train_lb_2, test_2, test_lb_2, 0.0003, 100)
