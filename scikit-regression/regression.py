import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge


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


def fit_predict_evaluate(regr, train, train_lab, test, test_lab):
    regr.fit(train, train_lab)
    preds = regr.predict(test)
    print("MSE:", mean_squared_error(test_lab, preds, multioutput='raw_values'))


def apply_models(train, train_lab, test, test_lab):
    print("=== Linear Regression ===")
    lr = LinearRegression()
    fit_predict_evaluate(lr, train, train_lab, test, test_lab)

    print("=== Decision Tree Regressor ===")
    dtr = DecisionTreeRegressor()
    fit_predict_evaluate(dtr, train, train_lab, test, test_lab)

    print("=== Support Vector Regressor ===")
    svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
    fit_predict_evaluate(svr, train, train_lab, test, test_lab)

    print("=== Ridge Regression ===")
    rr = Ridge()
    fit_predict_evaluate(rr, train, train_lab, test, test_lab)

    print("=== Beyesian Ridge Regression ===")
    br = BayesianRidge()
    fit_predict_evaluate(br, train, train_lab, test, test_lab)

    print("=== KNN Regressor ===")
    knnr = KNeighborsRegressor(n_neighbors=2)
    fit_predict_evaluate(knnr, train, train_lab, test, test_lab)


if __name__ == "__main__":
    train_1, train_lb_1 = read_data("artificial_2x_train.tsv")
    test_1, test_lb_1 = read_data("artificial_2x_test.tsv")
    apply_models(train_1, train_lb_1, test_1, test_lb_1)
    print('=' * 50)
    train_2, train_lb_2 = read_data("pragueestateprices_train.tsv")
    test_2, test_lb_2 = read_data("pragueestateprices_test.tsv")
    train_2, test_2 = data_manip(train_2, test_2)
    apply_models(train_2, train_lb_2, test_2, test_lb_2)


