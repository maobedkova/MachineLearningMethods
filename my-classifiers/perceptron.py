from common_functions import read_data, accuracy, data_manip


class Perceptron:
    def predict(self, row, weights):
        activation = weights[0]
        for i in range(len(row) - 1):
            activation += weights[i + 1] * row[i]
        return 1.0 if activation >= 0.0 else 0.0

    def train(self, train, l_rate, n_epochs):
        weights = [0.0 for i in range(len(train.iloc[0,]))]
        for epoch in range(n_epochs):
            for index, row in train.iterrows():
                prediction = self.predict(row, weights)
                error = row[-1] - prediction
                weights[0] = weights[0] + l_rate * error
                for i in range(len(row) - 1):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        return weights


def call_perceptron(train_data, train_label, test_data, test_label, l_rate, n_epochs):
    perc = Perceptron()
    train_data, test_data = data_manip(train_data, test_data)
    train_data["label"] = train_label
    weights = perc.train(train_data, l_rate, n_epochs)
    preds = []
    for index, row in test_data.iterrows():
        pred = perc.predict(row, weights)
        preds.append(pred)
    accuracy(test_label, preds)


if __name__ == "__main__":
    train_1, train_lb_1 = read_data("artificial_separable_train.csv")
    test_1, test_lb_1 = read_data("artificial_separable_test.csv")
    call_perceptron(train_1, train_lb_1, test_1, test_lb_1, 0.1, 5)
    print('='*50)
    train_2, train_lb_2 = read_data("artificial_with_noise_train.csv")
    test_2, test_lb_2 = read_data("artificial_with_noise_test.csv")
    call_perceptron(train_2, train_lb_2, test_2,test_lb_2, 0.01, 5)
    print('='*50)
    train_3, train_lb_3 = read_data("adult.data.csv")
    test_3, test_lb_3 = read_data("adult.test.csv")
    call_perceptron(train_3, train_lb_3, test_3, test_lb_3, 0.5, 3)
