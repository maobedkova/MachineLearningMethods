from common_functions import read_data, accuracy, data_manip
import math


class NaiveBayes:
    def calculate_prob(self, x, mean, stdev):
        if stdev == 0.0:
            stdev += 0.0001
        elif stdev == 1.0:
            stdev -= 0.0001
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculate_probs(self, row, mean, variance):
        probs = {}
        for label, col in mean.iterrows():
            probs[label] = 1
            for i in range(0, len(mean.iloc[label, ])):
                var_value = variance.iloc[label, i]
                mean_value = mean.iloc[label, i]
                probs[label] *= self.calculate_prob(row[i], mean_value, var_value)
        return probs

    def find_best(self, row, mean, variance):
        probs = self.calculate_probs(row, mean, variance)
        best_label = None
        best_prob = -1
        for label in probs:
            if best_label is None or probs[label] > best_prob:
                best_prob = probs[label]
                best_label = label
        return best_label

    def predict(self, test_data, mean, variance):
        preds = []
        for index, row in test_data.iterrows():
            pred = self.find_best(row, mean, variance)
            preds.append(pred)
        return preds


def call_naive_bayes(train_data, train_label, test_data, test_label):
    nb = NaiveBayes()
    train_data, test_data = data_manip(train_data, test_data)
    train_data["label"] = train_label
    mean = train_data.groupby('label').mean()
    variance = train_data.groupby('label').var()
    preds = nb.predict(test_data, mean, variance)
    accuracy(test_label, preds)


if __name__ == "__main__":
    train_1, train_lb_1 = read_data("artificial_separable_train.csv")
    test_1, test_lb_1 = read_data("artificial_separable_test.csv")
    call_naive_bayes(train_1, train_lb_1, test_1, test_lb_1)
    print('=' * 50)
    train_2, train_lb_2 = read_data("artificial_with_noise_train.csv")
    test_2, test_lb_2 = read_data("artificial_with_noise_test.csv")
    call_naive_bayes(train_2, train_lb_2, test_2, test_lb_2)
    print('=' * 50)
    train_3, train_lb_3 = read_data("adult.data.csv")
    test_3, test_lb_3 = read_data("adult.test.csv")
    call_naive_bayes(train_3, train_lb_3, test_3, test_lb_3)
