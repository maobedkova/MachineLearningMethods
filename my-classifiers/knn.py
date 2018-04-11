from common_functions import read_data, accuracy, data_manip
import math
import operator

class KNN:
    def eucl_dist(self, inst_1, inst_2, length):
        distance = 0
        for x in range(length):
            distance += pow((inst_1[x] - inst_2[x]), 2)
        return math.sqrt(distance)

    def get_neighbours(self, train, test_row, k):
        distances = []
        length = len(test_row) - 1
        for index, train_row in train.iterrows():
            dist = self.eucl_dist(test_row, train_row, length)
            distances.append((train_row, dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def predict(self, neighbors, labels):
        votes = [0] * len(labels)
        for x in range(len(neighbors)):
            for l in labels:
                vote = neighbors[x][-1]
                if vote == l:
                    votes[l] += 1
        return votes.index(max(votes))


def call_knn(train_data, train_label, test_data, test_label, k):
    knn = KNN()
    train_data, test_data = data_manip(train_data, test_data)
    train_data["label"] = train_label
    preds = []
    for index, test_row in test_data.iterrows():
        neighbors = knn.get_neighbours(train_data.sample(frac=1).reset_index(drop=True).iloc[:50, :], test_row, k)
        pred = knn.predict(neighbors, set(train_data["label"]))
        preds.append(pred)
    accuracy(test_label, preds)


if __name__ == "__main__":
    train_1, train_lb_1 = read_data("artificial_separable_train.csv")
    test_1, test_lb_1 = read_data("artificial_separable_test.csv")
    call_knn(train_1, train_lb_1, test_1, test_lb_1, 2)
    print('=' * 50)
    train_2, train_lb_2 = read_data("artificial_with_noise_train.csv")
    test_2, test_lb_2 = read_data("artificial_with_noise_test.csv")
    call_knn(train_2, train_lb_2, test_2, test_lb_2, 2)
    print('=' * 50)
    train_3, train_lb_3 = read_data("adult.data.csv")
    test_3, test_lb_3 = read_data("adult.test.csv")
    call_knn(train_3, train_lb_3, test_3, test_lb_3, 2)
