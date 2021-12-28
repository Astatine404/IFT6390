import numpy as np

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        return np.mean(iris[:, :-1], axis=0)

    def covariance_matrix(self, iris):
        return np.cov(iris[:, :-1], rowvar=False)

    def feature_means_class_1(self, iris):
        return np.mean(iris[iris[:, 4] == 1][:, :-1], axis=0)

    def covariance_matrix_class_1(self, iris):
        return np.cov(iris[iris[:, 4] == 1][:, :-1], rowvar=False)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def dist_func(self, x, Y, p=2):
        return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.n_classes = len(self.label_list)

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]
        counts = np.ones((num_test, self.n_classes))
        classes_pred = np.zeros(num_test)

        for (i, ex) in enumerate(test_data):
            dist_mat = self.dist_func(ex, self.train_inputs)
            neighbors_ind = np.array([j for j in range(len(dist_mat)) if dist_mat[j] < self.h])
            if len(neighbors_ind) == 0:
              classes_pred[i] = draw_rand_label(ex, self.label_list)
              continue

            for ind in neighbors_ind:
              counts[i, self.train_labels[ind].astype('int32') - 1] += 1
            
            classes_pred[i] = np.argmax(counts[i, :]) + 1

        return classes_pred.astype('int32')


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def dist_func(self, x, y, p=2):
        return (np.sum(np.abs(x - y) ** p, axis=1)) ** (1.0 / p)

    def rbf(self, x, Y):
        D = self.train_inputs.shape[1]
        return ((2 * np.pi) ** (D/2) * (self.sigma ** D)) ** (-1) * np.exp(-(self.dist_func(x, Y) ** 2) / (2 * (self.sigma ** 2)))

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.n_classes = len(self.label_list)

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]
        counts = np.ones((num_test, self.n_classes))
        classes_pred = np.zeros(num_test)

        for (i, ex) in enumerate(test_data):
            weights = self.rbf(ex, self.train_inputs)
            for (j, train_ex) in enumerate(self.train_inputs):
              counts[i, self.train_labels[j].astype('int32') - 1] += weights[j]
            
            classes_pred[i] = np.argmax(counts[i, :]) + 1

        return classes_pred.astype('int32')


def split_dataset(iris):
    train_data = iris[[i for i in range(iris.shape[0]) if i % 5 <= 2], :]
    val_data = iris[[i for i in range(iris.shape[0]) if i % 5 == 3], :]
    test_data = iris[[i for i in range(iris.shape[0]) if i % 5 == 4], :]
    return (train_data, val_data, test_data)

class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        model = HardParzen(h)
        
        model.train(self.x_train, self.y_train)
        
        pred = model.compute_predictions(self.x_val)

        error_flags = [1 if pred[i] != self.y_val[i] else 0 for i in range(len(pred))]
        error = np.sum(error_flags).astype('float64') / len(error_flags)
        return error

    def soft_parzen(self, sigma):
        model = SoftRBFParzen(sigma)
        
        model.train(self.x_train, self.y_train)
        
        pred = model.compute_predictions(self.x_val)

        error_flags = [1 if pred[i] != self.y_val[i] else 0 for i in range(len(pred))]
        error = np.sum(error_flags).astype('float64') / len(error_flags)
        return error


def get_test_errors(iris):
    errors = []
    train_data, val_data, test_data = split_dataset(iris)
    h_star = 0.5
    sigma_star = 0.2
    model = ErrorRate(train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1])
    errors.append(model.hard_parzen(h_star))
    errors.append(model.soft_parzen(sigma_star))
    return errors


def random_projections(X, A):
    return np.matmul(X, A) / (2 ** (1/2))
