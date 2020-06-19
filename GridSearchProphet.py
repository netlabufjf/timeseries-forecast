import itertools
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class GridSearch():
    """
    A GridSearch implementation for Prophet model hyperparameter tunning.
    """

    def __init__(self, estimator, params, scoring, cv=3):
        self.estimator = estimator
        self.params = self.__permute_dict(params) # permute all the params possibilities
        self.cv = cv
        self.scoring = self.__score_method(scoring)

        self.best_params_ = {}
        self.best_score_ = 0.0

    def __permute_dict(self, dictionary):
        """
        Permute a dictionary with lists to a list of dictionaries.
        """
        keys, values = zip(*dictionary.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return permutations_dicts

    def __score_method(self, scoring):
        """
        Choose the score method by string.
        """
        if scoring == 'mae':
            return mean_absolute_error
        elif scoring == 'mse':
            return mean_squared_error
        elif scoring == 'r2':
            return r2_score

    def __cross_validation(self, estimator, data, params):
        """
        Implements a time-series nested cross validation.
        """
        
        set_size = len(data) // self.cv
        actual_set_size = 0
        scores = []

        for _ in range(self.cv):
            actual_set_size += set_size # update the set size

            # select the train and test set size by the ratio of 80% and 20%
            train_size, test_size = (int(actual_set_size * 0.8), int(actual_set_size * 0.2))
            data_train, data_test = (data[:train_size], data[train_size:train_size+test_size])

            # fit and predict the data
            model = estimator(**params)
            model.fit(data_train)
            prediction = model.predict(data_test[['ds']])

            scores.append(self.scoring(data_test.y, prediction.yhat))

        # score mean of the all k-folds
        mean_score = sum(scores)/len(scores)

        return mean_score

    def fit(self, data):
        """
        Start the grid search over the parameters possibilities.
        """
        scores = []
        for params in tqdm(self.params):
            scores.append(self.__cross_validation(self.estimator, data, params))

        self.best_score_ = min(scores)
        best_index = scores.index(min(scores))
        self.best_params_ = self.params[best_index]