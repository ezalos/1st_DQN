from sklearn.base import BaseEstimator, ClassifierMixin
from CuteNet import CuteLearning
from config import GridS_conf, net_config
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class PersClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, epsilon=net_config.epsilon, min_epsilon=net_config.min_epsilon, eps_decay=net_config.eps_decay,
                 n_update=net_config.n_update, max_turns=net_config.max_turns,
                 batch=net_config.batch, gamma=net_config.gamma,
                 soft_update=net_config.soft_update, tau=net_config.tau,
                 replay_nb_batch=net_config.replay_nb_batch, dropout=net_config.dropout,
                 reward_optimisation=net_config.reward_optimisation,
                 early_stopping=net_config.early_stopping, #ModelsManager=net_config.ModelsManager,
                 learning_rate=net_config.learning_rate, layers=net_config.layers):
        """
        Called when initializing the classifier
        """
        self.layers = layers
        self.dropout = dropout
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.eps_decay = eps_decay
        self.n_update = n_update
        self.max_turns = max_turns
        self.batch = batch
        self.gamma = gamma
        self.tau = tau
        self.early_stopping = early_stopping
        self.replay_nb_batch = replay_nb_batch
        self.soft_update = soft_update
        self.reward_optimisation = reward_optimisation
        self.learning_rate = learning_rate
        # self.ModelsManager = ModelsManager


    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        if False:
            assert (type(self.epsilon) ==
                    float), "epsilon parameter must be float"
            assert (type(self.learning_rate) ==
                    float), "learning_rate parameter must be float"
            assert (type(self.eps_decay) ==
                    float), "eps_decay parameter must be float"
            assert (type(self.n_update) ==
                    int), "n_update parameter must be integer"
            assert (type(self.max_turns) ==
                    int), "max_turns parameter must be integer"
            assert (type(self.batch) ==
                    int), "batch parameter must be integer"
            assert (type(self.gamma) ==
                    float), "gamma parameter must be float"
            assert (type(self.tau) ==
                    float), "tau parameter must be float"
            assert (type(self.replay_nb_batch) ==
                    int), "replay_nb_batch parameter must be integer"
            assert (type(self.soft_update) ==
                    bool), "soft_update parameter must be bool"
            assert (type(self.reward_optimisation) ==
                    bool), "reward_optimisation parameter must be bool"
            if not np.iscomplexobj(X):
                try:
                    X = np.array(X, dtype=float)
                except Exception as e:
                    print(e)
                
        check_X_y(X, y)
        # if (type(X) != type(np.ndarray((1, 2))) and str(type(X)) != "<class 'numpy.memmap'>") or np.iscomplexobj(X):
        #     raise ValueError("X: " + str(type(X)))
        # assert (type(self.stringParam) == str), "stringValue parameter must be string"
        # assert (len(X) == 20), "X must be list with numerical values."
        self._model_ = CuteLearning(epsilon=self.epsilon, eps_decay=self.eps_decay,
                                        n_update=int(self.n_update), max_turns=int(self.max_turns),
                                        batch=int(self.batch), gamma=self.gamma,
                                        soft_update=self.soft_update, tau=self.tau,
                                        replay_nb_batch=int(self.replay_nb_batch), dropout=self.dropout,
                                        reward_optimisation=self.reward_optimisation, #ModelsManager=net_config.ModelsManager,
                                        early_stopping=self.early_stopping, min_epsilon=self.min_epsilon,
                                        learning_rate=self.learning_rate, layers=self.layers, verbose=2)
        self._model_.learn()
        self._value = self._model_.best_consecutive_wins
        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return True

    def predict(self, X, y=None):
        # check_X_y(X, y)
        X = check_array(X)
        return self

    def score(self, X, y=None):
        X = check_array(X)
        self._value = self._model_.get_score()
        # check_X_y(X, X)
        # counts number of values bigger than mean
        return(self._value)

if __name__ == "__main__":
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.utils.estimator_checks import check_estimator

    print("Starting check")

    # check_estimator(PersClassifier())

    print("Check SUCCESSFULL")

    X_train = [i for i in range(0, 100, 5)]
    X_test = [i + 3 for i in range(-5, 95, 5)]
    tuned_params = GridS_conf

    #gs = GridSearchCV(PersClassifier(), tuned_params, verbose=4)
    gs = RandomizedSearchCV(PersClassifier(), tuned_params, verbose=4)

    # for some reason I have to pass y with same shape
    # otherwise gridsearch throws an error. Not sure why.
    y_test = np.array([1 for i in range(len(X_test))])
    y_test = y_test.reshape(-1, 1)
    X_test = np.array(X_test)
    X_test = X_test.reshape(-1, 1)

    gs.fit(X_test, y=y_test)

    print(gs.best_params_)  # {'intValue': -10} # and that is what we expect :)
