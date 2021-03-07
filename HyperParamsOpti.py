from sklearn.base import BaseEstimator, ClassifierMixin
from CuteNet import CuteLearning
from config import GridS_conf, net_config
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class PersClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, gsc_epsilon=net_config.epsilon, gsc_min_epsilon=net_config.min_epsilon, gsc_eps_decay=net_config.eps_decay,
                 gsc_n_update=net_config.n_update, gsc_max_turns=net_config.max_turns,
                 gsc_batch=net_config.batch, gsc_gamma=net_config.gamma,
                 gsc_soft_update=net_config.soft_update, gsc_tau=net_config.tau,
                 gsc_replay_nb_batch=net_config.replay_nb_batch, gsc_dropout=net_config.dropout,
                 gsc_reward_optimisation=net_config.reward_optimisation,
                 gsc_early_stopping=net_config.early_stopping, #gsc_ModelsManager=net_config.ModelsManager,
                 gsc_learning_rate=net_config.learning_rate, gsc_layers=net_config.layers):
        """
        Called when initializing the classifier
        """
        self.gsc_layers = gsc_layers
        self.gsc_dropout = gsc_dropout
        self.gsc_epsilon = gsc_epsilon
        self.gsc_min_epsilon = gsc_min_epsilon
        self.gsc_eps_decay = gsc_eps_decay
        self.gsc_n_update = gsc_n_update
        self.gsc_max_turns = gsc_max_turns
        self.gsc_batch = gsc_batch
        self.gsc_gamma = gsc_gamma
        self.gsc_tau = gsc_tau
        self.gsc_early_stopping = gsc_early_stopping
        self.gsc_replay_nb_batch = gsc_replay_nb_batch
        self.gsc_soft_update = gsc_soft_update
        self.gsc_reward_optimisation = gsc_reward_optimisation
        self.gsc_learning_rate = gsc_learning_rate
        # self.gsc_ModelsManager = gsc_ModelsManager


    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        if False:
            assert (type(self.gsc_epsilon) ==
                    float), "gsc_epsilon parameter must be float"
            assert (type(self.gsc_learning_rate) ==
                    float), "gsc_learning_rate parameter must be float"
            assert (type(self.gsc_eps_decay) ==
                    float), "gsc_eps_decay parameter must be float"
            assert (type(self.gsc_n_update) ==
                    int), "gsc_n_update parameter must be integer"
            assert (type(self.gsc_max_turns) ==
                    int), "gsc_max_turns parameter must be integer"
            assert (type(self.gsc_batch) ==
                    int), "gsc_batch parameter must be integer"
            assert (type(self.gsc_gamma) ==
                    float), "gsc_gamma parameter must be float"
            assert (type(self.gsc_tau) ==
                    float), "gsc_tau parameter must be float"
            assert (type(self.gsc_replay_nb_batch) ==
                    int), "gsc_replay_nb_batch parameter must be integer"
            assert (type(self.gsc_soft_update) ==
                    bool), "gsc_soft_update parameter must be bool"
            assert (type(self.gsc_reward_optimisation) ==
                    bool), "gsc_reward_optimisation parameter must be bool"
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
        self._gsc_model_ = CuteLearning(epsilon=self.gsc_epsilon, eps_decay=self.gsc_eps_decay,
                                        n_update=int(self.gsc_n_update), max_turns=int(self.gsc_max_turns),
                                        batch=int(self.gsc_batch), gamma=self.gsc_gamma,
                                        soft_update=self.gsc_soft_update, tau=self.gsc_tau,
                                        replay_nb_batch=int(self.gsc_replay_nb_batch), dropout=self.gsc_dropout,
                                        reward_optimisation=self.gsc_reward_optimisation, #ModelsManager=net_config.ModelsManager,
                                        early_stopping=self.gsc_early_stopping, min_epsilon=self.gsc_min_epsilon,
                                        learning_rate=self.gsc_learning_rate, layers=self.gsc_layers, verbose=2)
        self._gsc_model_.learn()
        self._value = self._gsc_model_.best_consecutive_wins
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
        self._value = self._gsc_model_.get_score()
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
