from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from HyperParamsOpti import PersClassifier
import sklearn.datasets
from config import GridS_conf, net_config
import numpy as np
import random

data = sklearn.datasets.load_digits()
X = data["data"]
y = data["target"]


paramgrid = GridS_conf

random.seed(net_config.seed)

cv = EvolutionaryAlgorithmSearchCV(estimator=PersClassifier(),
                                   params=paramgrid,
                                   scoring=None,
                                   cv=2,
                                   verbose=10,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=10,
                                   n_jobs=1)
ret = cv.fit(X, y)
print(ret)