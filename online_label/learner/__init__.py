import numpy as np
import torch

import logging
logger = logging.getLogger(__name__)


class Learner(object):
    def __init__(self, config):
        self.config = config
        self.npr = np.random.RandomState(config.seed)
        self.calibrate = config.learner.calibrate
        self.semi_supervised = config.learner.semi_supervised
        self.risk_thres = config.learner.risk_thres
        self.use_cuda = torch.cuda.is_available()
        self.early_stop_scope = config.learner.early_stop_scope
        self.prototype_as_val = config.learner.prototype_as_val

    def save_state(self):
        raise NotImplementedError
        
    def load_state(self):
        raise NotImplementedError

    def fit_and_predict(self, features, prototype_targets, belief, n_annotation, ground_truth):
        raise NotImplementedError


class DummyLearner(Learner):
    def __init__(self, config):
        Learner.__init__(self, config)
        logger.info('No learner is used')

    def save_state(self):
        pass
        
    def load_state(self, state):
        pass
        
    def fit_and_predict(self, features, prototype_targets, belief, n_annotation, ground_truth):
        return None


def get_learner_class(config):
    from .nn_learner import LinearNNLearner
    if config.learner.algo == 'dummy':
        return DummyLearner
    elif config.learner.algo == 'mlp':
        return LinearNNLearner
    else:
        raise ValueError
