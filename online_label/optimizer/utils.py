import json
import numpy as np

import logging
logger = logging.getLogger(__name__)


class DirichletDist():
    def __init__(self, config, **kwargs):

        n_classes = config.n_classes
        prior_alpha = np.zeros((n_classes, n_classes))
        if config.worker.prior.type == 'predefined_homegeneous' or len(kwargs) == 0:
            logger.debug('Use Predefined worker prior')
            neg_density = config.worker.prior.predefined_params.neg_count / n_classes
            prior_alpha += config.worker.prior.predefined_params.neg_count / n_classes
            prior_alpha += np.eye(n_classes) * (config.worker.prior.predefined_params.pos_count - neg_density)
        else:
            logger.debug('Use worker prior: {}'.format(config.worker.prior.type))
            if config.worker.prior.type == 'homegeneous_workers':
                # Diagonal terms are initialized by the correctness of all the workers
                # Off-diagonal terms are initialized by the incorrecness of all the workers
                global_cm = kwargs['global_cm']
                off_diagonal_density = (global_cm.sum(1).mean() - global_cm.diagonal().mean()) / (n_classes-1)
                prior_alpha += off_diagonal_density
                np.fill_diagonal(prior_alpha, np.ones(n_classes) * global_cm.diagonal().mean())
            elif config.worker.prior.type == 'homegeneous_workers_diagonal_perfect':
                # Diagonal terms are initialized by the per-class correctness of all the workers
                # Off-diagonal terms are initialized by the per-class incorrecness of all the workers
                global_cm = kwargs['global_cm']
                off_diagonal_density = global_cm.sum(1) - global_cm.diagonal()
                off_diagonal_density = off_diagonal_density.reshape(-1, 1) / (n_classes - 1)
                prior_alpha += off_diagonal_density
                np.fill_diagonal(prior_alpha, global_cm.diagonal())
            elif config.worker.prior.type == 'homegeneous_workers_perfect':
                # Diagonal terms are initialized by the per-class correctness of all the workers
                # Off-diagonal terms are initialized by the per-class-per-prediction incorrecness of all the workers
                global_cm = kwargs['global_cm']
                prior_alpha = global_cm
            elif config.worker.prior.type == 'heterogeneous_workers':
                # Diagonal terms are initialized by the correctness of the workers
                # Off-diagonal terms are initialized by the incorrecness of the workers
                worker_cm = kwargs['worker_cm']
                off_diagonal_density = (worker_cm.sum(1).mean() - worker_cm.diagonal().mean()) / (n_classes-1)
                prior_alpha += off_diagonal_density
                np.fill_diagonal(prior_alpha, np.ones(n_classes) * worker_cm.diagonal().mean())
            elif config.worker.prior.type == 'heterogeneous_workers_diagonal_perfect':
                # Diagonal terms are initialized by the per-class correctness of the workers
                # Off-diagonal terms are initialized by the per-class incorrecness of the workers
                worker_cm = kwargs['worker_cm']
                off_diagonal_density = worker_cm.sum(1) - worker_cm.diagonal()
                off_diagonal_density = off_diagonal_density.reshape(-1, 1) / (n_classes - 1)
                prior_alpha += off_diagonal_density
                np.fill_diagonal(prior_alpha, worker_cm.diagonal())
            elif config.worker.prior.type == 'heterogeneous_workers_perfect':
                # Diagonal terms are initialized by the per-class correctness of all the workers
                # Off-diagonal terms are initialized by the per-class-per-prediction incorrecness of all the workers
                worker_cm = kwargs['worker_cm']
                prior_alpha = worker_cm
            else:
                raise ValueError

        self.prior_alpha = prior_alpha * config.worker.prior.strength
        self.n_classes = n_classes
        self.init_posterior()
        
    def init_posterior(self):
        self.posterior_alpha = np.zeros((self.n_classes, self.n_classes))

    def save_state(self):
        return json.dumps({'posterior_alpha': self.posterior_alpha.tolist(), 
                           'prior_alpha': self.prior_alpha.tolist()})

    def load_state(self, state):
        state = json.loads(state)
        self.posterior_alpha = state['posterior_alpha']

    def batch_confidence(self, belief):
        # belief: (n_samples, n_classes)
        if len(belief.shape) == 1:
            belief = belief.reshape(1, -1)

        cm = self.prior_alpha + self.posterior_alpha
        cm = cm / cm.sum(1, keepdims=True)
        c = (belief * cm.diagonal().reshape(1, -1)).sum(1).mean()
        return c

    def p_z_given_y(self, z):
        cm = self.prior_alpha + self.posterior_alpha
        cm = cm / cm.sum(1, keepdims=True)
        return cm[:, z]

    def update(self, eval_list):
        posterior_alpha = np.zeros((self.n_classes, self.n_classes))
        for i in eval_list:
            prob, z = i
            pred = np.argmax(prob + np.random.rand(self.n_classes)*1e-8)    # Add noise to break tie
            posterior_alpha[pred, z] += 1.
            
        self.posterior_alpha = posterior_alpha
