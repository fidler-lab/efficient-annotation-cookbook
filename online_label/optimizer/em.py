import json
import numpy as np
from collections import OrderedDict, defaultdict
from . import Optimizer
from .utils import DirichletDist

import logging
logger = logging.getLogger(__name__)


class EMOptimizer(Optimizer):
    
    def __init__(self, config, imagenet_data, workers):
        Optimizer.__init__(self, config)
        self.prev_learner_prob = None
        self.imagenet_data = imagenet_data
        self.__init_workers_estimation(workers)
        
    def __init_workers_estimation(self, workers):
        self.workers_estimated_m = {p: DirichletDist(self.config, 
                                                    global_cm=w.global_cm, 
                                                    worker_cm=w.m) for p, w in workers.items()}
        
    def reset_learner(self):
        logger.info('Reset learners')
        self.prev_learner_prob = None

    def save_state(self):
        state = {}
        if self.prev_learner_prob is not None :
            state.update({'prev_learner_prob': self.prev_learner_prob.tolist()})
        for k, v in self.workers_estimated_m.items():
            state.update({f'#workers-{k}': v.save_state()})
        return json.dumps(state)
            
    def load_state(self, state, workers):
        if state is not None:
            state = json.loads(state)
            self.prev_learner_prob = np.array(state['prev_learner_prob'])
            self.__init_workers_estimation(workers)

    def step(self, annotation_holder, aggregator, learner, imagenet_data):

        self.reset()

        if self.config.worker.known or annotation_holder.n_annotation == 0:
            y_posterior = self.infer_y_posterior(aggregator, annotation_holder)
            self.converged = True
        else:
            self._init_w_as_prior()
            self._update_worker_likelihood(annotation_holder)

            prev_y_posterior = None
            prev_z_likelihood = None

            # Expectation Maximization
            for i in range(self.config.optimizer.max_em_steps):

                # E step
                y_posterior = self.infer_y_posterior(aggregator, 
                                                     annotation_holder, 
                                                     self.prev_learner_prob)

                # M step
                self.m_step(annotation_holder, y_posterior)

                z_likelihood = self.calculate_z_likelihood(annotation_holder, y_posterior)
                logger.debug(f'[{i}] Z Likelihood = {z_likelihood.mean()}')
                if prev_z_likelihood is not None:
                    if z_likelihood.mean() - prev_z_likelihood.mean() < 0:
                        logger.warning('[Warning] Likelihood is decreasing after E step')


                    if self.config.optimizer.criterion == 'hard':
                        if i > 0 and prev_y_posterior is not None and np.all(prev_y_posterior.argmax(1) == y_posterior.argmax(1)):
                            self.converged = True
                            logger.info(f'[Hard constraint] Stop at step: {i}')
                            break
                    elif self.config.optimizer.criterion == 'soft':
                        diff_z_likelihood = abs(prev_z_likelihood.mean() - z_likelihood.mean())
                        if i > 0 and diff_z_likelihood < self.config.optimizer.likelihood_epsilon:
                            self.converged = True
                            logger.info(f'[Soft constraint] Stop at step: {i}')
                            break
                    else:
                        raise ValueError

                prev_z_likelihood = z_likelihood
                prev_y_posterior = y_posterior

            if not self.converged:
                logger.info(f'Use the EM steps at step: {self.config.optimizer.max_em_steps - 1}')


        # Learning
        y_posterior = self.infer_y_posterior(aggregator, 
                                             annotation_holder, 
                                             self.prev_learner_prob)
        learner_prob = self.fit_machine_learner(annotation_holder, imagenet_data, learner, y_posterior)

        # Use new learner to infer y posterior again
        y_posterior = self.infer_y_posterior(aggregator, annotation_holder, learner_prob)


        self.prev_learner_prob = learner_prob
        return {
            'y_posterior_risk': aggregator.compute_risk(y_posterior),
            'learner_prob': learner_prob, 
            'y_posterior': y_posterior
        }

    def infer_y_posterior(self, aggregator, annotation_holder, learner_prob=None):
        y_posterior = aggregator.aggregate(annotation_holder, 
                                           learner_prob=learner_prob)
        return y_posterior

    def fit_machine_learner(self, annotation_holder, imagenet_data, learner, y_posterior):

        logger.debug('Fitting Learner')
        feature_dict = imagenet_data.get_features_dict()
        n_annotation = np.array([len(v) for v in annotation_holder.annotation.values()])

        learner_prob = learner.fit_and_predict(feature_dict, 
                                        np.array(imagenet_data.get_y(imagenet_data.p_image_path)), 
                                        y_posterior, n_annotation, 
                                        np.array(imagenet_data.get_y(imagenet_data.image_path)))


        logger.debug('Finish fitting')
        return learner_prob

    def calculate_z_likelihood(self, annotation_holder, belief):
        y_pred = belief.argmax(1)
        likelihood = []

        image_path = list(annotation_holder.annotation.keys())
        for k, y in zip(image_path, y_pred):
            for a_i in annotation_holder.annotation[k]:
                _, z, j, _ = a_i
                likelihood.append(self.workers_estimated_m[j].p_z_given_y(z)[y])
        return np.array(likelihood)

    def m_step(self, annotation_holder, belief):

        image_path = list(annotation_holder.annotation.keys())
        for id, _ in self.workers_estimated_m.items():
            # Collect annotation from a worker 
            w_eval = []

            for i, k in enumerate(image_path):
                for a_k in annotation_holder.annotation[k]:
                    _, z, j, _ = a_k
                    if str(j) == str(id):
                        w_eval.append((belief[i], z))

            # Update the worker posterior
            before = self.workers_estimated_m[id].posterior_alpha.sum()
            self.workers_estimated_m[id].update(w_eval)
            after = self.workers_estimated_m[id].posterior_alpha.sum()
            
        self._update_worker_likelihood(annotation_holder)

    def _update_worker_likelihood(self, annotation_holder):
        
        # Update `p_z_given_y` in the annotation holder
        annotation = annotation_holder.annotation
        n_data = annotation_holder.n_data
        new_annotation = OrderedDict()

        for k, v in annotation_holder.annotation.items():
            anno_k = []
            for a_k in v:
                y, z, j, _ = a_k
                p_z_given_y = self.workers_estimated_m[j].p_z_given_y(z)
                anno_k.append((y, z, j, np.array(p_z_given_y)))

            new_annotation.update({k: anno_k})

        annotation_holder.annotation = new_annotation
        return annotation_holder

    def _init_w_as_prior(self):
        for estimated_m in self.workers_estimated_m.values():
            estimated_m.init_posterior()
