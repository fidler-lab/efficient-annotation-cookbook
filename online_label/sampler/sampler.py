import numpy as np

from collections import defaultdict

import logging
logger = logging.getLogger(__name__)


class Sampler(object):
    def __init__(self, config, annotation_holder, workers):
        self.npr = np.random.RandomState(config.seed)
        self.config = config
        self.annotation_holder = annotation_holder
        self.workers = workers
        self.n_workers = len(workers)
        self.risk_thres = config.risk_thres
        self.max_annotation_per_example = config.sampler.max_annotation_per_example
        self.max_annotation_per_worker = config.sampler.max_annotation_per_worker
        self.__init_annotation_stats()

    def __init_annotation_stats(self):
        self.n_annotation = np.array([len(v) for _, v in self.annotation_holder.annotation.items()])

        worker_n_annotation_counter = defaultdict(lambda: 0)
        for w in self.workers.values():
            worker_n_annotation_counter[w.id] = 0
        for _, vs in self.annotation_holder.annotation.items():
            for v in vs:
                worker_n_annotation_counter[v[2]] += 1
        self.worker_n_annotation_counter = worker_n_annotation_counter
        
    def load_state(self, annotation_holder, workers):
        self.annotation_holder = annotation_holder
        self.workers = workers
        self.__init_annotation_stats()

    def lower_risk_thres_if_necessary(self, risk, hit_size):
        unconfident_mask = risk > self.risk_thres
        exceed_max = self.n_annotation >= self.max_annotation_per_example
        valid_sample = np.logical_and(~exceed_max, unconfident_mask)

        # Avoid sample the same image in the same HIT
        new_risk_thres = self.risk_thres
        while valid_sample.sum() < hit_size:
            new_risk_thres -= 0.01
            unconfident_mask = risk > new_risk_thres
            valid_sample = np.logical_and(~exceed_max, unconfident_mask)

        if new_risk_thres != self.risk_thres:
            logger.debug(f'Lower the risk threshold to {new_risk_thres}')

        return valid_sample

    def stop(self, risk, **kwargs):
        raise NotImplementedError

    def sample(self, hit_size, n_hit, risk, **kwargs):
        raise NotImplementedError
