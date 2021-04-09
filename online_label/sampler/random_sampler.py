import numpy as np

from .sampler import Sampler

import logging
logger = logging.getLogger(__name__)


class RandomSampler(Sampler):

    def __init__(self, config, annotation_holder, workers, **kwargs):
        Sampler.__init__(self, config, annotation_holder, workers)

    def stop(self, risk, **kwargs):
        confident = risk < self.risk_thres
        exceed_max = self.n_annotation >= self.max_annotation_per_example
        logger.debug(f'Number of unconfident examples: {sum(confident)}')
        logger.debug(f'Number of examples exceeds budget: {sum(exceed_max)}')
        return np.all(np.logical_or(confident, exceed_max))

    def sample(self, hit_size, n_hit, risk, **kwargs):
        
        assert self.n_workers >= n_hit, 'You are launching more HITs than the size of worker pool'
        
        sampled_workers = self.npr.choice([w.id for w in self.workers.values()], n_hit, replace=n_hit>self.n_workers)
        data_idx = []
        worker_id = []
        for w in sampled_workers:
            # Check every time to avoid over sampling some examples
            unconfident_mask = self.lower_risk_thres_if_necessary(risk, hit_size)
            unconfident = np.where(unconfident_mask)[0]

            _data_idx = self.npr.choice(unconfident, hit_size, replace=False)
            _worker_id = np.repeat([w], hit_size)
            self.n_annotation[_data_idx] += 1
            data_idx.extend(_data_idx)
            worker_id.extend(_worker_id)

        return data_idx, worker_id
