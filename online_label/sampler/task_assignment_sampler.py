import numpy as np
from collections import defaultdict

from .sampler import Sampler

import logging
logger = logging.getLogger(__name__)


class GreedyTaskAssignmentSampler(Sampler):
    def __init__(self, config, annotation_holder, workers, optimizer):
        Sampler.__init__(self, config, annotation_holder, workers)
        self.optimizer = optimizer

    def stop(self, risk, **kwargs):
        confident = risk < self.risk_thres
        exceed_max = self.n_annotation >= self.max_annotation_per_example

        valid_worker_id = self._get_valid_workers()
        if len(valid_worker_id) == 0:
            logger.info(f'All worker annotate reach maximum number of annotation {self.max_annotation_per_worker}')

        return np.all(np.logical_or(confident, exceed_max)) or len(valid_worker_id) == 0

    def _get_valid_workers(self):
        valid_worker_id = [k for k, v in self.worker_n_annotation_counter.items() if v + self.config.online.hit_size <= self.max_annotation_per_worker]
        return valid_worker_id

    def sample(self, hit_size, n_hit, risk, y_posterior, **kwargs):
        
        assert self.n_workers >= n_hit, 'You are launching more HITs than the size of worker pool'

        valid_worker_id = self._get_valid_workers()
        if len(valid_worker_id) < self.n_workers:
            logger.debug(f'{self.n_workers - len(valid_worker_id)} already have maximum annotation {self.max_annotation_per_worker}')

        if len(valid_worker_id) < n_hit:
            logger.info(f'Reduce the `n_hit_per_step` to {len(valid_worker_id)} since we only have {len(valid_worker_id)} available now')
            n_hit = len(valid_worker_id)


        data_idx = []
        worker_id = []

        for i in range(n_hit):
            unconfident_mask = self.lower_risk_thres_if_necessary(risk, hit_size)
            unconfident = np.where(unconfident_mask)[0]
            _data_idx = self.npr.choice(unconfident, hit_size, replace=False)
            _y_posterior = y_posterior[_data_idx]

            #valid_worker_confidence = defaultdict()
            _valid_worker_ids = []
            _valid_worker_confidence = []
            for k in valid_worker_id:
                confidence = self.optimizer.workers_estimated_m[k].batch_confidence(_y_posterior)
                _valid_worker_ids.append(k)
                _valid_worker_confidence.append(confidence)


            _valid_worker_confidence = np.array(_valid_worker_confidence) + \
                                        self.npr.rand(len(_valid_worker_confidence)) * 1e-8
            idx = np.argmax(_valid_worker_confidence)
            _worker_id = _valid_worker_ids[idx]
            valid_worker_id.remove(_worker_id)

            data_idx.extend(_data_idx)
            worker_id.extend(np.repeat([_worker_id], hit_size))

            self.n_annotation[_data_idx] += 1
            self.worker_n_annotation_counter[_worker_id] += hit_size


        return data_idx, worker_id
