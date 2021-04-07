import os
import json
import uuid
import numpy as np

from data import REPO_DIR, imagenet100

import logging
logger = logging.getLogger(__name__)


class Worker(object):

    def __init__(self, config, known, seed, **kwargs):

        self.id = str(uuid.uuid4())
        self.config = config
        self.seed = seed
        self.npr = np.random.RandomState(seed)
        self.n_classes = config.n_classes
        self.known = known

        self.m = self.sample_confusion_matrix() # (actual classes, predict classes)

    def save_state(self):
        return json.dumps(dict(id=self.id, m=self.m.tolist()))

    def load_state(self, state):
        state = json.loads(state)
        self.id = state['id']
        self.m = np.array(state['m'])
    
    def annotate(self, true_y, qmask=None):
        m = self.m
        valid_options = range(self.n_classes)

        prob = m[true_y]
        prob = np.clip(prob, 0., 1.) 

        z = self.npr.choice(range(self.n_classes), p=prob)
        if self.known:
            p_z_given_y = m[:, z]
        else:
            p_z_given_y = None

        return z, p_z_given_y
        
    def sample_confusion_matrix(self):
        raise NotImplementedError


class UniformWorker(Worker):
    '''w/o class correlation
    '''
    def sample_confusion_matrix(self):
        config = self.config

        m = np.zeros((self.n_classes, self.n_classes))
        m += np.eye(self.n_classes) * self.npr.normal(config.worker.reliability.mean, 
                                                    config.worker.reliability.std, 
                                                    size=self.n_classes)
        m = np.clip(m, 0, 1)
        for i in range(self.n_classes):
            noise = self.npr.rand(self.n_classes)
            noise /= noise.sum()
            noise += noise[i] / (self.n_classes -  1)
            noise *= (1 - m[i, i])
            m[i] += -1*(np.eye(self.n_classes)-1)[i] * noise

        reliability = np.diag(m).mean()
        logger.debug(f'Reliability: {reliability}')
        return m


class PerfectWorker(Worker):
    def sample_confusion_matrix(self):
        m = np.eye(self.n_classes)
        m = np.clip(m, 0, 1)
        return m


class RealWorker(Worker):
    wnids = None
    worker_cm_info = json.load(open(os.path.join(REPO_DIR, 'data/group_workers.json'), 'r'))
    groups_path = os.path.join(REPO_DIR, 'data/groups.txt')
    global_cm = []
    for _k, _v in worker_cm_info['group_workers'].items():
        _v = np.array(_v)
        global_cm.append(_v.sum(0))
    global_cm = sum(global_cm)

    def __init__(self, config, known, seed, **kwargs):

        keep_indices = np.array([imagenet100.index(i.lower()) for i in self.wnids])
        self.keep_indices = keep_indices
        self.global_cm = self.global_cm[keep_indices, :][:, keep_indices]
        Worker.__init__(self, config, known, seed, **kwargs)

    def sample_confusion_matrix(self):
        m = self._sample_confusion_matrix()

        with open(self.groups_path) as fp:
            groups = fp.read()
            groups = groups.split('\n\n')
            groups.pop(-1)
            groups = [np.array(i.split('\n')) for i in groups]


        def __which_group(i):
            for g_idx, g in enumerate(groups):
                if i in g:
                    return g_idx


        # Add uniform noise in off-diagonal terms
        noise_level = 0.03      # According to the amt stats
        for i, i_wnid in enumerate(self.wnids):
            i_group = __which_group(i_wnid)
            same_group_mask = np.zeros(self.config.n_classes).astype(np.bool)
            same_group_mask[i] = True
            for j, j_wnid in enumerate(self.wnids):
                if i != j and i_group == __which_group(j_wnid):
                    same_group_mask[j] = True
                

            if (~same_group_mask).sum() > 0:
                density_to_spread = m[i, same_group_mask].sum()
                m[i, same_group_mask] = m[i, same_group_mask] * (1 - noise_level)
                m[i, ~same_group_mask] += density_to_spread * (noise_level) / max(sum(~same_group_mask), 1e-8)

        return m


class StructuredNoiseWorker(RealWorker):
    def _sample_confusion_matrix(self):
        
        cm = []
        for _, v in self.worker_cm_info['group_workers'].items():
            v = np.array(v)
            global_v = v.sum(0)
            idx = self.npr.choice(range(len(v)), 1)
            cm.append(global_v + v[idx][0] * 10)

        cm = sum(cm)
        if self.config.n_data_distraction_per_class > 0:
            m = np.zeros((self.config.n_classes, self.config.n_classes))
            m[:self.config.n_classes-1, :self.config.n_classes-1] = cm[self.keep_indices, :][:, self.keep_indices]

            drop_indices_mask = np.ones(cm.shape[0]).astype(np.bool)
            drop_indices_mask[self.keep_indices] = False

            m[-1, :self.config.n_classes-1] = cm[drop_indices_mask, :][:, self.keep_indices].sum(0)  # Last row
            m[:self.config.n_classes-1, -1] = cm[self.keep_indices, :][:, drop_indices_mask].sum(1)  # Last column
            m[-1, -1] = cm[drop_indices_mask, :][:, drop_indices_mask].sum()
        else:
            m = cm[self.keep_indices, :][:, self.keep_indices]

        assert len(np.where(m.sum(1)==0)[0]) == 0

        m = m / (m.sum(1, keepdims=True) + 1e-8)

        
        reliability = np.diag(m).mean()
        logger.debug(f'Reliability: {reliability:.2f}')
        return m


class UniformNoiseWorker(RealWorker):
    def _sample_confusion_matrix(self):
        
        cm = []
        for _, v in self.worker_cm_info['group_workers'].items():
            v = np.array(v)
            global_v = v.sum(0)
            idx = self.npr.choice(range(len(v)), 10)
            cm.append(global_v + v[idx][0] * 10)
        cm = sum(cm)

        imagenet100_name = self.worker_cm_info['imagenet100_name']

        m = cm[self.keep_indices, :][:, self.keep_indices]

        if self.config.n_data_distraction_per_class > 0:
            m = np.zeros((self.config.n_classes, self.config.n_classes))
            m[:self.config.n_classes-1, :self.config.n_classes-1] = cm[self.keep_indices, :][:, self.keep_indices]
            idx_to_drop_mask = np.ones(cm.shape[0]).astype(np.bool)
            idx_to_drop_mask[self.keep_indices] = False


            m[-1, :len(self.keep_indices)] = cm[idx_to_drop_mask, :][:, self.keep_indices].sum(0)
            m[:len(self.keep_indices), -1] = cm[self.keep_indices, :][:, idx_to_drop_mask].sum(1)
            m[-1, -1] = cm[idx_to_drop_mask, :][:, idx_to_drop_mask].sum()
        else:
            m = cm[self.keep_indices, :][:, self.keep_indices]
        assert len(np.where(m.sum(1)==0)[0]) == 0

        m = m / (m.sum(1, keepdims=True) + 1e-8)

        n_classes = len(self.wnids)
        _m = np.zeros((n_classes, n_classes))
        _m += ((1 - m.diagonal()) / (n_classes - 1)).reshape(-1, 1)
        np.fill_diagonal(_m, m.diagonal())
        m = _m
        
        reliability = np.diag(m).mean()
        logger.debug(f'Reliability: {reliability:.2f}')
        return m


def get_worker_class(config, wnids):

    if config.worker.type == 'perfect':
        worker_class = PerfectWorker
    elif config.worker.type == 'uniform':
        worker_class = UniformWorker
    elif config.worker.type == 'uniform_noise':
        worker_class = UniformNoiseWorker
        worker_class.wnids = wnids
    elif config.worker.type == 'structured_noise':
        worker_class = StructuredNoiseWorker
        worker_class.wnids = wnids
    else:
        raise ValueError

    return worker_class
