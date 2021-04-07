import json
import numpy as np
from .utils import NpEncoder, cal_ece
from collections import defaultdict


import logging
logger = logging.getLogger(__name__)


class BatchLogger(object):
    def __init__(self, config, y, n_data, wnids, p_image_path, image_path, log_image_path, d_image_path):
        self.config = config
        self.n_data = n_data
        self.wnids = wnids
        self.ground_truth_y = np.array(y)
        self.image_path = image_path
        self.log_image_path = log_image_path

        self.include_distraction = config.n_data_distraction_per_class > 0
        self.d_image_path = d_image_path

        self.info = defaultdict()
        self.info['p_image_path'] = p_image_path
        self.info['image_path'] = image_path
        self.info['log_image_path'] = log_image_path
        self.info['d_image_path'] = d_image_path
        self.info['ground_truth_y'] = self.ground_truth_y
        self.cur_step = 0.

    def save(self):
        json.dump(self.info, open(f'info-{self.cur_step}.json', 'w'), cls=NpEncoder)

    def load(self, filename):
        info = json.load(open(filename, 'w'))
        self.info = info
        
    def step(self,
             n_data,
             annotation, 
             sampled_image_path,
             step, 
             learner_prob,
             y_posterior,
             y_posterior_risk):


        self.cur_step = step
        n_annotation = sum([len(v) for v in annotation.values()])
        cost = n_annotation 
        logger.info(f'Step: {self.cur_step}')
        logger.info(f'Number of annotation: {cost}')
        logger.info(f'Average annotation: {cost / len(self.log_image_path)}')

        x_axis = {
            'step': step,
            'cost': cost,
            'cost-per-image': cost / self.n_data
        }

        self.info['data'] = {
                'step': int(step), 
                'cost': int(cost), 
                'cost-per-image': cost / self.n_data,
                'sampled_image_path': sampled_image_path, 
                'learner_prob': learner_prob, 
                'y_posterior': y_posterior.tolist(), 
                'y_posterior_risk': y_posterior_risk.tolist()
        }

        log_idx = np.array([self.image_path.index(p) for p in self.log_image_path])

        y_posterior = y_posterior[log_idx, :]
        y_posterior_risk = y_posterior_risk[log_idx]


        self.log_acc(y_posterior, 'aggregation', y_posterior_risk, x_axis)
        self.log_risk(y_posterior_risk, 'aggregation', x_axis)
        self.log_count(annotation, 'aggregation', x_axis)
        self.log_ece(y_posterior, 'aggregation', y_posterior_risk, x_axis)

        if learner_prob is not None:
            # learner_prob: [data_learner_prob, prototype_learner_prob]
            learner_prob = learner_prob[:self.n_data, :]
            learner_prob = learner_prob[log_idx]
            self.log_acc(learner_prob, 'learner_prob', y_posterior_risk, x_axis)
            self.log_ece(learner_prob, 'learner_prob', y_posterior_risk, x_axis)
            
    def log_ece(self, belief, prefix, risk, x_axis):

        n_data = self.n_data
        ground_truth_y = self.ground_truth_y
        config = self.config

        y_hat = belief.argmax(1)
        pred_prob = np.take_along_axis(belief, y_hat[:, np.newaxis], axis=1).squeeze()

        def _log_ece(mask, tag, info):
            if sum(mask) > 0:
                ece = cal_ece(pred_prob[mask], 
                              y_hat[mask], 
                              ground_truth_y[mask])
                logger.info(f'{prefix}\t{tag}/ece: {ece}')
                info.update({f'{prefix}/{tag}/ece': ece})

        info = {}

        all_mask = np.ones(len(risk)).astype(np.bool)
        valid_mask = risk < config.risk_thres
        invalid_mask = ~valid_mask

        if self.include_distraction:

            # Class of interest
            _log_ece(all_mask, 'all', info)
            _log_ece(valid_mask, 'valid', info)
            _log_ece(invalid_mask, 'invalid', info)

            # Distraction classes
            _log_ece(all_mask, 'all', info)
            _log_ece(valid_mask, 'valid', info)
            _log_ece(invalid_mask, 'invalid', info)

            _log_ece(all_mask, 'all+distraction', info)
            _log_ece(valid_mask, 'valid+distraction', info)
            _log_ece(invalid_mask, 'invalid+distraction', info)
        else:
            _log_ece(all_mask, 'all', info)
            _log_ece(valid_mask, 'valid', info)
            _log_ece(invalid_mask, 'invalid', info)


        self.info['data'].update(info)

    def log_count(self, annotation, prefix, x_axis):

        count = np.array([len(anno_i) for anno_i in annotation])
        n_data = self.n_data

        info = {f'{prefix}/num_labeled_data': len(np.where(count > 0)[0]) / n_data}
        logger.info(f'{prefix}\tnum_labeled_data: {len(np.where(count > 0)[0]) / n_data}')
        
        self.info['data'].update(info)

    def log_risk(self, risk, prefix, x_axis):
        valid_mask = (risk < self.config.risk_thres)

        logger.info(f'{prefix}\tvalid_num: {sum(valid_mask)}')
        info = {f'{prefix}/valid_num': sum(valid_mask)}

        self.info['data'].update(info)

    def log_acc(self, belief, prefix, risk, x_axis):

        n_data = self.n_data
        ground_truth_y = self.ground_truth_y
        config = self.config

        topK = [1, 5, 10]
        y_hat = np.argsort(belief, 1)[:, ::-1]
        topK_acc = {k: None for k in topK}
        for k in topK:
            acc = np.any(y_hat[:, :k] == ground_truth_y.reshape(-1, 1), axis=1)
            topK_acc[k] = acc


        def _log_acc(mask, tag, info):
            if sum(mask) > 0:
                for k in topK:
                    average_acc = np.mean(topK_acc[k][mask])
                    logger.info(f'{prefix}\t{tag}/top{k}: {average_acc}')
                    info.update({f'{prefix}/{tag}/top{k}': average_acc})


        def _log_num(mask, tag, info):
            if sum(mask) > 0:
                n_correct = len(np.where(y_hat[mask, 0] == \
                                        ground_truth_y[mask])[0])
                n_incorrect = len(np.where(y_hat[mask, 0] != \
                                        ground_truth_y[mask])[0])
                logger.info(f'{prefix}\t{tag}/n_correct: {n_correct}')
                info.update({f'{prefix}/{tag}/n_correct': n_correct})
                logger.info(f'{prefix}\t{tag}/n_incorrect: {n_incorrect}')
                info.update({f'{prefix}/{tag}/n_incorrect': n_incorrect})


        info = {}
        all_mask = np.ones(len(risk)).astype(np.bool)
        valid_mask = risk < config.risk_thres
        invalid_mask = ~valid_mask
        if self.include_distraction:
            distraction_mask = ground_truth_y == config.n_classes-1
            
            # Class of interest
            _log_acc((all_mask & ~distraction_mask), 'all', info)
            _log_num((all_mask & ~distraction_mask), 'all', info)

            _log_acc((valid_mask & ~distraction_mask), 'valid', info)
            _log_num((valid_mask & ~distraction_mask), 'valid', info)

            _log_acc((invalid_mask & ~distraction_mask), 'invalid', info)
            _log_num((invalid_mask & ~distraction_mask), 'invalid', info)

            # Distraction classes
            _log_acc((all_mask & distraction_mask), 'all_distraction', info)
            _log_num((all_mask & distraction_mask), 'all_distraction', info)

            _log_acc((valid_mask & distraction_mask), 'valid_distraction', info)
            _log_num((valid_mask & distraction_mask), 'valid_distraction', info)

            _log_acc((invalid_mask & distraction_mask), 'invalid_distraction', info)
            _log_num((invalid_mask & distraction_mask), 'invalid_distraction', info)

            # All 
            _log_acc(all_mask, 'all+distraction', info)
            _log_num(all_mask, 'all+distraction', info)

            _log_acc(valid_mask, 'valid+distraction', info)
            _log_num(valid_mask, 'valid+distraction', info)

            _log_acc(invalid_mask, 'invalid+distraction', info)
            _log_num(invalid_mask, 'invalid+distraction', info)

        else:
            _log_acc(all_mask, 'all', info)
            _log_num(all_mask, 'all', info)

            _log_acc(valid_mask, 'valid', info)
            _log_num(valid_mask, 'valid', info)

            _log_acc(invalid_mask, 'invalid', info)
            _log_num(invalid_mask, 'invalid', info)


        self.info['data'].update(info)
