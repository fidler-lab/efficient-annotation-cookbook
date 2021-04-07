import json
import numpy as np

import logging
logger = logging.getLogger(__name__)


class EarlyStopper():
    def __init__(self, hit_size, n_hits_per_step, risk_thres, n_data):
        self.risk_thres = risk_thres
        self.n_data = n_data
        self.cost_per_step = hit_size * n_hits_per_step
        self.num_valid_list = []

    def stop(self, num_valid, n_annotation):

        self.num_valid_list.append(num_valid)
        check_prev_n = int(max(5000 // self.cost_per_step, 5))
        
        # At least runs `check_prev_n` steps
        if len(self.num_valid_list) <= check_prev_n:
            return False

        # At least collect n_data/2 annotations
        if n_annotation < (self.n_data / 2.):
            return False

        max_so_far = max(self.num_valid_list)
        
        if len(self.num_valid_list) > 3:
            num_valid_list = np.array(self.num_valid_list)
            if not np.any(max_so_far == num_valid_list[-check_prev_n:]):
                # Decrease at least for `check_prev_n` steps
                return True
            else:
                return False


def __init(config, annotation_holder, optimizer, aggregator, learner, imagenet_data, batch_logger, start_step):

    logger.info(f'{"*"*20} Step: {start_step} {"*"*20}')

    # >>> Jointly optimize the worker skills and true labels
    info = optimizer.step(annotation_holder, aggregator, learner, imagenet_data)

    # >>> Log information
    if start_step == 0:
        batch_logger.step(annotation_holder.n_data, 
                        annotation_holder.annotation, 
                        sampled_image_path=[], 
                        step=start_step, 
                        **info)
    
    return info


def run_online_loop(config,
                    imagenet_data,
                    annotation_holder,
                    sampler,
                    aggregator,
                    learner,
                    optimizer,
                    batch_logger, 
                    save_state_fn, 
                    start_step):


    step = start_step
    early_stopper = EarlyStopper(config.online.hit_size, 
                                 config.online.n_hits_per_step, 
                                 config.risk_thres, 
                                 annotation_holder.n_data)
    info = __init(config, 
                  annotation_holder, 
                  optimizer, 
                  aggregator, 
                  learner, 
                  imagenet_data, 
                  batch_logger, 
                  start_step)

    for step in range(start_step+1, config.online.budget // config.online.hit_size+1):
        
        num_valid = sum(info['y_posterior_risk'] < config.risk_thres)
        if config.early_stop and early_stopper.stop(num_valid, annotation_holder.n_annotation):
            logger.info('Early stop since the number of valid examples decreases for certain steps')
            break

        if sampler.stop(info['y_posterior_risk']):
            logger.info('Examples either satisfy the risk criterion or reach maximum number of annotation')
            break

        logger.info(f'{"*"*20} Step: {step} {"*"*20}')
        
        logger.debug('Construct HITs')
        data_idx, worker_id = sampler.sample(config.online.hit_size, 
                                              config.online.n_hits_per_step, 
                                              info['y_posterior_risk'], 
                                              y_posterior=info['y_posterior'], 
                                              feature_dict=imagenet_data.get_features_dict())


        data_path = [imagenet_data.image_path[i] for i in data_idx]

        logger.debug('Workers Annotating')
        annotation_holder.collect_annotation(imagenet_data.a_data, data_path, worker_id, info['y_posterior'])
        
        # >>> Jointly optimize the worker skills and true labels
        info = optimizer.step(annotation_holder, aggregator, learner, imagenet_data)
        save_state_fn(config, annotation_holder.workers, annotation_holder, optimizer, learner, step=step)
        
        # >>> Log information
        batch_logger.step(annotation_holder.n_data, 
                          annotation_holder.annotation, 
                          sampled_image_path=data_path, 
                          step=step, 
                          **info)
