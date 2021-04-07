import os
import yaml
import hydra
import json
import numpy as np
from shutil import copyfile
from collections import OrderedDict

from online_label.online_loop import run_online_loop
from online_label.worker import get_worker_class
from online_label.sampler import get_sampler_class
from online_label.aggregator import get_aggregator_class
from online_label.learner import get_learner_class
from online_label.optimizer import get_optimizer_class
from online_label.annotation_holder import AnnotationHolder
from online_label.logger import BatchLogger
from data.imagenet import ImageNetData


import logging
logger = logging.getLogger(__name__)


def setup(config):
    '''Basic setup
    Including loading best hyper-params and saving the experiment configurations.
    '''

    ## Save experiment configurations
    logger.info(config.pretty())
    logger.info(f'Current working directory: {os.getcwd()}')

    with open('config.txt', 'w') as f:
        f.write(config.pretty())
        

def init_workers(config, wnids):

    worker_class = get_worker_class(config, wnids)

    workers = OrderedDict()
    for i in range(config.worker.n):
        w = worker_class(config=config, seed=config.seed+i, known=config.worker.known)
        workers.update({w.id: w})

    mean_reliability = sum([np.diag(w.m).mean() for w in workers.values()]) / config.worker.n
    logger.info(f'Average worker reliability: {mean_reliability}')
    return workers


def save_state(config, workers, annotation_holder, optimizer, learner, step):
    workers_str = json.dumps([w.save_state() for w in workers.values()])
    annotation_holder_str = annotation_holder.save_state()
    optimizer_str = optimizer.save_state()
    learner_str = learner.save_state()

    state = dict(workers_str=workers_str, 
                 annotation_holder_str=annotation_holder_str, 
                 optimizer_str=optimizer_str, 
                 learner_str=learner_str, 
                 step=step)

    if os.path.exists('latest_state.json'):
        copyfile('latest_state.json', 'backup_state.json')

    json.dump(state, open('latest_state.json', 'w'))


def load_state(workers, annotation_holder, optimizer, learner, sampler, filename):
    state = json.load(open(filename))
    workers_str = json.loads(state['workers_str'])

    for w, s in zip(workers.values(), workers_str):
        w.load_state(s)
    
    _workers = OrderedDict()
    for w in workers.values():
        _workers.update({w.id: w})
    workers = _workers
    
    annotation_holder_str = state['annotation_holder_str']
    annotation_holder.load_state(annotation_holder_str, workers)

    optimizer_str = state['optimizer_str']
    optimizer.load_state(optimizer_str, workers)
    
    learner_str = state['learner_str']
    learner.load_state(learner_str)
    
    sampler.load_state(annotation_holder, workers)
    return state['step']
    

# support pre-emption
def load_from_latest_state(config, workers, annotation_holder, optimizer, learner, sampler):

    resume = os.path.exists('latest_state.json')
    if resume:
        try:
            start_step = load_state(workers, annotation_holder, optimizer, learner, sampler, 'latest_state.json')
            logger.info('Log state from latest_state.json')
        except json.decoder.JSONDecodeError:
            start_step = load_state(workers, annotation_holder, optimizer, learner, sampler, 'backup_state.json')
            logger.info('Log state from backup_state.json')

        logger.info(f'From step {start_step}')
    else:
        save_state(config, workers, annotation_holder, optimizer, learner, step=0)
        start_step = 0

    return start_step


@hydra.main(config_path='online_label/config', config_name='config')
def main(config):

    setup(config)

    # >>> Data and Simulated Workers
    imagenet_data = ImageNetData(config)
    workers = init_workers(config, imagenet_data.wnids)
    annotation_holder = AnnotationHolder(config, 
                                         workers, 
                                         imagenet_data.image_path, 
                                         imagenet_data.imagenet_struc)

    # >>> Initialize Components in Online Labeling
    aggregator = get_aggregator_class(config)(config, imagenet_data.n)
    learner = get_learner_class(config)(config)
    optimizer = get_optimizer_class(config)(config, imagenet_data, workers)
    sampler = get_sampler_class(config)(config, annotation_holder, workers, optimizer=optimizer)

    
    # >>> Load the state
    start_step = load_from_latest_state(config, workers, annotation_holder, optimizer, learner, sampler)

    batch_logger = BatchLogger(config, 
                               imagenet_data.get_y(imagenet_data.log_image_path), 
                               imagenet_data.n, 
                               imagenet_data.wnids, 
                               imagenet_data.p_image_path, 
                               imagenet_data.image_path, 
                               imagenet_data.log_image_path, 
                               imagenet_data.d_image_path)

    
    # >>> Online Labeling
    run_online_loop(config,
                    imagenet_data,
                    annotation_holder,
                    sampler,
                    aggregator,
                    learner,
                    optimizer,
                    batch_logger, 
                    save_state, 
                    start_step)


if __name__ == '__main__':
    main()
