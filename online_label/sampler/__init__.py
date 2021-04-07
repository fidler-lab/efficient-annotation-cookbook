from .sampler import Sampler
from .random_sampler import RandomSampler 
from .task_assignment_sampler import GreedyTaskAssignmentSampler 


def get_sampler_class(config):
    if config.sampler.algo.lower() == 'random':
        return RandomSampler
    elif config.sampler.algo.lower() == 'greedy_task_assignment':
        return GreedyTaskAssignmentSampler
    else:
        raise ValueError
