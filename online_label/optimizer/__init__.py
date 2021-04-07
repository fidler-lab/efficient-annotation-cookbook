class Optimizer(object):
    def __init__(self, config):
        self.config = config
        self.converged = None
        self.max_step_reached = None

    def reset(self):
        self.converged = False
        self.max_step_reached = False
        
    def step(self, annotation_holder, aggregator, learner, imagenet_data):
        raise NotImplementedError


def get_optimizer_class(config):
    from .em import EMOptimizer
    return EMOptimizer
