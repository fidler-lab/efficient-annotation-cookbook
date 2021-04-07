from .aggregator import MjAggregator, BayesAggregator


def get_aggregator_class(config):
    if config.aggregator.algo.lower() == 'mj':
        return MjAggregator
    elif config.aggregator.algo.lower() == 'bayes':
        return BayesAggregator
    else: 
        raise ValueError
        