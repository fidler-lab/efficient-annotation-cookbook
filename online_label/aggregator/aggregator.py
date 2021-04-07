import numpy as np

epsilon = 1e-8


class Aggregator(object):
    def __init__(self, config, n_data):
        self.config = config
        self.algo = config.aggregator.algo
        self.n_classes = config.n_classes
        self.n_data = n_data

    def compute_risk(self, belief):
        assert np.max(belief) <= 1. and np.min(belief) >= 0.
        y_hat = np.argmax(belief, 1)
        confidence = np.take_along_axis(belief, y_hat.reshape(-1, 1), axis=1)
        confidence = confidence.reshape(-1)
        risk = 1 - confidence
        return risk
        
    def aggregate(self, annotation_holder, **kwargs):
        raise NotImplementedError

    def empty_belief(self):
        belief = np.zeros((self.n_data, self.n_classes))
        return belief


class MjAggregator(Aggregator):
    def __init__(self, config, n_data, **kwargs):
        Aggregator.__init__(self, config, n_data)

    def aggregate(self, annotation_holder, **kwargs):
        belief = self.empty_belief()

        image_path = list(annotation_holder.annotation.keys())
        for i, p in enumerate(image_path):
            for anta in annotation_holder.annotation[p]:
                _, z, _, _ = anta
                belief[i][z] += 1
        
        votes_per_data = np.sum(belief, 1, keepdims=True)
        belief = np.divide(belief, votes_per_data, 
                                out=np.zeros_like(belief), 
                                where=votes_per_data!=0)
        return belief


class BayesAggregator(Aggregator):
    def __init__(self, config, n_data, **kwargs):
        Aggregator.__init__(self, config, n_data)
        self.uniform_prior = np.ones((1, self.n_classes)) / self.n_classes

    def __normalize(self, log_p):
        if len(log_p.shape) == 1:
            log_p = log_p.reshape(1, -1)

        b = np.max(log_p, 1, keepdims=True)
        log_sum_p = b + np.log(np.exp(log_p - b).sum(1, keepdims=True))
        
        p = np.exp(log_p - log_sum_p)
        p = np.clip(p, 0., 1.)
        return p

    def bayes(self, annotation_holder, belief, prior):

        image_path = list(annotation_holder.annotation.keys())
        for i, p in enumerate(image_path):

            log_prior = np.log(prior[i] + epsilon) if len(prior) == len(image_path) else np.log(prior[0] + epsilon)
            
            log_likelihood = np.zeros_like(log_prior)
            for anta in annotation_holder.annotation[p]:
                y, _, _, p_z_given_y = anta
                log_likelihood += np.log(p_z_given_y + epsilon)

            log_p = log_prior + log_likelihood

            belief[i] = log_p

        return belief

    def aggregate(self, annotation_holder, prior=None, learner_prob=None, prototype_targets=None, **kwargs):

        belief = self.empty_belief()
        if prior is None:
            prior = self.uniform_prior
        unnormalized_worker_belief = self.bayes(annotation_holder, belief, prior)
        worker_belief = self.__normalize(unnormalized_worker_belief)

        if learner_prob is not None:
            n_data = annotation_holder.n_data
            learner_log_prob = np.log(learner_prob + epsilon)
            unnormalized_belief = unnormalized_worker_belief + learner_log_prob[:n_data, :]
            return self.__normalize(unnormalized_belief)
        else:
            return self.__normalize(unnormalized_worker_belief)
