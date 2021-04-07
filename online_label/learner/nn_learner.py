import json
import time
import math
import itertools
import numpy as np
from collections import Counter
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .utils import ModelWithTemperature, LinearModel
from . import Learner

import logging
logger = logging.getLogger(__name__)


class NNLearner(Learner):
    def __init__(self, config):
        Learner.__init__(self, config)
        self.batch_size = self.config.learner.batch_size
        self.max_epochs = self.config.learner.max_epochs
        self.best_model_loss = np.inf
        self.best_probs = None

    def save_state(self):
        if self.best_probs is not None:
            best_model_loss = self.best_model_loss
            return json.dumps(dict(best_model_loss=[float(best_model_loss)], best_probs=self.best_probs.tolist()))
    
    def load_state(self, state):
        if state is not None:
            state = json.loads(state)
            self.best_probs = np.array(state['best_probs'])
            self.best_model_loss = state['best_model_loss'][0]
            
    def init_learner(self, in_channels, out_channels):
        raise NotImplementedError

    def fit(self, in_channels, weighted_labeled_loader, val_loader):
        '''Fit the model on `weighted_labeled_loader` and validate on `val_loader`
        Return the best model based on the loss in val_loader and
                its corresponding loss in val_loader
        '''

        t1 = time.time()
        model = self.init_learner(in_channels, self.config.n_classes)
        best_model = self.init_learner(in_channels, self.config.n_classes)

        if self.use_cuda:
            model = model.cuda()

        lr = self.config.learner.lr_ratio * math.sqrt(min(weighted_labeled_loader.batch_size, 
                                                        len(weighted_labeled_loader.dataset)))
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.config.learner.weight_decay)
        
            
        def __train_step(model, inputs, targets):

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            optim.zero_grad()
            logits = model(inputs)
            loss = model.compute_loss(logits, targets)

            loss.backward()
            optim.step()
            return loss.cpu().item()
                

        @torch.no_grad()
        def __eval(model, loader):
            model.eval()
            loss_l = []
            targets_l = []
            for inputs, targets in loader:
                if self.use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                logits = model(inputs)
                loss = model.compute_loss(logits, targets, reduction='none')
                
                loss_l.extend(loss.cpu().numpy())
                targets_l.extend(targets.cpu().numpy())
            return loss_l, targets_l


        best_loss = np.inf
        best_epoch = 0
        for epoch in range(1, self.max_epochs+1):
            
            model.train()

            train_loss_average = []
            for inputs, targets in weighted_labeled_loader:
                loss = __train_step(model, inputs, targets)
                train_loss_average.append(loss)

            if epoch % 10 == 0:
                model.eval()
                val_loss_l, _ = __eval(model, val_loader)
                val_loss = np.mean(val_loss_l)
                logger.debug('[Epoch {epoch}] Val Loss: {val_loss}')

                if val_loss < best_loss:
                    best_model.load_state_dict(deepcopy(model.state_dict()))
                    best_loss = val_loss
                    best_epoch = epoch

        logger.debug(f'Use model at epoch {best_epoch}')
        logger.debug(f'Fiting nn takes {time.time()-t1} sec in {epoch} epoch')
        if self.use_cuda:
            best_model = best_model.cuda()

        return best_model, best_loss

    def get_train_val(self, features, prototype_targets, belief, n_annotation, ground_truth):
        '''
        features: dict with key "features" and "prototype_features"
        prototype_targets: np.ndarray
        belief: np.ndarray with shape (#data, #class)
        n_annotation: np.ndarray describing the number of annotation for each image
        ground_truth: np.ndarray
        '''

        n_prototypes = len(prototype_targets)
        n_data = len(features['features'])

        def __get_train_mask(risk, confident_mask):
            # Get training data
            at_least_one_annotation = n_annotation > 0
            logger.debug(f'Number of annotated examples (|W_i| > 0) : {at_least_one_annotation.sum()}')
            logger.debug(f'Number of confident examples: {confident_mask.sum()}')
            if self.semi_supervised == 'none':
                train_mask = np.copy(at_least_one_annotation)
            elif self.semi_supervised == 'pseudolabel':
                train_mask = np.copy(confident_mask)
            else:
                raise ValueError

            # Include prototypes
            at_least_one_annotation = np.concatenate([at_least_one_annotation, 
                                                    np.zeros(n_prototypes).astype(np.bool)])
            confident_mask = np.concatenate([confident_mask, np.ones(n_prototypes).astype(np.bool)])
            if self.prototype_as_val:
                train_mask = np.concatenate([train_mask, np.zeros(n_prototypes).astype(np.bool)])
            else:
                train_mask = np.concatenate([train_mask, np.ones(n_prototypes).astype(np.bool)])
                
            return train_mask, confident_mask

        prototype_prob = np.zeros([n_prototypes, self.config.n_classes])
        prototype_prob[range(n_prototypes), prototype_targets] = 1.

        # Get confident mask
        y_hat = belief.argmax(1)
        confidence = belief.max(1)
        risk = 1 - confidence
        confident_mask = risk < self.risk_thres

    
        # All data to use
        P = np.concatenate([belief, prototype_prob])
        Y = np.concatenate([y_hat, prototype_targets])
        X = np.concatenate([features['features'], features['prototype_features']])
        ground_truth = np.concatenate([ground_truth, prototype_targets])


        # Train
        train_mask, confident_mask = __get_train_mask(risk, confident_mask)

        prototype_mask = np.zeros(n_data + n_prototypes).astype(np.bool)
        prototype_mask[-n_prototypes:] = True

        # Val
        if self.prototype_as_val:
            val_mask = prototype_mask.copy()
        else:
            # Choose a balance set
            n_val = prototype_mask.sum()

            idx = self.npr.choice(np.where(train_mask)[0], n_val, replace=False)
            val_mask = np.zeros(n_data + n_prototypes).astype(np.bool)
            val_mask[idx] = True
            train_mask[idx] = False


        def __ensure_enough_in_train_and_val(train_mask, val_mask):
            val_Y = Y[val_mask]
            train_Y = Y[train_mask]

            train_counter = Counter(train_Y)
            val_counter = Counter(val_Y)
            for c in range(self.config.n_classes):
                train_c = train_counter[c]
                val_c = val_counter[c]
                if val_c < math.floor(self.config.n_prototype_per_class / 2):
                    logger.debug('Not enough class ({c}) in validation set')
                    n = math.floor(self.config.n_prototype_per_class / 2) - val_c

                    # data move from t to v
                    idx = np.where(Y[train_mask] == c)[0]
                    idx = self.npr.choice(idx, n, replace=False)
                    idx = np.where(train_mask)[0][idx]

                    val_mask[idx] = True
                    train_mask[idx] = False
                elif train_c < val_c:
                    # Move half of data from v to t
                    logger.debug(f'Not enough class ({c}) in train set')
                                
                    idx = np.where(Y[val_mask] == c)[0]
                    idx = self.npr.choice(idx, math.floor(val_c * 0.5), replace=False)
                    idx = np.where(val_mask)[0][idx]
                    
                    train_mask[idx] = True
                    val_mask[idx] = False
                    
            return train_mask, val_mask

            
        train_mask, val_mask = __ensure_enough_in_train_and_val(train_mask, val_mask)

        train_X = X[train_mask]
        train_Y = Y[train_mask]
        train_gt = ground_truth[train_mask]

        val_X = X[val_mask]
        val_Y = Y[val_mask]
        val_gt = ground_truth[val_mask]

        logger.debug(f'Train {len(train_X)}/Val {len(val_X)}')
        train_acc = (train_Y == train_gt).mean()
        val_acc = (val_Y == val_gt).mean()

        logger.debug(f'Accuracy of train dataset ({len(train_gt)}): {train_acc}')
        logger.debug(f'Accuracy of validation dataset ({len(val_gt)}): {val_acc}')

        return (X, Y, ground_truth), \
                (train_X, train_Y, train_gt, train_mask), \
                (val_X, val_Y, val_gt, val_mask)

    def create_dataloader_from_data(self, data, batch_size, shuffle=None, weighted=False):
        X, Y, _, _ = data
        tensor_X = torch.tensor(X)
        tensor_Y = torch.tensor(Y)
        dataset = TensorDataset(tensor_X, tensor_Y)

        batch_size = int(min(len(Y), batch_size))
        if batch_size == 0:
            return None

        if weighted:
            if len(Y.shape) == 2:
                # Y is a probability distribution
                Y = Y.argmax(1)
            y_counter = Counter(Y)
            if len(Y) == 0:
                weights = np.ones_like(Y)
            else:
                weights = np.array([1. / y_counter[t] for t in Y])
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=self.config.n_jobs)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.config.n_jobs)
        return loader

    def calibrate_model(self, model, val_loader):
        scaled_model = ModelWithTemperature(model, self.use_cuda)
        if self.use_cuda:
            scaled_model = scaled_model.cuda()
        scaled_model.set_temperature(val_loader)
        if scaled_model.before_ece < scaled_model.after_ece or scaled_model.before_nll < scaled_model.after_nll:
            logger.info('ECE or NLL in validation set goes higher than before. Use the uncalibrated model.')
            return model
        else:
            return scaled_model

    def predict(self, model, test_loader):
        with torch.no_grad():
            if self.use_cuda:
                model = model.cuda()

            model.eval()
            probs = []
            for inputs in test_loader:
                inputs = inputs[0]

                if self.use_cuda:
                    inputs = inputs.cuda()
                logits = model(inputs)
                prob = F.softmax(logits, dim=1)
                probs.append(prob.cpu())

            probs = torch.cat(probs, dim=0)
        return probs.cpu().numpy()

    def fit_and_predict(self, features, prototype_targets, belief, n_annotation, ground_truth):

        n_data = features['features'].shape[0]
        n_prototypes = features['prototype_features'].shape[0]
        in_channels = features['features'].shape[1]

        all_data, train_data, val_data = self.get_train_val(features, prototype_targets, belief, n_annotation, ground_truth)
        
        if 'cv' in self.calibrate:
            assert not self.prototype_as_val

            n_folds = int(self.calibrate.split('_')[1])

            _, _, _, train_mask = train_data
            _, _, _, val_mask = val_data
            mask = train_mask | val_mask

            X, Y, gt = all_data
            
            cv_X = X[mask]
            cv_Y = Y[mask]
            cv_gt = gt[mask]


            cv_idx = np.arange(len(cv_X))
            invalid = True
            while invalid:
                logger.debug(f'Split to {n_folds} folds')
                self.npr.shuffle(cv_idx)
                cv_splits = np.array_split(cv_idx, n_folds)
                # ensure at least one of the split is valid
                for cv_s in cv_splits:
                    cv_val_mask = np.zeros(len(cv_idx)).astype(np.bool)
                    cv_val_mask[cv_s] = True
                    cv_train_mask = ~cv_val_mask
                    if len(np.unique(cv_Y[cv_train_mask])) == self.config.n_classes and \
                        len(np.unique(cv_Y[cv_val_mask])) == self.config.n_classes:
                        invalid = False

            
            cv_best_loss_average = []
            cv_probs = []
            for i, cv_s in enumerate(cv_splits):
                logger.debug('{} fold'.format(i))

                cv_val_mask = np.zeros(len(cv_idx)).astype(np.bool)
                cv_val_mask[cv_s] = True
                cv_train_mask = ~cv_val_mask


                cv_train = (cv_X[cv_train_mask], 
                            cv_Y[cv_train_mask], 
                            cv_gt[cv_train_mask], 
                            None)
                
                cv_val = (cv_X[cv_val_mask], 
                        cv_Y[cv_val_mask], 
                        cv_gt[cv_val_mask], 
                        None)

                if len(np.unique(cv_train[1])) == self.config.n_classes and \
                    len(np.unique(cv_val[1])) == self.config.n_classes:

                    weighted_train_loader = self.create_dataloader_from_data(cv_train, self.batch_size, weighted=True)
                    val_loader = self.create_dataloader_from_data(cv_val, self.batch_size, shuffle=False)

                    cv_best_model, cv_best_loss = self.fit(in_channels, weighted_train_loader, val_loader)
                    cv_best_loss_average.append(cv_best_loss)
                    cv_best_model = self.calibrate_model(cv_best_model, val_loader)


                    dataset = TensorDataset(torch.tensor(np.concatenate([features['features'], features['prototype_features']])))
                    loader = DataLoader(dataset, 2**13)
                    probs = self.predict(cv_best_model, loader)
                    cv_probs.append(probs)

            probs = np.stack(cv_probs, 0).mean(0)
            best_model_loss = np.mean(cv_best_loss_average)

        elif self.calibrate == 'temperature':
            weighted_train_loader = self.create_dataloader_from_data(train_data, self.batch_size, weighted=True)
            val_loader = self.create_dataloader_from_data(val_data, self.batch_size, shuffle=False)
            best_model, best_model_loss = self.fit(in_channels, weighted_train_loader, val_loader)

            if self.calibrate == 'temperature':
                best_model = self.calibrate_model(best_model, val_loader)
            
            dataset = TensorDataset(torch.tensor(np.concatenate([features['features'], features['prototype_features']])))
            loader = DataLoader(dataset, 2**13)
            probs = self.predict(best_model, loader)
        else:
            raise ValueError


        if self.early_stop_scope == 'global':
            if best_model_loss < self.best_model_loss:
                logger.info(f'Find the model ({best_model_loss}) better than the previous one ({self.best_model_loss})')
                self.best_model_loss = best_model_loss
                self.best_probs = np.copy(probs)
            else:
                logger.info(f'Use the previous learned model ({self.best_model_loss})')
                probs = np.copy(self.best_probs)
           
        return probs


class LinearNNLearner(NNLearner):
    def __init__(self, config):
        NNLearner.__init__(self, config)
        self.n_hidden_layer = config.learner.n_hidden_layer
        self.hidden_size = config.learner.hidden_size
        
    def init_learner(self, in_channels, out_channels):
        model = LinearModel(in_channels, out_channels, self.hidden_size, self.n_hidden_layer)
        return model
