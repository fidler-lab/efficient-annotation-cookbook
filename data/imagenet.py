import os
import json
import numpy as np
from copy import deepcopy
from collections import Counter
from collections import OrderedDict # For python >= 3.7, dict() is fine

from . import REPO_DIR, imagenet100
from .utils import ImageNetStruc

import logging
logger = logging.getLogger(__name__)


class ImageNetData(object):

    imagenet_struc = ImageNetStruc()
    label_path = os.path.join(REPO_DIR, 'data/train_labeled.txt')
    feat_dir = os.path.join(REPO_DIR, 'data/features')
    
    def __init__(self, config):
        
        self.npr = np.random.RandomState(config.seed)
        self.wnids = config.wnids.split(' ')
        assert len(self.wnids) == config.n_classes

        self.config = config
        self.imagenet_struc.register_nodes(self.wnids)

        self.include_distraction = config.n_data_distraction_per_class > 0
        a_data, p_image_path, image_path, log_image_path, d_image_path, features_dict = self.load_data()

        ys = [a_data[p][1] for p in image_path if not self.include_distraction or a_data[p][1] != self.config.n_classes-1]        
        counter = Counter(ys)    
        n_max = counter.most_common()[0][1]
        n_min = counter.most_common()[-1][1]
        rho = n_max / n_min
        logger.info(f'Imbalance level: {rho:.2f}')

        self.n_p = len(p_image_path)
        self.n = len(image_path)
       
        self.a_data = a_data
        self.p_image_path = p_image_path
        self.image_path = image_path
        self.log_image_path = log_image_path
        self.d_image_path = d_image_path
        self.features_dict = features_dict
        logger.info(f'Number of data to label: {self.n}, number of prototype data: {self.n_p}')

    def get_y(self, path):
        y = []
        for p in path:
            y.append(self.a_data[p][1])
        return y

    def get_features_dict(self):
        return deepcopy(self.features_dict)

    def __load_a_data(self):
        
        a_data = OrderedDict()
        with open(self.label_path) as f:
            content = f.readlines()
            path = [i.split(' ')[0] for i in content]
            for p in path:
                a_data[p] = [p.split('/')[0], None]
        
        return a_data
        
    def add_distracting_images(self, a_data, p_image_path, image_path, log_image_path):

        logger.info('Include distracting images')
        
        other_class = [p for p, (wnid, _) in a_data.items() if wnid not in self.wnids]
        valid = [p for p in other_class if p not in p_image_path and p not in image_path]
        d_image_path = self.npr.choice(valid, 
                                       min(len(valid), self.config.n_classes * self.config.n_data_distraction_per_class), 
                                       replace=False).tolist()
        image_path = image_path + d_image_path
        log_image_path = log_image_path + d_image_path

        valid = [p for p in other_class if p not in p_image_path and \
                                            p not in image_path and \
                                            p not in d_image_path]
        d_p_image_path = self.npr.choice(valid, 
                                         min(len(valid), self.config.n_data_distraction_per_class), 
                                         replace=False).tolist()
        p_image_path = p_image_path + d_p_image_path

        logger.info(f'Number of distraction: {len(d_image_path)}')
        return p_image_path, image_path, log_image_path, d_image_path

    def load_prototype(self, a_data, selected=[]):
        
        # Random
        def __choose(id, n):
            valid = [p for p, (wnid, _) in a_data.items() if wnid == id and \
                                                        p not in selected]
            return self.npr.choice(valid, min(len(valid), n), replace=False).tolist()

        p_image_path = []
        for id in self.wnids:
            p_image_path.extend(__choose(id, self.config.n_prototype_per_class))

        return p_image_path

    def which_class(self, wnid):
        if wnid in self.wnids:
            return self.wnids.index(wnid)
        elif self.include_distraction:
            return self.config.n_classes - 1
        else:
            raise ValueError

    def load_random_image_path(self, a_data):
        p_image_path = self.load_prototype(a_data)
        
        def __choose(id, n):
            valid = [p for p, (wnid, _) in a_data.items() if wnid == id and \
                                                        p not in p_image_path]
            return self.npr.choice(valid, min(len(valid), n), replace=False).tolist()


        image_path = []
        for id in self.wnids:
            image_path.extend(__choose(id, self.config.n_data_per_class))

        log_image_path = image_path.copy()
        return p_image_path, image_path, log_image_path

    def load_data(self):
        '''
        return 
            a_data: (dict) key=image_path, value=[wnid, y]
            p_image_path: (list) paths of prototype images
            image_path: (list) paths of all accessible images (use to compute features)
            log_image_path: (list) paths of images of interests
            d_image_path: (list) paths of distracting images
            features_dict: (dict) {prototype_features: ?, features: ?}
        '''
        config = self.config
        
        # Load all data
        a_data = self.__load_a_data()
        a_n_data = len(a_data)
        a_image_path = list(a_data.keys())


        # Sample Prototype/Accessible/Log data
        p_image_path, image_path, log_image_path = self.load_random_image_path(a_data)


        # Include distraction data if needed
        if self.include_distraction:
            p_image_path, image_path, log_image_path, d_image_path = \
                self.add_distracting_images(a_data, 
                                            p_image_path, 
                                            image_path, 
                                            log_image_path)

            config.n_classes += 1   # Append other_class at the end
        else:
            d_image_path = []
        
        
        p = os.path.join(self.feat_dir, config.learner.features)
        assert os.path.exists(p), logger.warning(f'Features {p} does not exist')
        logger.info(f'Load features from {p}')

        a_features = np.load(p)
        p_features = np.array([a_features[a_image_path.index(p)] for p in p_image_path])
        features = np.array([a_features[a_image_path.index(p)] for p in image_path])
        features_dict = {
            'features': features, 
            'prototype_features': p_features
        }


        for k, v in a_data.items():
            wnid, _ = v
            if self.include_distraction:
                y = self.wnids.index(wnid) if wnid in self.wnids else config.n_classes - 1
            else:
                y = self.wnids.index(wnid) if wnid in self.wnids else None
            
            a_data[k] = [wnid, y]

        return a_data, p_image_path, image_path, log_image_path, d_image_path, features_dict
