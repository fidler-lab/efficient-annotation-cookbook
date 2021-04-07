import json
import numpy as np
from collections import OrderedDict
from copy import deepcopy

import logging
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class AnnotationHolder():
    def __init__(self, config, workers, image_path, imagenet_struc):
        self.config = config
        self.npr = np.random.RandomState(config.seed)
        self.imagenet_struc = imagenet_struc
        self.workers = workers

        self.n_data = len(image_path)
        self.annotation = OrderedDict()
        for p in image_path:
            self.annotation.update({p: []})
        self.n_annotation = 0

    def load_state(self, checkpoint_annotation_holder, workers):
    
        self.workers = workers

        checkpoint_annotation_holder = json.loads(checkpoint_annotation_holder)
        new_annotation = OrderedDict()

        for k in self.annotation.keys():
            anno_k = []
            for a in checkpoint_annotation_holder[k]:
                y, z, j, p_z_given_y = a
                anno_k.append((y, z, j, np.array(p_z_given_y)))

            new_annotation.update({k: anno_k})
            
        self.annotation = new_annotation
        n_annotation = sum([len(v) for v in self.annotation.values()])
        self.n_annotation = n_annotation

    def save_state(self):
        return json.dumps(deepcopy(self.annotation), cls=NumpyEncoder)

    def collect_annotation(self,
                           a_data,  
                           data_path, 
                           worker_id, 
                           belief):

        logger.info(f'Collect {len(data_path)} Annotation')
        for p, j in zip(data_path, worker_id):
            y = a_data[p][1]
            w = self.workers[j]
            z, p_z_given_y = w.annotate(y)
            self.annotation[p].append((int(y), int(z), j, p_z_given_y))

        self.n_annotation += len(data_path)

    def add_annotation(self, annotations):

        for anno_i in annotations:
            p, y, z, j = anno_i
            p_z_given_y = None

            self.annotation[p].append((int(y), int(z), j, p_z_given_y))

        self.n_annotation += len(annotations)
