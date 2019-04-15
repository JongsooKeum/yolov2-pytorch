import os
import numpy as np
from datasets import data as dataset
from models.yolov2 import YOLO as ConvNet
# from learning.optimizers import MomentumOptimizer as Optimizer
from learning.optimizers import AdamOptimizer as Optimizer
from learning.evaluators import RecallEvaluator as Evaluator

""" 1. Load and split datasets """
root_dir = os.path.join('data/face/') # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# Load anchors
anchors = dataset.load_json(os.path.join(trainval_dir, 'anchors.json'))

# Set image size and number of class
IM_SIZE = (416, 416)
NUM_CLASSES = 1

# Load trainval set and split into train/val sets
X_trainval, y_trainval = dataset.read_data(trainval_dir, IM_SIZE, order='CHW')
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.1) # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

""" 2. Set training hyperparameters"""
hp_d = dict()

# FIXME: Training hyperparameters
hp_d['batch_size'] = 2
hp_d['num_epochs'] = 50
hp_d['init_learning_rate'] = 1e-5
hp_d['learning_rate_patience'] = 10
hp_d['learning_rate_decay'] = 0.1
hp_d['score_threshold'] = 1e-4
hp_d['nms_flag'] = True

""" 3. Build graph, initialize a session and start training """
model = ConvNet([3, IM_SIZE[0], IM_SIZE[1]], NUM_CLASSES, anchors)
model.cuda()

evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

train_results = optimizer.train(save_dir='.', details=True, verbose=True, **hp_d)