import os, struct, sys
import glob, pickle
import math, re
# import cv2
import itertools, operator

import numpy as np
import pandas as pd
import math

from collections import defaultdict
from copy import deepcopy
from functools import reduce

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
# from skimage import transform




class Config:
    
    def __init__(self):
        self.NUM_BBOX = 5
        self.NUM_CLASS = 2
        
        self.GRID_H = 13
        self.GRID_W = 13
        
        self.BBOX_SIZE = 1 + 4 + self.NUM_CLASS

        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        

