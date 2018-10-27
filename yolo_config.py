import os, struct, sys
import glob, pickle
import math, re
# import cv2
import itertools, operator

import numpy as np
import pandas as pd

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
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
# from PIL import Image
# from skimage import transform

# from nltk.corpus import wordnet as wn

# import xml.etree.ElementTree as ET


class Config:
    
    def __init__(self):
        self.NUM_BOX = 5
        self.NUM_CLASS = 2
        
        self.GRID_H = 13
        self.GRID_W = 13

