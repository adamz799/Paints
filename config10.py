import cv2
import random
import time 
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

ITERATION = 4

pic_dir = '../pics/256_data/'

data_dir = pic_dir+ 'bw/'
label_dir = pic_dir+ 'color/'
block_dir = pic_dir+ 'block/'

file_list = os.listdir(data_dir)

device = torch.device('cuda')
device_ids = [0]



