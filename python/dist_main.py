import os
os.environ['DGLBACKEND']='pytorch'
from multiprocessing import Process
import argparse, time, math
import numpy as np
from functools import wraps
import tqdm
import numpy as np

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data.utils import load_graphs
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.distributed import DistDataLoader

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

if __name__ == '__main__':
    dgl.distributed.initialize('ip_config.txt')
    g = dgl.distributed.DistGraph('FB15k', part_config='data/FB15k.json')
    pb = g.get_partition_book()

    train_eids = pb.partid2eids(pb.partid)
    train_nids = pb.partid2nids(pb.partid)
    print(np.max(g.edata['tid']))
    
    
    
