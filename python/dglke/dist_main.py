import os
os.environ['DGLBACKEND']='pytorch'
from multiprocessing import Process
import argparse, time, math
import numpy as np
from functools import wraps
import tqdm
import random

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data.utils import load_graphs
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.distributed import DistDataLoader

import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from .utils.argparser import TrainArgParser
from .utils.misc import prepare_args
from .utils.logging import Logger
from .data import get_dataset, TrainDataset, TestDataset, ValidDataset
from .data.dataloader import KGETrainDataLoaderGenerator, KGEEvalDataLoaderGenerator
from .utils import EMB_INIT_EPS
from .nn.modules import KGEEncoder, TransREncoder
from .nn.modules import KGEDecoder, AttHDecoder, TransRDecoder
from .nn.loss import sLCWAKGELossGenerator
from .nn.loss import BCELoss, HingeLoss, LogisticLoss, LogsigmoidLoss
from .regularizer import Regularizer
from .nn.modules import TransEScore, TransRScore, DistMultScore, ComplExScore, RESCALScore, RotatEScore, SimplEScore
from .nn.metrics import RankingMetricsEvaluator
from functools import partial
import torch as th
from torch import nn
import time
from .nn.modules import KEModel

def dist_main():
    dgl.distributed.initialize('ip_config.txt')
    g = dgl.distributed.DistGraph('FB15k', part_config='data/FB15k.json')
    pb = g.get_partition_book()

    train_eids = pb.partid2eids(pb.partid)
    train_nids = pb.partid2nids(pb.partid)

    emb_init = (143.0 + 1.0) / 200
    init_func = [partial(th.nn.init.uniform_, a=-emb_init, b=emb_init), partial(th.nn.init.uniform_, a=-emb_init, b=emb_init)]

    encoder = KGEEncoder(hidden_dim=200,
                             n_entity=g.num_nodes(),
                             n_relation=np.max(g.edata['tid']),
                             init_func=init_func,
                             score_func='TransE')

    '''
    emb_init = (143.0 + 2.0) / 200
    score_func = TransEScore(143.0, dist_func='l1')
    loss_gen = sLCWAKGELossGenerator(neg_adversarial_sampling=args.neg_adversarial_sampling,
                                         adversarial_temperature=args.adversarial_temperature,
                                         pairwise=args.pairwise,
                                         label_smooth=args.label_smooth)
    criterion = LogsigmoidLoss()

    loss_gen.set_criterion(criterion)
    metrics_evaluator = RankingMetricsEvaluator(args.eval_filter)
    decoder = KGEDecoder(args.decoder,
                         score_func,
                         loss_gen,
                         metrics_evaluator)
    '''

    def sample(eids):
        s, d = g.find_edges(eids)
        rel = g.edata['tid'][eids]
        neg = random.sample(list(train_nids.numpy()), 16)
        data = {'head' : s, 'tail' : d, 'rel' : rel, 'neg' : neg}
        return data

    dataloader = DistDataLoader(
        dataset=train_eids.numpy(),
        batch_size=16,
        collate_fn=sample,
        shuffle=True,
        drop_last=False)

    if args.dgl_sparse:
        emb_optimizer = dgl.distributed.optim.SparseAdam([emb_layer.sparse_emb], lr=args.sparse_lr)
        print('optimize DGL sparse embedding:', emb_layer.sparse_emb)
    elif args.standalone:
        emb_optimizer = th.optim.SparseAdam(list(emb_layer.sparse_emb.parameters()), lr=args.sparse_lr)
        print('optimize Pytorch sparse embedding:', emb_layer.sparse_emb)
    else:
        emb_optimizer = th.optim.SparseAdam(list(emb_layer.module.sparse_emb.parameters()), lr=args.sparse_lr)
        print('optimize Pytorch sparse embedding:', emb_layer.module.sparse_emb)

    for step, blocks in enumerate(dataloader):
        tic_step = time.time()
        sample_time += tic_step - start

        # The nodes for input lies at the LHS side of the first block.
        # The nodes for output lies at the RHS side of the last block.
        batch_inputs = blocks[0].srcdata['features']
        batch_labels = blocks[-1].dstdata['labels']
        batch_labels = batch_labels.long()

        num_seeds += len(blocks[-1].dstdata[dgl.NID])
        num_inputs += len(blocks[0].srcdata[dgl.NID])
        blocks = [block.to(device) for block in blocks]
        batch_labels = batch_labels.to(device)
        # Compute loss and prediction
        start = time.time()
        batch_pred = model(blocks, batch_inputs)
        loss = loss_fcn(batch_pred, batch_labels)
        forward_end = time.time()
        optimizer.zero_grad()
        loss.backward()
        compute_end = time.time()
        forward_time += forward_end - start
        backward_time += compute_end - forward_end

        optimizer.step()
        update_time += time.time() - compute_end

        step_t = time.time() - tic_step
        step_time.append(step_t)
        iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
        if step % args.log_every == 0:
            acc = compute_acc(batch_pred, batch_labels)
            gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
            print('Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB | time {:.3f} s'.format(
                g.rank(), epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc, np.sum(step_time[-args.log_every:])))
        start = time.time()