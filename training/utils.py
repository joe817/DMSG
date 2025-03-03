import random
import numpy as np
import torch
import dgl
from random import sample
import os
import copy
import errno
import json

def assign_hyp_param(args, params):
    if args.method=='lwf':
        args.lwf_args = params
    if args.method == 'bare':
        args.bare_args = params
    if args.method == 'gem':
        args.gem_args = params
    if args.method == 'ewc':
        args.ewc_args = params
    if args.method == 'mas':
        args.mas_args = params
    if args.method == 'twp':
        args.twp_args = params
    if args.method in ['jointtrain', 'joint', 'Joint']:
        args.joint_args = params
    if args.method == 'ergnn':
        args.ergnn_args = params

def str2dict(value):
    return json.loads(value)



def compose_hyper_params(hyp_params):
    hyp_param_list = [{}]
    for hk in hyp_params:
        hyp_param_list_ = []
        hyp_p_current = hyp_params[hk] if isinstance(hyp_params[hk],list) else [hyp_params[hk]]
        for v in hyp_p_current:
            for hk_ in hyp_param_list:
                hk__ = copy.deepcopy(hk_)
                hk__.update({hk: v})
                hyp_param_list_.append(hk__)
        hyp_param_list = hyp_param_list_
    return hyp_param_list

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def set_seed(args=None):
    seed = 1 if not args else args.seed
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)

def remove_illegal_characters(name, replacement='_'):
    for c in ['-', '[' ,']' ,'{', '}', "'", ',', ':', ' ']:
        name = name.replace(c,replacement)
    return name


