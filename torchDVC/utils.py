from random import choice
from copy import deepcopy
import re
import logging
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from compressai.ops.bound_ops import LowerBound

lower_bound = LowerBound(1e-9)

log = logging.getLogger(__name__)

class logDict(dict):

    def __init__(self):
        super(logDict, self).__init__()

    def append(self, k, v):
        if k not in self.keys():
            self.__setitem__(k, [v])
        else:
            self.__getitem__(k).append(v)

    def extend(self, dict2):
        for k, v in dict2.items():
            if k not in self.keys():
                self.__setitem__(k, [])
            
            if isinstance(v, list):
                self.__getitem__(k).extend(v)
            else:
                self.__getitem__(k).append(v)

def seed_everything(seed):
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_ckpt(checkpoint, model, not_load_intra=False, mode="load", **kwargs):
    ckpt = torch.load(checkpoint, map_location=(lambda storage, loc: storage))
    if not_load_intra:
        print(f"Remove I codec params")
        ckpt['state_dict'] = {k: v for k, v in ckpt['state_dict'].items() if 'I_codec' not in k}

    if mode == "resume": # will load optmizer
        model.load_state_dict(ckpt['state_dict'], strict=True)
        kwargs['optim'].load_state_dict(ckpt['optimizer'])
        print("Resume the training from: ", checkpoint)
    elif mode == "load":
        model.load_state_dict(ckpt['state_dict'], strict=True)
        print("Load the model from: ", checkpoint)
    elif mode == "load_same":
        load_dict = {}
        model_dict = model.state_dict()
        for k, v in ckpt['state_dict'].items():
            if k in model_dict:
                if v.size() == model_dict[k].size():
                    load_dict[k] = v
                else:
                    print(f"Excldue parameter: {k}, size diff")
            else:
                print(f"Excldue parameter: {k}, no such key")

        for k in model_dict:
            if not (k in load_dict or 'I_codec' in k and not_load_intra):
                print(f"Miss key: {k}")

        model.load_state_dict(load_dict, strict=False)
        print("Load model without some keys from: ", checkpoint)
    else:
        raise ValueError(f"{mode} is not a valid retore option")

    return ckpt['step'] if 'step' in ckpt else 0
   
from coding_structure import training_pairs
def parse_pair(s, RNN):
    pair_list = []
    pairs = re.split('or', s)
    tp = training_pairs['RNN' if RNN else 'nonRNN']

    for pair in pairs:     
        pair_list.append(tp[pair.strip(" ")])
        
    return tuple(pair_list)

def get_header_process(config_path):
    def get_line(lines):
        for line in lines:
            yield line
      
    def until(line_gen, key: str):
        for line in line_gen:
            if key in line:
                break

    def parse_dict(line):
        raw_props = line.split(',')
        prop = {}
        for raw_prop in raw_props:
            if not raw_prop.strip(): continue
            key, value = raw_prop.split(':')
            value = value.strip()
            
            if value.isnumeric():
                value = int(value)
            elif re.match('^-?\d+(?:\.\d*)$', value) is not None:
                value = float(value)
            elif 'step' in value:
                value = parse_pair(value, prop['RNN'])

            prop[key.strip()] = value

        return prop

    with open(config_path, 'r') as f:
        header = {}
        process = []
        lines = f.read().split('\n')
        line_gen = get_line(lines)

        until(line_gen, "* Header")
        for line in line_gen:
            if "* End" in line: break
            if len(line) != 0 and line[0] != '#':
                code, prop = line.split('-')
                header[code.strip().lower()] = parse_dict(prop)

        until(line_gen, "* Process")
        for line in line_gen:
            if "* End" in line: break
            if len(line) != 0 and line[0] != '#':
                process.append(parse_dict(line))

    return header, process

def get_prop(curr_epoch, process):
    curr_prop = None
    for i, prop in enumerate(process):
        if curr_epoch <= prop['epoch'] or i == (len(process) - 1):
            curr_prop = deepcopy(prop)
            break

    pair = curr_prop['pairs']
    while isinstance(pair, tuple):
        pair = choice(pair)

    if not isinstance(pair[0], list): pair = [pair]
    curr_prop['pairs'] = pair

    return curr_prop

def show_model_size(net):
    print("========= Model Size =========")
    total = 0
    for name, module in net.named_children():
        sum = 0
        for param in module.parameters():
            sum += param.numel()
        total += sum
        print(f"{name}: {sum/1e6:.3f} M params")
    print("==============================")
    print(f"Total: {total/1e6:.3f} M params\n")

def estimate_bpp(likelihood, input=None):

    assert torch.is_tensor(input) and input.dim() > 2
    num_pixels = np.prod(input.size()[-2:])

    if torch.is_tensor(likelihood):
        likelihood = [likelihood]

    lll = 0
    for ll in likelihood:
        lll = lll + lower_bound(ll).log().flatten(1).sum(1)

    return lll / (-np.log(2.) * num_pixels)

def update_device(device):
    lower_bound.to(device)

def space_to_depth(x, size=2):
    b, c, h, w = x.size()
    assert not (h % size or w % size), f"{size} is invalid for {h}x{w}"
    ch_tensors = F.pixel_unshuffle(x, size)
    subs = ch_tensors.view(b, c, size*size, h//size, w//size).permute(0, 2, 1, 3, 4).split(1, dim=1)
    return [s[:, 0] for s in subs] # s.size() = b, 1, c, h, w

def depth_to_space(subs, size=2):
    n = len(subs)
    b, c, h, w = subs[0].size()
    assert n == size * size, f"{n} should equal to {size * size}"
    ch_tensors = torch.stack(subs, dim=1).permute(0, 2, 1, 3, 4).reshape(b, c * n, h, w)
    x = F.pixel_shuffle(ch_tensors, size)
    return x

import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)