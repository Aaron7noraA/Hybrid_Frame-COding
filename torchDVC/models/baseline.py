import torch
from torch import nn
from advance_modules import MODELS 
import json
import pandas as pd
from types import SimpleNamespace
from compressai.entropy_models import EntropyBottleneck
from compressai.models.utils import update_registered_buffers
from warnings import warn


def get_model_config(csv_path):
    table = pd.read_csv(csv_path, index_col=0).transpose().to_dict()
    table = {k.lower(): v for k, v in table.items()}
    return table

def get_module_args(json_path):
    # print("path: ", json_path)
    with open(json_path, 'r') as openfile: 
        args = json.load(openfile, object_hook=lambda d: SimpleNamespace(**d))
    return args

class Baseline(nn.Module):
    def __init__(self, config_path, **kwargs):
        super(Baseline, self).__init__()
        self.module_table = get_model_config(config_path)
        # print("JIJI: ",self.module_table)
        for code, val in self.module_table.items():
            # print("code: ", code)
            if isinstance(val['args_path'], str):
                args = get_module_args(val['args_path'])
            else:
                args = SimpleNamespace()
            if 'lmda' in kwargs:
                args.lmda = kwargs['lmda']
            self.__setattr__(val["name"], MODELS[val["type"]](args))

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck module(s)."""
        loss = {}
        for code, mprop in self.module_table.items():
            net = self.__getattr__(mprop['name'])
            aux_loss = sum(
                m.loss() for m in net.modules() if isinstance(m, EntropyBottleneck)
            )
            if aux_loss > 0.:
                loss[f'aux_{code}'] = aux_loss
        return loss

    def forward(self, phase, inputs, prop, header):
        data = {k: v for k,v in inputs.items()}
        for code in phase:
            m_header = header[code.lower()]
            mprop = self.module_table[code.lower()]
            net = self.__getattr__(mprop['name'])
            
            with torch.set_grad_enabled(code.isupper()):
                net.train(mode=code.isupper())
                net(data, {**m_header, **mprop})

        return data

    def load_RIFE_weight(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.RIFE.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
    def load_state_dict(self, path, rank=0):
        self.load_RIFE_weight(path, -1)
