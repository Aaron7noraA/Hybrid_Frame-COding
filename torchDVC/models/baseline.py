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

    # def forward(self, phase, inputs, prop, header):
    #     data = {k: v for k,v in inputs.items()}
    #     for code in phase:
    #         m_header = header[code.lower()]
    #         mprop = self.module_table[code.lower()]
    #         net = self.__getattr__(mprop['name'])
            
    #         with torch.set_grad_enabled(code.isupper()):
    #             net.train(mode=code.isupper())
    #             net(data, {**m_header, **mprop})

    #     return data
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
        # for k, m in self.named_modules():
        #     if isinstance(m, EntropyBottleneck):
        #         try:
        #             update_registered_buffers(
        #                 m,
        #                 k,
        #                 ["_quantized_cdf", "_offset", "_cdf_length"],
        #                 state_dict
        #             )
        #         except KeyError as e:
        #             warn(f"Load _cdf fail, no key in state_dict: {e.args}")

        # return super().load_state_dict(state_dict, strict)

# class Baseline(nn.Module):

#     def __init__(self, config_path, **kwargs):
#         super(Baseline, self).__init__()
#         self.module_table = get_model_config(config_path)
#         self.sub = False

#         for code, mprop in self.module_table.items():
#             if isinstance(mprop['args_path'], str):
#                 args = get_module_args(mprop['args_path'])
#             else:
#                 args = SimpleNamespace()
#             if mprop['type'] == "SubImageCoder":
#                 self.sub = True
#             if 'lmda' in kwargs:
#                 args.lmda = kwargs['lmda']
#             self.__setattr__(mprop['name'], MODELS[mprop['type']](args))

#     def aux_loss(self):
#         """Return the aggregated loss over the auxiliary entropy bottleneck module(s)."""
#         loss = {}
#         for code, mprop in self.module_table.items():
#             net = self.__getattr__(mprop['name'])
#             aux_loss = sum(
#                 m.loss() for m in net.modules() if isinstance(m, EntropyBottleneck)
#             )
#             if aux_loss > 0.:
#                 loss[f'aux_{code}'] = aux_loss
#         return loss

#     def update(self, force=False):
#         """Updates the entropy bottleneck(s) CDF values.

#         Needs to be called once after training to be able to later perform the
#         evaluation with an actual entropy coder.

#         Args:
#             force (bool): overwrite previous values (default: False)

#         Returns:
#             updated (bool): True if one of the EntropyBottlenecks was updated.

#         """
#         updated = False
#         for m in self.modules():
#             if not isinstance(m, EntropyBottleneck):
#                 continue
#             rv = m.update(force=force)
#             updated |= rv
#         return updated

#     def forward(self, phase, inputs, param, header):
#         data = {k: v for k, v in inputs.items()} # prevent overwriting the (key: value) pair in inputs
#         for code in phase:
#             # print("code: ", code)
#             # print("PPP: ", self.module_table)
#             # print("grad: ", code.isupper())
#             m_header = header[code.lower()]
#             mprop = self.module_table[code.lower()]
#             net = self.__getattr__(mprop['name'])
#             # print("NAME: ",mprop['name'])
#             with torch.set_grad_enabled(code.isupper()):
#                 net.train(mode=code.isupper())
#                 net(data, {**m_header, **param, 'sub': self.sub})
        
#         return data

#     def load_state_dict(self, state_dict, strict=True):
#         for k, m in self.named_modules():
#             if isinstance(m, EntropyBottleneck):
#                 try:
#                     update_registered_buffers(
#                         m,
#                         k,
#                         ["_quantized_cdf", "_offset", "_cdf_length"],
#                         state_dict
#                     )
#                 except KeyError as e:
#                     warn(f"Load _cdf fail, no key in state_dict: {e.args}")

#         return super().load_state_dict(state_dict, strict)