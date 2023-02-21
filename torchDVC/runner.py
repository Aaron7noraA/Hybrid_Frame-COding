import flowiz as fz
import numpy as np
import torch
from torch import nn
from util.psnr import psnr as psnr_fn
from util.psnr import mse2psnr
from util.ssim import MS_SSIM
from coding_structure import get_coding_pairs
from utils import logDict, get_header_process
from utils import lower_bound, estimate_bpp

"""
TODO: add ssim with smaller window size for sub-image during training
TODO: fix psnr_fn for batch size > 1 (training case)
"""

class Runner():

    def __init__(self, args, model):
        super(Runner, self).__init__()
        self.args = args
        self.forward_ = model
        self.model = model.module if isinstance(model, nn.DataParallel) else model
        self.mse_fn = nn.MSELoss(reduction='none')
        self.ssim_fn = MS_SSIM(data_range=1., reduction="none").to(args.device)
        self.lmda_to_qp = {256: 37, 512: 32, 1024: 27, 2048: 22} 
        
        if args.ssim:
            assert not self.model.sub, "Do not support ssim loss for subimage coding!!!"
            self.args.lmda /= args.ssim_factor
            self.lmda_to_qp = {k/args.ssim_factor: v for k, v in self.lmda_to_qp.items()}

        self.header, self.process = get_header_process(args.run_config)
        self.val_pairs = get_coding_pairs(args.intra_period, args.gop_size)

    def distortion_fn(self, pred, target):
        value = (1 - self.ssim_fn(pred, target)) if self.args.ssim else self.mse_fn(pred, target)
        return value

    @torch.no_grad()
    def quality_fn(self, pred, target, no_ssim=False):
        pred = self.frame_postprocess(pred)
        psnr = psnr_fn(pred, target)
        if no_ssim:
            return psnr
        else:
            ssim = self.ssim_fn(pred, target).mean()
            return psnr, ssim

    @staticmethod
    def frame_postprocess(frame):
        return (frame.clamp(0, 1) * 255).round() / 255

    @staticmethod
    def rate_fn(likelihood, input):
        rate_y = estimate_bpp(likelihood['y'], input=input).mean() # mean - over batch channel
        rate_z = estimate_bpp(likelihood['z'], input=input).mean()
        rate = rate_y + rate_z

        return rate, rate_y, rate_z

    def logging(self, data, target, param):
        log_dict = logDict()
        ftype = param['ftype']
        loss = 0.
        rate = 0.

        for k, v in data.items():
            if 'frame' in k and 'res' not in k:
                prefix, _ = k.split('_') # XXX_frame
                if 'sub~' in k:
                    assert self.model.sub, f"Using sub image - {k} under non-sub framework!!!"
                    mode = k.split('sub~')[-1][0]
                    psnr = self.quality_fn(v, data[f'xt_{mode}'], no_ssim=True)
                    frame = None
                    last_sub  = {'key': k, 'value': v}
                else:
                    psnr, ssim = self.quality_fn(v, target)
                    frame = v
                    log_dict.append(f"{ftype}/MS-SSIM_{prefix}", ssim)
                log_dict.append(f"{ftype}/PSNR_{prefix}", psnr)
                
                
            elif 'map' in k or 'mask' in k:
                log_dict.append(f"{ftype}/{k}_mean", v.mean())
                log_dict.append(f"{ftype}/{k}_std", v.std())
            elif 'likelihood' in k:
                r, ry, rz = self.rate_fn(v, input=target)
                rate += r
                _, sufix = k.split('_') # likelihood_XXX
                log_dict.append(f"{ftype}/rate_{sufix}", r)
                log_dict.append(f"{ftype}/rate_{sufix}_y", ry)
                log_dict.append(f"{ftype}/rate_{sufix}_z", rz)
            elif 'bpg_rate' in k:
                rate += v.mean()
            elif 'flow' in k:
                v = torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
                log_dict.append(f"{ftype}/{k}_mean", v.mean())
                log_dict.append(f"{ftype}/{k}_std", v.std())
            elif 'sg_loss' in k:
                log_dict.append(f"{ftype}/{k}", v)
                loss += v

        data['frame'] = frame
        if not isinstance(data['frame'], torch.Tensor):
            if 'sub_modes' in param:
                distortion = sum(self.distortion_fn(data[f'xt-sub~{mode}_frame' if f'xt-sub~{mode}_frame' in data else f'sub~{mode}-inp_frame'], data[f'xt_{mode}']) 
                                for mode in param['sub_modes']) / len(param['sub_modes'])
            else:
                mode = last_sub['key'].split('sub~')[-1][0]
                distortion = self.distortion_fn(last_sub['value'], data[f'xt_{mode}'])
            psnr = mse2psnr(distortion.mean())
            log_dict.append(f"{ftype}/PSNR", psnr)
        else:
            distortion = self.distortion_fn(data['frame'], target)
            psnr, ssim = self.quality_fn(data['frame'], target)
            log_dict.append(f"{ftype}/PSNR", psnr)
            log_dict.append(f"{ftype}/MS-SSIM", ssim)
        
        if not ('only_rate' in param):
            loss += (param['lmda'] * distortion).mean()
        loss += rate
        log_dict.append(f"{ftype}/rate", rate)
        log_dict.append(f"{ftype}/loss", loss)
        
        return log_dict

    def forward_intra_period(self, batch, bpgs, prop):
        print("prop: ", prop)
        batch = batch.permute([1, 0, 2, 3, 4]).to(self.args.device) # seq, B, C, H, W
        if bpgs:
            bpg = bpgs[0].permute([1, 0, 2, 3, 4]).to(self.args.device) # seq, B, C, H, W
            bpg_rate = bpgs[1].permute([1, 0]).to(self.args.device) # seq, B
        log_dict = logDict()
        recons = [None for _ in range(prop['num_frame'])]
        source = recons if prop['RNN'] else batch

        try:
            use_hidden = hasattr(self.model.MCNet.network, 'init_hidden')
        except:
            use_hidden = False


        if use_hidden:
            S, B, C, H, W = batch.size()
            init_hidden = hidden = self.model.MCNet.network.init_hidden(B, H, W)

        for p in prop['pairs']:
            # region - prepare inputs, targets
            inputs, param = {}, {'lmda': self.args.lmda}

            if len(p) == 1 and len(prop['imode']):
                if self.model.module_table['i']['type'] == 'BPG':
                    inputs['bpg_frame'] = bpg[1 if p[0] else 0]
                    inputs['bpg_rate'] = bpg_rate[1 if p[0] else 0]
                inputs['x1'] = inputs['xt'] = batch[p[0]]
                param['ftype'] = 'I'
            elif len(p) == 2 and len(prop['pmode']):
                param['ftype'] = 'P'
                inputs['x1'] = source[p[0]]
                inputs['xt'] = batch[p[1]]
            elif len(p) == 3 and len(prop['bmode']):
                param['ftype'] = 'B'
                inputs['x1'] = source[p[0]]
                inputs['x2'] = source[p[-1]]
                inputs['xt'] = batch[p[1]]
            else:
                continue
            # endregion

            if use_hidden:
                inputs['hidden'] = hidden if prop['RNN'] else init_hidden

            if prop['RNN'] < 2:
                for k, v in inputs.items():
                    if isinstance(v, tuple):
                        inputs[k] = tuple([i.detach() for i in v])
                    else:
                        inputs[k] = v.detach()

            # # print("modulation: ",f'{param["ftype"].lower()}mode')
            # print("prop: ", prop)
            # print("parameters: ", f'{param["ftype"].lower()}mode')
            data = self.forward_(prop[f'{param["ftype"].lower()}mode'], inputs, {**prop, **param}, self.header)
            # print("DATA:", len(data))
            log_ = self.logging(data, inputs['xt'], {**prop, **param})
            tar_idx = p[1] if len(p) > 1 else p[0]
            recons[tar_idx] = data['frame']
            log_dict[tar_idx] = log_

            if use_hidden and 'hidden' in data:
                hidden = data['hidden']
            
            if 'save_fn' in prop and not self.args.no_image:
                prop['save_fn'](p, prop, data)
        
        return log_dict

    @staticmethod
    def process_saved_data(data):           
        upload_dict = {}            
        for k, v in data.items():
            if 'ref' in k or 'bpg' in k or 'xt' in k:
                continue
            if 'flow' in k or 'frame' in k or 'map' in k or 'mask' in k:
                upload_dict[k] = v
            elif 'likelihood' in k:
                upload_dict[k + "_y"] = v['y']
                upload_dict[k + "_z"] = v['z']   
            # elif 'map' in k:
            #     for i in range(3):
            #         upload_dict[k + f"_{i}"] = v[:, [i]]

        return_dict = {}
        for k, v in upload_dict.items():
            if not isinstance(v, torch.Tensor):
                continue

            if 'flow' in k:
                x = np.transpose(fz.convert_from_flow(v[0].permute(1, 2, 0).cpu().numpy()) / 255, (2, 0, 1))    
                x = torch.from_numpy(x) 
            elif 'likelihood' in k:
                x = lower_bound(v).log() / -np.log(2.)
                x = x[0].mean(dim=0).cpu()
            elif 'map' in k:
                x = v[0].abs().mean(dim=0).cpu()
            else: # frame...
                x = v[0].clamp(0, 1).cpu()
            
            return_dict[k] = x

        return return_dict