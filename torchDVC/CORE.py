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

class CORE():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.mse_fn = nn.MSELoss(reduction='none')
        self.ssim_fn = MS_SSIM(data_range=1., reduction="none").to(args.device)
        self.header, self.process = get_header_process(args.run_config)

    @torch.no_grad()
    def quality_fn(self, pred, target, no_ssim=False):
        pred = self.frame_postprocess(pred)
        psnr = psnr_fn(pred, target)
        if no_ssim:
            return psnr
        else:
            ssim = self.ssim_fn(pred, target).mean()
            return psnr, ssim

    def distortion_fn(self, pred, target):
        value = (1 - self.ssim_fn(pred, target)) if self.args.ssim else self.mse_fn(pred, target)
        return value

    @staticmethod
    def frame_postprocess(frame):
        return (frame.clamp(0, 1) * 255).round() / 255

    @staticmethod
    def rate_fn(likelihood, input):
        rate_y = estimate_bpp(likelihood['y'], input=input).mean() # mean - over batch channel
        rate_z = estimate_bpp(likelihood['z'], input=input).mean()
        rate = rate_y + rate_z

        return rate, rate_y, rate_z

    def forward_a_sequence(self, batch, prop):
        frame_seq = batch.permute((1,0,2,3,4)).to(self.args.device) # seq, B, C, H, W
        log_dict = logDict()
        recons = [None for _ in range(prop["num_frame"])]
        source = recons if prop['RNN'] else frame_seq

        # print("pairs: ", prop['pairs'])
        for idx, p in enumerate(prop['pairs']):
            # print("process pair: ",p )

            inputs = {}
            param = {'lmda':self.args.lmda}
            if len(p) == 1 and len(prop['imode']): #intra frame
                inputs['x1'] = inputs['xt'] = frame_seq[p[0]]
                param['ftype'] = 'i'
            elif len(p) == 2 and len(prop['pmode']):
                raise NotImplementedError
            elif len(p) == 3 and len(prop['hmode']):
                inputs['x1'] = source[p[0]]
                inputs['xt'] = frame_seq[p[1]]
                inputs['x2'] = source[p[2]]
                param['ftype'] = 'h'
            elif len(p) == 3 and len(prop['bmode']):
                raise NotImplementedError('B frame mode not implemented !')
                # inputs['x1'] = source[p[0]]
                # inputs['xt'] = frame_seq[p[1]]
                # inputs['x2'] = source[p[2]]
                # param['ftype'] = 'b'
            else:
                raise NotImplementedError('No such coding type')
            
            if prop['RNN'] < 2:
                for k, v in inputs.items():
                    inputs[k] = v.detach()

            data = self.model(prop[param['ftype']+'mode'], inputs, {**prop, **param}, self.header)


            # Distoration calculate 
            log_ = self.logging(data, inputs['xt'], {**prop, **param})

            output_idx = p[1] if len(p) > 1 else p[0]
            recons[output_idx] = data['x_hat']
            log_dict[output_idx] = log_
            # print("final dict: ", log_dict)
        return log_dict


    def logging(self, data, target, param):
        log_dict = logDict()
        ftype = param['ftype']
        loss = 0.

        x_hat = data['x_hat']
        likelihood = data['likelihood']

        # PSNR MS-SSIM
        psnr, ssim = self.quality_fn(x_hat, target)
        log_dict.append(f"{ftype}/MS-SSIM", ssim)
        log_dict.append(f"{ftype}/PSNR", psnr)


        # rate estimation
        r, ry, rz = self.rate_fn(likelihood, input=target)
        # log_dict.append(f"{ftype}/rate", r)
        log_dict.append(f"{ftype}/rate_y", ry)
        log_dict.append(f"{ftype}/rate_z", rz)
        log_dict.append(f"{ftype}/Rate", r) # all rate

        # distoration
        distortion = self.distortion_fn(x_hat, target)
        loss += (param['lmda'] * distortion).mean()
        loss += r

        # This is to record total rate & loss in a pair
        log_dict.append(f"Loss", loss)

        return log_dict


            

        

