import sys, os, copy
from warnings import warn
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random
from compressai.zoo import cheng2020_anchor, mbt2018
from modules import SPyNet, GS, DCVC_GS, Refinement, ShrinkRefinement, FlowRefinement, DeepRefinement, RNNRefinement
from util.warp import torch_warp as sampler
from util.alignment import Alignment
from utils import space_to_depth, depth_to_space
from modules.Codecs import get_coder_from_args
from modules import IFNet

use_sub_motion = False

class RIFE(nn.Module):
    def __init__(self, args):
        super(RIFE, self).__init__()
        self.RIFE = IFNet()
        self.load_RIFE_weight('modules/RIFE/train_log') # change it later
    def load_RIFE_weight(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        # if rank <= 0:
        self.RIFE.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
    def forward(self, data, prop):
        scale=1
        scale_list=[4, 2, 1]
        TTA=False
        timestep=0.5,

        img0 = data['x1']
        B, C, H, W = img0.shape
        assert H%16 == 0 and W%16 == 0, "latent shape is wrong"
        y_h, y_w = H//16, W//16
        assert H%64 == 0 and W%64 == 0, "hyper-prior latent shape is wrong"
        z_h, z_w = y_h//4, y_w//4
        img1 = data['x2']
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.RIFE(imgs, scale_list, timestep=timestep)
        if TTA == False:
            data['x_hat'] = merged[2]
            data['likelihood'] = {'y':torch.zeros((B, C, y_h, y_w)).to(img0.device),
                                        'z':torch.zeros((B, C, z_h, z_w)).to(img0.device)}
            # return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.RIFE(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2

class BPG_Codec(nn.Module):

    def __init__(self, args) -> None:
        super(BPG_Codec, self).__init__()

    def forward(self, data, param) -> None:   
        pass
        

class Intra_Codec(nn.Module):

    def __init__(self, args) -> None:
        super(Intra_Codec, self).__init__()
        self.aligner = Alignment(divisor=64., mode='pad', padding_mode='replicate')
        if args.intra_mode == "cheng2020":
            self.network = cheng2020_anchor(getattr(args.lmda_to_quality, str(args.lmda)), metric='mse', pretrained=True, progress=True)
        elif args.intra_mode == "mbt2018":
            self.network = mbt2018(3, metric='mse', pretrained=True, progress=True)
        else:
            raise ValueError(f"Do not support intra mode = {args.intra_mode}")

    def forward(self, data, param) -> None: 
        # print("ENTER INTRA")  
        x = self.aligner.align(data['x1'])
        output = self.network(x)
        target_hat = self.aligner.resume(output['x_hat'])
        # data['xt_frame'] = target_hat.clamp(0., 1.)
        data['x_hat'] = target_hat.clamp(0., 1.)
        data['likelihood'] = output['likelihoods']


class MENet(nn.Module):

    def __init__(self, args):
        super(MENet, self).__init__()
        self.aligner = Alignment(divisor=16., mode='pad', padding_mode='replicate')
        self.network = SPyNet(level=4, trainable=False)

    def forward(self, data, param):
        x1 = self.aligner.align(data['x1'])
        xt = self.aligner.align(data['x2'])
        esti_flow = self.network(x1, xt)
        esti_flow = self.aligner.resume(esti_flow)
        data['esti_flow'] = esti_flow

        # if param['e_has_frame'] or param['test']:
        #     if use_sub_motion and param['sub']:
        #         data['sub~a-mc_frame'] = sampler(data['x1_a'], data['esti_flow']).clamp(0., 1.)
        #     else:
        #         data['mc_frame'] = sampler(data['x1'], data['esti_flow']).clamp(0., 1.)


class MCNet(nn.Module):

    def __init__(self, args):
        super(MCNet, self).__init__()
        self.aligner = Alignment(divisor=4., mode='pad', padding_mode='replicate')
        self.args = args
        self.freeze = False
        if args.mc_mode == "normal":
            self.network = Refinement(6, 64, 3)
            self.network.load_state_dict(torch.load("./modules/weights/mc.pth"))
        elif args.mc_mode == "shrink":
            self.network = ShrinkRefinement(6, 32, 64, 3)
            self.network.load_state_dict(torch.load("./modules/weights/smc.pth"))
        elif args.mc_mode == "deep":
            self.network = DeepRefinement(6, 32, 64, 3)
        elif args.mc_mode == "flow":
            self.network = FlowRefinement(level=4)
        elif args.mc_mode == "RNN":
            self.network = RNNRefinement(6, 32, 64, 3)
        else:
            raise ValueError(f"{args.mc} is invalid for MCNet")

    def forward(self, data, param):
        x1 = self.aligner.align(data['x1'])
        x2 = self.aligner.align(data['mc-hat_frame'])
       
        if self.args.mc_mode == "flow":
            flow = self.aligner.align(data['tar-hat_flow'])
            mc_flow = self.network(x1, x2, flow)
            mc_frame = sampler(data['x1'], mc_flow)
            data['mc_flow'] = self.aligner.resume(mc_flow)
        elif self.args.mc_mode == "RNN":
            flow = self.aligner.align(data['tar-hat_flow'])
            mc_frame, hidden = self.network(data['hidden'], flow, x1, x2)
            data['hidden'] = hidden
        else:
            mc_frame = self.network(x1, x2)
        
        mc_frame = self.aligner.resume(mc_frame)
        
        data['fuse_frame'] = mc_frame.clamp(0., 1.)

        if param['test']:
            data['fuse-err_map'] = data['xt'] - data['fuse_frame']

    def freeze_exceptMask(self):
        print("Freeze except Mask")
        self.freeze = True
        for name, m in self.network.named_children():
            if "gate" not in name:
                m.requires_grad_(False)
            else:
                print(f"Train: {name}")

    def unfreeze(self):
        print("Unfreeze")
        self.freeze = False
        self.network.requires_grad_(True)

class MotionCoder(nn.Module):

    def __init__(self, args):
        super(MotionCoder, self).__init__()
        coder = get_coder_from_args(args)
        if hasattr(args, 'downsample_8') and args.downsample_8:
            self.aligner = Alignment(divisor=32., mode='pad', padding_mode='replicate')
        else:
            self.aligner = Alignment(divisor=64., mode='pad', padding_mode='replicate')
        self.args = args

        self.network = coder(in_channels=2, out_channels=2, kernel_size=3)


    def forward(self, data, param):
        flow = self.aligner.align(data['esti_flow'])
        output = self.network(flow)
        target_hat = self.aligner.resume(output['x_hat'])
        data['tar-hat_flow'] = target_hat
        data['likelihood_m'] = output['likelihoods']

        if param['m_has_frame'] or param['test']:
            data['mc-hat_frame'] = sampler(data['x1'], target_hat).clamp(0., 1.)

            if param['test']:
                data['mc-hat-err_map'] = data['xt'] - data['mc-hat_frame']

        if self.args.quant == 'straight' and self.training:
            data['sg_loss'] = self.network.entropy_bottleneck.sg_loss + self.network.gaussian_conditional.sg_loss
            # data['sg_loss'] = self.network.gaussian_conditional.sg_loss

# class MotionCoder(nn.Module):

#     def __init__(self, args):
#         super(MotionCoder, self).__init__()
#         coder = get_coder_from_args(args)
#         if hasattr(args, 'downsample_8') and args.downsample_8:
#             self.aligner = Alignment(divisor=32., mode='pad', padding_mode='replicate')
#         else:
#             self.aligner = Alignment(divisor=64., mode='pad', padding_mode='replicate')
#         self.args = args

#         self.network = coder(in_channels=2, out_channels=2, kernel_size=3)


#     def forward(self, data, param):
#         flow = self.aligner.align(data['esti_flow'])
#         output = self.network(flow)
#         target_hat = self.aligner.resume(output['x_hat'])
#         data['tar-hat_flow'] = target_hat
#         data['likelihood_m'] = output['likelihoods']

#         if param['m_has_frame'] or param['test']:
#             data['mc-hat_frame'] = sampler(data['x1'], target_hat).clamp(0., 1.)

#             if param['test']:
#                 data['mc-hat-err_map'] = data['xt'] - data['mc-hat_frame']

#         if self.args.quant == 'straight' and self.training:
#             data['sg_loss'] = self.network.entropy_bottleneck.sg_loss + self.network.gaussian_conditional.sg_loss
#             # data['sg_loss'] = self.network.gaussian_conditional.sg_loss

class ResidualCoder(nn.Module):

    def __init__(self, args):
        super(ResidualCoder, self).__init__()
        coder = get_coder_from_args(args)
        self.aligner = Alignment(divisor=64., mode='pad', padding_mode='replicate')
        self.args = args
        if args.coding_type in ["res", "swin_gs"]:
            in_ch = 3
        elif args.coding_type in ["cond_res", "condres_res", "gs"]:
            in_ch = 6
        else:
            raise TypeError(f"Don't support coding type: {args.coding_type}")
        

        if args.architecture == "DCVCHyperPriorCoder":
            self.network = coder(in_channels=in_ch, out_channels=3, ks_ana=args.ks_ana, ks_syn=args.ks_syn)
        else:
            self.network = coder(in_channels=in_ch, out_channels=3, kernel_size=args.kernel_size)
        
        self.freeze = False
        if args.coding_type == "gs":
            # self.gs = GS(6, 32)
            self.gs = DCVC_GS(6, 32, 3)

    def forward(self, data, param):
        ref = data['mc-hat_frame'] if 'fuse_frame' not in data else data['fuse_frame']

        if self.args.coding_type == "res":
            res = data['xt'] - ref
        elif self.args.coding_type in ["cond_res", "gs"]:
            res = torch.cat([data['xt'], ref], dim=1)
        elif self.args.coding_type == "condres_res":
            res = data['xt'] - ref
            res = torch.cat([res, ref], dim=1)

        if self.args.coding_type in ["gs", "swin_gs"]:
            if "freeze_exceptGS" in param:
                if not self.freeze:
                    self.freeze_exceptGS()
            else:
                if self.freeze:
                    self.unfreeze()
        else:
            if 'only_entropy' in param:
                if not self.freeze:
                    self.freeze_exceptEntropy()
            else:
                if self.freeze:
                    self.unfreeze()

        if self.args.architecture == "SwinGSHyperPrior":
            output = self.network(self.aligner.align(data['xt']), 
                                  self.aligner.align(data['mc-hat_frame']))
            res_hat = self.aligner.resume(output['x_hat'])
            target_hat = res_hat + data['mc-hat_frame']
            data['x_token'] = output['x_token']
            data['xc_token'] = output['xc_token'] 
            data['tokenc'] = output['tokenc']
            data['gr_token'] = output['gr_token']
        else:
            res = self.aligner.align(res)
            output = self.network(res)
            res_hat = self.aligner.resume(output['x_hat'])

            if self.args.coding_type == "gs":
                target_hat = self.gs([res_hat, ref])
            else:
                target_hat = res_hat + ref
            
            if param['test']: 
                data['res_map'] = res_hat
        
        data['xt_frame'] = target_hat.clamp(0., 1.)
        data['likelihood_r'] = output['likelihoods']

    def freeze_exceptGS(self):
        print("Freeze except GS")
        self.freeze = True
        if self.args.architecture == "SwinGSHyperPrior":
            for name, m in self.network.named_children():
                if name not in ['embed_x', 'embed_xc', 'xc_window_embed', 'xc_kv', 'swin_gr', 'swin_gs', 'unembed']:
                    m.requires_grad_(False)
                else:
                    print(f"Train: {name}")
        else:
            self.network.requires_grad_(False)
            # self.network.g_a[0].requires_grad_(True)

    def freeze_exceptEntropy(self):
        print("Freeze except Entropy")
        self.freeze = True
        for name, m in self.network.named_children():
            if "h_s" not in name and 'gaussian_conditional' not in name: # and 'h_a' not in name and 'entropy_bottleneck' not in name:
                m.requires_grad_(False)
            else:
                print(f"Train: {name}")

    def unfreeze(self):
        print("Unfreeze")
        self.freeze = False
        self.network.requires_grad_(True)


MODELS = {
    "BPG": BPG_Codec,
    "Intra_Codec": Intra_Codec,
    "MENet": MENet,
    "MCNet": MCNet,
    "MotionCoder": MotionCoder,
    "ResidualCoder": ResidualCoder, 
    "RIFE":RIFE,
}