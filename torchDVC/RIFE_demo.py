import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
import warnings
from torchvision import transforms
import _thread
import skvideo.io
from queue import Queue, Empty
from modules.RIFE.pytorch_msssim import ssim_matlab
from dataset.dataset import DATASETS, SEQUENCES, seq_to_dataset
from dataset.dataloader import VideoDataBframe, VideoTestDataBframe
import torchvision.transforms as T
import torch.nn as nn
from models import Baseline


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target, psnr=False):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)

        return out

class Tester():

    def __init__(self, args, model=None):
        # super(Tester, self).__init__(args, model)
        # self.process = self.process[0]
        # self.process['pmode'] = self.process['pmode'].lower()
        # self.process['bmode'] = self.process['bmode'].lower()
        self.num_seq = args.num_seq

        # generate coding order
        self.coding_ord = []
        self.gen_coding_seq(1, self.num_seq)
        self.coding_ord = [0, self.num_seq-1] + self.coding_ord
    

    def tqdm_bar(self, mode, pbar):
        pbar.set_description(f"({mode}) phase={self.process['pmode']}", refresh=False)
        pbar.refresh()

    def gen_coding_seq(self, start, end):
        mid = (start+end)//2
        if mid == start:
            return 
        self.coding_ord.append(mid-1)
        self.gen_coding_seq(start, mid)
        self.gen_coding_seq(mid, end)

        

    # @torch.no_grad()
    # def test_one_batch(self, batch):   
    #     for seq_id in batch:




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--img', dest='img', type=str, default=None)
    parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
    parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
    parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
    parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
    parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
    parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
    parser.add_argument('--fps', dest='fps', type=int, default=None)
    parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
    parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
    parser.add_argument('--exp', dest='exp', type=int, default=1)
    parser.add_argument('--demo', dest='demo', type=int, default=1)
    parser.add_argument('--num_seq', dest='num_seq', type=int, default=7)
    parser.add_argument('--cutN', dest='cutN',  default=None)
    parser.add_argument("--model_config",     type=str, default="baseline.csv")
    parser.add_argument('--lmda',             type=int, default=2048)
    parser.add_argument('--device',           type=str, choices=["cuda", "cpu"], default="cuda")

    args = parser.parse_args()

    if args.demo:
        torch.set_grad_enabled(False)

    config_root = "./configs"
    args.model_config = os.path.join(config_root, "model", args.model_config)

    model = Baseline(args.model_config, lmda=args.lmda).to(args.device)
    model.load_state_dict('modules/RIFE/train_log', -1)

    # # import RIFE model 
    # from modules.RIFE.RIFE import *

    # # from modules.RIFE.RIFE import RIFE_Model
    # from utils import show_model_size
    # model = RIFE_Model()
    # model.load_model('modules/RIFE/train_log', -1)
    # model.eval()
    # # print(show_model_size(model.flownet))


    dataset_root = '/home/pc3501/Learned-Codec-Protocol/torchDVC/DATASET/'
    # dataset_root = '/home/pc3501/Learned-Codec-Protocol/torchDVC/DATASET/'

    transformer = transforms.Compose([
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip()
    ])


    train_dataset = VideoDataBframe(os.path.join(dataset_root, "vimeo_septuplet/"), 7, transform=transformer, cutN=args.cutN)
    train_loader = DataLoader(train_dataset, batch_size=3, num_workers=1)
    val_dataset = VideoTestDataBframe(os.path.join(dataset_root, "TestVideo"), intra_period=32, 
                                                mode="short", used_datasets=list(DATASETS.keys()), used_seqs=list(SEQUENCES.keys()))

    path, frame = next(iter(train_loader))
    # print("path: ",path)
    # for img in frame:
    #     print(img.shape)
    for idx, (path,frame) in enumerate(train_loader):
        frame = frame.permute((1,0,2,3,4)).to(args.device)
        img1 = frame[0]
        img2 = frame[2]
        mid = model.RIFE_interpolate(img1, img2, args.scale)
        img = T.ToPILImage()(mid[0])
    
    # print("merge: ",merged[2].shape)
    # img = T.ToPILImage()(merged[2][0])
        img.save(f'out_img/idx{idx}_2.png')
        img = T.ToPILImage()(frame[0][0])
        img.save(f'out_img/idx{idx}_1.png')
        img = T.ToPILImage()(frame[2][0])
        img.save(f'out_img/idx{idx}_3.png')

        crit = RateDistortionLoss()
        out = crit.psnr(mid.cpu(), frame[1].cpu())
        print(out)
    