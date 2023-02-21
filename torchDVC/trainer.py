from comet_ml import Experiment, ExistingExperiment
import argparse, os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataloader import VideoDataBframe, VideoTestDataBframe
from dataset.dataset import DATASETS, SEQUENCES, seq_to_dataset
from utils import logDict, get_prop, seed_everything
from CORE import CORE
from models import Baseline
from types import SimpleNamespace

class Trainer(CORE):

    def __init__(self, args, model, logger):
        super(Trainer, self).__init__(args, model)
        self.logger = logger
        self.cut = False
        self.curr_prop = get_prop(args.start_epoch, self.process)
        self.setup()
        self.configure_optimizers()

    def tqdm_bar(self, mode, pbar, loss):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, phase={self.curr_prop['pmode']}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def setup(self):

        # dataset_root = os.getcwd() + '/DATASET'
        dataset_root = '/home/pc3501/Learned-Codec-Protocol/torchDVC/DATASET/'

        self.transformer = transforms.Compose([
            transforms.RandomCrop((self.args.patch_size, self.args.patch_size)),
            transforms.RandomHorizontalFlip()
        ])

        print("Process: ", self.curr_prop)
        if 'train_cutN' in self.curr_prop:
            self.cut = True
            cutN = self.curr_prop['train_cutN']
        else: 
            self.cut = False
            cutN = None
        self.train_dataset = VideoDataBframe(os.path.join(dataset_root, "vimeo_septuplet/"), 7, transform=self.transformer, cutN=cutN)

        self.val_dataset = VideoTestDataBframe(os.path.join(dataset_root, "TestVideo"), intra_period=self.args.intra_period, 
                                                    mode="short", used_datasets=self.args.test_datasets, used_seqs=self.args.test_seqs)
    def fit(self, current_step=0):
        self.current_step = current_step
        self.current_epoch = self.args.start_epoch

        if not self.args.no_sanity: self.eval()

        for self.current_epoch in range(self.args.start_epoch, self.args.end_epoch+1):
            dataloader = self.train_dataloader()
    
            for batch in (pbar := tqdm(dataloader, ncols=100)):
                loss, aux_loss, log = self.training_step(batch)
                self.optimizer_step(loss + aux_loss)
                self.logger.log_metrics(log, step=self.current_step, epoch=self.current_epoch)
                self.tqdm_bar('train', pbar, loss.detach().cpu())
                self.current_step += 1
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_dir, f"epoch={self.current_epoch}.ckpt"))
            if self.current_epoch % self.args.per_val == 0:
                self.eval()
            self.training_epoch_end()
    
    def training_step(self, batch):
        path, batch = batch
        prop = get_prop(self.current_epoch, self.process)

        # over-write prop['pairs]

        prop['num_frame'] = 7
        prop['test'] = False 

        log_dict_per_idx = self.forward_a_sequence(batch, prop)
        log_dict = logDict()
        for log_ in log_dict_per_idx.values():
            log_dict.extend(log_)

        logs = {}
        for k, v in log_dict.items():
            logs['train/' + k] = torch.mean(torch.tensor(v)).item()

        losses = sum(log_dict['Loss'])/len(log_dict['Loss'])
        logs['train/loss'] = losses.item()

        aux_loss = 0.
        if self.args.aux:
            aux_ = self.model.aux_loss()
            for k, v in aux_.items():
                _, sufix = k.split('_') # aux_XXX
                logs['train/' + k] = v.item()
                aux_loss += v

        return losses, aux_loss, logs

    def optimizer_step(self, loss):
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optim.step()
        self.optim.zero_grad()


    def configure_optimizers(self):
        def aux_param(model, prefix=''):      
            for name, param in model.named_parameters(prefix=prefix, recurse=True):
                if 'quantiles' in name:
                    yield param
        
        def main_param(model, prefix=''):      
            for name, param in model.named_parameters(prefix=prefix, recurse=True):
                if 'quantiles' not in name:
                    yield param

        params = [dict(params=main_param(self.model.__getattr__(mprop['name'])), lr=self.header[code]['lr']) 
                    for code, mprop in self.model.module_table.items()]

        # print("JJJ: ",self.model.module_table.items())
        params.extend([dict(params=aux_param(self.model.__getattr__(mprop['name'])), lr=10*self.header[code]['lr']) 
                         for code, mprop in self.model.module_table.items()])
        self.optim = torch.optim.Adam(params)

    def train_dataloader(self):
        if 'train_cutN' in self.curr_prop or self.cut:
            self.setup(datasets=['train'])
            
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.curr_prop['batch_size'] * self.args.gpu,
                                  num_workers=self.args.num_workers,
                                  persistent_workers=True, 
                                  pin_memory=True,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                                batch_size=1,
                                num_workers=2, # data takes too much memory
                                persistent_workers=True, 
                                pin_memory=True,
                                shuffle=False)
        return val_loader


if __name__ == '__main__':
    seed_everything(888888)
    save_root = os.path.join(os.getenv('LOG', './'), 'torchDVC')
    config_root = "./configs"
    

    # region - args
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model_config",     type=str, default="baseliqe.csv")
    parser.add_argument("--run_config",       type=str, default="scratch.cfg")
    
    parser.add_argument('--experiment_name',  type=str, required=True)
    parser.add_argument('--experiment_key',   type=str, default=None)
    parser.add_argument('--project_name',     type=str, default="Learned_Codec_Protocol")
    parser.add_argument('--lmda',             type=int, default=2048)
    parser.add_argument('--i_lmda',           type=int, default=None)
    parser.add_argument('--intra_period',     type=int, default=32, help="intra period")
    parser.add_argument('--gop_size',         type=int, default=1,  help="gop size")
    parser.add_argument('--patch_size',       type=int, default=256)
    parser.add_argument('--ssim',             action="store_true", help="set MS-SSIM as distortion metrics")
    parser.add_argument('--ssim_factor',      type=float, default=64.)
    parser.add_argument('--aux',              type=int, default=1, help="set to optimize aux loss")
    parser.add_argument('--static',           action='store_true', help="set to use static scene for training")
    parser.add_argument('--static_rate',      type=float, default=0.5, help="chance to use static seqs")

    parser.add_argument('--device',           type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--gpu',              type=int, default=1)
    parser.add_argument('--no_sanity',        action='store_true')
    parser.add_argument('--no_comet',         action='store_true')
    parser.add_argument('--no_image',         action='store_true')
    parser.add_argument('--test_datasets',    type=str, nargs='+', choices=list(DATASETS.keys()), default=['UVG', 'HEVC-B'])
    parser.add_argument('--test_seqs',        type=str, nargs='+', choices=list(SEQUENCES.keys()), default=[])
    parser.add_argument('--per_val',          type=int, default=1)
    parser.add_argument('--per_save',         type=int, default=1)
    parser.add_argument('--restore',          type=str, default=None)
    parser.add_argument('--checkpoint',       type=str, default="")
    parser.add_argument('--not_load_I',       action="store_true", help="do not load I_codec")
    parser.add_argument('--num_workers',      type=int, default=8)
    parser.add_argument('--start_epoch',      type=int, default=0)
    parser.add_argument('--end_epoch',        type=int, default=1000)
    parser.add_argument('--save_dir',         type=str, default="")
    parser.add_argument('--final_copy',       type=str, default="")
    parser.add_argument('--ctx_only',         action="store_true", help="train context only")
    parser.add_argument('--print_model',         action="store_true", help="print model structure")
    parser.add_argument('--ctx_path',       type=str, default="")


    args = parser.parse_args()
    if args.i_lmda is None:
        args.i_lmda = args.lmda
    args.model_config = os.path.join(config_root, "model", args.model_config)
    args.run_config = os.path.join(config_root, "train", args.run_config)

    import json
    with open("./configs/keys.json", 'r') as f:
        personal = json.load(f)

    if args.experiment_key:
        comet_logger = ExistingExperiment(
            api_key=personal['api_key'],
            experiment_key=args.experiment_key,
            disabled=args.no_comet
        )
    else:
        comet_logger = Experiment(
            api_key=personal['api_key'],
            workspace=personal['workspace'],
            project_name=args.project_name,
            disabled=args.no_comet
        )

    exp_name = f"{args.experiment_name}"

    if args.save_dir == "":
        args.save_dir = os.path.join(save_root, args.project_name, f"{exp_name}_{comet_logger.get_key()}")

    if ~args.no_comet:
        os.makedirs(args.save_dir, exist_ok=True)

    from utils import update_device
    update_device(args.device)

    # currently I only use RIFE model
    model = Baseline(args.model_config, lmda=args.lmda).to(args.device)
    # print(model)
    # model.load_state_dict('modules/RIFE/train_log', -1)

    # if args.checkpoint != "":
    #     from utils import load_ckpt
    #     step = load_ckpt(args.checkpoint, model, args.not_load_I, args.restore, optim=trainer.optim)
    # else:
    #     print("Train from scratch")

    if args.print_model:
        print(model)
    # for name, v in model.named_parameters():
    #     print(name)

    
    trainer = Trainer(args, model, comet_logger)

    from utils import show_model_size
    show_model_size(model)

    comet_logger.set_name(exp_name)
    comet_logger.log_parameters(args)

    # for code, mprop in model.module_table.items():
    #      if isinstance(mprop['args_path'], str):
    #         comet_logger.log_code(mprop['args_path'])

    # for file in ["runner.py", "advance_modules.py", "coding_structure.py", "utils.py",
    #              args.model_config, args.run_config]:
    #     comet_logger.log_code(file)

    # for folder in ["./dataset", "./models", "./modules"]:
    #     comet_logger.log_code(folder=folder)

    trainer.fit(step if args.experiment_key else 0)
    if args.final_copy:
        os.makedirs(args.final_copy, exist_ok=True)
        trainer.save(os.path.join(args.final_copy, f"model_{args.lmda}.ckpt"))