import os, subprocess
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData
from torch.utils.data import DataLoader
from torchvision import transforms
from random import random
from torchvision.datasets.folder import default_loader as imgloader
to_tensor = transforms.ToTensor()

from dataset.dataset import DATASETS, seq_to_dataset

class VideoDataBframe(torchData):

    def __init__(self, root, numFrame, transform, cutN=None):
        super().__init__()
        print("ROOT: ", root)
        self.folder = glob(os.path.join(root, 'img/*/*/'))
        self.numFrame = numFrame
        self.transform = transform
        self.cutN = cutN
        print(f"Training dataset size: {len(self.folder[:self.cutN])}")

    def __len__(self):
        return len(self.folder[:self.cutN])

    def __getitem__(self, index):
        path = self.folder[index]
        imgs = []
        bpgs = []
        bpgs_rate = []

        for f in range(self.numFrame):
            file = path + str(f) + '.png'
            img = imgloader(file)
            imgs.append(to_tensor(img))
        
        frame = torch.stack(imgs)
        frame = self.transform(frame)

        return path, frame

class VideoTestDataBframe(torchData):

    def __init__(self, root, intra_period=32, mode="normal", used_datasets=['UVG', 'HEVC-B'], used_seqs=[], transform=None):
        super(VideoTestDataBframe, self).__init__()

        self.root = root
        self.transform = transform
        
        self.seq_len = {}
        for dataset, seqs in DATASETS.items():
            if dataset in used_datasets:
                for seqName, prop in seqs.items():
                    if used_seqs and seqName not in used_seqs:
                        continue
                    self.seq_len[seqName] = [((prop['frameNum'] - 1) // intra_period) * intra_period + 1, dataset]

        self.intra_list = []
        for seqName, (seqLen, dataset_name) in self.seq_len.items():
            if mode == "short":
                intra_num = 1
            else:
                intra_num = seqLen // intra_period

            for intra_idx in range(intra_num):
                self.intra_list.append([seqName, dataset_name,
                                        1 + intra_period * intra_idx,
                                        1 + intra_period * (intra_idx + 1)])

        # print("seq len: ",self.intra_list)

    def __len__(self):
        return len(self.intra_list)

    def __getitem__(self, idx):
        seq_name, dataset_name, frame_start, frame_end = self.intra_list[idx]
        # print(seq_name, dataset_name, frame_start, frame_end)
        seq_len = self.seq_len[seq_name]
        
        imgs = []
   
        for frame_idx in range(frame_start, frame_end + 1): # +1 => buffer future I-frame
            raw_path = f"{self.root}/raw_video_1080/{dataset_name}/{seq_name}/frame_{frame_idx}.png"
            imgs.append(to_tensor(imgloader(raw_path)))


        return seq_name, self.transform(stack(imgs))

class VideoDataIframe(torchData):

    def __init__(self, root, numFrame, qp, transform, cutN=None, static=False, static_rate=0.5):
        super().__init__()
        self.folder = glob(os.path.join(root, 'img/*/*/'))
        self.qp = qp
        self.numFrame = numFrame
        self.transform = transform
        self.cutN = cutN
        self.static = static
        self.static_rate = static_rate
        print(f"Training dataset size: {len(self.folder[:self.cutN])}")

    def __len__(self):
        return len(self.folder[:self.cutN])

    def __getitem__(self, index):
        path = self.folder[index]
        imgs = []
        bpgs = []
        bpgs_rate = []

        for f in range(self.numFrame):
            file = path + str(f) + '.png'
            img = imgloader(file)
            imgs.append(to_tensor(img))
            
            if f == 0:
                bpg_path = os.path.join(path.replace('img', 'IframeBPG'), f'IframeBPG_QP{self.qp}_{f+1}.png')

                # Compress data on-the-fly when they are not previously compressed.
                if not os.path.exists(bpg_path):     
                    # compress bin file anyway if there is not image, because the bin file may be wrong.
                    bin_path = os.path.join(path.replace('img', 'IframeBPG_bin'), f'IframeBPG_QP{self.qp}_{f+1}.bin')
                    print(f"Encode {bin_path} from {file}")
                    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
                    subprocess.call(f'bpgenc -f 444 -q {self.qp} -o {bin_path} {file}'.split(' '))
                    
                    print(f"Decode {bpg_path} from {bin_path}")               
                    os.makedirs(os.path.dirname(bpg_path), exist_ok=True)       
                    subprocess.call(f'bpgdec -o {bpg_path} {bin_path}'.split(' '))

                bpg = imgloader(bpg_path)
                bpgs.append(to_tensor(bpg))
                bpgs_rate.append(-1.)
        
        frame = torch.stack([*imgs, *bpgs])
        frame = self.transform(frame)
        imgs, bpgs = frame.split([self.numFrame, 1], dim=0)

        if self.static and random() < self.static_rate:
            imgs[1:] = imgs[0]

        return path, imgs, bpgs, torch.tensor(bpgs_rate)


class VideoTestDataIframe(torchData):

    def __init__(self, root, qp, intra_period=32, mode="normal", used_datasets=['UVG', 'HEVC-B'], used_seqs=[]):
        super(VideoTestDataIframe, self).__init__()

        self.root = root
        self.qp = qp
        
        self.seq_len = {}
        for dataset, seqs in DATASETS.items():
            if dataset in used_datasets:
                for seqName, prop in seqs.items():
                    if used_seqs and seqName not in used_seqs:
                        continue
                    self.seq_len[seqName] = [((prop['frameNum'] - 1) // intra_period) * intra_period + 1, dataset]

        self.intra_list = []
        for seqName, (seqLen, dataset_name) in self.seq_len.items():
            if mode == "short":
                intra_num = 1
            else:
                intra_num = seqLen // intra_period

            for intra_idx in range(intra_num):
                self.intra_list.append([seqName, dataset_name,
                                        1 + intra_period * intra_idx,
                                        1 + intra_period * (intra_idx + 1)])

    def __len__(self):
        return len(self.intra_list)

    def __getitem__(self, idx):
        seq_name, dataset_name, frame_start, frame_end = self.intra_list[idx]
        seq_len = self.seq_len[seq_name]
        
        imgs = []
        bpgs = []
        bpgs_rate = []
        state = []
        if frame_start == 1:
            state.append("first")
        
        if frame_end == seq_len:
            state.append("last")
   
        for frame_idx in range(frame_start, frame_end + 1): # +1 => buffer future I-frame
            raw_path = f"{self.root}/raw_video_1080/{dataset_name}/{seq_name}/frame_{frame_idx}.png"
            if frame_idx == frame_start or frame_idx == frame_end:
                bin_path = f"{self.root}/bpg/{self.qp}/bin/{dataset_name}/{seq_name}/frame_{frame_idx}.bin"
                img_path = f"{self.root}/bpg/{self.qp}/decoded/{dataset_name}/{seq_name}/frame_{frame_idx}.png"
                
                # Compress data on-the-fly when they are not previously compressed.
                
                if not os.path.exists(img_path):    
                    print(f'bpgenc -f 444 -q {self.qp} -o {bin_path} {raw_path}'.split(' ')) 
                    # compress bin file anyway if there is not image, because the bin file may be wrong.
                    print(f"Encode {bin_path} from {raw_path}")
                    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
                    subprocess.call(f'bpgenc -f 444 -q {self.qp} -o {bin_path} {raw_path}'.split(' '))
                    
                    print(f"Decode {img_path} from {bin_path}")               
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)       
                    subprocess.call(f'bpgdec -o {img_path} {bin_path}'.split(' '))

                bpgs.append(to_tensor(imgloader(img_path)))
                h, w = bpgs[-1].size()[-2:]
                rate = os.path.getsize(bin_path) * 8 / h / w
                bpgs_rate.append(rate)

            imgs.append(to_tensor(imgloader(raw_path)))


        return seq_name, stack(imgs), stack(bpgs), torch.tensor(bpgs_rate), frame_start, state


def getTestDatasets(root, qp, intra_period=32, mode="normal", used_datasets=['UVG', 'HEVC-B'], used_seqs=[], max_frame=-1, crop_size=None):
    datasets = {k: {} for k in used_datasets}
    for dataset_name, seqs in DATASETS.items():
        if dataset_name in used_datasets:
            for seq_name, prop in seqs.items():
                if used_seqs and seq_name not in used_seqs:
                    continue

                num_frame = prop['frameNum'] if max_frame == -1 else max_frame
                if mode == "short":
                    seq_len = intra_period + 1
                    intra_num = 1
                else:
                    seq_len = ((num_frame - 1) // intra_period) * intra_period + 1
                    intra_num = seq_len // intra_period

                datasets[dataset_name][seq_name] = VideoTestData(root, dataset_name, seq_name, qp, intra_period, intra_num, crop_size)

    return datasets


class VideoTestData(torchData):

    def __init__(self, root, dataset_name, seq_name, qp, intra_period, intra_num, crop_size=None):
        super(VideoTestData, self).__init__()

        self.root = root
        self.dataset_name = dataset_name
        self.seq_name = seq_name
        self.prop = DATASETS[dataset_name][seq_name]
        self.qp = qp
        self.intra_period = intra_period
        self.intra_num = intra_num        
        if crop_size:
            self.crop = True
            self.croper = transforms.CenterCrop(crop_size)
        else:
            self.crop = False

    def __len__(self):
        return self.intra_num

    def __getitem__(self, idx):        
        imgs = []
        bpgs = []
        bpgs_rate = []
        state = []
        if idx == 0:
            state.append("first")
        
        if idx == self.intra_num - 1:
            state.append("last")

        frame_start = idx * self.intra_period + 1 # +1 => frame idx: 1 ~ seq_len
        frame_end = (idx + 1) * self.intra_period + 1
        for frame_idx in range(frame_start, frame_end + 1): # +1 => buffer future I-frame
            raw_path = f"{self.root}/raw_video_1080/{self.dataset_name}/{self.seq_name}/frame_{frame_idx}.png"
            if frame_idx == frame_start or frame_idx == frame_end:
                bin_path = f"{self.root}/bpg/{self.qp}/bin/{self.dataset_name}/{self.seq_name}/frame_{frame_idx}.bin"
                img_path = f"{self.root}/bpg/{self.qp}/decoded/{self.dataset_name}/{self.seq_name}/frame_{frame_idx}.png"
                
                # Compress data on-the-fly when they are not previously compressed.
                if not os.path.exists(img_path):  
                    # compress bin file anyway if there is not image, because the bin file may be wrong. 
                    print(f"Encode {bin_path} from {raw_path}")
                    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
                    subprocess.call(f'bpgenc -f 444 -q {self.qp} -o {bin_path} {raw_path}'.split(' '))

                    print(f"Decode {img_path} from {bin_path}")               
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)       
                    subprocess.call(f'bpgdec -o {img_path} {bin_path}'.split(' '))

                bpgs.append(to_tensor(imgloader(img_path)))
                h, w = bpgs[-1].size()[-2:]
                rate = os.path.getsize(bin_path) * 8 / h / w
                bpgs_rate.append(rate)

            imgs.append(to_tensor(imgloader(raw_path)))

        imgs = stack(imgs)
        if self.crop:
            imgs = self.croper(imgs)

        return imgs, stack(bpgs), torch.tensor(bpgs_rate), frame_start, state


if __name__ == "__main__":

    dataset_root = os.getenv('DATASET')
    # test_dataset = VideoTestDataIframe(os.path.join(dataset_root, "TestVideo"), 2048, intra_period=32)

    # test_loader = DataLoader(test_dataset,
    #                          batch_size=1,
    #                          num_workers=0,
    #                          shuffle=False)

    # for seq_name, img, bpg, start_id, state in test_loader:
    #     print(seq_name, img.size(), bpg.size(), start_id, state)

    datasets = getTestDatasets(os.path.join(dataset_root, "TestVideo"), 2048, intra_period=32, mode="short", used_datasets=['UVG', 'HEVC-B'])
    from torchvision.utils import save_image

    for dataset_name, seq_datasets in datasets.items():
        for seq_name, dataset in seq_datasets.items():
            test_loader = DataLoader(dataset,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False)

            for i, (img, bpg, start_id, state) in enumerate(test_loader):
                print(seq_name, img.size(), bpg.size(), start_id, state)
                
                save_image(img[0], f"./img_{i}.png")
                save_image(bpg[0], f"./bpg_{i}.png")