# Learned Codec Protocol
> Create and experiment a pratical learning-based codec that has reasonable model size, MACs, encode/decode time, memory usage, and other necessary properties, ex: device interoperability, single modle for multiple rates, etc.

## Installation
1. git clone this repo
2. create conda env by 

        conda env create -f ./Requirements/learned_codec_protocol.yml
3. activate env: `Learned_Codec_Protocol`
4. add `compressai.pth` to your python `site-packages` for conda package reference

        # python `site-packages` usually locates at `~/.local/lib/python3.9/site-packages/`
        # the content of `compressai.pth` should be the location of compressai folder which is `YOUR_PATH/Learned_Codec_Protocol`
    
        
5. modify the forward function of `torch.nn.Identity` to 

        # locate at your_env_path/Learned_Codec_Protocol/lib/python3.9/site-packages/torch/nn/modules/linear.py 
        # or you can use IDE to reference

        def forward(self, input: Tensor, *args, **kwargs)
6. download [module weights](https://drive.google.com/drive/folders/1ILaXX1S8ocNuuIIKVoVZkscUGKPlNvfs?usp=sharing) to `./torchDVC/modules`
7. create `keys.json` under `./torchDVC/configs` with your [comet](https://www.comet.com/site/) account info

        {
            "api_key": "XXXXXXXXXXXXXXXX",
            "workspace": "XXXXXXXXXXX"
        }

8. create two environment variables: 
    * `DATASET` - where train/test datasets are located
    * `LOG` - where you want to store train results

## File Introductions

### Plots
* plot.py - plot R-D curves
* rd_plots.py - R-D curves class used by plot.py
* rd_points_IXXGXX_.py - store R-D curves if csv files are unavailable
* *.py - other plot functions

### compressai
> Base on [this repo](https://github.com/InterDigitalInc/CompressAI).

key files:
* entropy_models/entropy_models.py - define the models to calculate the entropy.
* layers/layers.py - define the layers used by other modules.
* models/google.py - define the codecs.
### torchDVC

* `analysis` - some scripts for read and summary csv files (test results)
* `configs` - 
    * model - (csv) store model structure (which moduels are used) 
    * module - (json) store modules arguments referenced by model csv
    * test - (cfg) store test header/process
    * train - (cfg) store train header/process
    * `keys.json` - please write your comet info for logging train process online
* `dataset` - dataloader for train/test
* `models` - define models
* `modules` - define modules
* `util` - some useful tools (old)
* advance_modules.py - define the modules used by model
* coding_structure.py - define the train/test pair used by trainer/tester
* runner.py - parent class of trainer and tester
* trainer.py - train script
* test.py - test script
* utils.py - some functions ex: train/test config parser, ckpt loader, etc.

## Command Example
./torchDVC
* Training : `python trainer.py --lmda 2048 --intra_period 32 --gop_size 1 --model_config baseline.csv --run_config baseline/scratch.cfg --start_epoch 0`

* Test : `python tester.py --lmda 2048 --intra_period 32 --gop_size 1 --model_config baseline.csv --run_config baseline.cfg --no_image --save_dir XXX --restore load --checkpoint XXX`

## Development Process
1. Add new modules in advance_modules.py and create json argument (if needed) under `./torchDVC/configs/modules`
2. Create new model csv file that use your new modules under `./torchDVC/configs/models`
3. Create new process under `./torchDVC/configs/test or train`
> <b>Reuse configs if possible, do not duplicate configs.</b>

> This is one of the benefits. We can easily add new technologies and control the training process (modules to be used/trained, batch size, RNN training or not, and train data etc.) without modifying the code.


### libbpg need to be installed
* Install dependencies:
```
  $ apt-get update -y
  $ apt-get install -y libturbojpeg
  $ apt-get install -y libsdl-image1.2-dev libsdl1.2-dev libjpeg8-dev emscripten
  $ apt-get install cmake yasm
  $ wget -O libpng-1.6.37.tar.xz "https://downloads.sourceforge.net/project/libpng/libpng16/1.6.37/libpng-1.6.37.tar.gz?ts=$(date +%s)"
  $ tar xf libpng-1.6.37.tar.xz
  $ cd libpng-1.6.37
  $ ./configure
  $ make -j
  $ make install
  $ apt-get purge libnuma-dev
```
* Install libbpg
```
  $ wget http://bellard.org/bpg/libbpg-0.9.5.tar.gz
  $ tar xzf libbpg-0.9.5.tar.gz
  $ mv libbpg-0.9.5/ libbpg
  $ cd libbpg 
  $ make -j 8
  $ make install
  $ apt-get install libnuma-dev
```