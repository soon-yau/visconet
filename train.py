import os
import argparse
from share import *
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from visconet.deepfashion import DeepFashionDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from ldm.modules.attention import CrossAttention
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

DEFAULT_CKPT = './models/control_sd21_ini.ckpt'

class SetupCallback(Callback):
    def __init__(self, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "project.yaml"))


def main(args):

    model_config = args.config
    resume_path = args.resume_path
    gpus = args.gpus
    proj_name = args.name
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    num_gpus = len(gpus)
    num_workers = num_gpus * batch_size

    logdir = os.path.join('./logs/', proj_name)

    if resume_path == '':
        resume_path = DEFAULT_CKPT
        reset_crossattn = True
    else:
        reset_crossattn = False

    logger_freq = 1000
    learning_rate = num_gpus * (batch_size / 4) * 5e-5
    sd_locked = True
    only_mid_control = False
    
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # initialize cross attention weights
    if reset_crossattn:
        for name, module in model.control_model.named_modules():
            if isinstance(module, CrossAttention):
                print(f"Found CrossAttention Layer: {name}")
                # Reset parameters of the CrossAttention layer
                if hasattr(module, 'reset_parameters'):
                    with torch.no_grad():
                        for param in module.parameters():
                            module.reset_parameters()  # Reset the parameters of the CrossAttention layer

    config = OmegaConf.load(model_config)

    # data
    dataset = instantiate_from_config(config.dataset.train)
    val_dataset = instantiate_from_config(config.dataset.val)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    
    # callbacks
    logger = ImageLogger(batch_frequency=logger_freq)
    setup_cb = SetupCallback(logdir=logdir, ckptdir=logdir, cfgdir=logdir, config=config)
    save_cb = ModelCheckpoint(dirpath=logdir,
                            save_last=True, 
                            every_n_train_steps=8000, 
                            monitor='val/loss_simple_ema')
    lr_monitor_cb = LearningRateMonitor(logging_interval='step')
    callbacks = [logger, save_cb, setup_cb, lr_monitor_cb]

    strategy = "ddp" if num_gpus > 1 else "auto"
    trainer = pl.Trainer(accelerator="gpu", devices=gpus, strategy=strategy,
                        precision=32, callbacks=callbacks, 
                        accumulate_grad_batches=4,
                        default_root_dir=logdir,
                        val_check_interval=8000,
                        #check_val_every_n_epoch=1,
                        num_sanity_val_steps=1,
                        max_epochs=max_epochs)

    # Train!
    trainer.fit(model, dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')

    # Adding arguments
    parser.add_argument('--name', type=str)
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--gpus', nargs='+', type=int, default=[1])
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=3)
    # Parsing arguments
    args = parser.parse_args()

    # Calling the main function with parsed arguments
    main(args)